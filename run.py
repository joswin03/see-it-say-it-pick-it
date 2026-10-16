"""Run the OpenCV + Apertus pipeline on every scene in a folder.

    python run.py --data heldout --out results_heldout.json

Needs Apertus served by vLLM (see notebooks/marimo_gpu.py and configs/opencv_apertus.yaml).
Besides the results file, writes a run directory with the scene graphs, annotated photos,
every raw model reply and a manifest.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import yaml

from src import manifest
from src.pipeline import answer_instruction, perceive_scene
from src.workcell import load_scene

REPO_DIR = Path(__file__).resolve().parent


def apply_override(config: dict, override: str) -> None:
    """Set one nested config field from ``section.key=value``, failing on unknown keys.

    Args:
        config: parsed config, changed in place.
        override: dotted path and YAML value, e.g. ``safety.enabled=false``.
    """
    path, _, value = override.partition("=")
    *sections, key = path.split(".")
    target = config
    for section in sections:
        target = target[section]
    if key not in target:
        raise KeyError(f"unknown config field {path}; available: {sorted(target)}")
    target[key] = yaml.safe_load(value)


def scene_dirs(data_dir: Path) -> list[Path]:
    """Scene folders (those holding an ``rgb.png``), sorted by name.

    Args:
        data_dir: folder of scene folders.

    Returns:
        Sorted list of scene folder paths.
    """
    found = sorted(path for path in data_dir.iterdir() if (path / "rgb.png").exists())
    if not found:
        raise FileNotFoundError(f"no scene folders with rgb.png in {data_dir}")
    return found


def scene_record(perceived: dict) -> dict:
    """JSON-serialisable view of a perceived scene for the run directory.

    Args:
        perceived: output of ``perceive_scene``.

    Returns:
        Dict with objects (without masks), the scene text and the naming log.
    """
    keep = ("id", "name", "pixel", "size_mm", "tray", "is_hand", "on_top_of", "under", "source_ids")
    return {
        "objects": [{key: item[key] for key in keep} for item in perceived["objects"]],
        "text": perceived["text"],
        "naming_log": perceived["naming_log"],
    }


def main() -> None:
    """Parse arguments, run every scene and instruction, write results and the run record."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="folder of scene folders")
    parser.add_argument("--out", type=Path, required=True, help="results JSON to write")
    parser.add_argument("--config", type=Path, default=REPO_DIR / "configs" / "opencv_apertus.yaml")
    parser.add_argument("--base-url", help="override the config's model endpoint")
    parser.add_argument("--runs-dir", type=Path, default=REPO_DIR / "runs")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config field (YAML value); recorded in the manifest",
    )
    parser.add_argument("--limit-scenes", type=int, help="only the first N scenes (debugging)")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for override in args.set:
        apply_override(config, override)
    endpoint = {key: config[key] for key in ("model_id", "base_url", "request_timeout_s")}
    if args.base_url:
        endpoint["base_url"] = args.base_url
    run_dir = args.runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-opencv-{args.data.name}"
    run_dir.mkdir(parents=True)

    results, decision_log, scenes = [], [], {}
    started = time.time()
    for scene_dir in scene_dirs(args.data)[: args.limit_scenes]:
        scene = load_scene(scene_dir)
        gripper_mm = scene.gripper_max_opening_mm or config["gripper_max_opening_mm"]
        perceived = perceive_scene(scene, config, endpoint)
        scenes[scene.name] = scene_record(perceived)
        cv2.imwrite(
            str(run_dir / f"{scene.name}_marked.jpg"),
            cv2.cvtColor(perceived["marked"], cv2.COLOR_RGB2BGR),
        )
        print(f"{scene.name}: {len(perceived['objects'])} objects")
        for instruction_id, _, text in scene.instructions:
            answer, log = answer_instruction(
                instruction_id, text, perceived, config, endpoint, gripper_mm
            )
            results.append(answer)
            decision_log.append({**log, "instruction": text, "answer": answer})
            print(f"  {instruction_id}: {answer['actions'][0]['action']}  {text}")

    submission = {"team": config["team"], "results": results}
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(submission, indent=2))
    temporary.replace(args.out)
    (run_dir / "results.json").write_text(json.dumps(submission, indent=2))
    (run_dir / "scenes.json").write_text(json.dumps(scenes, indent=2))
    (run_dir / "decisions.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in decision_log)
    )
    manifest.write(
        run_dir,
        args.config,
        {
            "data": str(args.data),
            "out": str(args.out),
            "wall_clock_s": round(time.time() - started),
            "endpoint": endpoint,
            "overrides": args.set,
        },
    )
    print(f"{len(results)} answers -> {args.out}; run record in {run_dir}")


if __name__ == "__main__":
    main()
