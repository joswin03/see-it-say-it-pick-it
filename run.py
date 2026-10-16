"""Run the prompting pipeline (src/graph.py) on every scene in a folder.

    python run.py --data heldout --out results_heldout.json
    python run.py --data dataset --base-url https://api.inference.cscs.ch/v1
    python run.py --data dataset --mock
    python run.py --data dataset --set stack_min_overlap=0.5

Writes outputs/<time>-<name>/ with results.json, manifest.json, the annotated photos, the scene
graphs in the format shared with the ground truth (scene_graphs.json) and one figure per scene
(scene_graphs/).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from PIL import Image

from src import manifest
from src.apertus import Apertus
from src.calibration import lens_distortion
from src.graph import build_graph
from src.scene_graph import graph_record
from src.scene_graph_figure import write_run_figure

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "starter"))

from scene_io import load_scene

DEFAULT_CONFIG = REPO_DIR / "configs" / "pipeline.yaml"
OUTPUTS_DIR = REPO_DIR / "outputs"
JPEG_QUALITY = 90


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Args:
        None.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="folder of scene folders")
    parser.add_argument("--out", type=Path, help="extra copy of results.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", default="pipeline")
    parser.add_argument("--base-url", help="override the config's endpoint")
    parser.add_argument("--model", help="override the config's model_id")
    parser.add_argument("--mock", action="store_true", help="no model: fixed replies")
    parser.add_argument(
        "--set", action="append", default=[], help="override any field, section.key=value (YAML)"
    )
    return parser.parse_args()


def main() -> None:
    """Annotate every photo once, decide every instruction, write results and a manifest.

    Args:
        None.

    Returns:
        None.
    """
    arguments = parse_arguments()
    load_dotenv(REPO_DIR / ".env")
    config = yaml.safe_load(arguments.config.read_text())
    overrides = {"base_url": arguments.base_url, "model_id": arguments.model}
    overrides = {key: value for key, value in overrides.items() if value}
    for setting in arguments.set:
        key, separator, value = setting.partition("=")
        if not separator:
            raise KeyError(f"invalid --set {setting!r}; expected section.key=value")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(f"unknown config field in --set {setting!r}")
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"unknown config field in --set {setting!r}")
        target[parts[-1]] = yaml.safe_load(value)
        overrides[key] = target[parts[-1]]
    config.update({key: value for key, value in overrides.items() if "." not in key})
    run_dir = OUTPUTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{arguments.name}"
    (run_dir / "annotated").mkdir(parents=True)
    (run_dir / "scene_graphs").mkdir()

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    graph = build_graph(
        config, Apertus(config, mock=arguments.mock, log_path=run_dir / "model_replies.jsonl")
    )
    scene_records = {}
    graph_records = {}
    results = []
    started = time.time()
    scene_dirs = sorted(path for path in arguments.data.iterdir() if (path / "rgb.png").is_file())
    if not scene_dirs:
        raise ValueError(f"no scene folders containing rgb.png in {arguments.data}")
    for scene_dir in scene_dirs:
        scene = load_scene(scene_dir)
        distortion = lens_distortion(scene_dir)
        marks, annotated, stacking, on_top = None, None, None, None
        for instruction_id, _category, text in scene.instructions:
            state = graph.invoke(
                {
                    "instruction_id": instruction_id,
                    "instruction": text,
                    "scene": scene,
                    "distortion": distortion,
                    "photo": scene.rgb,
                    "marks": marks,
                    "annotated": annotated,
                    "stacking": stacking,
                    "on_top": on_top,
                    "result": None,
                }
            )
            if marks is None:
                marks, annotated = state["marks"], state["annotated"]
                stacking, on_top = state["stacking"], state["on_top"]
                Image.fromarray(annotated).save(
                    run_dir / "annotated" / f"{scene.name}.jpg", quality=JPEG_QUALITY
                )
                print(f"{scene.name}: {len(marks)} objects marked")
                graph_records[scene.name] = graph_record(marks, on_top)
                (run_dir / "scene_graphs.json").write_text(
                    json.dumps(graph_records, indent=2) + "\n"
                )
                write_run_figure(
                    annotated,
                    graph_records[scene.name],
                    state["scene_text"],
                    scene.name,
                    run_dir / "scene_graphs" / f"{scene.name}.png",
                )
            scene_records[scene.name] = {
                "objects": [
                    {key: value for key, value in mark.items() if key != "mask"} for mark in marks
                ],
                "on_top": on_top,
                "text": state.get("scene_text", ""),
            }
            (run_dir / "scenes.json").write_text(json.dumps(scene_records, indent=2) + "\n")
            with (run_dir / "decisions.jsonl").open("a") as instruction_log:
                instruction_log.write(
                    json.dumps(
                        {"instruction_id": instruction_id, "attempts": state.get("attempts", [])}
                    )
                    + "\n"
                )
            results.append(state["result"])
            print(f"  {instruction_id}: {state['result']['actions'][0]['action']}  {text}")

    (run_dir / "results.json").write_text(
        json.dumps({"team": config["team"], "results": results}, indent=2) + "\n"
    )
    if arguments.out:
        shutil.copyfile(run_dir / "results.json", arguments.out)
    manifest.write(
        run_dir,
        arguments.config,
        {
            "overrides": overrides,
            "mock": arguments.mock,
            "data": str(arguments.data),
            "effective_config": config,
            "runtime_s": round(time.time() - started),
        },
    )
    print(f"{len(results)} answers in {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
