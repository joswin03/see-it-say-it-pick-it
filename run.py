"""Run the Apertus actor-judge pipeline on every scene in a folder.

python run.py --data heldout --out results_heldout.json
python run.py --data dataset --mock --limit 5
python run.py --data dataset --set loop.max_iterations=1 --name one-round
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src import manifest
from src.config import load_config
from src.graph import build_graph
from src.image import resize_for_model, to_data_url
from src.llm import ApertusClient

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "starter"))
from scene_io import load_scene

DEFAULT_CONFIG = REPO_DIR / "configs" / "actor_critic.yaml"
OUTPUTS_DIR = REPO_DIR / "outputs"
STEPS_PER_ITERATION = 2
FIXED_STEPS = 10


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="folder of scene folders")
    parser.add_argument("--out", type=Path, help="extra copy of results.json, e.g. for submission")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", help="run name, default: config file name")
    parser.add_argument("--set", action="append", default=[], help="override, section.key=value")
    parser.add_argument("--mock", action="store_true", help="no model: fixed replies")
    parser.add_argument("--limit", type=int, help="only the first N instructions")
    return parser.parse_args()


def scene_folders(data_dir: Path) -> list[Path]:
    """Find the scene folders, i.e. sub-folders that contain ``rgb.png``.

    Args:
        data_dir: dataset folder.

    Returns:
        Sorted scene folder paths.
    """
    return sorted(path for path in data_dir.iterdir() if (path / "rgb.png").is_file())


def write_json(path: Path, content: dict) -> None:
    """Write a dict as indented JSON, creating parent folders.

    Args:
        path: output file.
        content: JSON-serialisable dict.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n")


def main() -> None:
    """Run every instruction through the graph and write a run directory with a manifest.

    Returns:
        None.
    """
    arguments = parse_arguments()
    load_dotenv(REPO_DIR / ".env")
    config = load_config(arguments.config, arguments.set)
    run_name = arguments.name or arguments.config.stem
    run_dir = OUTPUTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_name}"
    run_dir.mkdir(parents=True)

    client = ApertusClient(config, mock=arguments.mock)
    graph = build_graph(config, client)
    recursion_limit = STEPS_PER_ITERATION * config["loop"]["max_iterations"] + FIXED_STEPS

    results = []
    remaining = arguments.limit
    started = time.time()
    for scene_dir in scene_folders(arguments.data):
        if remaining is not None and remaining <= 0:
            break
        scene = load_scene(scene_dir)
        image = resize_for_model(scene.rgb, config["image"]["max_side"])
        image_url = to_data_url(image.rgb, config["image"]["jpeg_quality"])
        scene_graph = None
        instructions = scene.instructions[:remaining] if remaining else scene.instructions
        for instruction_id, _category, text in instructions:
            state = graph.invoke(
                {
                    "instruction_id": instruction_id,
                    "instruction": text,
                    "image": image,
                    "image_url": image_url,
                    "scene": scene_graph,
                    "safety": None,
                    "candidate": None,
                    "iteration": 0,
                    "history": [],
                    "result": None,
                },
                {"recursion_limit": recursion_limit},
            )
            if scene_graph is None:
                scene_graph = state["scene"]
            results.append(state["result"])
            last_verdict = state["history"][-1]["judgement"]["verdict"]
            print(
                f"[{len(results)}] {instruction_id}: {state['result']['actions'][0]['action']}"
                f"  rounds={state['iteration']}  last_verdict={last_verdict}"
            )
        if remaining is not None:
            remaining -= len(instructions)

    submission = {"team": config["team"], "results": results}
    write_json(run_dir / "results.json", submission)
    if arguments.out:
        shutil.copyfile(run_dir / "results.json", arguments.out)
    manifest.write(
        run_dir,
        arguments.config,
        {
            "overrides": arguments.set,
            "mock": arguments.mock,
            "data": str(arguments.data),
            "answers": len(results),
            "runtime_s": round(time.time() - started),
        },
    )
    print(f"\n{len(results)} answers in {run_dir / 'results.json'}")
    print(f"score it: .venv/bin/python scripts/score_run.py {run_dir} --data {arguments.data}")


if __name__ == "__main__":
    main()
