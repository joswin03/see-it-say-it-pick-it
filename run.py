"""Run the two-node prompting pipeline (src/graph.py) on every scene in a folder.

    python run.py --data heldout --out results_heldout.json
    python run.py --data dataset --base-url https://api.inference.cscs.ch/v1
    python run.py --data dataset --mock

Writes outputs/<time>-<name>/ with results.json, manifest.json and the annotated photos.
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
from src.graph import build_graph

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR / "starter"))

from scene_io import load_scene

DEFAULT_CONFIG = REPO_DIR / "configs" / "two_node.yaml"
OUTPUTS_DIR = REPO_DIR / "outputs"
JPEG_QUALITY = 90


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="folder of scene folders")
    parser.add_argument("--out", type=Path, help="extra copy of results.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", default="two-node")
    parser.add_argument("--base-url", help="override the config's endpoint")
    parser.add_argument("--model", help="override the config's model_id")
    parser.add_argument("--mock", action="store_true", help="no model: fixed replies")
    return parser.parse_args()


def main() -> None:
    """Annotate every photo once, decide every instruction, write results and a manifest.

    Returns:
        None.
    """
    arguments = parse_arguments()
    load_dotenv(REPO_DIR / ".env")
    config = yaml.safe_load(arguments.config.read_text())
    overrides = {"base_url": arguments.base_url, "model_id": arguments.model}
    config.update({key: value for key, value in overrides.items() if value})
    run_dir = OUTPUTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{arguments.name}"
    (run_dir / "annotated").mkdir(parents=True)

    graph = build_graph(config, Apertus(config, mock=arguments.mock))
    results = []
    started = time.time()
    scene_dirs = sorted(path for path in arguments.data.iterdir() if (path / "rgb.png").is_file())
    for scene_dir in scene_dirs:
        scene = load_scene(scene_dir)
        marks, annotated = None, None
        for instruction_id, _category, text in scene.instructions:
            state = graph.invoke(
                {
                    "instruction_id": instruction_id,
                    "instruction": text,
                    "photo": scene.rgb,
                    "marks": marks,
                    "annotated": annotated,
                    "result": None,
                }
            )
            if marks is None:
                marks, annotated = state["marks"], state["annotated"]
                Image.fromarray(annotated).save(
                    run_dir / "annotated" / f"{scene.name}.jpg", quality=JPEG_QUALITY
                )
                print(f"{scene.name}: {len(marks)} objects marked")
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
            "overrides": {key: value for key, value in overrides.items() if value},
            "mock": arguments.mock,
            "data": str(arguments.data),
            "runtime_s": round(time.time() - started),
        },
    )
    print(f"{len(results)} answers in {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
