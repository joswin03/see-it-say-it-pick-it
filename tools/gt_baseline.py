"""The organisers' baseline prompt plus the ground-truth scene graph: Apertus with perfect
perception but none of our planning (no safety screen, no plan checks, no code-set pixels).

Uses ``baseline/baseline.py`` unchanged and inserts the scene before the instruction. Dev only:
reads the ground truth.

    python -m tools.gt_baseline --form json
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from src.reasoning.client import api_key
from src.workcell import load_scene
from tools.gt_scene_graph import OUT, REPO, perceive_ground_truth

sys.path.insert(0, str(REPO / "baseline"))
import baseline as organisers  # PROMPT, encode, ask, parse, rescale


def ask(base_url: str, model: str, image_b64: str, prompt: str, timeout: float) -> str:
    """``baseline.ask`` with the API key; the organisers' copy always sends "Bearer none"."""
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        {"type": "text", "text": prompt},
    ]
    body = {"model": model, "temperature": 0, "max_tokens": 400, "messages": [{"role": "user", "content": content}]}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def scene_text(scene, config: dict, form: str, scale: float) -> str:
    """The ground-truth scene graph as the baseline sees it.

    Args:
        scene: loaded scene.
        config: full pipeline config (for ``describe``).
        form: "text" or "json".
        scale: resize factor of the photo the baseline sends; JSON pixels are given in that photo,
            since the baseline scales the model's pixels back up by 1 / scale.

    Returns:
        The scene description.
    """
    if form == "text":  # no pixels: the model must find the objects in the photo
        return perceive_ground_truth(scene, config, "text")["text"]
    graph = json.loads(OUT.read_text())[scene.name]
    for item in graph["objects"]:
        item["pixel"] = [round(value * scale) for value in item["pixel"]]
    return json.dumps(graph)


def main() -> None:
    """Answer every dev instruction with the baseline prompt and the ground-truth scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form", choices=["text", "json"], required=True)
    parser.add_argument("--data", type=Path, default=REPO / "dataset")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs" / "scratch")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    args = parser.parse_args()
    config = yaml.safe_load((REPO / "configs" / "opencv_apertus.yaml").read_text())
    run_dir = args.runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-baseline-gt-{args.form}-{args.data.name}"
    run_dir.mkdir(parents=True)
    results, raw, scenes = [], [], {}
    for scene_dir in sorted(p for p in args.data.iterdir() if (p / "rgb.png").exists()):
        scene = load_scene(scene_dir)
        image, width, height, scale = organisers.encode(scene.rgb, 1024)  # the baseline's max side
        scenes[scene.name] = scene_text(scene, config, args.form, scale)
        for instruction_id, _, text in scene.instructions:
            prompt = organisers.PROMPT.format(w=width, h=height, text=text).replace(
                "\nInstruction:", f"\nObjects in the scene:\n{scenes[scene.name]}\n\nInstruction:", 1
            )
            reply = ask(args.base_url, args.model, image, prompt, 180)
            answer = organisers.parse(reply)
            answer = organisers.rescale(answer, scale) if isinstance(answer, dict) else {"actions": []}
            answer["instruction_id"] = instruction_id
            results.append(answer)
            raw.append({"instruction_id": instruction_id, "instruction": text, "reply": reply})
            print(f"{instruction_id}: {(answer.get('actions') or [{}])[0].get('action')}  {text}")
    (run_dir / "results.json").write_text(json.dumps({"team": "baseline", "results": results}, indent=2))
    (run_dir / "scenes.json").write_text(json.dumps(scenes, indent=2))
    (run_dir / "results.raw.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in raw))
    print(f"{len(results)} answers; run record in {run_dir}")


if __name__ == "__main__":
    main()
