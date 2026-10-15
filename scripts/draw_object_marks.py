"""Draw the object marks Apertus gave when the organisers' baseline asked it to label objects.

Reads ``results.raw.jsonl`` from a run of ``baseline/baseline.py`` whose prompt asked for
``{"objects": [{"label", "box"}]}`` (saved as ``prompt.txt`` in the run directory), parses the
first reply per photo with the organisers' ``baseline.parse`` and draws its boxes and labels
on the photo with Pillow. Nothing else is added.

    .venv/bin/python scripts/draw_object_marks.py outputs/20261015-235949-object-marking
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "baseline"))

import baseline

IMAGES_DIR = REPO_DIR / "outputs" / "annotated" / "object-marking"
MODEL_MAX_SIDE_PX = 1024
MARK_RGB = (255, 0, 255)
MARK_WIDTH_PX = 6
FONT_SIZE_PX = 44
JPEG_QUALITY = 90


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="run directory with results.raw.jsonl")
    parser.add_argument("--data", type=Path, default=REPO_DIR / "dataset")
    return parser.parse_args()


def marks(reply_text: str, scale: float) -> list[dict]:
    """Read the labelled boxes from one reply, in full-resolution pixels.

    Args:
        reply_text: raw model reply.
        scale: resized size divided by full size, as in ``baseline.encode``.

    Returns:
        Entries with ``label`` and ``box`` [u_min, v_min, u_max, v_max]; entries without a
        label or a four-number box are skipped.
    """
    reply = baseline.parse(reply_text)
    entries = reply.get("objects", []) if isinstance(reply, dict) else []
    found = []
    for entry in entries:
        box = entry.get("box") if isinstance(entry, dict) else None
        if not (isinstance(box, list) and len(box) == 4 and isinstance(entry.get("label"), str)):
            continue
        if not all(isinstance(value, (int, float)) for value in box):
            continue
        found.append({"label": entry["label"], "box": [round(value / scale) for value in box]})
    return found


def draw(photo: Image.Image, found: list[dict]) -> Image.Image:
    """Draw Apertus' labelled boxes on a copy of the photo.

    Args:
        photo: full-resolution photo.
        found: labelled boxes, full resolution.

    Returns:
        Annotated copy of the photo.
    """
    image = photo.convert("RGB")
    canvas = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=FONT_SIZE_PX)
    for mark in found:
        u_min, v_min, u_max, v_max = mark["box"]
        box = [min(u_min, u_max), min(v_min, v_max), max(u_min, u_max), max(v_min, v_max)]
        canvas.rectangle(box, outline=MARK_RGB, width=MARK_WIDTH_PX)
        canvas.text((box[0], box[1] - FONT_SIZE_PX - 6), mark["label"], fill=MARK_RGB, font=font)
    return image


def main() -> None:
    """Draw the first reply of every photo.

    Returns:
        None.
    """
    arguments = parse_arguments()
    first_reply: dict[str, str] = {}
    for line in (arguments.run_dir / "results.raw.jsonl").read_text().splitlines():
        record = json.loads(line)
        first_reply.setdefault(record["instruction_id"].rsplit("_i", 1)[0], record["reply"])
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for scene_name, reply in sorted(first_reply.items()):
        photo = Image.open(arguments.data / scene_name / "rgb.png")
        scale = min(1.0, MODEL_MAX_SIDE_PX / max(photo.size))
        found = marks(reply, scale)
        draw(photo, found).save(IMAGES_DIR / f"{scene_name}.jpg", quality=JPEG_QUALITY)
        labels = ", ".join(mark["label"] for mark in found) or "(reply could not be parsed)"
        print(f"{scene_name}: {labels}")


if __name__ == "__main__":
    main()
