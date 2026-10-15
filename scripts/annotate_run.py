"""Draw a run's own answers on every scene photo; nothing else is added.

Each photo shows a numbered marker at every answer's target_pixel (green: on the right
object, red: wrong) and a side panel with each instruction, the answer and the verdict.
Verdicts come from a report written by ``starter/evaluate.py --report``, so this script is
for diagnosis only and never part of the pipeline. No model is called; drawing uses Pillow.

    .venv/bin/python scripts/annotate_run.py outputs/<run>/report.json \\
        --data dataset --out-dir outputs/annotated/<run-name>
"""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_DIR = Path(__file__).resolve().parent.parent
RIGHT_RGB = (0, 170, 0)
WRONG_RGB = (220, 0, 0)
TEXT_RGB = (0, 0, 0)
WHITE_RGB = (255, 255, 255)
MARKER_RADIUS_PX = 34
MARKER_OUTLINE_PX = 4
MARKER_FONT_PX = 40
PANEL_WIDTH_PX = 1400
PANEL_MARGIN_PX = 40
PANEL_FONT_PX = 32
PANEL_LINE_PX = 44
PANEL_WRAP_CHARACTERS = 62
TITLE_FONT_PX = 46
JPEG_QUALITY = 90


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("report", type=Path, help="report.json from starter/evaluate.py --report")
    parser.add_argument("--data", type=Path, default=REPO_DIR / "dataset")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def answer_line(number: int, entry: dict) -> tuple[str, bool]:
    """Write one legend entry for an instruction.

    Args:
        number: marker number of the instruction in this scene.
        entry: one instruction of the evaluation report.

    Returns:
        The legend text and whether the answer scored.
    """
    right = entry["score"] > 0
    verdict = "RIGHT" if right else "WRONG"
    answer = entry.get("plan") or entry["answer"] or entry.get("error") or "no answer"
    text = (
        f'{number}. [{entry["category"]}] "{entry["instruction"]}" -> {answer}'
        f" | expected: {entry['expected']} | {verdict}"
    )
    if entry.get("pixel_on") is not None or entry.get("target_pixel"):
        text += f" (pixel on: {entry.get('pixel_on') or 'nothing'})"
    return text, right


def draw_markers(photo: Image.Image, entries: list[dict]) -> None:
    """Draw a numbered, coloured marker at every answer's target_pixel, in place.

    Args:
        photo: full-resolution photo.
        entries: report instructions of this scene, in order.

    Returns:
        None.
    """
    canvas = ImageDraw.Draw(photo)
    font = ImageFont.load_default(size=MARKER_FONT_PX)
    for number, entry in enumerate(entries, start=1):
        pixel = entry.get("target_pixel")
        if not pixel:
            continue
        u, v = int(pixel[0]), int(pixel[1])
        circle = [
            u - MARKER_RADIUS_PX,
            v - MARKER_RADIUS_PX,
            u + MARKER_RADIUS_PX,
            v + MARKER_RADIUS_PX,
        ]
        colour = RIGHT_RGB if entry["score"] > 0 else WRONG_RGB
        canvas.ellipse(circle, fill=colour, outline=WHITE_RGB, width=MARKER_OUTLINE_PX)
        canvas.text((u, v), str(number), fill=WHITE_RGB, font=font, anchor="mm")


def legend_panel(height: int, title: str, entries: list[dict]) -> Image.Image:
    """Render the side panel listing every instruction of the scene.

    Args:
        height: panel height in pixels (the photo height).
        title: first line of the panel.
        entries: report instructions of this scene, in order.

    Returns:
        White panel of width ``PANEL_WIDTH_PX`` with the legend text.
    """
    panel = Image.new("RGB", (PANEL_WIDTH_PX, height), WHITE_RGB)
    canvas = ImageDraw.Draw(panel)
    title_font = ImageFont.load_default(size=TITLE_FONT_PX)
    font = ImageFont.load_default(size=PANEL_FONT_PX)
    y = PANEL_MARGIN_PX
    canvas.text((PANEL_MARGIN_PX, y), title, fill=TEXT_RGB, font=title_font)
    y += 2 * PANEL_LINE_PX
    for number, entry in enumerate(entries, start=1):
        text, right = answer_line(number, entry)
        colour = RIGHT_RGB if right else WRONG_RGB
        for line in textwrap.wrap(text, PANEL_WRAP_CHARACTERS):
            canvas.text((PANEL_MARGIN_PX, y), line, fill=colour, font=font)
            y += PANEL_LINE_PX
        y += PANEL_LINE_PX // 2
    return panel


def main() -> None:
    """Annotate every scene of the report and write one JPEG per scene.

    Returns:
        None.
    """
    arguments = parse_arguments()
    report = json.loads(arguments.report.read_text())
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    by_scene: dict[str, list[dict]] = {}
    for entry in report["instructions"]:
        by_scene.setdefault(entry["instruction_id"].rsplit("_i", 1)[0], []).append(entry)
    for scene_name, entries in sorted(by_scene.items()):
        photo = Image.open(arguments.data / scene_name / "rgb.png").convert("RGB")
        draw_markers(photo, entries)
        right = sum(entry["score"] > 0 for entry in entries)
        title = f"{scene_name}: {right} of {len(entries)} answers right"
        annotated = Image.new("RGB", (photo.width + PANEL_WIDTH_PX, photo.height), WHITE_RGB)
        annotated.paste(photo, (0, 0))
        annotated.paste(legend_panel(photo.height, title, entries), (photo.width, 0))
        path = arguments.out_dir / f"{scene_name}.jpg"
        annotated.save(path, quality=JPEG_QUALITY)
        print(f"{path}: {right}/{len(entries)} right")


if __name__ == "__main__":
    main()
