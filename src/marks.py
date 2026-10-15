"""Objects marked by Apertus: read them from the reply, number them, draw them on the photo."""

import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BOX_RGB = (255, 0, 255)
BOX_WIDTH_PX = 6
FONT_SIZE_PX = 48
LABEL_GAP_PX = 6
BOX_VALUES = 4
NUMBER = r"\s*(-?\d+(?:\.\d+)?)\s*"
MARK_PATTERN = re.compile(
    r'"label"\s*:\s*"([^"]*)"\s*,\s*"box"\s*:\s*\[' + ",".join([NUMBER] * BOX_VALUES)
)


def read_marks(reply: dict | None, scale: float) -> list[dict]:
    """Turn the annotate reply into numbered objects with full-resolution boxes.

    Entries without a text label or without four numbers as box are skipped.

    Args:
        reply: parsed reply ``{"objects": [{"label", "box"}]}``, or None.
        scale: resized size divided by full size, from ``baseline.encode``.

    Returns:
        Objects numbered from 1, each with ``number``, ``id`` (e.g. ``red_cube_2``),
        ``label`` and ``box`` [u_min, v_min, u_max, v_max] in full-resolution pixels.
    """
    entries = reply.get("objects", []) if isinstance(reply, dict) else []
    marks = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
            continue
        box = entry.get("box")
        if not isinstance(box, list) or len(box) != BOX_VALUES:
            continue
        if not all(isinstance(value, (int, float)) for value in box):
            continue
        u_a, v_a, u_b, v_b = (round(value / scale) for value in box)
        number = len(marks) + 1
        name = re.sub(r"[^a-z0-9]+", "_", entry["label"].lower()).strip("_") or "object"
        marks.append(
            {
                "number": number,
                "id": f"{name}_{number}",
                "label": entry["label"],
                "box": [min(u_a, u_b), min(v_a, v_b), max(u_a, u_b), max(v_a, v_b)],
            }
        )
    return marks


def salvage_marks(raw: str) -> dict:
    """Recover labelled boxes from a reply that is not valid JSON.

    Apertus sometimes writes every label and box but breaks the JSON around them (for example
    a missing ``]`` after each box). Each ``"label": "...", "box": [n, n, n, n`` is read
    directly from the text, whether or not its brackets close.

    Args:
        raw: raw annotate reply.

    Returns:
        ``{"objects": [{"label", "box"}]}`` with every complete label and box found.
    """
    return {
        "objects": [
            {"label": match[0], "box": [float(value) for value in match[1:]]}
            for match in MARK_PATTERN.findall(raw)
        ]
    }


def draw_marks(rgb: np.ndarray, marks: list[dict]) -> np.ndarray:
    """Draw every object's box with its number and label on a copy of the photo.

    Args:
        rgb: full-resolution photo, (H, W, 3) uint8 RGB.
        marks: output of ``read_marks``.

    Returns:
        The annotated photo, (H, W, 3) uint8 RGB.
    """
    image = Image.fromarray(rgb)
    canvas = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=FONT_SIZE_PX)
    for mark in marks:
        canvas.rectangle(mark["box"], outline=BOX_RGB, width=BOX_WIDTH_PX)
        text_origin = (mark["box"][0], mark["box"][1] - FONT_SIZE_PX - LABEL_GAP_PX)
        canvas.text(text_origin, f"{mark['number']} {mark['label']}", fill=BOX_RGB, font=font)
    return np.array(image)


def objects_text(marks: list[dict]) -> str:
    """List the marked objects for the decide prompt, one per line.

    Args:
        marks: output of ``read_marks``.

    Returns:
        Lines like ``1: red cube``; ``(none)`` when nothing was marked.
    """
    return "\n".join(f"{mark['number']}: {mark['label']}" for mark in marks) or "(none)"


def box_centre(box: list[int]) -> list[int]:
    """Centre pixel of a box.

    Args:
        box: [u_min, v_min, u_max, v_max].

    Returns:
        [u, v].
    """
    return [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
