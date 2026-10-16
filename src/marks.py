"""Numbered objects on the photo: draw their boxes and list them for the decide prompt."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BOX_RGB = (255, 0, 255)
BOX_WIDTH_PX = 6
FONT_SIZE_PX = 48
LABEL_GAP_PX = 6


def draw_marks(rgb: np.ndarray, marks: list[dict]) -> np.ndarray:
    """Draw every object's box with its number and label on a copy of the photo.

    Args:
        rgb: full-resolution photo, (H, W, 3) uint8 RGB.
        marks: objects with ``number``, ``label`` and ``box`` [u_min, v_min, u_max, v_max].

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
