"""Refine a pick pixel: crop around the object's box, draw a numbered grid, let Apertus pick a cell.

Apertus places boxes only roughly but picks numbered cells reliably, so the pick pixel becomes
the centre of the cell it names inside a close-up of the input photo. Nothing is stored; the
crop exists only for this one call.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID_LINE_RGB = (255, 255, 0)
GRID_SHADOW_RGB = (0, 0, 0)
GRID_LINE_PX = 3
FONT_FRACTION = 0.08
MIN_FONT_PX = 16
TEXT_MARGIN_PX = 4


def crop_box(box: list[int], margin_fraction: float, width: int, height: int) -> list[int]:
    """Widen a box by a margin on every side and clip it to the photo.

    Args:
        box: [u_min, v_min, u_max, v_max] in full-resolution pixels.
        margin_fraction: margin added on each side, relative to the box size.
        width: photo width in pixels.
        height: photo height in pixels.

    Returns:
        The widened box [u_min, v_min, u_max, v_max], inside the photo.
    """
    margin_u = round((box[2] - box[0]) * margin_fraction)
    margin_v = round((box[3] - box[1]) * margin_fraction)
    return [
        max(0, box[0] - margin_u),
        max(0, box[1] - margin_v),
        min(width, box[2] + margin_u),
        min(height, box[3] + margin_v),
    ]


def gridded_crop(rgb: np.ndarray, region: list[int], grid: int) -> np.ndarray:
    """Cut a region from the photo and draw a numbered grid of cells on it.

    Cells are numbered from 1, row by row, from the top left.

    Args:
        rgb: full-resolution photo, (H, W, 3) uint8 RGB.
        region: [u_min, v_min, u_max, v_max] to cut out.
        grid: number of cells per row and per column.

    Returns:
        The crop with grid lines and cell numbers, (h, w, 3) uint8 RGB.
    """
    crop = Image.fromarray(rgb[region[1] : region[3], region[0] : region[2]])
    canvas = ImageDraw.Draw(crop)
    font = ImageFont.load_default(size=max(MIN_FONT_PX, round(FONT_FRACTION * min(crop.size))))
    cell_width, cell_height = crop.width / grid, crop.height / grid
    for index in range(1, grid):
        u, v = round(index * cell_width), round(index * cell_height)
        for colour, line_width in (
            (GRID_SHADOW_RGB, GRID_LINE_PX + 2),
            (GRID_LINE_RGB, GRID_LINE_PX),
        ):
            canvas.line([(u, 0), (u, crop.height)], fill=colour, width=line_width)
            canvas.line([(0, v), (crop.width, v)], fill=colour, width=line_width)
    for row in range(grid):
        for column in range(grid):
            origin = (
                round(column * cell_width) + TEXT_MARGIN_PX,
                round(row * cell_height) + TEXT_MARGIN_PX,
            )
            label = str(row * grid + column + 1)
            canvas.text(
                origin,
                label,
                fill=GRID_LINE_RGB,
                font=font,
                stroke_width=2,
                stroke_fill=GRID_SHADOW_RGB,
            )
    return np.array(crop)


def cell_centre(cell, region: list[int], grid: int) -> list[int] | None:
    """Full-resolution pixel at the centre of a numbered cell of the region.

    Args:
        cell: cell number from the reply (int or digit string), 1 to ``grid * grid``.
        region: [u_min, v_min, u_max, v_max] the grid was drawn on.
        grid: number of cells per row and per column.

    Returns:
        [u, v], or None when ``cell`` is not a valid cell number.
    """
    if isinstance(cell, str) and cell.strip().isdigit():
        cell = int(cell)
    if not isinstance(cell, int) or not 1 <= cell <= grid * grid:
        return None
    row, column = divmod(cell - 1, grid)
    cell_width = (region[2] - region[0]) / grid
    cell_height = (region[3] - region[1]) / grid
    return [
        round(region[0] + (column + 0.5) * cell_width),
        round(region[1] + (row + 0.5) * cell_height),
    ]
