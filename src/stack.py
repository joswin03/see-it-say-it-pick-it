"""Which marked object lies on which: overlapping boxes are shown up close and Apertus decides.

Boxes alone cannot tell which of two overlapping objects is above the other. For every pair
whose boxes overlap, a close-up with only those two boxes drawn is cut from the input photo
and Apertus says which one lies on top. Nothing is stored; each close-up exists for one call.
"""

import numpy as np

from src.marks import draw_marks


def overlap_fraction(box_a: list[int], box_b: list[int]) -> float:
    """Share of the smaller box that lies inside the other box.

    Args:
        box_a: [u_min, v_min, u_max, v_max].
        box_b: [u_min, v_min, u_max, v_max].

    Returns:
        Overlap area divided by the area of the smaller box, 0 to 1.
    """
    width = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    height = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return width * height / min(area_a, area_b)


def overlapping_pairs(marks: list[dict], min_overlap: float) -> list[tuple[dict, dict]]:
    """Pairs of marked objects whose boxes overlap enough that one may lie on the other.

    Args:
        marks: marked objects with ``box``.
        min_overlap: smallest ``overlap_fraction`` counted as overlapping.

    Returns:
        List of (object, object) pairs.
    """
    return [
        (mark, other)
        for index, mark in enumerate(marks)
        for other in marks[index + 1 :]
        if overlap_fraction(mark["box"], other["box"]) >= min_overlap
    ]


def pair_close_up(photo: np.ndarray, pair: tuple[dict, dict], margin_fraction: float) -> np.ndarray:
    """Draw both objects' numbered boxes and cut out the area around them.

    Args:
        photo: full-resolution photo, (H, W, 3) uint8 RGB.
        pair: two marked objects with ``number``, ``label`` and ``box``.
        margin_fraction: margin added on each side of the union of both boxes, relative to
            its size.

    Returns:
        The close-up, (h, w, 3) uint8 RGB.
    """
    first, second = pair
    union = [
        min(first["box"][0], second["box"][0]),
        min(first["box"][1], second["box"][1]),
        max(first["box"][2], second["box"][2]),
        max(first["box"][3], second["box"][3]),
    ]
    height, width = photo.shape[:2]
    margin_u = round((union[2] - union[0]) * margin_fraction)
    margin_v = round((union[3] - union[1]) * margin_fraction)
    left, top = max(0, union[0] - margin_u), max(0, union[1] - margin_v)
    right, bottom = min(width, union[2] + margin_u), min(height, union[3] + margin_v)
    drawn = draw_marks(photo, [first, second])
    return drawn[top:bottom, left:right]


def stacking_order(reply: dict | None, pair: tuple[dict, dict]) -> tuple[dict, dict] | None:
    """Read from the reply which of the two objects lies on the other.

    Args:
        reply: parsed reply ``{"on_top": <number or null>}``, or None.
        pair: the two marked objects that were shown.

    Returns:
        (upper, lower), or None when neither lies on the other or the reply is not one of
        the two.
    """
    on_top = reply.get("on_top") if isinstance(reply, dict) else None
    if isinstance(on_top, str) and on_top.strip().isdigit():
        on_top = int(on_top)
    first, second = pair
    if on_top == first["number"]:
        upper, lower = first, second
    elif on_top == second["number"]:
        upper, lower = second, first
    else:
        return None
    return upper, lower
