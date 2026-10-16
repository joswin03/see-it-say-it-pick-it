"""Spatial relations and scene text adapted from Joswin's OpenCV branch.

Uses only masks, calibration and visual object names; never reads instruction wording.
"""

import cv2
import numpy as np


def stacking(objects: list[dict], min_overlap_fraction: float) -> list[tuple[str, str]]:
    """Find which object lies on which, from 2D outlines.

    From above, an object lying on another hides part of it, so the lower object's outline
    has a bite whose shape the upper object fills. Measured as the fraction of A inside the
    filled convex hull of B: high for A on B, low for B on A.

    Args:
        objects: objects with ``id`` and ``mask``.
        min_overlap_fraction: fraction of the upper object inside the lower one's hull.

    Returns:
        List of (upper_id, lower_id) pairs.
    """
    hulls = {}
    for item in objects:
        points = cv2.findNonZero(item["mask"].astype(np.uint8))
        hull = np.zeros(item["mask"].shape, np.uint8)
        cv2.fillPoly(hull, [cv2.convexHull(points)], 1)
        hulls[item["id"]] = hull.astype(bool)
    pairs = []
    for upper in objects:
        for lower in objects:
            if upper is lower:
                continue
            inside = (upper["mask"] & hulls[lower["id"]]).sum() / upper["mask"].sum()
            reverse = (lower["mask"] & hulls[upper["id"]]).sum() / lower["mask"].sum()
            if inside >= min_overlap_fraction and inside > reverse:
                pairs.append((upper["id"], lower["id"]))
    return pairs


def describe(objects: list[dict], on_top: list[tuple[str, str]]) -> str:
    """Render the scene graph as text for the decision prompt.

    Nodes are grouped by tray and carry only what a decision needs (name, other names of merged
    pieces, size, flags). Edges and positions are stated as facts and ranks rather than pixels
    and millimetres: an 8B model reads a list position more reliably than it compares numbers.

    Args:
        objects: output of ``build_scene_graph``.
        on_top: Upper and lower object ids from the stacking node.

    Returns:
        Objects per tray, the stacking edges, the left-to-right and top-to-bottom orderings,
        the order within every name shared by several objects, and near-to-far neighbours.
    """
    lines = []
    for tray, label in (("grey_tray", "GREY TRAY (left)"), ("black_tray", "BLACK TRAY (right)")):
        lines.append(f"{label}:")
        in_tray = [item for item in objects if item["tray"] == tray]
        for item in in_tray:
            length, width = item["size_mm"]
            name = item["label"] + (
                f" (also seen as {', '.join(item['aliases'])})" if item.get("aliases") else ""
            )
            flags = [f"{length} x {width} mm"]
            if item["too_wide"]:
                flags.append("TOO WIDE TO GRIP, can hold things")
            if item["is_hand"]:
                flags.append("A PERSON'S HAND")
            lines.append(f"  {item['id']} (mark {item['number']}): {name}; {', '.join(flags)}")
        if not in_tray:
            lines.append("  (empty)")
    stacks = [
        f"  {item['id']} lies on {lower} (move {item['id']} first to reach {lower})"
        for upper, lower in on_top
        for item in objects
        if item["id"] == upper
    ]
    lines += ["STACKED:", *(stacks or ["  (nothing lies on anything)"])]
    lines.append("LEFT TO RIGHT: " + ", ".join(item["id"] for item in objects))
    by_row = sorted(objects, key=lambda item: item["pixel"][1])
    lines.append("TOP TO BOTTOM: " + ", ".join(item["id"] for item in by_row))
    for name in sorted({item["label"] for item in objects}):
        same = [item["id"] for item in objects if item["label"] == name]
        if len(same) > 1:
            lines.append(f"THE {len(same)} '{name}' OBJECTS, LEFT TO RIGHT: {', '.join(same)}")
    if len(objects) > 1:
        lines.append("NEAR TO FAR (closest first):")
    for item in objects:
        others = sorted(
            (other for other in objects if other is not item),
            key=lambda other: np.hypot(*np.subtract(item["table_xy"], other["table_xy"])),
        )
        if others:
            lines.append(f"  {item['id']}: {', '.join(other['id'] for other in others)}")
    return "\n".join(lines)


def graph_record(objects: list[dict], on_top: list[tuple[str, str]]) -> dict:
    """The scene graph in the format shared with the ground-truth graphs.

    Args:
        objects: marked objects with ``id``, ``label``, ``pixel``, ``box``, ``size_mm`` and
            ``tray``.
        on_top: (upper id, lower id) pairs.

    Returns:
        ``{"objects": [{"id", "name", "pixel", "box", "size_mm", "tray", "lies_on"}]}``, left to
        right; ``box`` is [u_min, v_min, u_max, v_max] in full-resolution pixels.
    """
    return {
        "objects": [
            {
                "id": item["id"],
                "name": item["label"],
                "pixel": item["pixel"],
                "box": item["box"],
                "size_mm": item["size_mm"],
                "tray": item["tray"],
                "lies_on": [lower for upper, lower in on_top if upper == item["id"]],
            }
            for item in sorted(objects, key=lambda item: item["pixel"][0])
        ]
    }
