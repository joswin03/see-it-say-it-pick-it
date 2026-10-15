"""Turn named regions into a scene graph: objects, sizes, stacking and spatial relations.

All geometry is computed here, in code, so Apertus reasons over facts it can read rather than
estimating pixels or distances from the photo.
"""

import re

import cv2
import numpy as np

HAND_WORDS = ("hand", "arm", "finger", "person", "human")


def size_mm(mask: np.ndarray, mm_per_work_pixel: float) -> tuple[int, int]:
    """Length and width of a region's minimum-area rotated rectangle.

    Args:
        mask: (h, w) bool at work scale.
        mm_per_work_pixel: table millimetres per work-scale pixel.

    Returns:
        (length_mm, width_mm), length >= width. The width is what the gripper must span.
    """
    points = cv2.findNonZero(mask.astype(np.uint8))
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(points)
    long_side, short_side = max(side_a, side_b), min(side_a, side_b)
    return round(long_side * mm_per_work_pixel), round(short_side * mm_per_work_pixel)


def touching(mask_a: np.ndarray, mask_b: np.ndarray, gap_pixels: int) -> bool:
    """Whether two masks touch or are separated by at most ``gap_pixels``.

    Args:
        mask_a: (h, w) bool.
        mask_b: (h, w) bool.
        gap_pixels: allowed gap.

    Returns:
        True if the dilated first mask overlaps the second.
    """
    kernel = np.ones((2 * gap_pixels + 1,) * 2, np.uint8)
    return bool((cv2.dilate(mask_a.astype(np.uint8), kernel).astype(bool) & mask_b).any())


def merge_same_object(regions: list[dict], gap_pixels: int) -> list[dict]:
    """Merge touching regions that Apertus gave the same name, unless split on purpose.

    A mug's white rim and green inside, or the fragments of a saucer's rim, come out as
    separate colour regions with one name. Regions cut apart by ``split_touching`` (three
    gears in a row) share a ``part`` and are never merged back.

    Args:
        regions: named regions (``name``, ``mask``, ``part`` ...), non-objects removed.
        gap_pixels: how close two regions must be to count as touching.

    Returns:
        New list of regions; a merged region keeps the first one's fields, the union mask and
        all source ids under ``source_ids``.
    """
    merged: list[dict] = []
    for region in regions:
        region = {**region, "source_ids": [region["id"]], "parts": {region["part"]}}
        for existing in merged:
            same_name = existing["name"] == region["name"]
            split_on_purpose = bool(existing["parts"] & region["parts"])
            if (
                same_name
                and not split_on_purpose
                and touching(existing["mask"], region["mask"], gap_pixels)
            ):
                existing["mask"] = existing["mask"] | region["mask"]
                existing["source_ids"] += region["source_ids"]
                existing["parts"] |= region["parts"]
                break
        else:
            merged.append(region)
    return merged


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


def object_id_base(name: str) -> str:
    """Make an identifier stem from a name: "Red gear" -> "red_gear".

    Args:
        name: object name from Apertus.

    Returns:
        Lower-case snake-case stem.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "object"


def build_scene_graph(
    regions: list[dict], work_scale: float, mm_per_work_pixel: float, config: dict
) -> list[dict]:
    """Assemble scene-graph objects from named regions.

    Args:
        regions: regions from ``find_regions`` with ``name`` and ``is_object`` added.
        work_scale: work-scale factor of the masks.
        mm_per_work_pixel: table millimetres per work-scale pixel.
        config: the ``scene_graph`` section of the pipeline config.

    Returns:
        Objects sorted left to right, each with ``id``, ``name``, ``pixel`` (full-res [u, v]),
        ``size_mm``, ``tray``, ``mask``, ``is_hand``, ``on_top_of`` and ``under`` (id lists),
        ``source_ids``.
    """
    from src.perception.regions import interior_point

    gap_pixels = max(1, round(config["merge_gap_mm"] / mm_per_work_pixel))
    kept = [region for region in regions if region["is_object"]]
    objects = merge_same_object(kept, gap_pixels)
    for item in objects:
        u, v = interior_point(item["mask"])
        item["pixel"] = [round(u / work_scale), round(v / work_scale)]
        item["size_mm"] = size_mm(item["mask"], mm_per_work_pixel)
        item["is_hand"] = any(word in item["name"] for word in HAND_WORDS)
    objects.sort(key=lambda item: item["pixel"][0])
    counts: dict[str, int] = {}
    for item in objects:
        base = object_id_base(item["name"])
        counts[base] = counts.get(base, 0) + 1
        item["id"] = f"{base}_{counts[base]}"
    for item in objects:
        item["on_top_of"], item["under"] = [], []
    by_id = {item["id"]: item for item in objects}
    for upper, lower in stacking(objects, config["stack_min_overlap_fraction"]):
        by_id[upper]["on_top_of"].append(lower)
        by_id[lower]["under"].append(upper)
    return objects


def describe(
    objects: list[dict], mm_per_pixel: float, nearest_count: int, gripper_mm: float
) -> str:
    """Render the scene graph as text for the decision prompt.

    Args:
        objects: output of ``build_scene_graph``.
        mm_per_pixel: table millimetres per full-resolution pixel.
        nearest_count: how many nearest neighbours to list per object.
        gripper_mm: maximum gripper opening; wider objects are flagged as not graspable.

    Returns:
        One line per object, then left-to-right and top-to-bottom orderings.
    """
    lines = []
    for item in objects:
        others = [other for other in objects if other is not item]
        distances = sorted(
            (
                round(np.hypot(*np.subtract(item["pixel"], other["pixel"])) * mm_per_pixel),
                other["id"],
            )
            for other in others
        )
        nearest = ", ".join(f"{oid} {mm} mm" for mm, oid in distances[:nearest_count])
        length, width = item["size_mm"]
        fields = [
            f"{item['id']}: {item['name']}",
            f"in the {item['tray'].replace('_', ' ')}",
            f"size {length} x {width} mm",
            f"image position x={item['pixel'][0]} y={item['pixel'][1]}",
        ]
        if item["on_top_of"]:
            fields.append("lying on " + ", ".join(item["on_top_of"]))
        if item["under"]:
            fields.append("underneath " + ", ".join(item["under"]) + " (covered)")
        if width > gripper_mm:
            fields.append(f"TOO WIDE TO GRIP (over {round(gripper_mm)} mm)")
        if item["is_hand"]:
            fields.append("A PERSON'S HAND")
        fields.append(f"nearest: {nearest or 'none'}")
        lines.append("- " + "; ".join(fields))
    left_to_right = ", ".join(item["id"] for item in objects)
    top_to_bottom = ", ".join(
        item["id"] for item in sorted(objects, key=lambda item: item["pixel"][1])
    )
    lines.append(f"Left to right in the photo: {left_to_right}")
    lines.append(f"Top to bottom in the photo: {top_to_bottom}")
    return "\n".join(lines)
