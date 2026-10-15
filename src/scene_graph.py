"""Turn named regions into a scene graph: objects, sizes, stacking and spatial relations.

All geometry is computed here, in code, so Apertus reasons over facts it can read rather than
estimating pixels or distances from the photo.
"""

import re

import cv2
import numpy as np

from src.perception.regions import interior_point
from src.workcell import Scene, table_xy

HAND_WORDS = ("hand", "arm", "finger", "person", "human")


def size_mm(mask: np.ndarray, work_scale: float, scene: Scene) -> tuple[int, int]:
    """Length and width of a region's minimum-area rotated rectangle on the table plane.

    The outline is projected onto the table with the calibration (``table_xy``) before the
    rectangle is fitted, so the size does not depend on where in the tilted view the object is.
    Tall objects still measure too large: their top is closer to the camera (no depth in 2D).

    Args:
        mask: (h, w) bool at work scale.
        work_scale: scale of ``mask`` relative to the full photo.
        scene: loaded scene with calibration.

    Returns:
        (length_mm, width_mm), length >= width. The width is what the gripper must span.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = (np.vstack(contours).reshape(-1, 2) + 0.5) / work_scale - 0.5
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(table_xy(scene, outline).astype(np.float32))
    return round(max(side_a, side_b)), round(min(side_a, side_b))


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


def absorb(target: dict, region: dict) -> None:
    """Fold ``region`` into ``target``: union mask, area, source ids and parts."""
    target["mask"] = target["mask"] | region["mask"]
    target["area_mm2"] += region["area_mm2"]
    target["source_ids"] += region["source_ids"]
    target["parts"] |= region["parts"]


def absorb_fragments(regions: list[dict], gap_pixels: int, fragment_max_mm2: float) -> list[dict]:
    """Fold each region too small to be an object into the larger region it touches most.

    A mug's handle, a sliver of its rim, a blemish on a mandarin or the shaded side face of a
    cube come out as small regions of another colour, and Apertus names them as objects of their
    own ("white handle", "red fruit", "grey block"). Smallest first, so chains of fragments end
    up in the object they belong to. A fragment that touches nothing larger stays as it is.

    Args:
        regions: regions with ``source_ids`` and ``parts`` set.
        gap_pixels: how close two regions must be to count as touching.
        fragment_max_mm2: area below which a region is a fragment.

    Returns:
        The regions that remain, larger ones with their fragments merged in.
    """
    kernel = np.ones((2 * gap_pixels + 1,) * 2, np.uint8)
    remaining = list(regions)
    for fragment in sorted(regions, key=lambda region: region["area_mm2"]):
        if fragment["area_mm2"] >= fragment_max_mm2:
            break
        near = cv2.dilate(fragment["mask"].astype(np.uint8), kernel).astype(bool)
        larger = [other for other in remaining if other["area_mm2"] > fragment["area_mm2"]]
        # a touching neighbour of the fragment's own colour wins (a rim sliver goes to its rim)
        contact = []
        for other in larger:
            touch = int((near & other["mask"]).sum())
            contact.append((touch > 0 and other["colour_group"] == fragment["colour_group"], touch))
        if not larger or max(contact)[1] == 0:
            continue
        absorb(larger[contact.index(max(contact))], fragment)
        remaining = [other for other in remaining if other is not fragment]
    return remaining


def same_kind(name_a: str, name_b: str, noun_groups: list[list[str]]) -> bool:
    """Whether two names mean the same object: equal, or nouns in one synonym group.

    The rim and inside of one mug are often named "white mug" and "yellow cup"; a saucer's
    rim and centre "orange plate" and "grey saucer".

    Args:
        name_a: object name from Apertus.
        name_b: object name from Apertus.
        noun_groups: lists of nouns that name the same kind of object.

    Returns:
        True if the names match.
    """
    if name_a == name_b:
        return True
    noun_a, noun_b = name_a.split()[-1], name_b.split()[-1]
    return any(noun_a in group and noun_b in group for group in noun_groups)


def merge_same_object(regions: list[dict], gap_pixels: int, config: dict) -> list[dict]:
    """Merge the regions of one object: small fragments first, then touching same-kind regions.

    A mug's white rim and green inside, or the fragments of a saucer's rim, come out as
    separate colour regions with one kind of name. Regions cut apart by ``split_touching``
    (three gears in a row) share a ``part`` and are never merged back by name.

    Args:
        regions: named regions (``name``, ``mask``, ``part``, ``area_mm2`` ...), non-objects
            removed.
        gap_pixels: how close two regions must be to count as touching.
        config: the ``scene_graph`` section, with ``fragment_max_mm2`` and
            ``same_object_nouns``.

    Returns:
        New list of regions; a merged region keeps the larger or first one's fields, the union
        mask and all source ids under ``source_ids``.
    """
    regions = [
        {**region, "source_ids": [region["id"]], "parts": {region["part"]}} for region in regions
    ]
    regions = absorb_fragments(regions, gap_pixels, config["fragment_max_mm2"])
    merged: list[dict] = []
    for region in regions:
        for existing in merged:
            split_on_purpose = bool(existing["parts"] & region["parts"])
            if (
                same_kind(existing["name"], region["name"], config["same_object_nouns"])
                and not split_on_purpose
                and touching(existing["mask"], region["mask"], gap_pixels)
            ):
                absorb(existing, region)
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
    regions: list[dict], work_scale: float, mm_per_work_pixel: float, config: dict, scene: Scene
) -> list[dict]:
    """Assemble scene-graph objects from named regions.

    Args:
        regions: regions from ``find_regions`` with ``name`` and ``is_object`` added.
        work_scale: work-scale factor of the masks.
        mm_per_work_pixel: table millimetres per work-scale pixel.
        config: the ``scene_graph`` section of the pipeline config.
        scene: loaded scene with calibration.

    Returns:
        Objects sorted left to right, each with ``id``, ``name``, ``pixel`` (full-res [u, v]),
        ``table_xy`` (mm), ``size_mm``, ``tray``, ``mask``, ``is_hand``, ``on_top_of`` and
        ``under`` (id lists),
        ``source_ids``.
    """
    gap_pixels = max(1, round(config["merge_gap_mm"] / mm_per_work_pixel))
    kept = [region for region in regions if region["is_object"]]
    objects = merge_same_object(kept, gap_pixels, config)
    for item in objects:
        u, v = interior_point(item["mask"])
        item["pixel"] = [round(u / work_scale), round(v / work_scale)]
        item["table_xy"] = table_xy(scene, [item["pixel"]])[0]
        item["size_mm"] = size_mm(item["mask"], work_scale, scene)
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


def describe(objects: list[dict], nearest_count: int, too_wide_mm: float) -> str:
    """Render the scene graph as text for the decision prompt.

    Args:
        objects: output of ``build_scene_graph``.
        nearest_count: how many nearest neighbours to list per object.
        too_wide_mm: measured width above which an object is flagged as not graspable.

    Returns:
        One line per object, then left-to-right and top-to-bottom orderings, and the
        left-to-right order within every name shared by several objects.
    """
    lines = []
    for item in objects:
        others = [other for other in objects if other is not item]
        distances = sorted(
            (
                round(np.hypot(*np.subtract(item["table_xy"], other["table_xy"]))),
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
            # "underneath X" was read backwards by Apertus; name the covering object explicitly
            fields.append("covered by " + ", ".join(item["under"]) + " (lying on top of it)")
        if width > too_wide_mm:
            fields.append("TOO WIDE TO GRIP")
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
    names = sorted({item["name"] for item in objects})
    for name in names:
        same = [item["id"] for item in objects if item["name"] == name]
        if len(same) > 1:
            lines.append(
                f"The {len(same)} objects named '{name}', left to right: {', '.join(same)}"
            )
    return "\n".join(lines)
