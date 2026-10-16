"""Turn named OpenCV regions into numbered objects: merge pieces of one object, set box and pixel.

A mug's rim and inside, or a saucer's rim fragments, come out of OpenCV as separate regions;
they are merged here before numbering. The pick pixel is the point deepest inside the mask,
the safest place for the gripper.
"""

import re

import cv2
import numpy as np

from src.calibration import Scene, table_xy
from src.regions import interior_point


def size_mm(
    mask: np.ndarray, work_scale: float, scene: Scene, distortion: np.ndarray
) -> tuple[int, int]:
    """Length and width of a region's minimum-area rectangle on the table plane.

    Args:
        mask: (h, w) bool at work scale.
        work_scale: scale of ``mask`` relative to the full photo.
        scene: loaded scene with calibration.
        distortion: lens distortion from ``calibration.lens_distortion``.

    Returns:
        (length_mm, width_mm), length >= width.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = (np.vstack(contours).reshape(-1, 2) + 0.5) / work_scale - 0.5
    table_points = table_xy(scene, distortion, outline).astype(np.float32)
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(table_points)
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
    """Fold ``region`` into ``target`` in place: union mask, area and parts.

    Args:
        target: region that stays.
        region: region that is merged into it.

    Returns:
        None.
    """
    target["mask"] = target["mask"] | region["mask"]
    target["area_mm2"] += region["area_mm2"]
    target["parts"] |= region["parts"]


def absorb_fragments(regions: list[dict], gap_pixels: int, fragment_max_mm2: float) -> list[dict]:
    """Fold each region too small to be an object into the larger region it touches most.

    Smallest first, so chains of fragments end up in their object. A touching neighbour of the
    fragment's own colour group wins (a rim sliver goes to its rim). A fragment touching nothing
    larger stays.

    Args:
        regions: regions with ``mask``, ``area_mm2``, ``colour_group`` and ``parts``.
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
        contact = [
            (
                int((near & other["mask"]).sum()) > 0
                and other["colour_group"] == fragment["colour_group"],
                int((near & other["mask"]).sum()),
            )
            for other in larger
        ]
        if not larger or max(contact)[1] == 0:
            continue
        absorb(larger[contact.index(max(contact))], fragment)
        remaining = [other for other in remaining if other is not fragment]
    return remaining


def same_kind(name_a: str, name_b: str, noun_groups: list[list[str]]) -> bool:
    """Whether two names mean the same object: equal, or last nouns in one synonym group.

    Args:
        name_a: object name.
        name_b: object name.
        noun_groups: lists of nouns that name the same kind of object, e.g. cup and mug.

    Returns:
        True if the names match.
    """
    if name_a == name_b:
        return True
    noun_a, noun_b = name_a.split()[-1], name_b.split()[-1]
    return any(noun_a in group and noun_b in group for group in noun_groups)


def merge_same_object(regions: list[dict], gap_pixels: int, config: dict) -> list[dict]:
    """Merge the regions of one object: small fragments first, then touching same-kind regions.

    Regions that OpenCV split apart on purpose (touching same-colour gears) share a ``part`` and
    are never merged back by name.

    Args:
        regions: named object regions (``name``, ``mask``, ``part``, ``area_mm2``, ...).
        gap_pixels: how close two regions must be to count as touching.
        config: the ``objects`` config section with ``fragment_max_mm2`` and
            ``same_object_nouns``.

    Returns:
        Merged regions; a merged region keeps the first one's name and the union mask.
    """
    regions = [
        {**region, "parts": {region["part"]}, "aliases": list(region.get("aliases", []))}
        for region in regions
    ]
    regions = absorb_fragments(regions, gap_pixels, config["fragment_max_mm2"])
    merged: list[dict] = []
    for region in regions:
        for existing in merged:
            if (
                same_kind(existing["name"], region["name"], config["same_object_nouns"])
                and not existing["parts"] & region["parts"]
                and touching(existing["mask"], region["mask"], gap_pixels)
            ):
                for alias in [region["name"], *region["aliases"]]:
                    if alias != existing["name"] and alias not in existing["aliases"]:
                        existing["aliases"].append(alias)
                absorb(existing, region)
                break
        else:
            merged.append(region)
    return merged


def build_marks(
    regions: list[dict],
    work_scale: float,
    mm_per_work_pixel: float,
    config: dict,
    scene: Scene,
    distortion: np.ndarray,
) -> list[dict]:
    """Merge named regions into objects and number them left to right.

    Args:
        regions: named regions with ``is_object``; non-objects are dropped.
        work_scale: scale of the masks relative to the full photo.
        mm_per_work_pixel: table millimetres per work-scale pixel.
        config: the ``objects`` config section with the derived ``too_wide_mm`` (gripper
            opening plus margin, set in ``src.graph``).
        scene: loaded scene with calibration.
        distortion: lens distortion from ``calibration.lens_distortion``.

    Returns:
        Objects with ``number``, ``id`` (e.g. ``red_gear_2``), ``label``, ``box``
        [u_min, v_min, u_max, v_max] and ``pixel`` [u, v] (deepest interior point), all in
        full-resolution pixels, ``width_mm`` on the table and ``too_wide`` (wider than
        ``too_wide_mm``, so not graspable).
    """
    gap_pixels = max(1, round(config["merge_gap_mm"] / mm_per_work_pixel))
    kept = [region for region in regions if region["is_object"]]
    merged = merge_same_object(kept, gap_pixels, config)
    objects = []
    for region in merged:
        u, v = interior_point(region["mask"])
        x, y, width, height = cv2.boundingRect(region["mask"].astype(np.uint8))
        length_mm, width_mm = size_mm(region["mask"], work_scale, scene, distortion)
        objects.append(
            {
                "label": region["name"],
                "width_mm": width_mm,
                "too_wide": width_mm > config["too_wide_mm"]
                or region["name"].split()[-1] in config.get("unpickable_nouns", []),
                "size_mm": [length_mm, width_mm],
                "table_xy": table_xy(
                    scene, distortion, np.array([[u / work_scale, v / work_scale]])
                )[0].tolist(),
                "tray": region["tray"],
                "aliases": region["aliases"],
                "mask": region["mask"],
                "is_hand": bool(set(region["name"].split()) & set(config.get("hand_nouns", []))),
                "pixel": [round(u / work_scale), round(v / work_scale)],
                "box": [
                    round(x / work_scale),
                    round(y / work_scale),
                    round((x + width) / work_scale),
                    round((y + height) / work_scale),
                ],
            }
        )
    objects.sort(key=lambda item: item["pixel"][0])
    marks = []
    for number, item in enumerate(objects, start=1):
        base = re.sub(r"[^a-z0-9]+", "_", item["label"].lower()).strip("_") or "object"
        marks.append({"number": number, "id": f"{base}_{number}", **item})
    return marks
