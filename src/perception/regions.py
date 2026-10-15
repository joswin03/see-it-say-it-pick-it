"""Find the two trays and the object regions in one workcell photo with OpenCV.

The camera looks straight down on a grey plastic tray (left) and a black tray (right). Both are
nearly unsaturated, so an object is any pixel that is clearly brighter, darker or more colourful
than the tray it lies in. A foreground blob is then split by colour class, so a yellow gear lying
on a red gear becomes two regions. Whether two touching regions are one object (the white rim and
green inside of a mug) is decided later, when Apertus names them.

Everything runs on a downscaled copy (``work_scale``); pixels returned are full resolution.
"""

import cv2
import numpy as np

NEUTRAL_WHITE = "white"
NEUTRAL_GREY = "grey"


def to_work_scale(rgb: np.ndarray, work_scale: float) -> np.ndarray:
    """Downscale the photo for segmentation.

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        work_scale: factor in (0, 1].

    Returns:
        (h, w, 3) uint8 downscaled photo.
    """
    height, width = rgb.shape[:2]
    size = (round(width * work_scale), round(height * work_scale))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected component of a boolean mask.

    Args:
        mask: (h, w) bool.

    Returns:
        (h, w) bool, empty if the mask is empty.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 2:
        return np.zeros_like(mask, dtype=bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def find_trays(hsv: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    """Locate the black tray, then take the grey tray as the area to its left.

    The black tray is one large dark rectangle and is found reliably even when cluttered. The
    grey tray floor is often hidden by objects, so it is not segmented directly: it is the
    band left of the black tray, between the black tray's top and bottom rows.

    Args:
        hsv: (h, w, 3) uint8 photo in OpenCV HSV at work scale.
        config: the ``perception`` section of the pipeline config.

    Returns:
        Dict with ``black_tray`` and ``grey_tray`` masks, each (h, w) bool.
    """
    saturation, value = hsv[..., 1], hsv[..., 2]
    kernel = np.ones((config["morph_kernel_px"],) * 2, np.uint8)
    floor = grey_floor(hsv, config)
    dark = (
        (value < config["black_tray_max_value"])
        & (saturation < config["black_tray_max_saturation"])
        & ~floor
    )
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    black_blob = largest_component(dark)
    if not black_blob.any():
        raise ValueError("no black tray found: no large dark region in the photo")
    x, y, width, height = cv2.boundingRect(black_blob.astype(np.uint8))
    black_tray = np.zeros_like(black_blob)
    black_tray[y : y + height, x : x + width] = True
    grey_tray = np.zeros_like(black_blob)
    grey_tray[y : y + height, :x] = True
    return {"black_tray": black_tray, "grey_tray": grey_tray}


def grey_floor(hsv: np.ndarray, config: dict) -> np.ndarray:
    """Pixels that look like the bare grey tray plastic.

    Args:
        hsv: (h, w, 3) uint8 OpenCV HSV.
        config: the ``perception`` section of the pipeline config.

    Returns:
        (h, w) bool.
    """
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_low, hue_high = config["grey_floor_hue"]
    sat_low, sat_high = config["grey_floor_saturation"]
    value_low, value_high = config["grey_floor_value"]
    return (
        (hue >= hue_low)
        & (hue <= hue_high)
        & (saturation >= sat_low)
        & (saturation < sat_high)
        & (value > value_low)
        & (value < value_high)
    )


def background_plane(value: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Fit brightness = a*u + b*v + c to the bare tray, to follow its lighting gradient.

    Args:
        value: (h, w) uint8 HSV value channel.
        support: (h, w) bool pixels believed to be bare tray.

    Returns:
        (h, w) float32 expected tray brightness at every pixel.
    """
    rows, columns = np.nonzero(support)
    design = np.column_stack([columns, rows, np.ones_like(rows)]).astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(design, value[rows, columns].astype(np.float64), rcond=None)
    grid_rows, grid_columns = np.indices(value.shape)
    plane = coefficients[0] * grid_columns + coefficients[1] * grid_rows + coefficients[2]
    return plane.astype(np.float32)


def foreground(hsv: np.ndarray, trays: dict[str, np.ndarray], config: dict) -> np.ndarray:
    """Mark pixels that belong to an object rather than to a tray.

    Args:
        hsv: (h, w, 3) uint8 OpenCV HSV at work scale.
        trays: output of ``find_trays``.
        config: the ``perception`` section of the pipeline config.

    Returns:
        (h, w) bool, cleaned with an open/close and with blobs below the minimum area removed
        by the caller.
    """
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_low, hue_high = config["grey_floor_hue"]
    off_floor_hue = (hue < hue_low) | (hue > hue_high)
    bare_black = (
        trays["black_tray"]
        & (value < config["black_tray_max_value"])
        & (saturation < config["black_tray_max_saturation"])
    )
    black_level = background_plane(value, bare_black)
    on_black = (value >= black_level + config["black_object_min_contrast"]) | (
        saturation >= config["black_object_min_saturation"]
    )
    on_grey = ~grey_floor(hsv, config) & (
        (saturation >= config["grey_object_min_saturation"])
        | (value >= config["grey_object_min_value"])
        | (value <= config["grey_object_max_value"])
        | (off_floor_hue & (saturation >= config["grey_object_min_tint_saturation"]))
    )
    mask = (on_black & trays["black_tray"]) | (on_grey & trays["grey_tray"])
    kernel = np.ones((config["morph_kernel_px"],) * 2, np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def colour_classes(hsv: np.ndarray, config: dict) -> tuple[np.ndarray, list[str]]:
    """Assign every pixel a named colour class.

    Saturated pixels get a hue class; unsaturated ones are white or grey by brightness.

    Args:
        hsv: (h, w, 3) uint8 OpenCV HSV.
        config: the ``perception`` section of the pipeline config.

    Returns:
        (h, w) int32 class index map, and the class names indexed by it.
    """
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    names = [*config["colour_hue_ranges"], NEUTRAL_WHITE, NEUTRAL_GREY]
    classes = np.full(hue.shape, names.index(NEUTRAL_GREY), np.int32)
    classes[value >= config["white_min_value"]] = names.index(NEUTRAL_WHITE)
    colourful = saturation >= config["colour_min_saturation"]
    for index, (name, ranges) in enumerate(config["colour_hue_ranges"].items()):
        ranges = ranges if isinstance(ranges[0], list) else [ranges]
        in_class = np.zeros_like(colourful)
        for low, high in ranges:
            in_class |= (hue >= low) & (hue <= high)
        classes[colourful & in_class] = index
    return classes, names


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """Pixel deepest inside a mask: safest for grounding and for the gripper.

    Args:
        mask: (h, w) bool, non-empty.

    Returns:
        (u, v) column and row of the maximum of the distance transform.
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    row, column = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(column), int(row)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes enclosed by a mask (a gear's hub, a highlight on a fruit).

    Args:
        mask: (h, w) bool.

    Returns:
        (h, w) bool with every enclosed hole filled.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def split_touching(mask: np.ndarray, core_fraction: float) -> list[np.ndarray]:
    """Split touching objects of one colour (three gears in a row) at their narrow necks.

    Cores are the parts of the mask deeper than ``core_fraction`` of its maximum depth; every
    pixel goes to the nearest core. A single round or elongated object has one core and is
    returned unchanged.

    Args:
        mask: (h, w) bool, one connected same-colour part.
        core_fraction: depth threshold relative to the deepest pixel, in (0, 1).

    Returns:
        List of (h, w) bool masks, one per core.
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    cores = (distance > core_fraction * distance.max()).astype(np.uint8)
    core_count, core_labels = cv2.connectedComponents(cores)
    if core_count <= 2:
        return [mask]
    distance_to_core = np.stack(
        [
            cv2.distanceTransform((core_labels != index).astype(np.uint8), cv2.DIST_L2, 5)
            for index in range(1, core_count)
        ]
    )
    nearest = np.argmin(distance_to_core, axis=0)
    return [mask & (nearest == index) for index in range(core_count - 1)]


def distance_to_box_edge(u: int, v: int, box: tuple[int, int, int, int]) -> int:
    """Distance from a pixel to the nearest edge of an axis-aligned box.

    Args:
        u: column.
        v: row.
        box: (x, y, width, height) as from ``cv2.boundingRect``.

    Returns:
        Pixels to the closest edge (0 or negative if outside).
    """
    x, y, width, height = box
    return min(u - x, x + width - 1 - u, v - y, y + height - 1 - v)


def colour_groups(class_names: list[str]) -> list[list[int]]:
    """Group colour classes that segment together: each hue alone, white and grey as one.

    White and grey are merged because a plate's pattern or a part's shading alternates
    between them; splitting there would only fragment one object.

    Args:
        class_names: names returned by ``colour_classes``.

    Returns:
        List of lists of class indices.
    """
    neutral = [class_names.index(NEUTRAL_WHITE), class_names.index(NEUTRAL_GREY)]
    return [[index] for index in range(len(class_names)) if index not in neutral] + [neutral]


def find_regions(rgb: np.ndarray, config: dict, mm2_per_work_pixel: float) -> dict:
    """Segment a photo into trays and colour-consistent object regions.

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        config: the ``perception`` section of the pipeline config.
        mm2_per_work_pixel: table area covered by one work-scale pixel, from calibration.

    Returns:
        Dict with ``trays`` (name -> (h, w) bool), ``work_scale`` and ``regions``: a list of
        dicts with ``id`` (int from 1), ``mask`` ((h, w) bool at work scale), ``colour``,
        ``colour_group`` (int), ``part`` (regions split from one same-colour part share it),
        ``area_mm2``, ``pixel`` (full-resolution [u, v]), ``tray`` and ``blob``; and ``blobs``:
        blob id -> {"mask": (h, w) bool, "region_ids": [...]} for every connected foreground
        blob that kept at least one region.
    """
    work_scale = config["work_scale"]
    small = to_work_scale(rgb, work_scale)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    trays = find_trays(hsv, config)
    mask = foreground(hsv, trays, config)
    classes, class_names = colour_classes(hsv, config)
    min_pixels = config["min_region_area_mm2"] / mm2_per_work_pixel
    work_pixels_per_mm = 1 / np.sqrt(mm2_per_work_pixel)
    min_depth_pixels = config["min_thickness_mm"] / 2 * work_pixels_per_mm
    edge_margin_pixels = config["tray_edge_margin_mm"] * work_pixels_per_mm
    tray_boxes = {name: cv2.boundingRect(tray.astype(np.uint8)) for name, tray in trays.items()}
    small_kernel = np.ones((3, 3), np.uint8)
    regions = []
    blobs = {}
    blob_count, blob_labels = cv2.connectedComponents(mask.astype(np.uint8))
    for blob in range(1, blob_count):
        blob_mask = blob_labels == blob
        if blob_mask.sum() < min_pixels:
            continue
        for group_index, group in enumerate(colour_groups(class_names)):
            part = blob_mask & np.isin(classes, group)
            part = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_OPEN, small_kernel)
            part_count, part_labels = cv2.connectedComponents(part)
            for part_index in range(1, part_count):
                part_mask = part_labels == part_index
                if part_mask.sum() < min_pixels:
                    continue
                for region_mask in split_touching(fill_holes(part_mask), config["core_fraction"]):
                    depth = cv2.distanceTransform(region_mask.astype(np.uint8), cv2.DIST_L2, 5)
                    if region_mask.sum() < min_pixels or depth.max() < min_depth_pixels:
                        continue
                    u, v = interior_point(region_mask)
                    tray = max(trays, key=lambda name: (trays[name] & region_mask).sum())
                    if distance_to_box_edge(u, v, tray_boxes[tray]) < edge_margin_pixels:
                        continue
                    majority = np.bincount(classes[region_mask], minlength=len(class_names))
                    regions.append(
                        {
                            "id": len(regions) + 1,
                            "mask": region_mask,
                            "colour": class_names[int(np.argmax(majority))],
                            "colour_group": group_index,
                            "part": (blob, group_index, part_index),
                            "area_mm2": round(float(region_mask.sum() * mm2_per_work_pixel)),
                            "pixel": [round(u / work_scale), round(v / work_scale)],
                            "tray": tray,
                            "blob": blob,
                        }
                    )
                    blobs.setdefault(blob, {"mask": blob_mask, "region_ids": []})
                    blobs[blob]["region_ids"].append(regions[-1]["id"])
    return {"trays": trays, "work_scale": work_scale, "regions": regions, "blobs": blobs}
