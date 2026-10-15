"""Find and name objects with an open VLM alone (Qwen3-VL), no OpenCV: the comparison run.

Qwen3-VL returns boxes and points in a 0-1000 frame relative to the image. The VLM also boxes
the two trays once; each object's tray is the tray box containing its point (geometry, not a
label the VLM guesses per object). The pick pixel is the VLM's point on the object, which lands
on it more reliably than a box centre. Each box becomes a filled rectangle mask at work scale,
so the rest of the pipeline is unchanged; sizes and stacking are therefore box-coarse.
"""

import numpy as np

from src.reasoning.client import ask_json, image_url

RELATIVE_FRAME = 1000
TRAYS = ["grey_tray", "black_tray"]


def detection_schema(max_objects: int) -> dict:
    """JSON schema for the detection reply.

    Args:
        max_objects: cap on the number of objects.

    Returns:
        JSON schema dict.
    """
    box = {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer"}}
    point = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}}
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "bbox_2d", "point_2d"],
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 40},
            "bbox_2d": box,
            "point_2d": point,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["grey_tray_bbox_2d", "black_tray_bbox_2d", "objects"],
        "properties": {
            "grey_tray_bbox_2d": box,
            "black_tray_bbox_2d": box,
            "objects": {"type": "array", "maxItems": max_objects, "items": item},
        },
    }


def to_unit(values: list[int]) -> list[float]:
    """Clip 0-1000 VLM coordinates and scale them to 0-1.

    Args:
        values: coordinates in the VLM's relative frame.

    Returns:
        The same coordinates in [0, 1].
    """
    return [min(max(value, 0), RELATIVE_FRAME) / RELATIVE_FRAME for value in values]


def tray_of(point: list[float], tray_boxes: dict[str, list[float]]) -> str:
    """The tray whose box contains a point, else the tray whose box centre is nearest.

    Args:
        point: (x, y) in [0, 1].
        tray_boxes: tray name -> [x1, y1, x2, y2] in [0, 1].

    Returns:
        Tray name.
    """
    x, y = point
    for name, (x1, y1, x2, y2) in tray_boxes.items():
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return name
    return min(
        tray_boxes,
        key=lambda name: (
            (x - (tray_boxes[name][0] + tray_boxes[name][2]) / 2) ** 2
            + (y - (tray_boxes[name][1] + tray_boxes[name][3]) / 2) ** 2
        ),
    )


def detect_regions(rgb: np.ndarray, endpoint: dict, config: dict, work_scale: float) -> tuple:
    """Ask the VLM for every object and turn its boxes into named regions.

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        endpoint: the VLM's endpoint settings for ``ask_json``.
        config: the ``vlm_detection`` section of the pipeline config, including its ``prompt``.
        work_scale: work-scale factor for the masks.

    Returns:
        List of regions (``id``, ``mask`` (h, w) bool, ``colour``, ``part``, ``pixel``, ``tray``,
        ``name``, ``is_object``) and the raw reply.
    """
    height, width = rgb.shape[:2]
    work_shape = (round(height * work_scale), round(width * work_scale))
    reply, raw = ask_json(
        endpoint,
        config["prompt"],
        [image_url(rgb, config["image_max_side_px"])],
        detection_schema(config["max_objects"]),
        config["max_tokens"],
    )
    tray_boxes = {name: to_unit(reply[f"{name}_bbox_2d"]) for name in TRAYS}
    regions = []
    for index, item in enumerate(reply["objects"], start=1):
        x1, y1, x2, y2 = to_unit(item["bbox_2d"])
        point = to_unit(item["point_2d"])
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        mask = np.zeros(work_shape, bool)
        mask[
            round(top * work_shape[0]) : max(
                round(bottom * work_shape[0]), round(top * work_shape[0]) + 1
            ),
            round(left * work_shape[1]) : max(
                round(right * work_shape[1]), round(left * work_shape[1]) + 1
            ),
        ] = True
        regions.append(
            {
                "id": index,
                "mask": mask,
                "colour": "",
                "part": ("vlm", index),
                "pixel": [
                    min(round(point[0] * width), width - 1),
                    min(round(point[1] * height), height - 1),
                ],
                "tray": tray_of(point, tray_boxes),
                "name": item["name"].strip().lower(),
                "is_object": True,
            }
        )
    return regions, raw
