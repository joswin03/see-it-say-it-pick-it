"""Find and name objects with an open VLM alone (Qwen3-VL), no OpenCV: the comparison run.

Qwen3-VL returns boxes in a 0-1000 frame relative to the image. Each box becomes a filled
rectangle mask at work scale, so the rest of the pipeline (scene graph, planning) is unchanged.
Boxes are coarser than OpenCV outlines: sizes include background and stacking is judged from
box overlap.
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
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "tray", "bbox_2d"],
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 40},
            "tray": {"enum": TRAYS},
            "bbox_2d": box,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["objects"],
        "properties": {"objects": {"type": "array", "maxItems": max_objects, "items": item}},
    }


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
    regions = []
    for index, item in enumerate(reply["objects"], start=1):
        x1, y1, x2, y2 = (
            min(max(value, 0), RELATIVE_FRAME) / RELATIVE_FRAME for value in item["bbox_2d"]
        )
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
                "pixel": [round((left + right) / 2 * width), round((top + bottom) / 2 * height)],
                "tray": item["tray"],
                "name": item["name"].strip().lower(),
                "is_object": True,
            }
        )
    return regions, raw
