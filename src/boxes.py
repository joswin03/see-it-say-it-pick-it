"""Ablation front end: Apertus marks every object with a label and a box on the whole photo.

This is the region proposal of the earlier prompting pipeline (branch ``dev/vaibhav-2``),
selected with ``perception.backend: apertus_boxes`` in place of OpenCV regions and per-crop
naming. The reply is constrained to a JSON schema; boxes are scaled back to the full photo and
each box becomes the object's mask, so everything after perception (stacking, scene text,
safety, decide, plan checks) is unchanged and the two front ends can be compared directly.
"""

import re

import cv2
import numpy as np

from src.apertus import Apertus
from src.calibration import Scene, table_xy
from src.objects import size_mm
from src.regions import find_trays, to_work_scale

BOX_VALUES = 4


def boxes_schema(max_objects: int, max_label_chars: int) -> dict:
    """JSON schema for the box reply.

    Args:
        max_objects: cap on the number of marked objects.
        max_label_chars: cap on one label.

    Returns:
        JSON schema dict.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["objects"],
        "properties": {
            "objects": {
                "type": "array",
                "maxItems": max_objects,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["label", "box"],
                    "properties": {
                        "label": {"type": "string", "maxLength": max_label_chars},
                        "box": {
                            "type": "array",
                            "minItems": BOX_VALUES,
                            "maxItems": BOX_VALUES,
                            "items": {"type": "integer"},
                        },
                    },
                },
            }
        },
    }


def detect_boxes(photo: np.ndarray, apertus: Apertus, config: dict) -> list[dict]:
    """Ask Apertus for every object's label and box, in full-resolution pixels.

    Boxes with no area after clipping to the photo are dropped; corners given in either order
    are sorted.

    Args:
        photo: (H, W, 3) uint8 RGB photo.
        apertus: model client.
        config: full pipeline config (``boxes`` section, ``prompts.boxes``, image size).

    Returns:
        ``[{"label", "box": [u_min, v_min, u_max, v_max]}]`` in full-resolution pixels.
    """
    settings = config["boxes"]
    height, width = photo.shape[:2]
    scale = min(1.0, config["image_max_side_px"] / max(height, width))
    prompt = config["prompts"]["boxes"].format(
        w=round(width * scale), h=round(height * scale), labels=", ".join(settings["object_labels"])
    )
    schema = boxes_schema(settings["max_objects"], settings["max_label_chars"])
    reply, _raw = apertus.ask_json("boxes", prompt, schema, settings["max_tokens"], photo)
    boxes = []
    for entry in reply["objects"]:
        u_a, v_a, u_b, v_b = (round(value / scale) for value in entry["box"])
        u_min, u_max = sorted((min(max(u_a, 0), width), min(max(u_b, 0), width)))
        v_min, v_max = sorted((min(max(v_a, 0), height), min(max(v_b, 0), height)))
        if u_max > u_min and v_max > v_min and entry["label"].strip():
            boxes.append({"label": entry["label"].strip(), "box": [u_min, v_min, u_max, v_max]})
    return boxes


def marks_from_boxes(
    boxes: list[dict], photo: np.ndarray, scene: Scene, distortion: np.ndarray, config: dict
) -> list[dict]:
    """Turn labelled boxes into numbered objects with the fields the rest of the pipeline uses.

    Args:
        boxes: output of ``detect_boxes``.
        photo: (H, W, 3) uint8 RGB photo.
        scene: loaded scene with calibration.
        distortion: lens distortion from ``calibration.lens_distortion``.
        config: full pipeline config (``perception`` and ``objects`` sections).

    Returns:
        Objects numbered left to right, as from ``objects.build_marks``: ``mask`` is the filled
        box at work scale, ``pixel`` the box centre, ``tray`` the tray holding most of the box.
    """
    perception, objects_config = config["perception"], config["objects"]
    work_scale = perception["work_scale"]
    small = to_work_scale(photo, work_scale)
    trays = find_trays(cv2.cvtColor(small, cv2.COLOR_RGB2HSV), perception)
    objects = []
    for entry in boxes:
        u_min, v_min, u_max, v_max = entry["box"]
        mask = np.zeros(small.shape[:2], bool)
        mask[
            round(v_min * work_scale) : max(
                round(v_max * work_scale), round(v_min * work_scale) + 1
            ),
            round(u_min * work_scale) : max(
                round(u_max * work_scale), round(u_min * work_scale) + 1
            ),
        ] = True
        length_mm, width_mm = size_mm(mask, work_scale, scene, distortion)
        centre = [(u_min + u_max) // 2, (v_min + v_max) // 2]
        words = entry["label"].lower().split()
        objects.append(
            {
                "label": entry["label"],
                "width_mm": width_mm,
                "too_wide": width_mm > objects_config["too_wide_mm"]
                or words[-1] in objects_config["unpickable_nouns"],
                "size_mm": [length_mm, width_mm],
                "table_xy": table_xy(scene, distortion, np.array([centre], float))[0].tolist(),
                "tray": max(trays, key=lambda name: int((trays[name] & mask).sum())),
                "aliases": [],
                "mask": mask,
                "is_hand": bool(set(words) & set(objects_config["hand_nouns"])),
                "pixel": centre,
                "box": entry["box"],
            }
        )
    objects.sort(key=lambda item: item["pixel"][0])
    marks = []
    for number, item in enumerate(objects, start=1):
        base = re.sub(r"[^a-z0-9]+", "_", item["label"].lower()).strip("_") or "object"
        marks.append({"number": number, "id": f"{base}_{number}", **item})
    return marks
