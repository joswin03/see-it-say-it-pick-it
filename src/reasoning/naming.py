"""Apertus names each OpenCV region from a close-up crop with the region outlined.

This is Set-of-Mark prompting (Yang et al. 2023, arXiv 2310.11441) one mark at a time: OpenCV
says *where*, Apertus says *what*. Asking about one outlined region per call is more reliable
for an 8B model than reading many numbered marks from one cluttered image.
"""

import cv2
import numpy as np

from src.reasoning.client import ask_json, image_url

NAMING_PROMPT = """This is a close-up from a camera looking straight down at a robot workcell table.
The area outlined in cyan was found by colour segmentation. Its colour is mostly {colour}; it is
about {length_mm} x {width_mm} mm and lies in the {tray}.

What is the outlined thing? Answer with its colour followed by the object type, 2 to 3 words,
for example "red cube", "yellow gear", "white mug", "orange mandarin". Object types seen in
workcells include: {examples}.
Set is_object to false if the outline is not a separate physical object: a reflection, a shadow,
part of the tray wall or the table, or a highlight. A human hand or arm is an object.
If the outline covers only part of an object, name the whole object."""

NAMING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "is_object"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 40},
        "is_object": {"type": "boolean"},
    },
}

OUTLINE_BGR_THICKNESS = 3
OUTLINE_RGB = (0, 255, 255)


def outlined_crop(
    rgb: np.ndarray, mask: np.ndarray, work_scale: float, padding_fraction: float
) -> np.ndarray:
    """Cut a padded box around one region from the full photo and outline the region in it.

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        mask: (h, w) bool region mask at work scale.
        work_scale: scale of ``mask`` relative to ``rgb``.
        padding_fraction: context added on each side, relative to the box size.

    Returns:
        (crop_h, crop_w, 3) uint8 crop with the region outline drawn.
    """
    full_mask = cv2.resize(
        mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    x, y, width, height = cv2.boundingRect(full_mask)
    pad_x, pad_y = int(width * padding_fraction), int(height * padding_fraction)
    left, top = max(0, x - pad_x), max(0, y - pad_y)
    right, bottom = min(rgb.shape[1], x + width + pad_x), min(rgb.shape[0], y + height + pad_y)
    crop = rgb[top:bottom, left:right].copy()
    contours, _ = cv2.findContours(
        full_mask[top:bottom, left:right], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    thickness = max(OUTLINE_BGR_THICKNESS, round(max(crop.shape[:2]) / 200))
    cv2.drawContours(crop, contours, -1, OUTLINE_RGB, thickness)
    return crop


def name_region(
    rgb: np.ndarray, region: dict, work_scale: float, endpoint: dict, config: dict
) -> dict:
    """Ask Apertus what one region is.

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        region: one region from ``find_regions`` with ``size_mm`` added by the scene graph.
        work_scale: scale of the region mask relative to ``rgb``.
        endpoint: model endpoint settings for ``ask_json``.
        config: the ``naming`` section of the pipeline config.

    Returns:
        Dict with ``name`` (lower case), ``is_object`` (False also when the name is a fixture
        such as the tray or the table, which are places rather than objects) and ``raw``.
    """
    crop = outlined_crop(rgb, region["mask"], work_scale, config["crop_padding_fraction"])
    length_mm, width_mm = region["size_mm"]
    prompt = NAMING_PROMPT.format(
        colour=region["colour"],
        length_mm=length_mm,
        width_mm=width_mm,
        tray=region["tray"].replace("_", " "),
        examples=", ".join(config["example_names"]),
    )
    reply, raw = ask_json(
        endpoint,
        prompt,
        [image_url(crop, config["crop_max_side_px"])],
        NAMING_SCHEMA,
        config["max_tokens"],
    )
    name = reply["name"].strip().lower()
    is_fixture = any(word in name.split() for word in config["fixture_words"])
    return {"name": name, "is_object": reply["is_object"] and not is_fixture, "raw": raw}
