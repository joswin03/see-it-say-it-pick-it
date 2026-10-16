"""Apertus names one OpenCV region at a time, from a close-up with the region outlined.

OpenCV says where a region is, Apertus says what it is. One outlined region per call is easier
for an 8B model than reading many marks in one cluttered photo.
"""

import logging

import cv2
import numpy as np

from src.apertus import Apertus
from src.structured import ModelReplyError

NAMING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "other_names", "is_object"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 40},
        "other_names": {
            "type": "array",
            "maxItems": 2,
            "items": {"type": "string", "minLength": 1, "maxLength": 40},
        },
        "is_object": {"type": "boolean"},
    },
}

COLOUR_WORDS = {
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "white",
    "grey",
    "gray",
    "black",
    "silver",
    "cyan",
}

OUTLINE_RGB = (0, 255, 255)
MIN_OUTLINE_PX = 3
OUTLINE_PER_CROP_PX = 200
UNKNOWN_NAME = "unknown object"
LOGGED_REPLY_CHARACTERS = 300
logger = logging.getLogger(__name__)


def outlined_crop(
    rgb: np.ndarray, mask: np.ndarray, padding_fraction: float, min_side_pixels: int
) -> np.ndarray:
    """Cut a padded box around one region from the full photo and outline the region in it.

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        mask: (h, w) bool region mask at work scale.
        padding_fraction: context added on each side, relative to the box size.
        min_side_pixels: the crop is at least this wide and high, so a small fragment is shown
            with enough surroundings to be recognised.

    Returns:
        (crop_h, crop_w, 3) uint8 crop with the region outline drawn.
    """
    full_mask = cv2.resize(
        mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    x, y, width, height = cv2.boundingRect(full_mask)
    pad_x = max(int(width * padding_fraction), (min_side_pixels - width) // 2)
    pad_y = max(int(height * padding_fraction), (min_side_pixels - height) // 2)
    left, top = max(0, x - pad_x), max(0, y - pad_y)
    right, bottom = min(rgb.shape[1], x + width + pad_x), min(rgb.shape[0], y + height + pad_y)
    crop = rgb[top:bottom, left:right].copy()
    contours, _ = cv2.findContours(
        full_mask[top:bottom, left:right], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    thickness = max(MIN_OUTLINE_PX, round(max(crop.shape[:2]) / OUTLINE_PER_CROP_PX))
    cv2.drawContours(crop, contours, -1, OUTLINE_RGB, thickness)
    return crop


def name_region(
    apertus: Apertus, template: str, fields: dict, crop: np.ndarray, fixture_words: list[str]
) -> tuple[str, bool, list[str]]:
    """Ask Apertus what one outlined region is.

    A reply that is not valid JSON keeps the region as "unknown object" (logged): it was found
    in the photo, so decide may still ask about it.

    Args:
        apertus: model client.
        template: naming prompt template.
        fields: prompt fields (colour, size, tray, labels).
        crop: output of ``outlined_crop``.
        fixture_words: words that make a name a place rather than an object (tray, table, ...).

    Returns:
        (name in lower case, whether it is a separate physical object, aliases).
    """
    try:
        prompt = template.format(
            **fields, examples=", ".join(apertus.config["naming"]["example_names"])
        )
        reply, raw = apertus.ask_json(
            "name",
            prompt,
            NAMING_SCHEMA,
            apertus.config["naming"]["max_tokens"],
            crop,
            apertus.config["naming"]["crop_max_side_px"],
        )
    except ModelReplyError as error:
        logger.warning("name reply invalid: %s: %s", error, error.raw[:LOGGED_REPLY_CHARACTERS])
        return UNKNOWN_NAME, True, []
    name = reply.get("name") if isinstance(reply, dict) else None
    if not isinstance(name, str) or not name.strip():
        logger.warning("name reply held no name: %s", raw[:LOGGED_REPLY_CHARACTERS])
        return UNKNOWN_NAME, True, []
    name = name.strip().lower()
    if not COLOUR_WORDS & set(name.split()):
        name = f"{fields['colour']} {name}"
    aliases = list(dict.fromkeys(other.strip().lower() for other in reply.get("other_names", [])))
    is_fixture = any(word in name.split() for word in fixture_words)
    return (
        name,
        bool(reply.get("is_object", True)) and not is_fixture,
        [alias for alias in aliases if alias != name],
    )
