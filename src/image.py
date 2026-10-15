"""Resize, encode and mark workcell photos for the model."""

import base64
from dataclasses import dataclass

import cv2
import numpy as np

MARKER_COLOUR_RGB = (255, 0, 0)


@dataclass
class ModelImage:
    """A photo as sent to the model, with the factor back to full resolution."""

    rgb: np.ndarray
    width: int
    height: int
    scale: float


def resize_for_model(rgb: np.ndarray, max_side: int) -> ModelImage:
    """Shrink a photo so its long side is at most ``max_side`` pixels.

    Args:
        rgb: full-resolution photo, shape [H, W, 3], uint8 RGB.
        max_side: largest allowed width or height in pixels.

    Returns:
        The resized photo with its size and the scale ``small / full``.
    """
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        size = (round(width * scale), round(height * scale))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return ModelImage(rgb=rgb, width=rgb.shape[1], height=rgb.shape[0], scale=scale)


def to_data_url(rgb: np.ndarray, jpeg_quality: int) -> str:
    """Encode a photo as a JPEG data URL for an OpenAI-compatible chat message.

    Args:
        rgb: photo, shape [H, W, 3], uint8 RGB.
        jpeg_quality: JPEG quality from 0 to 100.

    Returns:
        ``data:image/jpeg;base64,...`` string.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    encoded, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not encoded:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


def draw_target(rgb: np.ndarray, pixel: list[int], radius: int, thickness: int) -> np.ndarray:
    """Draw a circled cross at a pixel so the judge can see where the plan points.

    Args:
        rgb: photo, shape [H, W, 3], uint8 RGB; not modified.
        pixel: ``[u, v]`` in the same photo.
        radius: circle radius and half the cross length, in pixels.
        thickness: line thickness in pixels.

    Returns:
        A marked copy of the photo.
    """
    marked = rgb.copy()
    u, v = pixel
    cv2.circle(marked, (u, v), radius, MARKER_COLOUR_RGB, thickness)
    cv2.line(marked, (u - radius, v), (u + radius, v), MARKER_COLOUR_RGB, thickness)
    cv2.line(marked, (u, v - radius), (u, v + radius), MARKER_COLOUR_RGB, thickness)
    return marked
