"""Access to the organisers' scene loader and calibration (``starter/scene_io.py``)."""

import sys
from pathlib import Path

import numpy as np

STARTER_DIR = Path(__file__).resolve().parent.parent / "starter"
sys.path.insert(0, str(STARTER_DIR))

from scene_io import Scene, load_scene, pixel_to_table

__all__ = ["Scene", "load_scene", "mm_per_pixel"]


def mm_per_pixel(scene: Scene) -> float:
    """Table distance covered by one full-resolution pixel at the image centre.

    The camera looks almost straight down, so this scale holds across the table to within a
    few percent.

    Args:
        scene: a loaded scene with calibration.

    Returns:
        Millimetres per pixel on the table plane.
    """
    width, height = scene.size
    centre = pixel_to_table(scene, width / 2, height / 2)
    neighbour = pixel_to_table(scene, width / 2 + 1, height / 2)
    return float(np.linalg.norm(neighbour - centre))
