"""Access to the organisers' scene loader and calibration (``starter/scene_io.py``)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

STARTER_DIR = Path(__file__).resolve().parent.parent / "starter"
sys.path.insert(0, str(STARTER_DIR))

import scene_io
from scene_io import TABLE_Z_MM, Scene, calibration_path, pixel_to_table

__all__ = ["Scene", "load_scene", "mm_per_pixel", "table_xy"]


def load_scene(folder) -> Scene:
    """Load a scene with the organisers' loader and add its lens distortion.

    The starter's ``Scene`` has no distortion field; ``table_xy`` needs it.

    Args:
        folder: scene folder.

    Returns:
        Scene with an extra ``distortion`` array (OpenCV order k1, k2, p1, p2, k3).
    """
    scene = scene_io.load_scene(folder)
    intrinsics = json.loads(calibration_path(folder).read_text())["camera"]["intrinsics"]
    scene.distortion = np.array(intrinsics.get("distortion_opencv", [0.0] * 5), dtype=float)
    return scene


def table_xy(scene: Scene, points: np.ndarray) -> np.ndarray:
    """Where pixels land on the table plane: lens distortion removed, camera tilt included.

    The same ray-plane intersection as ``pixel_to_table``, for many points at once and with the
    lens distortion undone first, so sizes and distances are right anywhere in the photo, not
    only at the image centre.

    Args:
        scene: loaded scene with calibration and ``distortion``.
        points: (N, 2) full-resolution pixels [u, v].

    Returns:
        (N, 2) table x, y in the UR5 base frame, mm.
    """
    undistorted = cv2.undistortPoints(
        np.asarray(points, np.float64).reshape(-1, 1, 2), scene.K, scene.distortion, P=scene.K
    ).reshape(-1, 2)
    rotation, origin = scene.base_T_camera[:3, :3], scene.base_T_camera[:3, 3]
    homogeneous = np.column_stack([undistorted, np.ones(len(undistorted))])
    rays = homogeneous @ (rotation @ np.linalg.inv(scene.K)).T
    reach = (TABLE_Z_MM - origin[2]) / rays[:, 2]
    return (origin + reach[:, None] * rays)[:, :2]


def mm_per_pixel(scene: Scene) -> float:
    """Table distance covered by one full-resolution pixel at the image centre.

    Used only for thresholds and crop sizes; sizes and distances use ``table_xy``, because
    the camera is tilted by about 8 degrees and the scale changes by up to 8% across the photo.

    Args:
        scene: a loaded scene with calibration.

    Returns:
        Millimetres per pixel on the table plane.
    """
    width, height = scene.size
    centre = pixel_to_table(scene, width / 2, height / 2)
    neighbour = pixel_to_table(scene, width / 2 + 1, height / 2)
    return float(np.linalg.norm(neighbour - centre))
