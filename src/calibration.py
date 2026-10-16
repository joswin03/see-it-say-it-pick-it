"""Camera calibration helpers on top of the organisers' scene loader: pixels to table millimetres.

``starter/scene_io.py`` gives the intrinsics and the camera pose; the lens distortion is read
from the same ``calibration.json``. Sizes and distances are measured on the table plane, so they
do not depend on where in the slightly tilted view an object lies.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "starter"))

from scene_io import TABLE_Z_MM, Scene, calibration_path, pixel_to_table

NO_DISTORTION = [0.0] * 5


def lens_distortion(scene_dir: Path) -> np.ndarray:
    """Read the lens distortion coefficients of a scene's camera.

    Args:
        scene_dir: scene folder (its own or its parent's ``calibration.json`` is used).

    Returns:
        (5,) float array in OpenCV order k1, k2, p1, p2, k3; zeros when none are given.
    """
    calibration = json.loads(calibration_path(scene_dir).read_text())
    intrinsics = calibration["camera"]["intrinsics"]
    return np.array(intrinsics.get("distortion_opencv", NO_DISTORTION), dtype=float)


def table_xy(scene: Scene, distortion: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Where pixels land on the table plane, with lens distortion removed and camera tilt included.

    Args:
        scene: loaded scene with ``K`` and ``base_T_camera``.
        distortion: output of ``lens_distortion``.
        points: (N, 2) full-resolution pixels [u, v].

    Returns:
        (N, 2) table x, y in the robot base frame, mm.
    """
    undistorted = cv2.undistortPoints(
        np.asarray(points, np.float64).reshape(-1, 1, 2), scene.K, distortion, P=scene.K
    ).reshape(-1, 2)
    rotation, origin = scene.base_T_camera[:3, :3], scene.base_T_camera[:3, 3]
    homogeneous = np.column_stack([undistorted, np.ones(len(undistorted))])
    rays = homogeneous @ (rotation @ np.linalg.inv(scene.K)).T
    reach = (TABLE_Z_MM - origin[2]) / rays[:, 2]
    return (origin + reach[:, None] * rays)[:, :2]


def mm_per_pixel(scene: Scene) -> float:
    """Table distance covered by one full-resolution pixel at the image centre.

    Used for thresholds and crop sizes only; sizes use ``table_xy``.

    Args:
        scene: loaded scene with calibration.

    Returns:
        Millimetres per pixel on the table plane.
    """
    width, height = scene.size
    centre = pixel_to_table(scene, width / 2, height / 2)
    neighbour = pixel_to_table(scene, width / 2 + 1, height / 2)
    return float(np.linalg.norm(neighbour - centre))
