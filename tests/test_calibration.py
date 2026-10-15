"""CPU checks for table-plane measurement from the calibration (no model, no GPU)."""

import cv2
import numpy as np

from src.scene_graph import size_mm
from src.workcell import TABLE_Z_MM, load_scene, pixel_to_table, table_xy

SCENE = load_scene("dataset/scene_01")


def test_table_xy_matches_starter_projection_without_distortion() -> None:
    plain = load_scene("dataset/scene_01")
    plain.distortion = np.zeros(5)
    pixels = [[100, 100], [1224, 1024], [2300, 1900]]
    expected = [pixel_to_table(plain, u, v)[:2] for u, v in pixels]
    assert np.allclose(table_xy(plain, pixels), expected, atol=1e-6)


def square_mask(centre_pixel: list[int], side_mm: float) -> np.ndarray:
    """Work-scale (0.25) mask of a square of ``side_mm`` lying flat on the table at a pixel."""
    centre = table_xy(SCENE, [centre_pixel])[0]
    half = side_mm / 2
    corners = [
        centre + [dx, dy] for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))
    ]
    world = np.column_stack([corners, np.full(4, TABLE_Z_MM)])
    camera_T_base = np.linalg.inv(SCENE.base_T_camera)
    rvec, _ = cv2.Rodrigues(camera_T_base[:3, :3])
    pixels, _ = cv2.projectPoints(world, rvec, camera_T_base[:3, 3], SCENE.K, SCENE.distortion)
    mask = np.zeros((512, 612), np.uint8)
    cv2.fillPoly(mask, [np.round(pixels.reshape(-1, 2) * 0.25).astype(np.int32)], 1)
    return mask.astype(bool)


def test_flat_square_measures_the_same_at_top_and_bottom_of_the_tilted_view() -> None:
    for pixel in ([1224, 250], [1224, 1800], [300, 1000]):
        length, width = size_mm(square_mask(pixel, 80), 0.25, SCENE)
        assert abs(length - 80) <= 3 and abs(width - 80) <= 3, (pixel, length, width)
