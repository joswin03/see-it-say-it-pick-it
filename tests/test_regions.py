"""CPU checks for the OpenCV rules that separate same-colour stacks."""

import numpy as np

from src.perception.regions import split_by_texture

SMOOTH, GRAINY = 0.5, 2.0


def test_smooth_object_on_grainy_one_is_split_off() -> None:
    mask = np.zeros((100, 160), bool)
    mask[20:80, 10:150] = True
    texture = np.full(mask.shape, GRAINY, np.float32)
    texture[25:75, 55:105] = SMOOTH
    pieces = split_by_texture(mask, texture, 1.0, 1000, 5)
    assert len(pieces) == 2
    assert pieces[0][50, 80] and not pieces[1][50, 80]


def test_single_smooth_object_with_grainy_rim_stays_whole() -> None:
    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    texture = np.full(mask.shape, SMOOTH, np.float32)
    texture[20:24, 20:80] = GRAINY
    assert len(split_by_texture(mask, texture, 1.0, 1000, 5)) == 1
