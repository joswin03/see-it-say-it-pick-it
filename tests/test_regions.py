"""CPU checks for the OpenCV rules that separate same-colour stacks."""

import numpy as np

from src.perception.regions import drop_attached_slivers, split_by_texture

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


def test_sliver_on_an_object_is_dropped_but_a_lone_thin_object_is_kept() -> None:
    block = np.zeros((100, 200), bool)
    block[20:80, 20:80] = True
    shadow = np.zeros_like(block)
    shadow[20:80, 80:86] = True
    pen = np.zeros_like(block)
    pen[40:46, 120:190] = True
    regions = [{"id": i + 1, "mask": m} for i, m in enumerate([block, shadow, pen])]
    kept = drop_attached_slivers(regions, 5)
    assert [r["mask"] is m for r, m in zip(kept, [block, pen])] == [True, True]
    assert [r["id"] for r in kept] == [1, 2]
