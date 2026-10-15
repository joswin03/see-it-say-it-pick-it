"""CPU checks for merging the regions of one object into one scene-graph object."""

import numpy as np

from src.scene_graph import merge_same_object

CONFIG = {"fragment_max_mm2": 1100, "same_object_nouns": [["cup", "mug"], ["plate", "saucer"]]}


def region(region_id, name, box, area_mm2, colour_group, part=None):
    mask = np.zeros((100, 200), bool)
    top, bottom, left, right = box
    mask[top:bottom, left:right] = True
    return {
        "id": region_id,
        "name": name,
        "mask": mask,
        "area_mm2": area_mm2,
        "colour_group": colour_group,
        "part": part or (region_id, colour_group, 1),
    }


def test_mug_rim_and_inside_with_different_names_become_one_object() -> None:
    inside = region(1, "yellow cup", (20, 80, 20, 80), 6000, 2)
    rim = region(2, "white mug", (20, 80, 80, 100), 1500, 6)
    objects = merge_same_object([inside, rim], 2, CONFIG)
    assert len(objects) == 1 and sorted(objects[0]["source_ids"]) == [1, 2]


def test_small_fragment_joins_the_object_it_touches() -> None:
    cube = region(1, "red block", (20, 80, 20, 80), 6000, 0)
    side_face = region(2, "grey block", (20, 80, 80, 86), 300, 6)
    gear = region(3, "yellow gear", (20, 60, 150, 190), 1400, 2)
    objects = merge_same_object([cube, side_face, gear], 2, CONFIG)
    assert [sorted(item["source_ids"]) for item in objects] == [[1, 2], [3]]
    assert objects[0]["name"] == "red block"


def test_fragment_prefers_a_touching_neighbour_of_its_own_colour() -> None:
    part = region(1, "grey metal part", (20, 80, 20, 100), 8000, 6)
    rim = region(2, "orange plate", (0, 20, 100, 200), 2000, 1)
    sliver = region(3, "orange cube", (20, 30, 100, 110), 400, 1)
    objects = merge_same_object([part, rim, sliver], 2, CONFIG)
    assert sorted(next(item for item in objects if item["id"] == 2)["source_ids"]) == [2, 3]


def test_small_gear_lying_on_a_large_gear_stays_separate() -> None:
    large = region(1, "red gear", (10, 90, 10, 90), 5500, 0)
    small = region(2, "yellow gear", (30, 60, 30, 60), 1300, 2)
    assert len(merge_same_object([large, small], 2, CONFIG)) == 2


def test_gears_split_on_purpose_are_not_merged_back() -> None:
    left = region(1, "red gear", (20, 80, 20, 80), 5500, 0, part=(4, 0, 1))
    right = region(2, "red gear", (20, 80, 80, 140), 5500, 0, part=(4, 0, 1))
    assert len(merge_same_object([left, right], 2, CONFIG)) == 2
