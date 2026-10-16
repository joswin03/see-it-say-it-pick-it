"""CPU check that the ground-truth scene graph feeds the pipeline like perception would."""

from pathlib import Path

import yaml

from src.workcell import load_scene
from tools.gt_scene_graph import perceive_ground_truth

CONFIG = yaml.safe_load(Path("configs/opencv_apertus.yaml").read_text())


def test_ground_truth_scene_06_as_text_and_json() -> None:
    scene = load_scene(Path("dataset/scene_06"))
    text = perceive_ground_truth(scene, CONFIG, "text")
    by_id = {item["id"]: item for item in text["objects"]}
    assert by_id["red_gear_2"]["under"] == ["yellow_gear"]
    assert by_id["red_gear_1"]["name"] == "red gear"
    assert "yellow_gear lies on red_gear_2" in text["text"]
    assert "saucer: saucer; " in text["text"] and "TOO WIDE TO GRIP" in text["text"]
    as_json = perceive_ground_truth(scene, CONFIG, "json")["text"]
    assert '"lies_on_top_of": "red_gear_2"' in as_json
    assert '"graspable": false, "can_hold_things": true}' in as_json
