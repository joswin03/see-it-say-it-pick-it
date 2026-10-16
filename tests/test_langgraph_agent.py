"""CPU check of the agent's tools on the ground-truth scene graph of scene_06."""

import json
from pathlib import Path

import yaml

from src.workcell import load_scene
from tools.gt_scene_graph import perceive_ground_truth
from tools.langgraph_agent import scene_tools

CONFIG = yaml.safe_load(Path("configs/opencv_apertus.yaml").read_text())


def test_scene_06_tools() -> None:
    objects = perceive_ground_truth(load_scene(Path("dataset/scene_06")), CONFIG, "json")["objects"]
    tools = {t.name: t for t in scene_tools(objects, CONFIG["scene_graph"]["too_wide_mm"])}
    on_top = json.loads(tools["what_is_on_top"].invoke({"object_id": "red_gear_2"}))
    assert on_top["objects_on_top_of_it"] == ["yellow_gear"] and on_top["lies_on_top_of"] == []
    assert json.loads(tools["what_is_on_top"].invoke({"object_id": "yellow_gear"}))["lies_on_top_of"] == ["red_gear_2"]
    assert json.loads(tools["can_grip"].invoke({"object_id": "saucer"}))["graspable"] is False
    assert json.loads(tools["can_grip"].invoke({"object_id": "red_cube"}))["graspable"] is True
    assert "error" in json.loads(tools["can_grip"].invoke({"object_id": "blue_gear"}))
