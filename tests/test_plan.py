"""CPU checks for the code that turns Apertus's moves into scored actions (no model, no GPU).

python -m pytest tests -q
"""

from src import pipeline
from src.pipeline import (
    clear_tops_first,
    drop_redundant_holds,
    moves_to_steps,
    plan_problem,
)
from src.reasoning.client import ModelReplyError, close_stalled_reply

GEAR = {"id": "red_gear_1", "size_mm": (88, 86), "under": ["yellow_gear_1"]}
SMALL = {"id": "yellow_gear_1", "size_mm": (48, 47), "under": []}
PLATE = {"id": "white_plate_1", "size_mm": (189, 188), "under": []}
OBJECTS = [GEAR, SMALL, PLATE]


def test_object_on_top_is_moved_aside_first() -> None:
    steps = clear_tops_first(
        moves_to_steps([{"object": "red_gear_1", "destination": "grey_tray"}]), OBJECTS
    )
    assert [step["object"] for step in steps] == [
        "yellow_gear_1",
        "aside",
        "red_gear_1",
        "grey_tray",
    ]


def test_hold_continued_by_same_object_is_dropped() -> None:
    moves = [
        {"object": "red_gear_1", "destination": "hold"},
        {"object": "red_gear_1", "destination": "black_tray"},
    ]
    assert drop_redundant_holds(moves) == moves[1:]


def test_bad_plans_are_rejected() -> None:
    assert plan_problem([], OBJECTS, 120) is not None
    assert plan_problem([{"object": "red_gear_1", "destination": "red_gear_1"}], OBJECTS, 120)
    assert plan_problem([{"object": "white_plate_1", "destination": "aside"}], OBJECTS, 120)
    assert plan_problem([{"object": "none", "destination": "aside"}], OBJECTS, 120)
    assert plan_problem([{"object": "red_gear_1", "destination": "hold"}], OBJECTS, 120) is None


def test_stalled_reply_is_closed_only_when_complete() -> None:
    schema = {"required": ["decision", "moves"]}
    assert close_stalled_reply('{"decision": "plan", "moves": []\n \n ', schema) == {
        "decision": "plan",
        "moves": [],
    }
    assert close_stalled_reply('{"decision": "plan", "mov', schema) is None


def test_cut_off_safety_reply_is_retried_with_feedback(monkeypatch) -> None:
    calls = []

    def check(text, scene_text, endpoint, config, feedback):
        calls.append(feedback)
        if feedback is None:
            raise ModelReplyError("reply cut off at 500 tokens", "{ ...")
        reply = {"manner": "push, knock or sweep off", "involves_person": False}
        return {**reply, "contradicts_itself": False, "message": "unsafe"}, "raw"

    monkeypatch.setattr(pipeline, "check_instruction", check)
    config = {"safety": {"enabled": True}, "decision": {"max_attempts": 2}}
    perceived = {"objects": OBJECTS, "text": ""}
    answer, log = pipeline.answer_instruction("i1", "push it off", perceived, config, {}, 85)
    assert calls[0] is None and "cut off" in calls[1]
    assert answer["actions"] == [{"action": "refuse", "reason": "unsafe"}]
    assert len(log["safety"]) == 2
