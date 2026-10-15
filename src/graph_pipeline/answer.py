"""Answers in the official result schema, checked against ``schema/result.schema.json``."""

import json
from pathlib import Path

import jsonschema

RESULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "result.schema.json"
RESULT_VALIDATOR = jsonschema.Draft202012Validator(json.loads(RESULT_SCHEMA_PATH.read_text()))
MESSAGE_FIELD = {"ask_user": "question", "refuse": "reason"}
GIVE_UP_QUESTION = "I could not form an executable plan ({problem}). Could you rephrase it?"


def no_plan_answer(instruction_id: str, action: str, message: str) -> dict:
    """Build an ``ask_user`` or ``refuse`` answer.

    Args:
        instruction_id: id of the instruction.
        action: ``"ask_user"`` or ``"refuse"``.
        message: the question or the reason.

    Returns:
        Result dict with ``target_pixel`` null and one action.
    """
    return {
        "instruction_id": instruction_id,
        "target_pixel": None,
        "actions": [{"action": action, MESSAGE_FIELD[action]: message}],
    }


def give_up_answer(instruction_id: str, problem: str) -> dict:
    """Ask the user to rephrase when no executable plan was found within the attempts.

    Args:
        instruction_id: id of the instruction.
        problem: why the last plan was rejected.

    Returns:
        ``ask_user`` result dict.
    """
    return no_plan_answer(instruction_id, "ask_user", GIVE_UP_QUESTION.format(problem=problem))


def checked(result: dict) -> dict:
    """Check an answer against the official schema and the scorer's plan rules.

    Args:
        result: answer dict.

    Returns:
        The answer unchanged when valid; otherwise an explicit refusal naming the error.
    """
    errors = [error.message for error in RESULT_VALIDATOR.iter_errors(result)]
    actions = result.get("actions") or [{}]
    if actions[0].get("action") in ("pick", "place"):
        if actions[0]["action"] != "pick":
            errors.append("a plan must start with pick")
        if result.get("target_pixel") is None:
            errors.append("a plan needs target_pixel")
    if not errors:
        return result
    reason = f"No valid answer was produced ({errors[0]})."
    return no_plan_answer(result["instruction_id"], "refuse", reason)
