"""Turn the decide reply into one answer in the official result schema."""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "starter"))

from evaluate import answer_type

from src.marks import box_centre

MESSAGE_FIELD = {"ask_user": "question", "refuse": "reason"}


def refusal(instruction_id: str, reason: str) -> dict:
    """Build an explicit refusal, used when the reply cannot be turned into a valid answer.

    Args:
        instruction_id: id of the instruction.
        reason: why no valid answer was produced.

    Returns:
        Schema-valid refusal.
    """
    return {
        "instruction_id": instruction_id,
        "target_pixel": None,
        "actions": [{"action": "refuse", "reason": reason}],
    }


def place_name(value, by_number: dict[int, dict], places: list[str]) -> str | None:
    """Resolve a place given by the model: a place name or the number of an object.

    Args:
        value: the reply's ``object`` field of a place action.
        by_number: marked objects by number.
        places: allowed place names from the config.

    Returns:
        The place name or object id, or None if it is neither.
    """
    if isinstance(value, str) and value in places:
        return value
    return object_id(value, by_number)


def object_id(value, by_number: dict[int, dict]) -> str | None:
    """Resolve an object number given by the model (int or digit string) to its id.

    Args:
        value: the reply's ``object`` field.
        by_number: marked objects by number.

    Returns:
        The object id, or None if the number is not a marked object.
    """
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if isinstance(value, int) and value in by_number:
        return by_number[value]["id"]
    return None


def to_result(instruction_id: str, reply: dict | None, marks: list[dict], places: list[str]):
    """Convert the decide reply into a result; target_pixel is the centre of the first pick.

    Args:
        instruction_id: id of the instruction.
        reply: parsed reply ``{"actions": [...]}``, or None.
        marks: marked objects of the scene.
        places: allowed place names.

    Returns:
        Schema-valid result dict; anything that cannot be resolved becomes an explicit
        refusal naming the problem.
    """
    actions = reply.get("actions") if isinstance(reply, dict) else None
    if not isinstance(actions, list) or not actions or not isinstance(actions[0], dict):
        return refusal(instruction_id, "The model reply held no actions.")
    first = actions[0].get("action")
    if first in MESSAGE_FIELD:
        field = MESSAGE_FIELD[first]
        result = {
            "instruction_id": instruction_id,
            "target_pixel": None,
            "actions": [{"action": first, field: str(actions[0].get(field, ""))}],
        }
    else:
        by_number = {mark["number"]: mark for mark in marks}
        steps = []
        for action in actions:
            kind = action.get("action") if isinstance(action, dict) else None
            if kind == "pick":
                name = object_id(action.get("object"), by_number)
            elif kind == "place":
                name = place_name(action.get("object"), by_number, places)
            else:
                return refusal(instruction_id, f"The plan has an unknown action ({kind}).")
            if name is None:
                return refusal(instruction_id, f"The plan names no marked object ({action}).")
            steps.append({"action": kind, "object": name})
        first_pick = next((step for step in steps if step["action"] == "pick"), None)
        target = next((m for m in marks if first_pick and m["id"] == first_pick["object"]), None)
        result = {
            "instruction_id": instruction_id,
            "target_pixel": box_centre(target["box"]) if target else None,
            "actions": steps,
        }
    _kind, error = answer_type(result)
    if error:
        return refusal(instruction_id, f"No valid answer was produced ({error}).")
    return result
