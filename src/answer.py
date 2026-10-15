"""Turn the actor's candidate into one answer in the official result schema."""

import json
from pathlib import Path

import jsonschema

RESULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "result.schema.json"
RESULT_VALIDATOR = jsonschema.Draft202012Validator(json.loads(RESULT_SCHEMA_PATH.read_text()))


def to_full_resolution(pixel: list[int], scale: float) -> list[int]:
    """Map a pixel from the resized photo back to the full-resolution photo.

    Args:
        pixel: ``[u, v]`` in the resized photo.
        scale: resized size divided by full size.

    Returns:
        ``[u, v]`` in the full photo.
    """
    return [round(coordinate / scale) for coordinate in pixel]


def refusal(instruction_id: str, reason: str) -> dict:
    """Build an explicit refusal, used when no valid answer could be produced.

    Args:
        instruction_id: id of the instruction.
        reason: why the pipeline could not answer.

    Returns:
        Schema-valid result dict.
    """
    return {
        "instruction_id": instruction_id,
        "target_pixel": None,
        "actions": [{"action": "refuse", "reason": reason}],
    }


def candidate_to_result(instruction_id: str, candidate: dict, scale: float) -> dict:
    """Convert a candidate to a result entry and check it against ``result.schema.json``.

    Args:
        instruction_id: id of the instruction.
        candidate: ``Candidate`` as a dict, pixels in the resized photo.
        scale: resized size divided by full size.

    Returns:
        Schema-valid result dict; an invalid candidate becomes an explicit refusal naming the
        schema error.
    """
    decision = candidate["decision"]
    if decision == "ask_user":
        result = {
            "instruction_id": instruction_id,
            "target_pixel": None,
            "actions": [{"action": "ask_user", "question": candidate["question"]}],
        }
    elif decision == "refuse":
        result = refusal(instruction_id, candidate["reason"])
    else:
        target_pixel = candidate["target_pixel"]
        result = {
            "instruction_id": instruction_id,
            "target_pixel": to_full_resolution(target_pixel, scale) if target_pixel else None,
            "actions": [
                {"action": step["action"], "object": step["object"]}
                for step in candidate["actions"]
            ],
        }
    errors = [error.message for error in RESULT_VALIDATOR.iter_errors(result)]
    starts_with_pick = decision != "plan" or (
        result["actions"] and result["actions"][0]["action"] == "pick"
    )
    if not starts_with_pick:
        errors.append("a plan must start with pick")
    if errors:
        return refusal(instruction_id, f"No valid answer was produced ({errors[0]}).")
    return result
