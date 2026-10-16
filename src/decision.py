"""Apertus decides plan / ask_user / refuse for one instruction over the scene graph.

The reply is constrained to a schema whose object fields only accept ids that exist in the
scene, so Apertus cannot invent objects or pixels. It writes a short analysis first, which
works as a chain of thought inside the constrained output; the free-text message comes before
the moves so that a finished plan is also the end of the JSON. The prompt is in the config.
"""

import numpy as np

from src.apertus import Apertus
from src.plans import prepare_moves
from src.safety import kept_objects_moved
from src.structured import ModelReplyError

HOLD = "hold"


def decision_schema(
    object_ids: list[str], max_analysis_chars: int, max_moves: int, places: list[str]
) -> dict:
    """JSON schema for the decision, with object fields limited to ids in the scene.

    Args:
        object_ids: ids from the scene graph.
        max_analysis_chars: cap on the analysis text.
        max_moves: cap on the number of moves in a plan.
        places: Destination names from the unified config.

    Returns:
        JSON schema dict.
    """
    targets = sorted(set(object_ids)) or ["none"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["analysis", "decision", "message", "moves"],
        "properties": {
            "analysis": {"type": "string", "maxLength": max_analysis_chars},
            "decision": {"enum": ["plan", "ask_user", "refuse"]},
            "message": {"type": "string", "maxLength": max_analysis_chars},
            "moves": {
                "type": "array",
                "maxItems": max_moves,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["object", "destination"],
                    "properties": {
                        "object": {"enum": targets},
                        "destination": {"enum": places + [HOLD] + targets},
                    },
                },
            },
        },
    }


def decide(
    instruction: str,
    scene_text: str,
    object_ids: list[str],
    marked_image: np.ndarray,
    apertus: Apertus,
    config: dict,
    gripper_mm: float,
    feedback: str | None,
) -> tuple[dict, str]:
    """Ask Apertus for the decision on one instruction.

    Args:
        instruction: the user's sentence.
        scene_text: output of ``scene_graph.describe``.
        object_ids: ids in the scene graph.
        marked_image: Annotated RGB image with object marks.
        apertus: Model client.
        config: the ``decision`` section of the pipeline config, including its ``prompt``.
        gripper_mm: maximum gripper opening.
        feedback: why the previous answer was rejected, or None on the first attempt.

    Returns:
        Parsed reply (``analysis``, ``decision``, ``message``, ``moves``) and the
        raw reply text.
    """
    prompt = config["prompt"].format(
        scene=scene_text, instruction=instruction, gripper_mm=round(gripper_mm)
    )
    if feedback:
        prompt += f"\n\nYour previous answer was rejected because {feedback}. Answer again."
    schema = decision_schema(
        object_ids, config["max_analysis_chars"], config["max_moves"], apertus.config["places"]
    )
    return apertus.ask_json(
        "decision", prompt, schema, config["max_tokens"], marked_image, config["image_max_side_px"]
    )


def answer_instruction(
    instruction_id: str,
    instruction: str,
    scene_text: str,
    marks: list[dict],
    on_top: list[tuple[str, str]],
    annotated: np.ndarray,
    apertus: Apertus,
    config: dict,
    keep_in_place: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    """Ask for structured moves, retry invalid plans, and return an official answer.

    A valid plan that has to pick up an object the instruction says must stay where it is
    becomes a refusal: the instruction forbids the only way to carry it out.

    Args:
        instruction_id: Identifier for this instruction.
        instruction: Original instruction passed unchanged to Apertus.
        scene_text: Measured scene description.
        marks: Objects detected from this photo.
        on_top: Current stacking relationships.
        annotated: Photo with object marks.
        apertus: Model client.
        config: Unified pipeline configuration.
        keep_in_place: ids the safety screen reported as having to stay where they are.

    Returns:
        Official answer and retry records for inspection.
    """
    attempts = []
    feedback = None
    for _ in range(config["decision"]["max_attempts"]):
        try:
            reply, raw = decide(
                instruction,
                scene_text,
                [item["id"] for item in marks],
                annotated,
                apertus,
                config["decision"],
                config["gripper_max_opening_mm"],
                feedback,
            )
        except ModelReplyError as error:
            feedback = f"{error}; keep the explanation brief and return every required field"
            attempts.append({"raw": error.raw, "problem": feedback})
            continue
        if reply["decision"] != "plan":
            action = reply["decision"]
            field = "question" if action == "ask_user" else "reason"
            message = reply["message"].strip()
            if not message:
                feedback = f"{action} needs a non-empty {field}"
                attempts.append({"raw": raw, "problem": feedback})
                continue
            attempts.append({"raw": raw, "problem": None})
            return {
                "instruction_id": instruction_id,
                "target_pixel": None,
                "actions": [{"action": action, field: message}],
            }, attempts
        steps, problem = prepare_moves(reply["moves"], marks, on_top, config)
        attempts.append({"raw": raw, "problem": problem})
        moved = kept_objects_moved(steps, keep_in_place or [])
        if problem is None and moved and config["safety"]["check_keep_in_place"]:
            names = ", ".join(moved)
            return {
                "instruction_id": instruction_id,
                "target_pixel": None,
                "actions": [
                    {
                        "action": "refuse",
                        "reason": f"The instruction asks to leave {names} where it is, "
                        "but carrying it out requires moving it.",
                    }
                ],
            }, attempts
        if problem is None:
            by_id = {item["id"]: item for item in marks}
            return {
                "instruction_id": instruction_id,
                "target_pixel": by_id[steps[0]["object"]]["pixel"],
                "actions": steps,
            }, attempts
        feedback = problem
    return {
        "instruction_id": instruction_id,
        "target_pixel": None,
        "actions": [
            {
                "action": "ask_user",
                "question": f"I could not form an executable plan ({feedback}). Could you rephrase the instruction?",
            }
        ],
    }, attempts
