"""Apertus decides plan / ask_user / refuse for one instruction over the scene graph.

The reply is constrained to a schema whose object fields only accept ids that exist in the
scene, so Apertus cannot invent objects or pixels. It writes a short analysis first, which
works as a chain of thought inside the constrained output; the free-text message comes before
the moves so that a finished plan is also the end of the JSON. The prompt is in the config.
"""

from src.reasoning.client import ask_json

PLACES = ["grey_tray", "black_tray", "aside"]
HOLD = "hold"


def decision_schema(object_ids: list[str], max_analysis_chars: int, max_moves: int) -> dict:
    """JSON schema for the decision, with object fields limited to ids in the scene.

    Args:
        object_ids: ids from the scene graph.
        max_analysis_chars: cap on the analysis text.
        max_moves: cap on the number of moves in a plan.

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
                        "destination": {"enum": PLACES + [HOLD] + targets},
                    },
                },
            },
        },
    }


def decide(
    instruction: str,
    scene_text: str,
    object_ids: list[str],
    marked_image: str,
    endpoint: dict,
    config: dict,
    gripper_mm: float,
    feedback: str | None,
) -> tuple[dict, str]:
    """Ask Apertus for the decision on one instruction.

    Args:
        instruction: the user's sentence.
        scene_text: output of ``scene_graph.describe``.
        object_ids: ids in the scene graph.
        marked_image: data URL of the photo with ids drawn.
        endpoint: model endpoint settings for ``ask_json``.
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
    schema = decision_schema(object_ids, config["max_analysis_chars"], config["max_moves"])
    return ask_json(endpoint, prompt, [marked_image], schema, config["max_tokens"])
