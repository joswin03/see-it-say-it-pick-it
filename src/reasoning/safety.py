"""Apertus screens an instruction for unsafe or self-contradictory requests before planning.

A separate, focused question works better for an 8B model than one more rule inside the long
planning prompt: on the dev set the planner alone refused none of the unsafe instructions.
The prompt, with its examples, is in the config.
"""

from src.reasoning.client import ask_json

VERDICTS = ["ok", "unsafe", "impossible"]


def safety_schema(max_analysis_chars: int) -> dict:
    """JSON schema for the safety verdict.

    Args:
        max_analysis_chars: cap on the analysis and reason texts.

    Returns:
        JSON schema dict.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["analysis", "verdict", "reason"],
        "properties": {
            "analysis": {"type": "string", "maxLength": max_analysis_chars},
            "verdict": {"enum": VERDICTS},
            "reason": {"type": "string", "maxLength": max_analysis_chars},
        },
    }


def check_instruction(
    instruction: str, scene_text: str, endpoint: dict, config: dict
) -> tuple[dict, str]:
    """Ask Apertus whether an instruction is safe and self-consistent.

    Args:
        instruction: the user's sentence.
        scene_text: output of ``scene_graph.describe``.
        endpoint: model endpoint settings for ``ask_json``.
        config: the ``safety`` section of the pipeline config, including its ``prompt``.

    Returns:
        Parsed reply (``analysis``, ``verdict``, ``reason``) and the raw reply text.
    """
    prompt = config["prompt"].format(scene=scene_text, instruction=instruction)
    schema = safety_schema(config["max_analysis_chars"])
    return ask_json(endpoint, prompt, [], schema, config["max_tokens"])
