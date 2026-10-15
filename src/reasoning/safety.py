"""Apertus screens an instruction for unsafe or self-contradictory requests before planning.

A focused question with structured fields works better for an 8B model than an open "is this
safe?" or one more rule inside the long planning prompt: on the dev set the planner alone
refused none of the unsafe instructions. Apertus fills in what the instruction asks (the manner
of moving, whether a person is involved, whether it contradicts itself, whether it names an
object at all); ``screen_result`` turns those facts into ask / refuse / continue. The prompt,
with its generic examples, is in the config.
"""

from src.reasoning.client import ask_json

ORDINARY_MANNER = "pick and place"
MANNERS = [
    ORDINARY_MANNER,
    "throw or toss",
    "push, knock or sweep off",
    "pull or slide out from under",
    "other dangerous handling",
]


def safety_schema(max_text_chars: int) -> dict:
    """JSON schema for the safety screen.

    Args:
        max_text_chars: cap on the analysis and message texts.

    Returns:
        JSON schema dict.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "analysis",
            "names_an_object",
            "manner",
            "involves_person",
            "contradicts_itself",
            "message",
        ],
        "properties": {
            "analysis": {"type": "string", "maxLength": max_text_chars},
            "names_an_object": {"type": "boolean"},
            "manner": {"enum": MANNERS},
            "involves_person": {"type": "boolean"},
            "contradicts_itself": {"type": "boolean"},
            "message": {"type": "string", "maxLength": max_text_chars},
        },
    }


def check_instruction(
    instruction: str, scene_text: str, endpoint: dict, config: dict
) -> tuple[dict, str]:
    """Ask Apertus to describe what an instruction asks for.

    Args:
        instruction: the user's sentence.
        scene_text: output of ``scene_graph.describe``.
        endpoint: model endpoint settings for ``ask_json``.
        config: the ``safety`` section of the pipeline config, including its ``prompt``.

    Returns:
        Parsed reply (fields of ``safety_schema``) and the raw reply text.
    """
    prompt = config["prompt"].format(scene=scene_text, instruction=instruction)
    schema = safety_schema(config["max_analysis_chars"])
    return ask_json(endpoint, prompt, [], schema, config["max_tokens"])


def screen_result(reply: dict) -> tuple[str, str] | None:
    """Turn the screen's facts into a final action, or None to go on to planning.

    Args:
        reply: parsed reply of ``check_instruction``.

    Returns:
        ("ask_user", question) or ("refuse", reason), or None if nothing stops the plan.
    """
    message = reply["message"].strip()
    if not reply["names_an_object"]:
        return "ask_user", message or "Which object do you mean?"
    if reply["manner"] != ORDINARY_MANNER:
        return "refuse", message or f"Moving objects this way ({reply['manner']}) is unsafe."
    if reply["involves_person"]:
        return "refuse", message or "I do not take objects from or hand them to a person."
    if reply["contradicts_itself"]:
        return "refuse", message or "The instruction contradicts itself."
    return None
