"""Apertus screens an instruction for unsafe requests before planning.

A focused question with structured fields works better for an 8B model than an open "is this
safe?" or one more rule inside the long planning prompt: on the dev set the planner alone
refused none of the unsafe instructions. Apertus first copies the words that say how to move
the object, then fills in what they ask (the manner of moving, whether it takes from a person's
hand); ``screen_result`` turns those facts into refuse / continue. Self-contradicting
instructions are left to the planner: asked here, the 8B model flagged ordinary dev
instructions as contradicting themselves or as keeping their own target in place. The prompt, with its generic examples, is in the config.
"""

from src.apertus import Apertus

ORDINARY_MANNER = "pick and place"
MANNERS = [
    ORDINARY_MANNER,
    "throw or toss",
    "push, knock or sweep off",
    "pull or slide out from under",
    "other dangerous handling",
]


def safety_schema(max_text_chars: int, max_action_chars: int) -> dict:
    """JSON schema for the safety screen.

    Field names sort alphabetically in the order Apertus should fill them: the action words are
    copied before the manner is chosen.

    Args:
        max_text_chars: cap on the analysis and message texts.
        max_action_chars: cap on the copied action words.

    Returns:
        JSON schema dict.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action_words",
            "analysis",
            "involves_person",
            "manner",
            "message",
        ],
        "properties": {
            "action_words": {"type": "string", "maxLength": max_action_chars},
            "analysis": {"type": "string", "maxLength": max_text_chars},
            "manner": {"enum": MANNERS},
            "involves_person": {"type": "boolean"},
            "message": {"type": "string", "maxLength": max_text_chars},
        },
    }


def check_instruction(
    instruction: str,
    scene_text: str,
    apertus: Apertus,
    config: dict,
    feedback: str | None = None,
) -> tuple[dict, str]:
    """Ask Apertus to describe what an instruction asks for.

    Args:
        instruction: the user's sentence.
        scene_text: output of ``scene_graph.describe``.
        apertus: Model client.
        config: the ``safety`` section of the pipeline config, including its ``prompt``.
        feedback: why the previous reply was rejected, or None on the first attempt.

    Returns:
        Parsed reply (fields of ``safety_schema``) and the raw reply text.
    """
    prompt = config["prompt"].format(
        scene=scene_text,
        instruction=instruction,
        max_chars=config["max_analysis_chars"],
        max_action_chars=config["max_action_chars"],
    )
    if feedback:
        prompt += f"\n\nYour previous answer was rejected because {feedback}. Answer again."
    schema = safety_schema(config["max_analysis_chars"], config["max_action_chars"])
    return apertus.ask_json("safety", prompt, schema, config["max_tokens"])


def screen_result(reply: dict) -> tuple[str, str] | None:
    """Turn the screen's facts into a final action, or None to go on to planning.

    Args:
        reply: parsed reply of ``check_instruction``.

    Returns:
        ("refuse", reason), or None if nothing stops the plan.
    """
    message = reply["message"].strip()
    if reply["manner"] != ORDINARY_MANNER:
        return "refuse", message or f"Moving objects this way ({reply['manner']}) is unsafe."
    if reply["involves_person"]:
        return "refuse", message or "I do not take objects from or hand them to a person."
    return None
