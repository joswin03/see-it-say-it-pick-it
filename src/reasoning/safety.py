"""Apertus screens an instruction for unsafe requests before planning.

A focused question with structured fields works better for an 8B model than an open "is this
safe?" or one more rule inside the long planning prompt: on the dev set the planner alone
refused none of the unsafe instructions. Apertus fills in what the instruction asks (the manner
of moving, whether it takes from a person's hand); ``screen_result`` turns those facts into
refuse / continue. Self-contradicting instructions are left to the planner: asked here, the 8B
model flagged 6 to 7 ordinary dev instructions as contradicting themselves. The prompt,
with its generic examples, is in the config.
"""

import re

from src.reasoning.client import ask_json

# The organisers' safety primer (docs/safety_primer.md) names these as reasons to refuse. They
# are matched on the wording first: the 8B screen labels "throw" or "push off" as an ordinary
# pick and place in about half the runs.
UNSAFE_WORDING = [
    (r"\b(throw|toss|fling|hurl|chuck|lob)", "Throwing an object is uncontrolled motion."),
    (r"\bdrop\w*\b.*\b(from|height|high|above)\b", "Dropping an object from height is unsafe."),
    (
        r"\b(push|shove|knock|sweep|flick|kick)\w*\b.*\boff\b",
        "Pushing or knocking an object off is uncontrolled motion.",
    ),
    (
        r"\b(pull|yank|slide|drag|jerk|tug)\w*\b.*\bfrom (under|beneath)",
        "Pulling an object out from under another makes the top one fall.",
    ),
    (
        r"\b(out of|from) (my|your|his|her|their|someone's|a person's) (hand|hands|fingers|grip)\b",
        "I do not take objects out of a person's hand.",
    ),
    (
        r"\b(under|into|onto|on) (my|your|his|her|their|someone's|a person's) (hand|hands|arm)\b",
        "Moving onto or into a person is unsafe.",
    ),
]

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
            "manner",
            "involves_person",
            "message",
        ],
        "properties": {
            "analysis": {"type": "string", "maxLength": max_text_chars},
            "manner": {"enum": MANNERS},
            "involves_person": {"type": "boolean"},
            "message": {"type": "string", "maxLength": max_text_chars},
        },
    }


def check_instruction(
    instruction: str, scene_text: str, endpoint: dict, config: dict, feedback: str | None = None
) -> tuple[dict, str]:
    """Ask Apertus to describe what an instruction asks for.

    Args:
        instruction: the user's sentence.
        scene_text: output of ``scene_graph.describe``.
        endpoint: model endpoint settings for ``ask_json``.
        config: the ``safety`` section of the pipeline config, including its ``prompt``.
        feedback: why the previous reply was rejected, or None on the first attempt.

    Returns:
        Parsed reply (fields of ``safety_schema``) and the raw reply text.
    """
    prompt = config["prompt"].format(
        scene=scene_text, instruction=instruction, max_chars=config["max_analysis_chars"]
    )
    if feedback:
        prompt += f"\n\nYour previous answer was rejected because {feedback}. Answer again."
    schema = safety_schema(config["max_analysis_chars"])
    return ask_json(endpoint, prompt, [], schema, config["max_tokens"])


def unsafe_wording(instruction: str) -> str | None:
    """Reason to refuse if the instruction names an unsafe way of moving, else None.

    Args:
        instruction: the user's sentence.

    Returns:
        A one-sentence reason, or None.
    """
    for pattern, reason in UNSAFE_WORDING:
        if re.search(pattern, instruction.lower()):
            return reason
    return None


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
