"""Apertus screens an instruction for unsafe requests before planning.

A focused question with structured fields works better for an 8B model than an open "is this
safe?" or one more rule inside the long planning prompt: on the dev set the planner alone
refused none of the unsafe instructions. Apertus first copies the words that say how to move
the object, then fills in what they ask (the manner of moving, whether it takes from a person's
hand, which objects must stay where they are); ``screen_result`` turns those facts into
refuse / continue, and ``kept_objects_moved`` later checks the finished plan against the objects
that must stay. An open "contradicts itself" question is not asked: the 8B model flagged 6 to 7
ordinary dev instructions that way. The prompt, with its generic examples, is in the config.
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


def safety_schema(object_ids: list[str], max_text_chars: int, max_action_chars: int) -> dict:
    """JSON schema for the safety screen.

    Field names sort alphabetically in the order Apertus should fill them: the action words are
    copied before the manner is chosen.

    Args:
        object_ids: ids from the scene graph, the only values ``keep_in_place`` accepts.
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
            "keep_in_place",
            "manner",
            "message",
        ],
        "properties": {
            "action_words": {"type": "string", "maxLength": max_action_chars},
            "analysis": {"type": "string", "maxLength": max_text_chars},
            "keep_in_place": {
                "type": "array",
                "uniqueItems": True,
                "maxItems": len(object_ids),
                "items": {"enum": sorted(object_ids) or ["none"]},
            },
            "manner": {"enum": MANNERS},
            "involves_person": {"type": "boolean"},
            "message": {"type": "string", "maxLength": max_text_chars},
        },
    }


def check_instruction(
    instruction: str,
    scene_text: str,
    object_ids: list[str],
    apertus: Apertus,
    config: dict,
    feedback: str | None = None,
) -> tuple[dict, str]:
    """Ask Apertus to describe what an instruction asks for.

    Args:
        instruction: the user's sentence.
        scene_text: output of ``scene_graph.describe``.
        object_ids: ids in the scene graph.
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
    schema = safety_schema(object_ids, config["max_analysis_chars"], config["max_action_chars"])
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


def kept_objects_moved(steps: list[dict], keep_in_place: list[str]) -> list[str]:
    """Find objects the instruction says must stay where they are but the plan picks up.

    Args:
        steps: executable pick/place steps after stack clearing.
        keep_in_place: ids Apertus reported as having to stay where they are.

    Returns:
        Sorted ids that are both kept and picked; empty when the plan respects the instruction.
    """
    picked = {step["object"] for step in steps if step["action"] == "pick"}
    return sorted(picked & set(keep_in_place))
