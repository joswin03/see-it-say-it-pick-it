"""Apertus decides plan / ask_user / refuse for one instruction over the scene graph.

The reply is constrained to a schema whose object fields only accept ids that exist in the
scene, so Apertus cannot invent objects or pixels. It writes a short analysis first, which
works as a chain of thought inside the constrained output.
"""

from src.reasoning.client import ask_json

DECISION_PROMPT = """You plan actions for a robot arm with a two-finger gripper above a table.
The camera looks straight down. The table holds a grey tray (left in the photo) and a black
tray (right in the photo). The photo shows the objects with their ids drawn on them.

Scene graph, measured by the vision system (trust it for positions, sizes and stacking):
{scene}

Object names come from a vision model and may use synonyms or a slightly different colour:
match the instruction by meaning (cube = block, mug = cup, mandarin = orange, part = plate-like
metal or plastic piece). An object is missing only if nothing in the scene can be what the
instruction names.

Places to put things: grey_tray (the grey tray, left), black_tray (the black tray, right),
aside (free table space, to clear an object off another), or an object id above that can hold
things (a plate, a saucer).

Robot facts:
- The gripper holds one object at a time and opens at most {gripper_mm} mm. Objects marked
  TOO WIDE TO GRIP (plates, saucers) cannot be picked; every other object can.
- An object that is underneath another cannot be picked before the one on top is moved aside.
- An object can only be put into or onto something larger than itself.
- Left, right, top and bottom mean as seen in the photo (x grows to the right, y downwards).

Instruction: "{instruction}"

Decide:
- plan: the instruction names objects that exist and can be handled. This is the normal case:
  most instructions are fine. If the instruction says "all", "both" or "everything", or picks
  one object by position or relation ("on the left", "closest to the mug", "second from the
  right", "the one on the other gear"), it is NOT ambiguous: resolve it from the scene graph.
  To reach an object that is underneath another, first pick the one on top and place it aside.
- ask_user: only if the words fit several objects equally and nothing in the instruction tells
  them apart (e.g. "the red gear" when there are three red gears), or the instruction points
  at something without naming it ("that one", "this").
- refuse: only for a concrete reason: the named object is not in the scene; the action is
  physically impossible (too wide for the gripper, does not fit into the container, the
  instruction contradicts itself or forbids the only way to do it); or the action is unsafe
  (throwing, pushing or knocking objects off, pulling an object out from under another,
  taking or handing objects from or to a person's hand, moving near a person's hand).

A plan is a list of moves in order. Each move picks one object and puts it at a destination;
use destination "hold" when the instruction only asks to pick an object up ("pick up X").
Never move an object onto itself."""

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
        "required": ["analysis", "decision", "moves", "question", "reason"],
        "properties": {
            "analysis": {"type": "string", "maxLength": max_analysis_chars},
            "decision": {"enum": ["plan", "ask_user", "refuse"]},
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
            "question": {"type": "string", "maxLength": max_analysis_chars},
            "reason": {"type": "string", "maxLength": max_analysis_chars},
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
        config: the ``decision`` section of the pipeline config.
        gripper_mm: maximum gripper opening.
        feedback: why the previous answer was rejected, or None on the first attempt.

    Returns:
        Parsed reply (``analysis``, ``decision``, ``moves``, ``question``, ``reason``) and the
        raw reply text.
    """
    prompt = DECISION_PROMPT.format(
        scene=scene_text, instruction=instruction, gripper_mm=round(gripper_mm)
    )
    if feedback:
        prompt += f"\n\nYour previous answer was rejected because {feedback}. Answer again."
    schema = decision_schema(object_ids, config["max_analysis_chars"], config["max_moves"])
    return ask_json(endpoint, prompt, [marked_image], schema, config["max_tokens"])
