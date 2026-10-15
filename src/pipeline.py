"""OpenCV + Apertus pipeline: perceive a scene once, then answer each instruction.

perceive_scene: OpenCV regions -> Apertus names each region -> scene graph (code).
answer_instruction: Apertus decides over the scene graph -> code checks the plan and sets the
target pixel from the scene graph.
"""

import cv2
import numpy as np

from src.perception.regions import find_regions
from src.reasoning.client import ModelReplyError, image_url
from src.reasoning.decision import decide
from src.reasoning.naming import name_region
from src.scene_graph import build_scene_graph, describe, size_mm
from src.workcell import Scene, mm_per_pixel

LABEL_RGB = (255, 255, 0)
LABEL_SHADOW_RGB = (0, 0, 0)


def marked_photo(rgb: np.ndarray, objects: list[dict], work_scale: float) -> np.ndarray:
    """Draw every object's outline and id on a copy of the photo (Set-of-Mark image).

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        objects: scene-graph objects with ``mask`` at work scale.
        work_scale: work-scale factor of the masks.

    Returns:
        (H, W, 3) uint8 annotated photo.
    """
    image = rgb.copy()
    font_scale = rgb.shape[1] / 1000
    for item in objects:
        full_mask = cv2.resize(
            item["mask"].astype(np.uint8),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, LABEL_RGB, max(2, round(1 / work_scale)))
        u, v = item["pixel"]
        for colour, thickness in ((LABEL_SHADOW_RGB, 10), (LABEL_RGB, 3)):
            cv2.putText(image, item["id"], (u, v), 0, font_scale, colour, thickness)
    return image


def perceive_scene(scene: Scene, config: dict, endpoint: dict, gripper_mm: float) -> dict:
    """Find, name and relate the objects in one scene.

    Args:
        scene: loaded scene.
        config: full pipeline config.
        endpoint: model endpoint settings.
        gripper_mm: maximum gripper opening.

    Returns:
        Dict with ``objects`` (scene graph), ``text`` (its description), ``marked`` (annotated
        photo, (H, W, 3) uint8), ``marked_url`` and ``naming_log`` (one entry per region).
    """
    perception = config["perception"]
    scale_mm = mm_per_pixel(scene)
    mm_per_work_pixel = scale_mm / perception["work_scale"]
    found = find_regions(scene.rgb, perception, mm_per_work_pixel**2)
    naming_log = []
    for region in found["regions"]:
        region["size_mm"] = size_mm(region["mask"], mm_per_work_pixel)
        named = name_region(scene.rgb, region, found["work_scale"], endpoint, config["naming"])
        region["name"], region["is_object"] = named["name"], named["is_object"]
        naming_log.append(
            {
                "region": region["id"],
                "colour": region["colour"],
                "pixel": region["pixel"],
                "size_mm": region["size_mm"],
                "tray": region["tray"],
                "reply": named["raw"],
            }
        )
    objects = build_scene_graph(
        found["regions"], found["work_scale"], mm_per_work_pixel, config["scene_graph"]
    )
    marked = marked_photo(scene.rgb, objects, found["work_scale"])
    return {
        "objects": objects,
        "text": describe(objects, scale_mm, config["scene_graph"]["nearest_count"], gripper_mm),
        "marked": marked,
        "marked_url": image_url(marked, config["decision"]["image_max_side_px"]),
        "naming_log": naming_log,
    }


def clear_tops_first(steps: list[dict], objects: list[dict]) -> list[dict]:
    """Insert pick-and-place-aside for any object lying on one that is picked.

    Args:
        steps: pick/place steps from Apertus.
        objects: scene-graph objects.

    Returns:
        Steps where every picked object is uncovered first.
    """
    by_id = {item["id"]: item for item in objects}
    moved: set[str] = set()
    result: list[dict] = []
    for step in steps:
        if step["action"] == "pick" and step["object"] in by_id:
            for upper in by_id[step["object"]]["under"]:
                if upper not in moved:
                    result += [
                        {"action": "pick", "object": upper},
                        {"action": "place", "object": "aside"},
                    ]
                    moved.add(upper)
            moved.add(step["object"])
        result.append(step)
    return result


def plan_problem(steps: list[dict], objects: list[dict], gripper_mm: float) -> str | None:
    """Explain why a plan is not executable, or return None if it is.

    Args:
        steps: pick/place steps.
        objects: scene-graph objects.
        gripper_mm: maximum gripper opening.

    Returns:
        A one-sentence problem description, or None.
    """
    by_id = {item["id"]: item for item in objects}
    if not steps:
        return "the plan has no steps"
    if steps[0]["action"] != "pick":
        return "the plan must start with a pick"
    held = None
    for step in steps:
        target = step["object"]
        if step["action"] == "pick":
            if target not in by_id:
                return f"'{target}' is a place, not an object that can be picked"
            if by_id[target]["size_mm"][1] > gripper_mm:
                return f"{target} is too wide for the {round(gripper_mm)} mm gripper"
            if held is not None:
                return f"the gripper still holds {held} when picking {target}"
            held = target
        else:
            if held is None:
                return f"placing on {target} with nothing in the gripper"
            if target == held:
                return f"{held} cannot be placed on itself"
            held = None
    return None


def to_answer(instruction_id: str, reply: dict, objects: list[dict], config: dict) -> dict:
    """Convert Apertus's decision into a result in the challenge schema.

    Args:
        instruction_id: id of the instruction.
        reply: parsed decision reply.
        objects: scene-graph objects.
        config: full pipeline config.

    Returns:
        Answer dict matching ``schema/result.schema.json``.
    """
    if reply["decision"] == "ask_user":
        question = reply["question"].strip() or "Which object do you mean?"
        return {
            "instruction_id": instruction_id,
            "target_pixel": None,
            "actions": [{"action": "ask_user", "question": question}],
        }
    if reply["decision"] == "refuse":
        reason = reply["reason"].strip() or reply["analysis"].strip()
        return {
            "instruction_id": instruction_id,
            "target_pixel": None,
            "actions": [{"action": "refuse", "reason": reason}],
        }
    steps = reply["steps"]
    if config["decision"]["enforce_clear_top"]:
        steps = clear_tops_first(steps, objects)
    by_id = {item["id"]: item for item in objects}
    first_pick = next(step["object"] for step in steps if step["action"] == "pick")
    return {
        "instruction_id": instruction_id,
        "target_pixel": by_id[first_pick]["pixel"],
        "actions": [{"action": step["action"], "object": step["object"]} for step in steps],
    }


def answer_instruction(
    instruction_id: str, text: str, perceived: dict, config: dict, endpoint: dict, gripper_mm: float
) -> tuple[dict, dict]:
    """Answer one instruction; retry once with feedback if the plan is not executable.

    Args:
        instruction_id: id of the instruction.
        text: the instruction.
        perceived: output of ``perceive_scene``.
        config: full pipeline config.
        endpoint: model endpoint settings.
        gripper_mm: maximum gripper opening.

    Returns:
        The answer and a log entry with every raw reply.
    """
    objects = perceived["objects"]
    object_ids = [item["id"] for item in objects]
    attempts = []
    feedback = None
    for _ in range(config["decision"]["max_attempts"]):
        try:
            reply, raw = decide(
                text,
                perceived["text"],
                object_ids,
                perceived["marked_url"],
                endpoint,
                config["decision"],
                gripper_mm,
                feedback,
            )
        except ModelReplyError as error:
            attempts.append({"raw": error.raw, "problem": str(error)})
            feedback = f"your {error}; keep the analysis to two or three sentences"
            continue
        problem = (
            plan_problem(reply["steps"], objects, gripper_mm)
            if reply["decision"] == "plan"
            else None
        )
        attempts.append({"raw": raw, "problem": problem})
        if problem is None:
            answer = to_answer(instruction_id, reply, objects, config)
            return answer, {"instruction_id": instruction_id, "attempts": attempts}
        feedback = problem
    final_problem = attempts[-1]["problem"]
    answer = {
        "instruction_id": instruction_id,
        "target_pixel": None,
        "actions": [
            {
                "action": "ask_user",
                "question": f"I could not form an executable plan ({final_problem}). "
                "Could you rephrase the instruction?",
            }
        ],
    }
    return answer, {"instruction_id": instruction_id, "attempts": attempts, "gave_up": True}
