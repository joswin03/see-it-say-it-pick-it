"""OpenCV + Apertus pipeline: perceive a scene once, then answer each instruction.

perceive_scene: OpenCV regions -> Apertus names each region -> scene graph (code).
answer_instruction: Apertus decides over the scene graph -> code checks the plan and sets the
target pixel from the scene graph.
"""

import cv2
import numpy as np

from src.perception.regions import find_regions
from src.reasoning.client import ModelReplyError, image_url
from src.reasoning.decision import HOLD, decide
from src.reasoning.naming import name_region
from src.reasoning.safety import check_instruction, screen_result
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


def name_or_mark_unknown(
    scene: Scene, region: dict, config: dict, endpoint: dict, scale_mm: float
) -> dict:
    """Name a region; a cut-off or unparsable reply keeps it as "unknown object".

    The region is kept rather than dropped because it was found in the photo; the planner can
    still ask about it. The error goes into the naming log.

    Args:
        scene: loaded scene.
        region: region or blob to name.
        config: full pipeline config.
        endpoint: model endpoint settings.
        scale_mm: table millimetres per full-resolution pixel.

    Returns:
        Output of ``name_region``, or the same fields for an unknown object.
    """
    try:
        return name_region(
            scene.rgb,
            region,
            config["perception"]["work_scale"],
            scale_mm,
            endpoint,
            config["naming"],
        )
    except ModelReplyError as error:
        return {
            "name": "unknown object",
            "is_object": True,
            "raw": f"{error}: {error.raw}",
        }


def naming_entry(region: dict, named: dict) -> dict:
    """Log record for one naming call.

    Args:
        region: the region that was named.
        named: output of ``name_region``.

    Returns:
        JSON-serialisable dict.
    """
    return {
        "region": region["id"],
        "colour": region["colour"],
        "pixel": region["pixel"],
        "size_mm": region["size_mm"],
        "tray": region["tray"],
        "reply": named["raw"],
    }


def perceive_scene(scene: Scene, config: dict, endpoint: dict) -> dict:
    """Find, name and relate the objects in one scene.

    Args:
        scene: loaded scene.
        config: full pipeline config.
        endpoint: model endpoint settings.

    Returns:
        Dict with ``objects`` (scene graph), ``text`` (its description), ``marked`` (annotated
        photo, (H, W, 3) uint8), ``marked_url`` and ``naming_log`` (one entry per naming call).
    """
    perception = config["perception"]
    scale_mm = mm_per_pixel(scene)
    mm_per_work_pixel = scale_mm / perception["work_scale"]
    found = find_regions(scene.rgb, perception, mm_per_work_pixel**2)
    naming_log = []
    for region in found["regions"]:
        region["size_mm"] = size_mm(region["mask"], perception["work_scale"], scene)
        named = name_or_mark_unknown(scene, region, config, endpoint, scale_mm)
        region["name"], region["is_object"] = named["name"], named["is_object"]
        naming_log.append(naming_entry(region, named))
    objects = build_scene_graph(
        found["regions"], found["work_scale"], mm_per_work_pixel, config["scene_graph"], scene
    )
    marked = marked_photo(scene.rgb, objects, found["work_scale"])
    return {
        "objects": objects,
        "text": describe(
            objects,
            config["scene_graph"]["nearest_count"],
            config["scene_graph"]["too_wide_mm"],
        ),
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


def drop_redundant_holds(moves: list[dict]) -> list[dict]:
    """Remove a "hold" move that the next move continues with the same object.

    Apertus sometimes writes "pick X and hold it" followed by "move X to the tray"; that is one
    move, not two.

    Args:
        moves: list of {"object", "destination"}.

    Returns:
        Moves without holds that are immediately continued.
    """
    kept = []
    for index, move in enumerate(moves):
        continued = index + 1 < len(moves) and moves[index + 1]["object"] == move["object"]
        if move["destination"] == HOLD and continued:
            continue
        kept.append(move)
    return kept


def moves_to_steps(moves: list[dict]) -> list[dict]:
    """Expand moves into the challenge's pick/place actions.

    Args:
        moves: list of {"object", "destination"}; destination "hold" means pick only.

    Returns:
        List of {"action", "object"} steps.
    """
    steps = []
    for move in moves:
        steps.append({"action": "pick", "object": move["object"]})
        if move["destination"] != HOLD:
            steps.append({"action": "place", "object": move["destination"]})
    return steps


def plan_problem(moves: list[dict], objects: list[dict], too_wide_mm: float) -> str | None:
    """Explain why a plan is not executable, or return None if it is.

    Args:
        moves: moves from Apertus.
        objects: scene-graph objects.
        too_wide_mm: measured width above which an object cannot be gripped.

    Returns:
        A one-sentence problem description, or None.
    """
    by_id = {item["id"]: item for item in objects}
    if not moves:
        return "a plan needs at least one move"
    for index, move in enumerate(moves):
        if move["object"] not in by_id:
            return f"'{move['object']}' is not an object in the scene"
        if move["object"] == move["destination"]:
            return f"{move['object']} cannot be moved onto itself"
        if by_id[move["object"]]["size_mm"][1] > too_wide_mm:
            return f"{move['object']} is marked TOO WIDE TO GRIP"
        if move["destination"] == HOLD and index != len(moves) - 1:
            return "only the last move can end with the object held"
    return None


def to_answer(instruction_id: str, reply: dict, objects: list[dict], config: dict) -> dict:
    """Convert Apertus's decision into a result in the challenge schema.

    Args:
        instruction_id: id of the instruction.
        reply: parsed decision reply, with redundant holds already dropped from its moves.
        objects: scene-graph objects.
        config: full pipeline config.

    Returns:
        Answer dict matching ``schema/result.schema.json``.
    """
    if reply["decision"] == "ask_user":
        question = reply["message"].strip() or "Which object do you mean?"
        return {
            "instruction_id": instruction_id,
            "target_pixel": None,
            "actions": [{"action": "ask_user", "question": question}],
        }
    if reply["decision"] == "refuse":
        reason = reply["message"].strip() or reply["analysis"].strip()
        return {
            "instruction_id": instruction_id,
            "target_pixel": None,
            "actions": [{"action": "refuse", "reason": reason}],
        }
    steps = moves_to_steps(reply["moves"])
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
    """Answer one instruction: safety check, then plan with one retry on a bad plan.

    The safety screen runs first when ``safety.enabled`` is set (see ``screen_result``), with
    one retry on a cut-off reply. If every reply is unusable, the log says so ("unchecked") and
    the planner, which also knows the safety rules, decides alone.

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
    log = {"instruction_id": instruction_id}
    if config["safety"]["enabled"]:
        feedback, raws = None, []
        for _ in range(config["decision"]["max_attempts"]):
            try:
                screen, raw = check_instruction(
                    text, perceived["text"], endpoint, config["safety"], feedback
                )
                stop = screen_result(screen)
                raws.append(raw)
                break
            except ModelReplyError as error:
                # a stalled reply is almost always an analysis that ran into its length cap
                raws.append(f"unchecked, {error}: {error.raw}")
                feedback, stop = f"your {error}; keep the analysis to one or two sentences", None
        log["safety"] = raws[0] if len(raws) == 1 else raws
        if stop is not None:
            action, message = stop
            field = "question" if action == "ask_user" else "reason"
            answer = {
                "instruction_id": instruction_id,
                "target_pixel": None,
                "actions": [{"action": action, field: message}],
            }
            return answer, log
    attempts = []
    log["attempts"] = attempts
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
        reply["moves"] = drop_redundant_holds(reply["moves"])
        problem = (
            plan_problem(reply["moves"], objects, config["scene_graph"]["too_wide_mm"])
            if reply["decision"] == "plan"
            else None
        )
        attempts.append({"raw": raw, "problem": problem})
        if problem is None:
            answer = to_answer(instruction_id, reply, objects, config)
            return answer, log
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
    return answer, {**log, "gave_up": True}
