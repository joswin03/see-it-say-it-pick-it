"""LangGraph pipeline: annotate the photo, check stacking, screen safety, decide.

annotate: OpenCV finds the object regions, Apertus names each from an outlined close-up, code
merges pieces of one object and numbers the objects; their boxes are drawn on the photo. With
``perception.backend: apertus_boxes`` (ablation) Apertus instead marks labelled boxes on the
whole photo (``src/boxes.py``).
stack: convex-hull geometry establishes which objects lie on others.
Both run once per photo; later instructions of the same photo start at safety.
safety: Apertus screens each instruction for dangerous handling.
decide: Apertus plans moves using the measured spatial scene graph; code validates the moves,
clears blocking objects, retries invalid plans and sets the first pick pixel.
"""

from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.apertus import Apertus
from src.boxes import detect_boxes, marks_from_boxes
from src.calibration import mm_per_pixel
from src.decision import answer_instruction
from src.marks import draw_marks
from src.naming import name_region, outlined_crop
from src.objects import build_marks, size_mm
from src.regions import find_regions
from src.safety import check_instruction, screen_result
from src.scene_graph import describe, stacking
from src.structured import ModelReplyError

NO_STACKING = "(no object lies on another)"
OPENCV = "opencv"
APERTUS_BOXES = "apertus_boxes"
BACKENDS = {OPENCV, APERTUS_BOXES}


class PipelineState(TypedDict):
    """Everything one instruction carries through the graph."""

    instruction_id: str
    instruction: str
    scene: object
    distortion: np.ndarray
    photo: np.ndarray
    marks: list[dict] | None
    annotated: np.ndarray | None
    stacking: str | None
    on_top: list[tuple[str, str]] | None
    result: dict | None
    scene_text: str
    attempts: list[dict]


def resolve_grasp_limit(config: dict) -> None:
    """Derive the practical too-wide threshold from the gripper opening and the allowance.

    An object counts as unpickable when its measured mask width exceeds the challenge's
    85 mm opening plus a configured mask allowance (120 mm practical limit). The allowance
    is a documented engineering choice, not a validated grasp-width computation; plates
    and saucers are additionally excluded by name.

    Args:
        config: full run config; ``config["objects"]["too_wide_mm"]`` is set in place.

    Returns:
        None.
    """
    objects = config["objects"]
    objects["too_wide_mm"] = config["gripper_max_opening_mm"] + objects["grasp_width_margin_mm"]


def build_graph(config: dict, apertus: Apertus) -> CompiledStateGraph:
    """Assemble and compile the pipeline graph.

    The practical ``objects.too_wide_mm`` limit is derived here from the gripper opening
    plus margin (see ``resolve_grasp_limit``).

    Args:
        config: run config with ``prompts``, ``places``, the
            ``perception``, ``naming`` and ``objects`` sections, ``stack_min_overlap``,
            structured decision and safety settings.
        apertus: model client.

    Returns:
        Compiled graph; invoke it with a ``PipelineState`` dict.
    """
    prompts = config["prompts"]
    resolve_grasp_limit(config)
    if config["perception"]["backend"] not in BACKENDS:
        raise ValueError(f"perception.backend must be one of {sorted(BACKENDS)}")

    def annotate(state: PipelineState) -> dict:
        """Find regions with OpenCV, let Apertus name each one, merge pieces, number objects.

        With ``perception.backend: apertus_boxes`` Apertus marks labelled boxes instead.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``marks`` and ``annotated``.
        """
        scene, photo = state["scene"], state["photo"]
        perception, naming = config["perception"], config["naming"]
        if perception["backend"] == APERTUS_BOXES:
            boxes = detect_boxes(photo, apertus, config)
            marks = marks_from_boxes(boxes, photo, scene, state["distortion"], config)
            return {"marks": marks, "annotated": draw_marks(photo, marks)}
        scale_mm = mm_per_pixel(scene)
        mm_per_work_pixel = scale_mm / perception["work_scale"]
        found = find_regions(photo, perception, mm_per_work_pixel**2)
        min_side_pixels = round(naming["crop_min_side_mm"] / scale_mm)
        named = []
        for region in found["regions"]:
            length_mm, width_mm = size_mm(
                region["mask"], found["work_scale"], scene, state["distortion"]
            )
            crop = outlined_crop(
                photo, region["mask"], naming["crop_padding_fraction"], min_side_pixels
            )
            fields = {
                "colour": region["colour"],
                "length_mm": length_mm,
                "width_mm": width_mm,
                "tray": region["tray"].replace("_", " "),
            }
            name, is_object, aliases = name_region(
                apertus, prompts["name"], fields, crop, naming["fixture_words"]
            )
            named.append({**region, "name": name, "is_object": is_object, "aliases": aliases})
        marks = build_marks(
            named,
            found["work_scale"],
            mm_per_work_pixel,
            config["objects"],
            scene,
            state["distortion"],
        )
        return {"marks": marks, "annotated": draw_marks(photo, marks)}

    def stack(state: PipelineState) -> dict:
        """Find which objects lie on others from mask overlap with convex hulls.

        Args:
            state: pipeline state after ``annotate``.

        Returns:
            State update with ``stacking`` (one fact per stacked pair, as text) and ``on_top``
            ((upper id, lower id) pairs).
        """
        on_top = stacking(state["marks"], config["stack_min_overlap"])
        return {
            "stacking": "\n".join(f"{upper} lies on {lower}" for upper, lower in on_top)
            or NO_STACKING,
            "on_top": on_top,
        }

    def decide(state: PipelineState) -> dict:
        """Ask Apertus to plan, ask or refuse on the annotated photo, by object number.

        Structured moves are validated and retried with feedback.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``result`` in the official schema.
        """
        result, attempts = answer_instruction(
            state["instruction_id"],
            state["instruction"],
            state["scene_text"],
            state["marks"],
            state["on_top"],
            state["annotated"],
            apertus,
            config,
        )
        return {"result": result, "attempts": state["attempts"] + attempts}

    def safety(state: PipelineState) -> dict:
        """Screen every instruction with Apertus before generating moves.

        Args:
            state: Perceived scene and unchanged instruction.

        Returns:
            Scene text, audit records, and a refusal when Apertus identifies unsafe handling.
        """
        scene_text = describe(state["marks"], state["on_top"])
        update = {"scene_text": scene_text, "result": None, "attempts": []}
        if not config["safety"]["enabled"]:
            return update
        feedback = None
        for _ in range(config["decision"]["max_attempts"]):
            try:
                screen, raw = check_instruction(
                    state["instruction"],
                    scene_text,
                    apertus,
                    config["safety"],
                    feedback,
                )
            except ModelReplyError as error:
                feedback = f"{error}; keep the explanation brief and return every required field"
                update["attempts"].append(
                    {"stage": "safety", "raw": error.raw, "problem": feedback}
                )
                continue
            update["attempts"].append({"stage": "safety", "raw": raw, "problem": None})
            stop = screen_result(screen)
            if stop:
                update["result"] = {
                    "instruction_id": state["instruction_id"],
                    "target_pixel": None,
                    "actions": [{"action": "refuse", "reason": stop[1]}],
                }
            return update
        update["result"] = {
            "instruction_id": state["instruction_id"],
            "target_pixel": None,
            "actions": [
                {
                    "action": "ask_user",
                    "question": "The safety check could not produce a valid answer. Could you clarify the intended handling?",
                }
            ],
        }
        return update

    def route_safety(state: PipelineState) -> str:
        """Stop on a safety decision or continue to planning.

        Args:
            state: State after safety screening.

        Returns:
            End marker or decide node name.
        """
        return END if state["result"] is not None else "decide"

    def route_start(state: PipelineState) -> str:
        """Skip annotate and stack when this photo is already done.

        Args:
            state: initial pipeline state.

        Returns:
            ``"safety"`` or ``"annotate"``.
        """
        return "safety" if state["marks"] is not None else "annotate"

    graph = StateGraph(PipelineState)
    graph.add_node("annotate", annotate)
    graph.add_node("stack", stack)
    graph.add_node("decide", decide)
    graph.add_node("safety", safety)
    graph.add_conditional_edges("safety", route_safety, ["decide", END])
    graph.add_conditional_edges(START, route_start, ["annotate", "safety"])
    graph.add_edge("annotate", "stack")
    graph.add_edge("stack", "safety")
    graph.add_edge("decide", END)
    return graph.compile()
