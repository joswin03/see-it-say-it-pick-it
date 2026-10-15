"""Three-node LangGraph pipeline, prompting only: annotate the photo, decide, refine the pixel.

annotate: Apertus marks every object (label and box); the boxes are drawn and numbered on the
photo. Runs once per photo; later instructions of the same photo start at decide.
decide: Apertus sees the annotated photo, the numbered object list and the instruction, and
answers with object numbers; code turns that into the official answer.
refine: for plans, when ``zoom`` is on, Apertus picks a cell in a gridded close-up of the first
picked object, and the pick pixel becomes that cell's centre.
"""

import logging
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.answer import to_result
from src.apertus import Apertus
from src.marks import draw_marks, objects_text, read_marks, salvage_marks
from src.zoom import cell_centre, crop_box, gridded_crop

LOGGED_REPLY_CHARACTERS = 300
logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """Everything one instruction carries through the graph."""

    instruction_id: str
    instruction: str
    photo: np.ndarray
    marks: list[dict] | None
    annotated: np.ndarray | None
    result: dict | None


def build_graph(config: dict, apertus: Apertus) -> CompiledStateGraph:
    """Assemble and compile the three-node graph.

    Args:
        config: run config with ``prompts``, ``places``, ``object_labels``,
            ``salvage_unparsed_marks`` and the ``zoom`` settings.
        apertus: model client.

    Returns:
        Compiled graph; invoke it with a ``PipelineState`` dict.
    """
    prompts = config["prompts"]

    def annotate(state: PipelineState) -> dict:
        """Ask Apertus to mark every object, then draw the numbered boxes on the photo.

        A reply that is not valid JSON is logged; with ``salvage_unparsed_marks`` its labels
        and boxes are read from the text instead (``salvage_marks``), also logged.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``marks`` and ``annotated``.
        """
        fields = {"labels": ", ".join(config["object_labels"])}
        reply, raw, scale = apertus.ask("annotate", prompts["annotate"], fields, state["photo"])
        if reply is None:
            logger.warning("annotate reply held no JSON: %s", raw[:LOGGED_REPLY_CHARACTERS])
            if config["salvage_unparsed_marks"]:
                reply = salvage_marks(raw)
                logger.warning("salvaged %d objects from the text", len(reply["objects"]))
        marks = read_marks(reply, scale)
        return {"marks": marks, "annotated": draw_marks(state["photo"], marks)}

    def decide(state: PipelineState) -> dict:
        """Ask Apertus to plan, ask or refuse on the annotated photo, by object number.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``result`` in the official schema.
        """
        fields = {
            "objects": objects_text(state["marks"]),
            "text": state["instruction"],
            "labels": ", ".join(config["object_labels"]),
        }
        reply, raw, _scale = apertus.ask("decide", prompts["decide"], fields, state["annotated"])
        if reply is None:
            logger.warning("decide reply held no JSON: %s", raw[:LOGGED_REPLY_CHARACTERS])
        result = to_result(state["instruction_id"], reply, state["marks"], config["places"])
        return {"result": result}

    def refine(state: PipelineState) -> dict:
        """Move target_pixel to the cell Apertus picks in a gridded close-up of the first pick.

        The crop is cut from the original photo around the first picked object's box; an
        invalid reply keeps the box centre and is logged.

        Args:
            state: pipeline state after ``decide``, holding a plan.

        Returns:
            State update with ``result``, its target_pixel refined when the reply is valid.
        """
        result = state["result"]
        first = result["actions"][0]["object"]
        mark = next(mark for mark in state["marks"] if mark["id"] == first)
        height, width = state["photo"].shape[:2]
        grid = config["zoom_grid"]
        region = crop_box(mark["box"], config["zoom_margin_fraction"], width, height)
        if region[2] - region[0] < grid or region[3] - region[1] < grid:
            logger.warning("zoom skipped: box of %s is too small to grid", first)
            return {"result": result}
        crop = gridded_crop(state["photo"], region, grid)
        fields = {"cells": grid * grid, "label": mark["label"]}
        reply, raw, _scale = apertus.ask("zoom", prompts["zoom"], fields, crop)
        cell = reply.get("cell") if isinstance(reply, dict) else None
        pixel = cell_centre(cell, region, grid)
        if pixel is None:
            logger.warning("zoom reply gave no valid cell: %s", raw[:LOGGED_REPLY_CHARACTERS])
            return {"result": result}
        return {"result": {**result, "target_pixel": pixel}}

    def route_start(state: PipelineState) -> str:
        """Skip annotate when this photo is already annotated.

        Args:
            state: initial pipeline state.

        Returns:
            ``"decide"`` or ``"annotate"``.
        """
        return "decide" if state["marks"] is not None else "annotate"

    def route_after_decide(state: PipelineState) -> str:
        """Refine the pixel of a plan when zoom is on; asks and refusals end here.

        Args:
            state: pipeline state after ``decide``.

        Returns:
            ``"refine"`` or ``END``.
        """
        result = state["result"]
        is_plan = result["target_pixel"] is not None and result["actions"][0]["action"] == "pick"
        return "refine" if config["zoom"] and is_plan else END

    graph = StateGraph(PipelineState)
    graph.add_node("annotate", annotate)
    graph.add_node("decide", decide)
    graph.add_node("refine", refine)
    graph.add_conditional_edges(START, route_start, ["annotate", "decide"])
    graph.add_edge("annotate", "decide")
    graph.add_conditional_edges("decide", route_after_decide, ["refine", END])
    graph.add_edge("refine", END)
    return graph.compile()
