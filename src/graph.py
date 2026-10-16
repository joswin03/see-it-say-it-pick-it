"""LangGraph pipeline, prompting only: annotate the photo, check stacking, decide.

annotate: Apertus marks every object (label from the workcell list, and a box); the boxes are
drawn and numbered on the photo.
stack: for every pair of overlapping boxes, Apertus says on a close-up which object lies on
the other.
Both run once per photo; later instructions of the same photo start at decide.
decide: Apertus sees the annotated photo, the numbered object list, the stacking facts and the
instruction, and answers with object numbers; code turns that into the official answer.
"""

import logging
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.answer import to_result
from src.apertus import Apertus
from src.marks import draw_marks, objects_text, read_marks, salvage_marks
from src.stack import overlapping_pairs, pair_close_up, stacking_line

LOGGED_REPLY_CHARACTERS = 300
NO_STACKING = "(no object lies on another)"
logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """Everything one instruction carries through the graph."""

    instruction_id: str
    instruction: str
    photo: np.ndarray
    marks: list[dict] | None
    annotated: np.ndarray | None
    stacking: str | None
    result: dict | None


def build_graph(config: dict, apertus: Apertus) -> CompiledStateGraph:
    """Assemble and compile the pipeline graph.

    Args:
        config: run config with ``prompts``, ``places``, ``object_labels``,
            ``salvage_unparsed_marks``, ``stack_min_overlap`` and ``stack_margin_fraction``.
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

    def stack(state: PipelineState) -> dict:
        """Ask, for every pair of overlapping boxes, which object lies on the other.

        Args:
            state: pipeline state after ``annotate``.

        Returns:
            State update with ``stacking``: one fact per stacked pair, as text.
        """
        lines = []
        for pair in overlapping_pairs(state["marks"], config["stack_min_overlap"]):
            close_up = pair_close_up(state["photo"], pair, config["stack_margin_fraction"])
            fields = {
                "first": f"{pair[0]['number']} ({pair[0]['label']})",
                "second": f"{pair[1]['number']} ({pair[1]['label']})",
            }
            reply, raw, _scale = apertus.ask("stack", prompts["stack"], fields, close_up)
            if reply is None:
                logger.warning("stack reply held no JSON: %s", raw[:LOGGED_REPLY_CHARACTERS])
            line = stacking_line(reply, pair)
            if line:
                lines.append(line)
        return {"stacking": "\n".join(lines) or NO_STACKING}

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
            "stacking": state["stacking"],
        }
        reply, raw, _scale = apertus.ask("decide", prompts["decide"], fields, state["annotated"])
        if reply is None:
            logger.warning("decide reply held no JSON: %s", raw[:LOGGED_REPLY_CHARACTERS])
        result = to_result(state["instruction_id"], reply, state["marks"], config["places"])
        return {"result": result}

    def route_start(state: PipelineState) -> str:
        """Skip annotate and stack when this photo is already done.

        Args:
            state: initial pipeline state.

        Returns:
            ``"decide"`` or ``"annotate"``.
        """
        return "decide" if state["marks"] is not None else "annotate"

    graph = StateGraph(PipelineState)
    graph.add_node("annotate", annotate)
    graph.add_node("stack", stack)
    graph.add_node("decide", decide)
    graph.add_conditional_edges(START, route_start, ["annotate", "decide"])
    graph.add_edge("annotate", "stack")
    graph.add_edge("stack", "decide")
    graph.add_edge("decide", END)
    return graph.compile()
