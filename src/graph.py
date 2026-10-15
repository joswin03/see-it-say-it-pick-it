"""LangGraph pipeline: describe the scene, check safety, then an actor-judge loop over plans."""

import json
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.answer import candidate_to_result
from src.image import ModelImage, draw_target, to_data_url
from src.llm import ApertusClient
from src.schemas import Candidate, Judgement, SafetyAssessment, SceneDescription

UNPARSED_SAFETY = {"reasoning": "safety reply did not match the schema", "decision": None}
UNPARSED_SCENE = {"objects": [], "trays": [], "person_visible": None, "error": "unparsed reply"}


class PipelineState(TypedDict):
    """Everything one instruction carries through the graph."""

    instruction_id: str
    instruction: str
    image: ModelImage
    image_url: str
    scene: dict | None
    safety: dict | None
    candidate: dict | None
    iteration: int
    history: Annotated[list[dict], operator.add]
    model_calls: Annotated[list[dict], operator.add]
    result: dict | None


def candidate_score(candidate: dict, judgement: dict, weights: dict) -> float:
    """Combine the judge's sub-scores into one score from 0 to 10.

    Args:
        candidate: ``Candidate`` as a dict.
        judgement: ``Judgement`` as a dict.
        weights: ``decision``, ``target`` and ``plan`` weights from the config.

    Returns:
        Weighted mean for a plan; the decision score alone for ask_user or refuse.
    """
    if candidate["decision"] != "plan":
        return float(judgement["decision_score"])
    weighted = (
        weights["decision"] * judgement["decision_score"]
        + weights["target"] * judgement["target_score"]
        + weights["plan"] * judgement["plan_score"]
    )
    return weighted / (weights["decision"] + weights["target"] + weights["plan"])


def build_graph(config: dict, client: ApertusClient) -> CompiledStateGraph:
    """Assemble and compile the pipeline graph.

    Args:
        config: full run config (prompts, loop limits, judge weights, image settings).
        client: Apertus client used by every node.

    Returns:
        Compiled LangGraph graph; invoke it with a ``PipelineState`` dict.
    """
    prompts = config["prompts"]
    loop = config["loop"]
    image_config = config["image"]

    def describe_scene(state: PipelineState) -> dict:
        """List the objects, trays and any person in the photo.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``scene`` and the raw model reply.
        """
        image = state["image"]
        prompt = prompts["scene"].format(width=image.width, height=image.height)
        parsed, raw = client.ask(prompt, state["image_url"], SceneDescription)
        scene = parsed.model_dump() if parsed else UNPARSED_SCENE
        return {"scene": scene, "model_calls": [{"node": "describe_scene", "raw": raw}]}

    def check_safety(state: PipelineState) -> dict:
        """Decide whether the instruction calls for a plan, a question or a refusal.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``safety`` and the raw model reply.
        """
        prompt = prompts["safety"].format(
            scene=json.dumps(state["scene"]), instruction=state["instruction"]
        )
        parsed, raw = client.ask(prompt, state["image_url"], SafetyAssessment)
        safety = parsed.model_dump() if parsed else UNPARSED_SAFETY
        return {"safety": safety, "model_calls": [{"node": "check_safety", "raw": raw}]}

    def actor(state: PipelineState) -> dict:
        """Propose an answer, revising the previous one with the judge's feedback.

        Args:
            state: current pipeline state.

        Returns:
            State update with the new ``candidate``, the next ``iteration`` and the raw reply.
        """
        image = state["image"]
        feedback = ""
        if state["history"]:
            last = state["history"][-1]
            feedback = prompts["actor_feedback"].format(
                previous=json.dumps(last["candidate"]),
                score=f"{last['score']:.1f}",
                critique=last["judgement"]["feedback"],
            )
        prompt = prompts["actor"].format(
            width=image.width,
            height=image.height,
            scene=json.dumps(state["scene"]),
            safety=json.dumps(state["safety"]),
            instruction=state["instruction"],
            feedback=feedback,
        )
        parsed, raw = client.ask(prompt, state["image_url"], Candidate)
        candidate = (
            parsed.model_dump()
            if parsed
            else {
                "reasoning": "actor reply did not match the schema",
                "decision": "refuse",
                "target_object": None,
                "target_pixel": None,
                "actions": [],
                "question": None,
                "reason": "The planner could not produce a valid answer.",
            }
        )
        iteration = state["iteration"] + 1
        return {
            "candidate": candidate,
            "iteration": iteration,
            "model_calls": [{"node": "actor", "iteration": iteration, "raw": raw}],
        }

    def judge(state: PipelineState) -> dict:
        """Score the current candidate, showing the judge a cross at its target pixel.

        Args:
            state: current pipeline state.

        Returns:
            State update appending ``{iteration, candidate, judgement, score}`` to ``history``.
        """
        image = state["image"]
        candidate = state["candidate"]
        image_url = state["image_url"]
        if candidate["decision"] == "plan" and candidate["target_pixel"]:
            marked = draw_target(
                image.rgb,
                candidate["target_pixel"],
                image_config["marker_radius_px"],
                image_config["marker_thickness_px"],
            )
            image_url = to_data_url(marked, image_config["jpeg_quality"])
        prompt = prompts["judge"].format(
            width=image.width,
            height=image.height,
            scene=json.dumps(state["scene"]),
            instruction=state["instruction"],
            candidate=json.dumps(candidate),
        )
        parsed, raw = client.ask(prompt, image_url, Judgement)
        judgement = (
            parsed.model_dump()
            if parsed
            else {
                "reasoning": "judge reply did not match the schema",
                "decision_score": 0,
                "target_score": 0,
                "plan_score": 0,
                "feedback": "none",
            }
        )
        score = candidate_score(candidate, judgement, config["judge_weights"])
        entry = {
            "iteration": state["iteration"],
            "candidate": candidate,
            "judgement": judgement,
            "score": score,
        }
        return {
            "history": [entry],
            "model_calls": [{"node": "judge", "iteration": state["iteration"], "raw": raw}],
        }

    def finalize(state: PipelineState) -> dict:
        """Keep the best-scoring candidate (latest on ties) as the official answer.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``result`` in the official schema.
        """
        best = max(reversed(state["history"]), key=lambda entry: entry["score"])
        result = candidate_to_result(
            state["instruction_id"], best["candidate"], state["image"].scale
        )
        return {"result": result}

    def route_start(state: PipelineState) -> str:
        """Skip the scene step when the scene of this photo is already known.

        Args:
            state: initial pipeline state.

        Returns:
            Name of the first node to run.
        """
        return "check_safety" if state["scene"] is not None else "describe_scene"

    def route_after_judge(state: PipelineState) -> str:
        """Stop when the judge accepts the candidate or the iteration budget is spent.

        Args:
            state: pipeline state after the judge.

        Returns:
            ``"finalize"`` or ``"actor"``.
        """
        accepted = state["history"][-1]["score"] >= loop["accept_score"]
        if accepted or state["iteration"] >= loop["max_iterations"]:
            return "finalize"
        return "actor"

    graph = StateGraph(PipelineState)
    graph.add_node("describe_scene", describe_scene)
    graph.add_node("check_safety", check_safety)
    graph.add_node("actor", actor)
    graph.add_node("judge", judge)
    graph.add_node("finalize", finalize)
    graph.add_conditional_edges(START, route_start, ["describe_scene", "check_safety"])
    graph.add_edge("describe_scene", "check_safety")
    graph.add_edge("check_safety", "actor")
    graph.add_edge("actor", "judge")
    graph.add_conditional_edges("judge", route_after_judge, ["actor", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()
