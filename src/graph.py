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

UNPARSED_SAFETY = {"analysis": "safety reply did not match the schema", "decision": None}
ACTOR_UNPARSED_ANALYSIS = "actor reply did not match the schema"
ACTOR_UNPARSED_FEEDBACK = (
    "Your previous reply was not one complete JSON object following the schema. Answer again."
)
JUDGE_UNPARSED_FEEDBACK = "The reviewer reply could not be read. Check your answer once more."
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
    result: dict | None


def build_graph(config: dict, client: ApertusClient) -> CompiledStateGraph:
    """Assemble and compile the pipeline graph.

    Args:
        config: full run config (prompts, loop limit, image settings).
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
            State update with ``scene``.
        """
        image = state["image"]
        prompt = prompts["scene"].format(width=image.width, height=image.height)
        parsed = client.ask(prompt, state["image_url"], SceneDescription)
        scene = parsed.model_dump() if parsed else UNPARSED_SCENE
        return {"scene": scene}

    def check_safety(state: PipelineState) -> dict:
        """Decide whether the instruction calls for a plan, a question or a refusal.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``safety``.
        """
        prompt = prompts["safety"].format(
            scene=json.dumps(state["scene"]), instruction=state["instruction"]
        )
        parsed = client.ask(prompt, state["image_url"], SafetyAssessment)
        safety = parsed.model_dump() if parsed else UNPARSED_SAFETY
        return {"safety": safety}

    def actor(state: PipelineState) -> dict:
        """Propose an answer, revising the previous one with the judge's feedback.

        Args:
            state: current pipeline state.

        Returns:
            State update with the new ``candidate`` and the next ``iteration``.
        """
        image = state["image"]
        feedback = ""
        if state["history"]:
            last = state["history"][-1]
            feedback = prompts["actor_feedback"].format(
                previous=json.dumps(last["candidate"]),
                critique=last["judgement"]["what_to_change"],
            )
        prompt = prompts["actor"].format(
            width=image.width,
            height=image.height,
            scene=json.dumps(state["scene"]),
            safety=json.dumps(state["safety"]),
            instruction=state["instruction"],
            feedback=feedback,
        )
        parsed = client.ask(prompt, state["image_url"], Candidate)
        candidate = (
            parsed.model_dump()
            if parsed
            else {
                "analysis": ACTOR_UNPARSED_ANALYSIS,
                "decision": "refuse",
                "plan_steps": [],
                "question": None,
                "reason": "The planner could not produce a valid answer.",
                "target_object": None,
                "target_pixel": None,
            }
        )
        iteration = state["iteration"] + 1
        return {"candidate": candidate, "iteration": iteration}

    def judge(state: PipelineState) -> dict:
        """Judge the current candidate right or wrong, showing a cross at its target pixel.

        A candidate that only stands in for an unparsable actor reply is wrong without a call.

        Args:
            state: current pipeline state.

        Returns:
            State update appending ``{iteration, candidate, judgement}`` to ``history``.
        """
        image = state["image"]
        candidate = state["candidate"]
        if candidate["analysis"] == ACTOR_UNPARSED_ANALYSIS:
            judgement = {
                "analysis": ACTOR_UNPARSED_ANALYSIS,
                "verdict": "wrong",
                "what_to_change": ACTOR_UNPARSED_FEEDBACK,
            }
            entry = {
                "iteration": state["iteration"],
                "candidate": candidate,
                "judgement": judgement,
            }
            return {"history": [entry]}
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
        parsed = client.ask(prompt, image_url, Judgement)
        judgement = (
            parsed.model_dump()
            if parsed
            else {
                "analysis": "judge reply did not match the schema",
                "verdict": "wrong",
                "what_to_change": JUDGE_UNPARSED_FEEDBACK,
            }
        )
        entry = {"iteration": state["iteration"], "candidate": candidate, "judgement": judgement}
        return {"history": [entry]}

    def finalize(state: PipelineState) -> dict:
        """Turn the latest candidate into the official answer.

        The loop stops at the first "right" verdict, so the latest candidate is the judged-right
        one, or the last revision when every round was judged wrong. A stand-in for an
        unparsable actor reply is used only when no round produced a parsable answer.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``result`` in the official schema.
        """
        parsed = [
            entry["candidate"]
            for entry in state["history"]
            if entry["candidate"]["analysis"] != ACTOR_UNPARSED_ANALYSIS
        ]
        final_candidate = parsed[-1] if parsed else state["history"][-1]["candidate"]
        result = candidate_to_result(state["instruction_id"], final_candidate, state["image"].scale)
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
        """Stop when the judge says "right" or the round limit is reached.

        Args:
            state: pipeline state after the judge.

        Returns:
            ``"finalize"`` or ``"actor"``.
        """
        judged_right = state["history"][-1]["judgement"]["verdict"] == "right"
        if judged_right or state["iteration"] >= loop["max_iterations"]:
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
