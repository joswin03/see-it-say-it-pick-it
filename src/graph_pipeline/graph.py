"""LangGraph pipeline: OpenCV finds regions, Apertus names them and decides, code checks.

Scene nodes (find_regions, name_regions, build_scene) run once per photo; the instruction
nodes (check_safety, plan, check_plan, finalize) run per instruction. Vision and geometry are
Joswin's modules (``src/perception``, ``src/scene_graph``, ``src/pipeline``), used unchanged.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph_pipeline.answer import checked, give_up_answer, no_plan_answer
from src.graph_pipeline.llm import ApertusClient
from src.graph_pipeline.schemas import RegionName, SafetyScreen, decision_model
from src.perception.regions import find_regions
from src.pipeline import drop_redundant_holds, marked_photo, plan_problem, to_answer
from src.reasoning.client import image_url
from src.reasoning.naming import outlined_crop
from src.reasoning.safety import screen_result
from src.scene_graph import build_scene_graph, describe, size_mm
from src.workcell import Scene, mm_per_pixel

UNKNOWN_NAME = "unknown object"
UNPARSED_DECISION_PROBLEM = (
    "your reply was not one complete JSON object in the requested format; "
    "keep the analysis to two or three sentences"
)


class PipelineState(TypedDict):
    """Everything one instruction carries through the graph."""

    instruction_id: str
    instruction: str
    scene: Scene
    gripper_mm: float
    regions: list[dict] | None
    work_scale: float | None
    perceived: dict | None
    safety: dict | None
    stop: tuple[str, str] | None
    decision: dict | None
    problem: str | None
    attempts: int
    result: dict | None


def build_graph(config: dict, client: ApertusClient) -> CompiledStateGraph:
    """Assemble and compile the pipeline graph.

    Args:
        config: full run config (perception, naming, scene_graph, safety, decision sections).
        client: Apertus client used by every model step.

    Returns:
        Compiled LangGraph graph; invoke it with a ``PipelineState`` dict.
    """
    perception = config["perception"]
    naming = config["naming"]
    scene_graph = config["scene_graph"]
    safety = config["safety"]
    decision = config["decision"]

    def find_regions_node(state: PipelineState) -> dict:
        """Segment the photo into trays and object regions with OpenCV, sizes in mm.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``regions`` (masks at work scale) and ``work_scale``.
        """
        scene = state["scene"]
        mm_per_work_pixel = mm_per_pixel(scene) / perception["work_scale"]
        found = find_regions(scene.rgb, perception, mm_per_work_pixel**2)
        for region in found["regions"]:
            region["size_mm"] = size_mm(region["mask"], perception["work_scale"], scene)
        return {"regions": found["regions"], "work_scale": found["work_scale"]}

    def name_regions(state: PipelineState) -> dict:
        """Ask Apertus what each region is, from a crop with the region outlined.

        A reply that does not parse keeps the region as "unknown object": it was found in the
        photo, so the planner may still ask about it.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``regions``, each with ``name`` and ``is_object`` added.
        """
        scene = state["scene"]
        min_side_pixels = round(naming["crop_min_side_mm"] / mm_per_pixel(scene))
        named = []
        for region in state["regions"]:
            crop = outlined_crop(
                scene.rgb,
                region["mask"],
                perception["work_scale"],
                naming["crop_padding_fraction"],
                min_side_pixels,
            )
            length_mm, width_mm = region["size_mm"]
            prompt = naming["prompt"].format(
                colour=region["colour"],
                length_mm=length_mm,
                width_mm=width_mm,
                tray=region["tray"].replace("_", " "),
                examples=", ".join(naming["example_names"]),
            )
            reply = client.ask(prompt, image_url(crop, naming["crop_max_side_px"]), RegionName)
            if reply is None:
                named.append({**region, "name": UNKNOWN_NAME, "is_object": True})
                continue
            name = reply.name.strip().lower()
            is_fixture = any(word in name.split() for word in naming["fixture_words"])
            named.append({**region, "name": name, "is_object": reply.is_object and not is_fixture})
        return {"regions": named}

    def build_scene(state: PipelineState) -> dict:
        """Turn named regions into the scene graph, its text and the marked photo.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``perceived``: ``objects``, ``text`` and ``marked_url``.
        """
        scene = state["scene"]
        mm_per_work_pixel = mm_per_pixel(scene) / perception["work_scale"]
        objects = build_scene_graph(
            state["regions"], state["work_scale"], mm_per_work_pixel, scene_graph, scene
        )
        marked = marked_photo(scene.rgb, objects, state["work_scale"])
        perceived = {
            "objects": objects,
            "text": describe(objects, scene_graph["nearest_count"], scene_graph["too_wide_mm"]),
            "marked_url": image_url(marked, decision["image_max_side_px"]),
        }
        return {"perceived": perceived}

    def check_safety(state: PipelineState) -> dict:
        """Ask Apertus what the instruction asks for; code turns the facts into a stop or not.

        A reply that does not parse leaves the instruction unchecked (logged by the client);
        the planner, which also knows the safety rules, then decides alone.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``safety`` (the screen's facts) and ``stop`` (action, message).
        """
        if not safety["enabled"]:
            return {"safety": None, "stop": None}
        prompt = safety["prompt"].format(
            scene=state["perceived"]["text"], instruction=state["instruction"]
        )
        reply = client.ask(prompt, None, SafetyScreen)
        if reply is None:
            return {"safety": None, "stop": None}
        facts = reply.model_dump()
        return {"safety": facts, "stop": screen_result(facts)}

    def plan(state: PipelineState) -> dict:
        """Ask Apertus to plan, ask or refuse over the scene graph and the marked photo.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``decision`` (None if the reply did not parse) and ``attempts``.
        """
        perceived = state["perceived"]
        prompt = decision["prompt"].format(
            scene=perceived["text"],
            instruction=state["instruction"],
            gripper_mm=round(state["gripper_mm"]),
        )
        if state["problem"]:
            prompt += decision["feedback"].format(problem=state["problem"])
        object_ids = [item["id"] for item in perceived["objects"]]
        reply = client.ask(prompt, perceived["marked_url"], decision_model(object_ids))
        return {
            "decision": reply.model_dump() if reply else None,
            "attempts": state["attempts"] + 1,
        }

    def check_plan(state: PipelineState) -> dict:
        """Check a plan in code: known objects, graspable, not onto itself, hold only last.

        Args:
            state: current pipeline state.

        Returns:
            State update with the cleaned ``decision`` and ``problem`` (None if executable).
        """
        reply = state["decision"]
        if reply is None:
            return {"problem": UNPARSED_DECISION_PROBLEM}
        if reply["decision"] != "plan":
            return {"problem": None}
        moves = drop_redundant_holds(reply["moves"])
        problem = plan_problem(moves, state["perceived"]["objects"], scene_graph["too_wide_mm"])
        return {"decision": {**reply, "moves": moves}, "problem": problem}

    def finalize(state: PipelineState) -> dict:
        """Turn the outcome into one answer in the official schema.

        Args:
            state: current pipeline state.

        Returns:
            State update with ``result``: the safety stop, the checked plan (stacked objects
            cleared first, pixel from the scene graph) or, after the last failed attempt, a
            request to rephrase.
        """
        instruction_id = state["instruction_id"]
        if state["stop"] is not None:
            result = no_plan_answer(instruction_id, *state["stop"])
        elif state["problem"] is None:
            result = to_answer(
                instruction_id, state["decision"], state["perceived"]["objects"], config
            )
        else:
            result = give_up_answer(instruction_id, state["problem"])
        return {"result": checked(result)}

    def route_start(state: PipelineState) -> str:
        """Skip the scene nodes when this photo's scene graph is already known.

        Args:
            state: initial pipeline state.

        Returns:
            Name of the first node to run.
        """
        return "check_safety" if state["perceived"] is not None else "find_regions"

    def route_after_safety(state: PipelineState) -> str:
        """Stop at a refusal from the safety screen, otherwise plan.

        Args:
            state: pipeline state after the safety screen.

        Returns:
            ``"finalize"`` or ``"plan"``.
        """
        return "finalize" if state["stop"] is not None else "plan"

    def route_after_check(state: PipelineState) -> str:
        """Retry the plan with the problem as feedback while attempts remain.

        Args:
            state: pipeline state after the plan check.

        Returns:
            ``"plan"`` or ``"finalize"``.
        """
        if state["problem"] is not None and state["attempts"] < decision["max_attempts"]:
            return "plan"
        return "finalize"

    graph = StateGraph(PipelineState)
    graph.add_node("find_regions", find_regions_node)
    graph.add_node("name_regions", name_regions)
    graph.add_node("build_scene", build_scene)
    graph.add_node("check_safety", check_safety)
    graph.add_node("plan", plan)
    graph.add_node("check_plan", check_plan)
    graph.add_node("finalize", finalize)
    graph.add_conditional_edges(START, route_start, ["find_regions", "check_safety"])
    graph.add_edge("find_regions", "name_regions")
    graph.add_edge("name_regions", "build_scene")
    graph.add_edge("build_scene", "check_safety")
    graph.add_conditional_edges("check_safety", route_after_safety, ["plan", "finalize"])
    graph.add_edge("plan", "check_plan")
    graph.add_conditional_edges("check_plan", route_after_check, ["plan", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()
