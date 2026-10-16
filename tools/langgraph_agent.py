"""Apertus as a LangGraph tool-calling agent over the ground-truth scene graph.

Tests tool calling: the prompt lists only each object's id, tray and pixel (JSON), so Apertus must call
``what_is_on_top`` and ``can_grip`` (JSON answers) to learn stacking and grip, then answers in the
pipeline's decision schema. Code checks the plan and sets the target pixel, as in the pipeline.
Dev only: reads the ground truth.

    python -m tools.langgraph_agent --base-url https://api.inference.cscs.ch/v1
"""

import argparse
import collections
import json
import time
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from openai import BadRequestError, LengthFinishReasonError
from pydantic import ValidationError

from src.pipeline import drop_redundant_holds, plan_problem, to_answer
from src.reasoning.client import api_key, close_stalled_reply
from src.reasoning.decision import decision_schema
from src.scene_graph import too_wide
from src.workcell import load_scene
from tools.gt_scene_graph import REPO, perceive_ground_truth


def scene_tools(objects: list[dict], too_wide_mm: float) -> list:
    """The two tools over one scene.

    Args:
        objects: scene-graph objects from ``perceive_ground_truth``.
        too_wide_mm: measured width above which an object cannot be gripped.

    Returns:
        LangChain tools.
    """
    by_id = {item["id"]: item for item in objects}

    def unknown(object_id: str) -> str:
        return json.dumps({"error": f"'{object_id}' is not an object id", "object_ids": list(by_id)})

    @tool
    def what_is_on_top(object_id: str) -> str:
        """Say which objects this object lies on, and which objects lie on top of it."""
        if object_id not in by_id:
            return unknown(object_id)
        item = by_id[object_id]
        return json.dumps(
            {"id": object_id, "lies_on_top_of": item["on_top_of"], "objects_on_top_of_it": item["under"]}
        )

    @tool
    def can_grip(object_id: str) -> str:
        """Say whether the gripper can pick this object, and whether it can hold other objects."""
        if object_id not in by_id:
            return unknown(object_id)
        wide = too_wide(by_id[object_id], too_wide_mm)
        return json.dumps({"id": object_id, "graspable": not wide, "can_hold_things": wide})

    return [what_is_on_top, can_grip]


def build_graph(llm: ChatOpenAI, answer_llm, tools: list, answer_prompt: str):
    """agent <-> tools while the model calls known tools, then one schema-constrained answer turn.

    Apertus does not stop calling tools by itself: to answer it calls a decision ("plan",
    "refuse") as if it were a tool, or puts its answer text in a tool call. Both end the tool loop
    and are passed to the answer turn as text; the log marks them (``unknown_tool``, ``malformed``).
    """
    names = {item.name for item in tools}

    def agent(state: MessagesState) -> dict:
        try:
            return {"messages": [llm.bind_tools(tools).invoke(state["messages"])]}
        except ValidationError as exc:  # tool-call arguments that are not a JSON object
            return {"messages": [AIMessage(str(exc.errors()[0]["input"]), response_metadata={"malformed": True})]}

    def route(state: MessagesState) -> str:
        calls = state["messages"][-1].tool_calls
        return "tools" if calls and all(call["name"] in names for call in calls) else "answer"

    def answer(state: MessagesState) -> dict:
        *history, last = state["messages"]
        if last.tool_calls:  # unknown tool: keep the attempted call as text, unanswered calls are rejected
            calls = " ".join(f"{call['name']}: {json.dumps(call['args'])}" for call in last.tool_calls)
            last = AIMessage(f"{last.content} {calls}".strip(), response_metadata={"unknown_tool": True})
        request = HumanMessage(answer_prompt)
        return {"messages": [request, answer_llm.invoke([*history, last, request])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("answer", answer)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", "answer"])
    graph.add_edge("tools", "agent")
    graph.add_edge("answer", END)
    return graph.compile()


def parse_answer(message: AIMessage) -> tuple[dict | None, str | None]:
    """The decision from the answer turn, or None and the problem."""
    try:
        reply = json.loads(message.content)
    except json.JSONDecodeError as error:
        return None, f"answer is not valid JSON ({error})"
    return reply, None


def message_log(message) -> dict:
    """One message as a JSON-serialisable log entry."""
    if isinstance(message, AIMessage):
        return {
            "ai": message.content,
            "tool_calls": [{"name": call["name"], "args": call["args"]} for call in message.tool_calls],
            "invalid_tool_calls": message.invalid_tool_calls,
            "malformed": message.response_metadata.get("malformed", False),
        }
    if isinstance(message, ToolMessage):
        return {"tool": message.name, "result": message.content}
    return {message.type: message.content}


def main() -> None:
    """Answer every dev instruction with the tool-calling agent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "dataset")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs" / "scratch")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    args = parser.parse_args()
    config = yaml.safe_load((REPO / "configs" / "opencv_apertus.yaml").read_text())
    settings, decision = config["agent"], config["decision"]
    endpoint = {"model": args.model, "base_url": args.base_url, "api_key": api_key(), "temperature": 0,
                "timeout": config["request_timeout_s"]}
    llm = ChatOpenAI(**endpoint, max_tokens=settings["max_tokens"])
    too_wide_mm = config["scene_graph"]["too_wide_mm"]
    run_dir = args.runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-langgraph-gt-{args.data.name}"
    run_dir.mkdir(parents=True)
    results, raw = [], []
    for scene_dir in sorted(p for p in args.data.iterdir() if (p / "rgb.png").exists()):
        scene = load_scene(scene_dir)
        objects = perceive_ground_truth(scene, config, "json")["objects"]
        scene_json = json.dumps([{"id": item["id"], "tray": item["tray"], "pixel": item["pixel"]} for item in objects])
        schema = decision_schema([item["id"] for item in objects], decision["max_analysis_chars"], decision["max_moves"])
        answer_llm = ChatOpenAI(**endpoint, max_tokens=decision["max_tokens"]).bind(
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}}
        )
        graph = build_graph(llm, answer_llm, scene_tools(objects, too_wide_mm), settings["answer_prompt"])
        for instruction_id, _, text in scene.instructions:
            messages = [SystemMessage(settings["prompt"].format(scene=scene_json)), HumanMessage(f'Instruction: "{text}"')]
            reply, problem = None, None
            try:  # stream, so the log keeps the messages before an error
                for state in graph.stream(
                    {"messages": messages}, {"recursion_limit": 2 * settings["max_steps"] + 2}, stream_mode="values"
                ):
                    messages = state["messages"]
                reply, problem = parse_answer(messages[-1])
            except GraphRecursionError:
                problem = f"still calling tools after {settings['max_steps']} turns"
            except BadRequestError as exc:  # vLLM's tool-call parser rejected what Apertus wrote
                problem = f"server: {exc}"
            except LengthFinishReasonError as exc:  # the openai client raises instead of returning the cut-off answer
                reply = close_stalled_reply(exc.completion.choices[0].message.content or "", schema)
                problem = None if reply else "answer cut off at the token limit"
            if reply and reply["decision"] == "plan":
                reply["moves"] = drop_redundant_holds(reply["moves"])
                problem = plan_problem(reply["moves"], objects, too_wide_mm)
            if problem is None:
                answer = to_answer(instruction_id, reply, objects, config)
            else:
                answer = {
                    "instruction_id": instruction_id,
                    "target_pixel": None,
                    "actions": [{"action": "ask_user", "question": f"I could not form an executable plan ({problem}). "
                                 "Could you rephrase the instruction?"}],
                }
            results.append(answer)
            raw.append(
                {
                    "instruction_id": instruction_id,
                    "instruction": text,
                    "problem": problem,
                    "messages": [message_log(m) for m in messages[1:]],
                }
            )
            calls = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
            print(f"{instruction_id}: {answer['actions'][0]['action']:8} {' '.join(calls) or '(no tool call)'}  {text}")
    (run_dir / "results.json").write_text(json.dumps({"team": config["team"], "results": results}, indent=2))
    (run_dir / "results.raw.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in raw))
    print_summary(raw, {item.name for item in scene_tools([], 0)})
    print(f"{len(results)} answers; run record in {run_dir}")


def print_summary(raw: list[dict], names: set[str]) -> None:
    """Tool-use counts over the run."""
    calls = [[c["name"] for m in entry["messages"] for c in m.get("tool_calls", [])] for entry in raw]
    malformed = sum(m.get("malformed", False) for entry in raw for m in entry["messages"])
    bad_ids = sum("error" in m.get("result", "") for entry in raw for m in entry["messages"])
    print(
        f"instructions with a tool call: {sum(bool(c) for c in calls)}/{len(raw)}; "
        f"calls: {dict(collections.Counter(name for c in calls for name in c))}; "
        f"unknown-tool calls: {sum(name not in names for c in calls for name in c)}; "
        f"malformed calls: {malformed}; unknown object ids: {bad_ids}; "
        f"no usable answer: {sum(entry['problem'] is not None for entry in raw)}/{len(raw)}"
    )


if __name__ == "__main__":
    main()
