"""Draw the pipeline's LangGraph graph as PNG (300 dpi) and PDF, from the compiled graph.

Nodes and edges come from ``build_graph`` (built with the mock client, no model call); the
layout, colours and edge labels below only decide how they are drawn. Orange nodes call
Apertus, blue nodes are code; a dashed arrow is a conditional edge.

    .venv/bin/python figures/draw_graph.py
"""

import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langchain_core.runnables.graph import Graph
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from src.apertus import Apertus
from src.graph import build_graph

CONFIG_PATH = REPO_DIR / "configs" / "pipeline.yaml"
OUTPUT_STEM = REPO_DIR / "figures" / "pipeline_graph"
DPI = 300
NODE_WIDTH = 2.5
NODE_HEIGHT = 0.62
APERTUS_FILL = "#fbe9dd"
APERTUS_EDGE = "#b5541c"
CODE_FILL = "#e8eef7"
CODE_EDGE = "#3b5b8c"
TERMINAL_FILL = "#ededed"
TERMINAL_EDGE = "#555555"
EDGE_COLOUR = "#333333"
POSITIONS = {
    "__start__": (0.0, 5.5),
    "annotate": (0.0, 4.4),
    "stack": (0.0, 3.3),
    "safety": (0.0, 2.2),
    "decide": (0.0, 1.1),
    "__end__": (0.0, 0.0),
}
CAPTIONS = {
    "__start__": "START\ninstruction + photo",
    "annotate": "annotate (OpenCV + Apertus)\nregions, names, once per photo",
    "stack": "stack (convex-hull geometry)\nwhich object lies on which",
    "safety": "safety (Apertus)\naction words, manner, person",
    "decide": "decide (Apertus)\nvalidated moves / ask / refuse",
    "__end__": "END\nanswer in official schema",
}
APERTUS_NODES = {"annotate", "safety", "decide"}
SWITCHED_NODES: set[str] = set()
SIDE_ROUTES = {("__start__", "safety"): ("left", 0.5), ("safety", "__end__"): ("right", 0.7)}
CURVATURE: dict[tuple[str, str], float] = {}
EDGE_LABELS = {("__start__", "safety"): ("photo already\nannotated", (-3.35, 3.8))}


def node_style(node: str) -> tuple[str, str, str]:
    """Fill colour, border colour and line style of a node.

    Args:
        node: node id.

    Returns:
        (fill, border, line style).
    """
    if node.startswith("__"):
        return TERMINAL_FILL, TERMINAL_EDGE, "-"
    fill, border = (APERTUS_FILL, APERTUS_EDGE) if node in APERTUS_NODES else (CODE_FILL, CODE_EDGE)
    return fill, border, "--" if node in SWITCHED_NODES else "-"


def border_point(node: str, towards: tuple[float, float]) -> tuple[float, float]:
    """Point on a node's box border in the direction of another point.

    Args:
        node: node id.
        towards: (x, y) the edge goes to or comes from.

    Returns:
        (x, y) on the border.
    """
    x, y = POSITIONS[node]
    dx, dy = towards[0] - x, towards[1] - y
    if dx == 0 and dy == 0:
        return x, y
    scale = min(
        (NODE_WIDTH / 2) / abs(dx) if dx else float("inf"),
        (NODE_HEIGHT / 2) / abs(dy) if dy else float("inf"),
    )
    return x + dx * scale, y + dy * scale


def draw(graph: Graph, path_stem: Path) -> None:
    """Draw every node and edge of the graph and save PNG and PDF.

    Args:
        graph: ``CompiledStateGraph.get_graph()`` result.
        path_stem: output path without extension.

    Returns:
        None.
    """
    figure, axes = plt.subplots(figsize=(7, 7.5))
    for node in graph.nodes:
        x, y = POSITIONS[node]
        fill, border, style = node_style(node)
        box = FancyBboxPatch(
            (x - NODE_WIDTH / 2, y - NODE_HEIGHT / 2),
            NODE_WIDTH,
            NODE_HEIGHT,
            boxstyle="round,pad=0.04",
            facecolor=fill,
            edgecolor=border,
            linestyle=style,
            linewidth=1.6,
        )
        axes.add_patch(box)
        axes.text(x, y, CAPTIONS[node], ha="center", va="center", fontsize=9)
    for edge in graph.edges:
        source, target = POSITIONS[edge.source], POSITIONS[edge.target]
        route = SIDE_ROUTES.get((edge.source, edge.target))
        if route:
            side, curvature = route
            direction = -1 if side == "left" else 1
            start = (source[0] + direction * NODE_WIDTH / 2, source[1])
            end = (target[0] + direction * NODE_WIDTH / 2, target[1])
            going_up = target[1] > source[1]
            sign = 1 if (side == "left") != going_up else -1
            connection = f"arc3,rad={sign * curvature}"
        else:
            start, end = border_point(edge.source, target), border_point(edge.target, source)
            connection = f"arc3,rad={CURVATURE.get((edge.source, edge.target), 0.0)}"
        arrow = FancyArrowPatch(
            start,
            end,
            connectionstyle=connection,
            arrowstyle="-|>",
            mutation_scale=14,
            linestyle="--" if edge.conditional else "-",
            color=EDGE_COLOUR,
            linewidth=1.1,
        )
        axes.add_patch(arrow)
        label = EDGE_LABELS.get((edge.source, edge.target))
        if label:
            text, position = label
            axes.text(*position, text, fontsize=8, ha="left", va="center")
    axes.set_xlim(-3.4, 3.4)
    axes.set_ylim(-0.6, 6.0)
    axes.axis("off")
    axes.set_title(
        "Perception cached per photo; Apertus screens and plans each instruction",
        fontsize=10,
    )
    figure.savefig(path_stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    """Build the graph with the mock client and draw it.

    Args:
        None.

    Returns:
        None.
    """
    config = yaml.safe_load(CONFIG_PATH.read_text())
    graph = build_graph(config, Apertus(config, mock=True)).get_graph()
    missing = set(graph.nodes) - set(POSITIONS)
    if missing:
        raise KeyError(f"no layout position for nodes {sorted(missing)}")
    draw(graph, OUTPUT_STEM)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')} and {OUTPUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
