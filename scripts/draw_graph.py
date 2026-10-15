"""Draw the pipeline graph as a PNG (and Mermaid source), straight from the compiled LangGraph.

.venv/bin/python scripts/draw_graph.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from src.config import load_config
from src.graph import build_graph
from src.llm import ApertusClient

CONFIG_PATH = REPO_DIR / "configs" / "actor_critic.yaml"
GRAPH_DIR = REPO_DIR / "outputs" / "graph"
DPI = 300
NODE_WIDTH = 2.6
NODE_HEIGHT = 0.55
ROW_SPACING = 1.2
SIDE_EDGE_CURVATURE = 0.5
NODE_COLOUR = "#e8e4ff"
TERMINAL_COLOUR = "#c9c0fc"
EDGE_COLOUR = "#333333"


def node_label(node_id: str) -> str:
    """Turn a node id into the text shown in its box.

    Args:
        node_id: LangGraph node id, e.g. ``__start__`` or ``check_safety``.

    Returns:
        Display label, e.g. ``START`` or ``check_safety``.
    """
    return node_id.strip("_").upper() if node_id.startswith("__") else node_id


def draw_png(graph, path: Path, loop_label: str, skip_label: str) -> None:
    """Render nodes top to bottom in insertion order; skips and loops are drawn as side arcs.

    Args:
        graph: ``CompiledStateGraph.get_graph()`` result.
        path: PNG file to write.
        loop_label: text placed on edges that go back up (the retry loop).
        skip_label: text placed on edges that skip nodes going down.

    Returns:
        None.
    """
    node_ids = list(graph.nodes)
    row_of = {node_id: row for row, node_id in enumerate(node_ids)}
    centre_of = {node_id: (0.0, -row * ROW_SPACING) for node_id, row in row_of.items()}
    figure_height = len(node_ids) * ROW_SPACING + 1
    figure, axes = plt.subplots(figsize=(7, figure_height))
    for node_id, (x, y) in centre_of.items():
        colour = TERMINAL_COLOUR if node_id.startswith("__") else NODE_COLOUR
        box = FancyBboxPatch(
            (x - NODE_WIDTH / 2, y - NODE_HEIGHT / 2),
            NODE_WIDTH,
            NODE_HEIGHT,
            boxstyle="round,pad=0.05",
            facecolor=colour,
            edgecolor=EDGE_COLOUR,
        )
        axes.add_patch(box)
        axes.text(x, y, node_label(node_id), ha="center", va="center", fontsize=12)
    for edge in graph.edges:
        source_x, source_y = centre_of[edge.source]
        target_x, target_y = centre_of[edge.target]
        row_step = row_of[edge.target] - row_of[edge.source]
        goes_back = row_step < 0
        if row_step == 1:
            start = (source_x, source_y - NODE_HEIGHT / 2)
            end = (target_x, target_y + NODE_HEIGHT / 2)
            connection = "arc3,rad=0"
        else:
            side = 1 if goes_back else -1
            start = (source_x + side * NODE_WIDTH / 2, source_y)
            end = (target_x + side * NODE_WIDTH / 2, target_y)
            connection = f"arc3,rad={SIDE_EDGE_CURVATURE}"
        arrow = FancyArrowPatch(
            start,
            end,
            connectionstyle=connection,
            arrowstyle="-|>",
            mutation_scale=16,
            linestyle="--" if edge.conditional else "-",
            color=EDGE_COLOUR,
        )
        axes.add_patch(arrow)
        label_y = (source_y + target_y) / 2
        if goes_back:
            axes.text(NODE_WIDTH / 2 + 0.75, label_y, loop_label, fontsize=9, va="center")
        elif row_step > 1:
            axes.text(
                -NODE_WIDTH / 2 - 1.05, label_y, skip_label, fontsize=9, va="center", ha="right"
            )
    axes.set_xlim(-NODE_WIDTH - 2.6, NODE_WIDTH + 2.6)
    axes.set_ylim(-(len(node_ids) - 1) * ROW_SPACING - 0.8, 0.8)
    axes.axis("off")
    axes.set_title("Apertus actor-judge pipeline (dashed = conditional edge)", fontsize=12)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    """Build the graph with a mock client and write ``graph.png`` and ``graph.mmd``.

    Returns:
        None.
    """
    config = load_config(CONFIG_PATH, [])
    graph = build_graph(config, ApertusClient(config, mock=True)).get_graph()
    loop = config["loop"]
    loop_label = f"retry with feedback\nverdict wrong\nand round < {loop['max_iterations']}"
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    (GRAPH_DIR / "graph.mmd").write_text(graph.draw_mermaid())
    draw_png(graph, GRAPH_DIR / "graph.png", loop_label, "scene of this photo\nalready described")
    print(f"Wrote {GRAPH_DIR / 'graph.png'} and {GRAPH_DIR / 'graph.mmd'}")


if __name__ == "__main__":
    main()
