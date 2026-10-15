"""Draw the pipeline graph: ASCII in the terminal, Mermaid source and a viewable HTML page.

.venv/bin/python scripts/draw_graph.py
"""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from src.config import load_config
from src.graph import build_graph
from src.llm import ApertusClient

CONFIG_PATH = REPO_DIR / "configs" / "actor_critic.yaml"
GRAPH_DIR = REPO_DIR / "outputs" / "graph"
HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline Graph</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>body {{ font-family: sans-serif; background: #fff; margin: 2rem; }}</style>
</head><body><h1>Apertus actor-judge pipeline</h1>
<pre class="mermaid">
{mermaid}
</pre><script>mermaid.initialize({{ startOnLoad: true }});</script></body></html>
"""


def main() -> None:
    """Build the graph with a mock client and write its drawings to ``outputs/graph/``.

    Returns:
        None.
    """
    config = load_config(CONFIG_PATH, [])
    graph = build_graph(config, ApertusClient(config, mock=True)).get_graph()
    mermaid = graph.draw_mermaid()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    (GRAPH_DIR / "graph.mmd").write_text(mermaid)
    (GRAPH_DIR / "graph.html").write_text(HTML_TEMPLATE.format(mermaid=mermaid))
    print(graph.draw_ascii())
    print(f"Mermaid: {GRAPH_DIR / 'graph.mmd'}\nOpen in a browser: {GRAPH_DIR / 'graph.html'}")


if __name__ == "__main__":
    main()
