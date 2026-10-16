"""Plot the perception ablation: OpenCV regions vs Apertus boxes, from run outputs only.

Left panel: the official script score of each mode (points). Right panel: the scene-graph
comparison with the ground truth (percent). Each bar is the mean over that mode's runs, labelled
with its value; each run is a dot, so run-to-run spread is visible. Numbers come from every
run's ``manifest.json`` (score) and ``scene_graph_comparison.json`` (totals).

    .venv/bin/python figures/plot_ablation.py --opencv outputs/<run> \\
        --boxes outputs/<run_a> outputs/<run_b>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
OUT_STEM = "ablation_perception"
MODES = {"opencv": "OpenCV regions + Apertus naming", "boxes": "Apertus boxes on the whole photo"}
MODE_COLOURS = {"opencv": "#2a78d6", "boxes": "#eb6834"}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#1f1f1e"
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3df"
SCORE_METRICS = [
    ("total", "Script total", 65),
    ("grounding", "Grounding", 35),
    ("safety", "Safety", 30),
]
GRAPH_METRICS = [
    ("matched", "true_objects", "Objects found"),
    ("matched", "our_objects", "Objects real"),
    ("kind_right", "matched", "Kind right"),
    ("edges_found", "true_edges", "Stacking found"),
]
BAR_WIDTH = 0.36
BAR_GAP = 0.02
DOT_SIZE = 22
FIGURE_SIZE_IN = (12, 4.8)
DPI = 300
PERCENT = 100


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--opencv", type=Path, nargs="+", required=True, help="run directories")
    parser.add_argument("--boxes", type=Path, nargs="+", required=True, help="run directories")
    return parser.parse_args()


def run_values(run_dir: Path) -> dict[str, float]:
    """Script scores and scene-graph percentages of one run.

    Args:
        run_dir: run directory with ``manifest.json`` and ``scene_graph_comparison.json``.

    Returns:
        Metric name -> value (points for scores, percent for the scene-graph metrics).
    """
    score = json.loads((run_dir / "manifest.json").read_text())["score"]
    totals = json.loads((run_dir / "scene_graph_comparison.json").read_text())["totals"]
    values = {key: float(score[key]) for key, _label, _maximum in SCORE_METRICS}
    for numerator, denominator, label in GRAPH_METRICS:
        values[label] = PERCENT * totals[numerator] / totals[denominator]
    return values


def draw_panel(
    axis: plt.Axes, runs: dict[str, list[dict]], keys: list[str], labels: list[str], unit: str
) -> None:
    """Draw grouped bars (mean per mode) with one dot per run and a value label per bar.

    Args:
        axis: matplotlib axis.
        runs: mode -> list of per-run value dicts.
        keys: metric keys in display order.
        labels: x tick labels, same order.
        unit: suffix of the value labels ("" for points, "%" for percent).

    Returns:
        None.
    """
    positions = np.arange(len(keys))
    for offset, mode in zip((-0.5, 0.5), MODES):
        x = positions + offset * (BAR_WIDTH + BAR_GAP)
        means = [np.mean([run[key] for run in runs[mode]]) for key in keys]
        bars = axis.bar(x, means, BAR_WIDTH, color=MODE_COLOURS[mode], label=MODES[mode], zorder=2)
        for bar, mean in zip(bars, means):
            axis.annotate(
                f"{mean:.1f}{unit}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_PRIMARY,
            )
        for run in runs[mode]:
            axis.scatter(
                x,
                [run[key] for key in keys],
                s=DOT_SIZE,
                color=SURFACE,
                edgecolors=TEXT_PRIMARY,
                linewidths=1,
                zorder=3,
            )
    axis.set_xticks(positions, labels)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axis.grid(axis="y", color=GRID, zorder=0)
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)


def main() -> None:
    """Read the runs, draw both panels and write PDF and PNG into ``figures/``.

    Returns:
        None.
    """
    arguments = parse_arguments()
    runs = {
        "opencv": [run_values(path) for path in arguments.opencv],
        "boxes": [run_values(path) for path in arguments.boxes],
    }
    figure, (score_axis, graph_axis) = plt.subplots(1, 2, figsize=FIGURE_SIZE_IN, facecolor=SURFACE)
    draw_panel(
        score_axis,
        runs,
        [key for key, _label, _maximum in SCORE_METRICS],
        [f"{label}\n(of {maximum})" for _key, label, maximum in SCORE_METRICS],
        "",
    )
    score_axis.set_title("Dev script score (points)", color=TEXT_PRIMARY, fontsize=10)
    graph_labels = [label for _numerator, _denominator, label in GRAPH_METRICS]
    draw_panel(graph_axis, runs, graph_labels, graph_labels, "%")
    graph_axis.set_ylim(0, PERCENT * 1.12)
    graph_axis.set_title("Scene graph vs ground truth (percent)", color=TEXT_PRIMARY, fontsize=10)
    opencv_total = np.mean([run["total"] for run in runs["opencv"]])
    boxes_total = np.mean([run["total"] for run in runs["boxes"]])
    better, worse = ("OpenCV regions", "Apertus boxes")
    if boxes_total > opencv_total:
        better, worse = worse, better
    figure.suptitle(
        f"{better} beat {worse} on the dev set: "
        f"{max(opencv_total, boxes_total):.1f} vs {min(opencv_total, boxes_total):.1f} of 65 points",
        color=TEXT_PRIMARY,
        fontsize=12,
    )
    handles, labels = score_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    figure.text(
        0.99,
        0.01,
        f"bars: mean; dots: single runs (OpenCV {len(runs['opencv'])}, "
        f"Apertus boxes {len(runs['boxes'])})",
        ha="right",
        fontsize=7,
        color=TEXT_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    for suffix in ("pdf", "png"):
        out_path = FIGURES_DIR / f"{OUT_STEM}.{suffix}"
        figure.savefig(out_path, dpi=DPI, facecolor=SURFACE)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
