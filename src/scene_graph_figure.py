"""Draw scene graphs in the shared format on the photo; used by run.py and the comparison script.

Each object is a label at its pick pixel, coloured by tray, and each "lies on" edge is a red
arrow from the upper to the lower object. Drawing only; nothing here changes a result.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

TRAY_COLOURS = {"grey_tray": "#1565C0", "black_tray": "#2E7D32"}
OTHER_COLOUR = "#6A1B9A"
STACK_COLOUR = "#D32F2F"
RUN_FIGURE_SIZE_IN = (22, 9)
COMPARISON_FIGURE_SIZE_IN = (22, 9.5)
PHOTO_WIDTH_RATIO = 1.25
TEXT_WIDTH_RATIO = 1.0
ARROW_LINE_WIDTH = 3.5
ARROW_HEAD_SCALE = 30
ARROW_SHRINK_POINTS = 18
LABEL_FONT_SIZE = 9
TEXT_FONT_SIZE = 8.5
DPI = 150
LEGEND = "blue: grey tray, green: black tray, red arrow: lies on"


def draw_graph(axis: plt.Axes, photo: np.ndarray, graph: dict, title: str) -> None:
    """Draw one scene graph on a photo.

    Args:
        axis: matplotlib axis to draw into.
        photo: (H, W, 3) uint8 RGB photo.
        graph: ``{"objects": [{"id", "pixel", "tray", "lies_on"}, ...]}``.
        title: axis title.

    Returns:
        None.
    """
    axis.imshow(photo)
    pixels = {item["id"]: item["pixel"] for item in graph["objects"]}
    for item in graph["objects"]:
        axis.annotate(
            item["id"],
            item["pixel"],
            fontsize=LABEL_FONT_SIZE,
            color="white",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": TRAY_COLOURS.get(item["tray"], OTHER_COLOUR),
                "edgecolor": "none",
            },
        )
        for lower in item["lies_on"]:
            axis.add_patch(
                FancyArrowPatch(
                    item["pixel"],
                    pixels[lower],
                    arrowstyle="-|>",
                    mutation_scale=ARROW_HEAD_SCALE,
                    color=STACK_COLOUR,
                    linewidth=ARROW_LINE_WIDTH,
                    shrinkA=ARROW_SHRINK_POINTS,
                    shrinkB=ARROW_SHRINK_POINTS,
                )
            )
    stacked = sum(len(item["lies_on"]) for item in graph["objects"])
    axis.set_title(f"{title}: {len(graph['objects'])} objects, {stacked} lies-on edge(s)")
    axis.set_axis_off()


def write_run_figure(
    photo: np.ndarray, graph: dict, scene_text: str, title: str, out_path: Path
) -> None:
    """Write the pipeline's scene graph on its photo next to the text Apertus received.

    Args:
        photo: (H, W, 3) uint8 RGB annotated photo.
        graph: the scene graph in the shared format.
        scene_text: scene description given to Apertus.
        title: figure title, the scene name.
        out_path: PNG file to write.

    Returns:
        None.
    """
    figure, (photo_axis, text_axis) = plt.subplots(
        1,
        2,
        figsize=RUN_FIGURE_SIZE_IN,
        gridspec_kw={"width_ratios": [PHOTO_WIDTH_RATIO, TEXT_WIDTH_RATIO]},
    )
    draw_graph(photo_axis, photo, graph, f"{title} ({LEGEND})")
    text_axis.text(0, 1, scene_text, family="monospace", fontsize=TEXT_FONT_SIZE, va="top")
    text_axis.set_title("Scene graph text given to Apertus")
    text_axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI)
    plt.close(figure)


def write_comparison_figure(
    photo: np.ndarray, ours: dict, truth: dict, title: str, out_path: Path
) -> None:
    """Write the pipeline's scene graph and the ground-truth graph side by side.

    Args:
        photo: (H, W, 3) uint8 RGB photo.
        ours: the pipeline's scene graph in the shared format.
        truth: the ground-truth scene graph in the shared format.
        title: figure title, the scene name and its scores.
        out_path: PNG file to write.

    Returns:
        None.
    """
    figure, (ours_axis, truth_axis) = plt.subplots(1, 2, figsize=COMPARISON_FIGURE_SIZE_IN)
    draw_graph(ours_axis, photo, ours, "Pipeline")
    draw_graph(truth_axis, photo, truth, "Ground truth")
    figure.suptitle(f"{title} ({LEGEND})")
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI)
    plt.close(figure)
