"""Draw the failure-case panel from a scored run and the dataset.

For each case (scene, instruction) the panel crops the photo around the pipeline's answer
pixel (first pick of the plan in ``scene_graphs.json``) and the expected object's ground-truth
mask centroid (from ``dataset/ground_truth/masks/``): a red ring marks what the pipeline
answered, a green ring the ground-truth object, so a reader sees both at a glance. The same
panel serves the report and the slides.

    .venv/bin/python figures/failure_cases.py --run outputs/<timestamp>-<name>

Writes ``figures/failure_cases.pdf`` and ``figures/failure_cases.png`` (300 dpi).
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_DIR / "figures"
OUT_STEM = "failure_cases"
ANSWER_RGB = (214, 39, 40)
EXPECTED_RGB = (44, 160, 44)
TEXT_PRIMARY = "#1f1f1e"
DPI = 300
FIGURE_SIZE_IN = (13.33, 4.4)
CROP_MIN_W_PX = 1000
CROP_MIN_H_PX = 760
CROP_MARGIN_PX = 130
TITLE_SIZE = 13.5
TITLE_WRAP = 34
MARKER_SIZE = 1600
MARKER_LINE_PX = 3.5

CASES = [
    {
        "scene": "scene_07",
        "instruction_id": "scene_07_i5",
        "expected_object": "mug",
    },
    {
        "scene": "scene_08",
        "instruction_id": "scene_08_i2",
        "expected_object": "small_part",
    },
    {
        "scene": "scene_09",
        "instruction_id": "scene_09_i2",
        "expected_object": "red_gear_2",
    },
]


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Args:
        None.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="scored run directory")
    parser.add_argument("--data", type=Path, default=REPO_DIR / "dataset", help="dataset directory")
    return parser.parse_args()


def answer_pixel(run_dir: Path, instruction_id: str) -> tuple[int, int]:
    """First pick pixel of the run's answer, from the run's own scene graph.

    Args:
        run_dir: scored run directory with ``report.json`` and ``scene_graphs.json``.
        instruction_id: instruction whose answer pixel is wanted.

    Returns:
        ``(u, v)`` pixel of the first object the run's plan picks.
    """
    report = json.loads((run_dir / "report.json").read_text())
    entry = next(e for e in report["instructions"] if e["instruction_id"] == instruction_id)
    scene = instruction_id.rsplit("_i", 1)[0]
    graphs = json.loads((run_dir / "scene_graphs.json").read_text())
    objects = {o["id"]: o["pixel"] for o in graphs[scene]["objects"]}
    first_object = re.match(r"pick\(([^)]+)\)", entry["plan"]).group(1)
    return tuple(objects[first_object])


def expected_centroid(data_dir: Path, scene: str, object_name: str) -> tuple[int, int]:
    """Ground-truth mask centroid of one object.

    Args:
        data_dir: dataset directory with ``ground_truth/masks``.
        scene: scene name such as ``scene_07``.
        object_name: object name as in ``ground_truth.json`` targets.

    Returns:
        ``(u, v)`` centroid of the object's mask.
    """
    masks_dir = data_dir / "ground_truth" / "masks"
    names = json.loads((masks_dir / f"{scene}.json").read_text())["objects"]
    mask = np.array(Image.open(masks_dir / f"{scene}.png"))
    wanted = object_name.replace("_", " ")
    candidates = {value: name for value, name in names.items() if name.replace("_", " ") == wanted}
    if not candidates:
        candidates = {
            value: name
            for value, name in names.items()
            if wanted in name.replace("_", " ") or name.replace("_", " ") in wanted
        }
    if len(candidates) != 1:
        raise KeyError(
            f"{scene}: cannot match {object_name!r} to one mask, candidates {candidates}"
        )
    value = int(next(iter(candidates)))
    vs, us = np.nonzero(mask == value)
    return int(us.mean()), int(vs.mean())


def crop_window(points: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Smallest fixed-size window containing every point, clamped to the photo.

    Args:
        points: ``(u, v)`` points that must be visible.
        shape: ``(height, width)`` of the photo.

    Returns:
        ``(u0, v0, u1, v1)`` crop window.
    """
    height, width = shape
    us = [p[0] for p in points]
    vs = [p[1] for p in points]
    u0, u1 = min(us) - CROP_MARGIN_PX, max(us) + CROP_MARGIN_PX
    v0, v1 = min(vs) - CROP_MARGIN_PX, max(vs) + CROP_MARGIN_PX
    if u1 - u0 < CROP_MIN_W_PX:
        grow = (CROP_MIN_W_PX - (u1 - u0)) / 2
        u0, u1 = u0 - grow, u1 + grow
    if v1 - v0 < CROP_MIN_H_PX:
        grow = (CROP_MIN_H_PX - (v1 - v0)) / 2
        v0, v1 = v0 - grow, v1 + grow
    if u0 < 0:
        u1 -= u0
        u0 = 0
    if v0 < 0:
        v1 -= v0
        v0 = 0
    u1 = min(u1, width)
    v1 = min(v1, height)
    u0 = max(0, min(u0, u1 - CROP_MIN_W_PX))
    v0 = max(0, min(v0, v1 - CROP_MIN_H_PX))
    return int(u0), int(v0), int(u1), int(v1)


def draw_case(axis: plt.Axes, case: dict, instruction: str, run_dir: Path, data_dir: Path) -> None:
    """Draw one failure case on an axis, cropped around both markers.

    Args:
        axis: matplotlib axis to draw on.
        case: one entry of ``CASES``.
        instruction: the instruction text for the title.
        run_dir: scored run directory.
        data_dir: dataset directory.

    Returns:
        None.
    """
    photo = np.array(Image.open(data_dir / case["scene"] / "rgb.png").convert("RGB"))
    answer = answer_pixel(run_dir, case["instruction_id"])
    expected = expected_centroid(data_dir, case["scene"], case["expected_object"])
    u0, v0, u1, v1 = crop_window([answer, expected], photo.shape[:2])
    axis.imshow(photo[v0:v1, u0:u1])
    for point, colour in ((answer, ANSWER_RGB), (expected, EXPECTED_RGB)):
        axis.scatter(
            [point[0] - u0],
            [point[1] - v0],
            s=MARKER_SIZE,
            facecolors="none",
            edgecolors=[tuple(c / 255 for c in colour)],
            linewidths=MARKER_LINE_PX,
            zorder=3,
        )
    axis.set_title(
        textwrap.fill(f"{case['instruction_id']}: \u201c{instruction}\u201d", TITLE_WRAP),
        fontsize=TITLE_SIZE,
        color=TEXT_PRIMARY,
        loc="left",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    """Build the panel and write the figure as PDF and PNG.

    Args:
        None.

    Returns:
        None.
    """
    arguments = parse_arguments()
    ground_truth = json.loads((arguments.data / "ground_truth" / "ground_truth.json").read_text())
    text_by_id = {e["instruction_id"]: e["instruction"] for e in ground_truth["instructions"]}

    figure, axes = plt.subplots(1, len(CASES), figsize=FIGURE_SIZE_IN, dpi=DPI)
    for axis, case in zip(axes, CASES, strict=True):
        draw_case(axis, case, text_by_id[case["instruction_id"]], arguments.run, arguments.data)
    figure.subplots_adjust(wspace=0.03)
    for suffix in (".pdf", ".png"):
        figure.savefig(FIGURES_DIR / (OUT_STEM + suffix), dpi=DPI, bbox_inches="tight")
        print(f"wrote {FIGURES_DIR / (OUT_STEM + suffix)}")


if __name__ == "__main__":
    main()
