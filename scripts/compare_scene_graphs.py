"""Compare a run's scene graphs with the ground-truth scene graphs; for evaluation only.

Objects are matched one to one by bounding-box overlap (greedy, highest IoU first, at least
``MATCH_MIN_IOU``). For every matched pair the script checks the name (object kind and colour),
the tray, the size on the table and whether the pick pixel lies in the true box; edges are
compared after mapping ids through the matches. Writes into the run directory:
``scene_graph_comparison.json`` (per scene and totals), ``scene_graph_comparison.txt`` (a
readable table) and ``scene_graph_comparison/<scene>.png`` (both graphs side by side).

    .venv/bin/python scripts/compare_scene_graphs.py outputs/<run>
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "starter"))

from scene_io import load_scene

from src.scene_graph_figure import write_comparison_figure

TRUTH_PATH = REPO_DIR / "dataset" / "coco_annotations" / "scene_graphs.json"
MATCH_MIN_IOU = 0.3
GEAR_WORDS = {"gear", "cog", "cogwheel", "wheel"}
PART_WORDS = {"part", "bracket", "component", "block", "piece"}
EXPECTED_NAMES = {
    "red cube": ("red", {"cube", "block"}),
    "red gear": ("red", GEAR_WORDS),
    "yellow gear": ("yellow", GEAR_WORDS),
    "mug": (None, {"mug", "cup"}),
    "mandarin": ("orange", {"mandarin", "orange", "fruit", "tangerine"}),
    "banana": ("yellow", {"banana", "fruit"}),
    "plate": (None, {"plate", "dish"}),
    "saucer": (None, {"saucer", "plate", "dish"}),
    "grey part": ("grey", PART_WORDS),
    "white part": ("white", PART_WORDS),
    "small part": ("white", PART_WORDS),
    "hand": (None, {"hand", "arm", "finger"}),
}


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="outputs/<run> with scene_graphs.json")
    parser.add_argument("--truth", type=Path, default=TRUTH_PATH)
    parser.add_argument("--data", type=Path, default=REPO_DIR / "dataset")
    return parser.parse_args()


def box_iou(box_a: list[int], box_b: list[int]) -> float:
    """Intersection over union of two [u_min, v_min, u_max, v_max] boxes.

    Args:
        box_a: first box.
        box_b: second box.

    Returns:
        IoU from 0 to 1.
    """
    width = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    height = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = width * height
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def match_objects(ours: list[dict], truth: list[dict]) -> dict[str, str]:
    """Match true objects to the pipeline's objects one to one by box overlap.

    Args:
        ours: the pipeline's objects.
        truth: the ground-truth objects.

    Returns:
        True id -> pipeline id.
    """
    pairs = sorted(
        (
            (box_iou(true_item["box"], our_item["box"]), true_item["id"], our_item["id"])
            for true_item in truth
            for our_item in ours
        ),
        reverse=True,
    )
    matches: dict[str, str] = {}
    for overlap, true_id, our_id in pairs:
        if overlap >= MATCH_MIN_IOU and true_id not in matches and our_id not in matches.values():
            matches[true_id] = our_id
    return matches


def name_check(true_name: str, our_item: dict) -> tuple[bool, bool]:
    """Whether the pipeline's name has the right object kind and colour.

    Args:
        true_name: ground-truth name, e.g. "red gear".
        our_item: the matched pipeline object (``name``).

    Returns:
        (kind right, colour right); colour counts as right when the truth names none.
    """
    colour, kind_words = EXPECTED_NAMES[true_name]
    words = set(our_item["name"].split())
    return bool(words & kind_words), colour is None or colour in words


def nearest(objects: list[dict], object_id: str) -> str | None:
    """Id of the object whose pick pixel is closest to the given object's.

    Args:
        objects: objects with ``id`` and ``pixel``.
        object_id: the anchor object.

    Returns:
        Nearest other object's id, or None if the anchor is alone.
    """
    anchor = next(item for item in objects if item["id"] == object_id)
    others = [item for item in objects if item["id"] != object_id]
    if not others:
        return None
    return min(others, key=lambda item: math.dist(item["pixel"], anchor["pixel"]))["id"]


def compare_scene(ours: dict, truth: dict) -> dict:
    """All comparison counts for one scene.

    Args:
        ours: the pipeline's scene graph.
        truth: the ground-truth scene graph.

    Returns:
        Counts, per-object findings and the id matches.
    """
    our_by_id = {item["id"]: item for item in ours["objects"]}
    matches = match_objects(ours["objects"], truth["objects"])
    findings = []
    counts = {
        "true_objects": len(truth["objects"]),
        "our_objects": len(ours["objects"]),
        "matched": len(matches),
        "kind_right": 0,
        "colour_right": 0,
        "tray_right": 0,
        "pixel_in_true_box": 0,
        "nearest_same": 0,
        "size_error_mm": 0,
    }
    for true_item in truth["objects"]:
        our_id = matches.get(true_item["id"])
        if our_id is None:
            findings.append(f"MISSED {true_item['id']}")
            continue
        our_item = our_by_id[our_id]
        kind_right, colour_right = name_check(true_item["name"], our_item)
        counts["kind_right"] += kind_right
        counts["colour_right"] += colour_right
        counts["tray_right"] += our_item["tray"] == true_item["tray"]
        u, v = our_item["pixel"]
        u_min, v_min, u_max, v_max = true_item["box"]
        counts["pixel_in_true_box"] += u_min <= u <= u_max and v_min <= v <= v_max
        counts["size_error_mm"] += abs(our_item["size_mm"][0] - true_item["size_mm"][0]) + abs(
            our_item["size_mm"][1] - true_item["size_mm"][1]
        )
        true_nearest = nearest(truth["objects"], true_item["id"])
        counts["nearest_same"] += matches.get(true_nearest) == nearest(ours["objects"], our_id)
        if not (kind_right and colour_right):
            findings.append(f"NAMED {true_item['id']} as '{our_item['name']}'")
        if our_item["tray"] != true_item["tray"]:
            findings.append(f"TRAY {true_item['id']}: {our_item['tray']} not {true_item['tray']}")
    unmatched = sorted(set(our_by_id) - set(matches.values()))
    findings += [f"EXTRA {our_id} ('{our_by_id[our_id]['name']}')" for our_id in unmatched]
    true_edges = {
        (matches.get(item["id"], f"missing:{item['id']}"), matches.get(lower, f"missing:{lower}"))
        for item in truth["objects"]
        for lower in item["lies_on"]
    }
    our_edges = {(item["id"], lower) for item in ours["objects"] for lower in item["lies_on"]}
    counts["true_edges"] = len(true_edges)
    counts["our_edges"] = len(our_edges)
    counts["edges_found"] = len(true_edges & our_edges)
    findings += [f"EDGE MISSED {upper} lies on {lower}" for upper, lower in true_edges - our_edges]
    findings += [f"EDGE EXTRA {upper} lies on {lower}" for upper, lower in our_edges - true_edges]
    return {"counts": counts, "findings": findings, "matches": matches}


def ratio(numerator: int, denominator: int) -> str:
    """Format a count as 'n/d = p%'.

    Args:
        numerator: count of successes.
        denominator: count of cases.

    Returns:
        Readable ratio.
    """
    share = numerator / denominator if denominator else 0.0
    return f"{numerator}/{denominator} = {share:.0%}"


def summary_lines(totals: dict) -> list[str]:
    """Readable totals over all scenes.

    Args:
        totals: summed counts.

    Returns:
        One line per metric.
    """
    matched = totals["matched"]
    mean_size_error = totals["size_error_mm"] / (2 * matched) if matched else 0.0
    return [
        f"objects found (recall)            {ratio(matched, totals['true_objects'])}",
        f"objects real (precision)          {ratio(matched, totals['our_objects'])}",
        f"object kind right                 {ratio(totals['kind_right'], matched)}",
        f"colour right                      {ratio(totals['colour_right'], matched)}",
        f"tray right                        {ratio(totals['tray_right'], matched)}",
        f"pick pixel inside the true box    {ratio(totals['pixel_in_true_box'], matched)}",
        f"same nearest neighbour            {ratio(totals['nearest_same'], matched)}",
        f"mean size error per side          {mean_size_error:.1f} mm",
        f"lies-on edges found (recall)      {ratio(totals['edges_found'], totals['true_edges'])}",
        f"lies-on edges real (precision)    {ratio(totals['edges_found'], totals['our_edges'])}",
    ]


def main() -> None:
    """Compare every scene of a run and write the report and figures into the run directory.

    Returns:
        None.
    """
    arguments = parse_arguments()
    ours_all = json.loads((arguments.run_dir / "scene_graphs.json").read_text())
    truth_all = json.loads(arguments.truth.read_text())
    figure_dir = arguments.run_dir / "scene_graph_comparison"
    figure_dir.mkdir(exist_ok=True)
    report: dict = {"scenes": {}, "totals": {}}
    lines = []
    for scene_name in sorted(set(ours_all) & set(truth_all)):
        result = compare_scene(ours_all[scene_name], truth_all[scene_name])
        report["scenes"][scene_name] = result
        counts = result["counts"]
        for key, value in counts.items():
            report["totals"][key] = report["totals"].get(key, 0) + value
        headline = (
            f"{scene_name}: objects {counts['matched']}/{counts['true_objects']} found, "
            f"{counts['our_objects'] - counts['matched']} extra; "
            f"edges {counts['edges_found']}/{counts['true_edges']} found, "
            f"{counts['our_edges'] - counts['edges_found']} extra"
        )
        lines += [headline, *(f"    {finding}" for finding in result["findings"])]
        photo = load_scene(arguments.data / scene_name).rgb
        write_comparison_figure(
            photo,
            ours_all[scene_name],
            truth_all[scene_name],
            headline,
            figure_dir / f"{scene_name}.png",
        )
    lines = ["TOTALS", *summary_lines(report["totals"]), "", *lines]
    report["summary"] = summary_lines(report["totals"])
    (arguments.run_dir / "scene_graph_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    text = "\n".join(lines) + "\n"
    (arguments.run_dir / "scene_graph_comparison.txt").write_text(text)
    print(re.sub(r"\n{3,}", "\n\n", text))


if __name__ == "__main__":
    main()
