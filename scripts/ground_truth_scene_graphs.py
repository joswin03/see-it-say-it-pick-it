"""Ground-truth scene graphs of the dev scenes, for evaluation only; the pipeline never reads them.

Ported from Joswin's ``tools/gt_scene_graph.py`` (branch ``dev/joswin-opencv``, revision
``2c1d170``). Objects come from Mac's hand-drawn COCO masks in
``dataset/coco_annotations/_annotations.coco.json``; names are mapped to the vocabulary of
``dataset/ground_truth``. The masks are modal and never overlap, so the "lies on" edges are
written by hand from the dev instructions (``LIES_ON``). Pick pixel, size on the table and
tray are measured with the same helpers as the pipeline, so both graphs use one format:
``{"objects": [{"id", "name", "pixel", "box", "size_mm", "tray", "lies_on"}]}``.

    .venv/bin/python scripts/ground_truth_scene_graphs.py
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "starter"))

from scene_io import load_scene

from src.calibration import lens_distortion
from src.objects import size_mm
from src.regions import find_trays, interior_point, to_work_scale

COCO_PATH = REPO_DIR / "dataset" / "coco_annotations" / "_annotations.coco.json"
OUT_PATH = REPO_DIR / "dataset" / "coco_annotations" / "scene_graphs.json"
CONFIG_PATH = REPO_DIR / "configs" / "pipeline.yaml"
SCENE_NAME_CHARACTERS = len("scene_00")
TABLE = "table"
RENAMES = {
    30: "red_cube",
    34: "red_cube",
    46: "red_gear_1",
    48: "red_gear_3",
    26: "grey_part_1",
    25: "grey_part_2",
    27: "white_part",
    1: "banana",
    4: "saucer",
    5: "plate",
    18: "saucer",
    16: "red_gear_1",
    15: "red_gear_2",
    17: "red_cube",
    50: "red_gear",
    52: "red_cube",
    21: "red_gear_1",
    20: "red_gear_2",
    22: "red_cube",
    36: "grey_part",
    38: "saucer",
    39: "red_cube",
    41: "red_gear_1",
    42: "red_gear_2",
    40: "red_gear_3",
    6: "hand",
    7: "saucer",
    10: "red_gear_1",
    9: "red_gear_2",
}
LIES_ON = {
    "scene_06": {"yellow_gear": ["red_gear_2"]},
    "scene_07": {"mug": ["red_gear"]},
    "scene_08": {"red_cube": ["red_gear_1", "red_gear_2"]},
    "scene_09": {"red_cube": ["grey_part", "red_gear_3"]},
    "scene_10": {"mug": ["hand"]},
}


def decode_rle(segmentation: dict) -> np.ndarray:
    """Decode a compressed COCO RLE mask (pycocotools' format) without pycocotools.

    Args:
        segmentation: ``{"size": [h, w], "counts": str}``.

    Returns:
        (h, w) bool mask.
    """
    text, counts, position = segmentation["counts"], [], 0
    while position < len(text):
        value, shift, more = 0, 0, True
        while more:
            character = ord(text[position]) - 48
            value |= (character & 0x1F) << 5 * shift
            more = character & 0x20
            position += 1
            shift += 1
            if not more and character & 0x10:
                value |= -1 << 5 * shift
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    height, width = segmentation["size"]
    flat = np.zeros(height * width, bool)
    start = 0
    for index, length in enumerate(counts):
        flat[start : start + length] = index % 2 == 1
        start += length
    return flat.reshape(width, height).T


def load_masks(coco: dict) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Decode every annotation under its ground-truth name, checking it against its box.

    Args:
        coco: the parsed COCO file.

    Returns:
        Scene name -> (object id, (H, W) bool mask) per object.
    """
    categories = {category["id"]: category["name"] for category in coco["categories"]}
    scenes = {image["id"]: image["file_name"][:SCENE_NAME_CHARACTERS] for image in coco["images"]}
    by_scene: dict[str, list[tuple[str, np.ndarray]]] = {}
    for annotation in coco["annotations"]:
        mask = decode_rle(annotation["segmentation"])
        rows, columns = np.nonzero(mask)
        box = [columns.min(), rows.min(), np.ptp(columns) + 1, np.ptp(rows) + 1]
        if not np.allclose(box, annotation["bbox"], atol=2):
            raise ValueError(f"annotation {annotation['id']}: mask does not fit its box")
        name = RENAMES.get(annotation["id"], categories[annotation["category_id"]])
        by_scene.setdefault(scenes[annotation["image_id"]], []).append((name, mask))
    return by_scene


def tray_of(mask: np.ndarray, trays: dict[str, np.ndarray]) -> str:
    """The tray holding at least half of an object, otherwise the table.

    Args:
        mask: (h, w) bool at work scale.
        trays: ``find_trays`` output at the same scale.

    Returns:
        "grey_tray", "black_tray" or "table".
    """
    inside = {name: int((tray & mask).sum()) for name, tray in trays.items()}
    best = max(inside, key=inside.get)
    return best if 2 * inside[best] >= mask.sum() else TABLE


def scene_graph(scene_name: str, annotations: list[tuple[str, np.ndarray]], config: dict) -> dict:
    """Ground-truth scene graph of one dev scene.

    Args:
        scene_name: e.g. "scene_06".
        annotations: (object id, (H, W) bool mask) per object.
        config: the pipeline config (``perception.work_scale`` and tray thresholds).

    Returns:
        The scene graph in the shared format, left to right.
    """
    scene_dir = REPO_DIR / "dataset" / scene_name
    scene = load_scene(scene_dir)
    distortion = lens_distortion(scene_dir)
    work_scale = config["perception"]["work_scale"]
    small = to_work_scale(scene.rgb, work_scale)
    trays = find_trays(cv2.cvtColor(small, cv2.COLOR_RGB2HSV), config["perception"])
    work_size = (small.shape[1], small.shape[0])
    lies_on = LIES_ON.get(scene_name, {})
    names = [name for name, _ in annotations]
    if len(set(names)) != len(names):
        raise ValueError(f"{scene_name}: duplicate object ids {names}")
    objects = []
    for name, mask in annotations:
        u, v = interior_point(np.pad(mask, 1))
        rows, columns = np.nonzero(mask)
        work_mask = cv2.resize(mask.astype(np.uint8), work_size, interpolation=cv2.INTER_NEAREST)
        objects.append(
            {
                "id": name,
                "name": re.sub(r"_\d+$", "", name).replace("_", " "),
                "pixel": [u - 1, v - 1],
                "box": [
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()) + 1,
                    int(rows.max()) + 1,
                ],
                "size_mm": list(size_mm(work_mask > 0, work_scale, scene, distortion)),
                "tray": tray_of(work_mask > 0, trays),
                "lies_on": lies_on.get(name, []),
            }
        )
    return {"objects": sorted(objects, key=lambda item: item["pixel"][0])}


def main() -> None:
    """Write the ground-truth scene graphs of all annotated dev scenes.

    Returns:
        None.
    """
    config = yaml.safe_load(CONFIG_PATH.read_text())
    by_scene = load_masks(json.loads(COCO_PATH.read_text()))
    graphs = {name: scene_graph(name, by_scene[name], config) for name in sorted(by_scene)}
    OUT_PATH.write_text(json.dumps(graphs, indent=2) + "\n")
    print(f"wrote {OUT_PATH.relative_to(REPO_DIR)} ({len(graphs)} scenes)")


if __name__ == "__main__":
    main()
