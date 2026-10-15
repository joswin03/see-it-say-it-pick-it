"""Ground-truth scene graphs from our COCO annotations (``dataset/coco_annotations``).

Renames the Roboflow labels to the ground-truth vocabulary (``dataset/ground_truth``), in place,
redraws the ``annotated/`` previews with those names, then writes one scene graph per scene:
``{"objects": [{"id", "pixel", "on", "tray"}]}``. For
evaluating our scene graphs only; the pipeline never reads this.

    python -m tools.gt_scene_graph

``perceive_ground_truth`` stands in for the pipeline's perception in ``run.py --gt-scene-graph``,
to measure how Apertus plans when perception is perfect.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.perception.regions import find_trays, interior_point, to_work_scale
from src.pipeline import marked_photo
from src.reasoning.client import image_url
from src.scene_graph import HAND_WORDS, describe, size_mm
from src.workcell import Scene, load_scene, table_xy

REPO = Path(__file__).resolve().parent.parent
COCO = REPO / "dataset" / "coco_annotations" / "_annotations.coco.json"
OUT = REPO / "dataset" / "coco_annotations" / "scene_graphs.json"

# Annotation id -> ground-truth name. Numbers follow dataset/ground_truth/masks where the ground
# truth has them, otherwise left to right; a single object of a kind has no number. Keyed by id
# so that a second run changes nothing (scene 03 swaps red_gear_1 and red_gear_3).
RENAMES = {
    30: "red_cube",  # scene 01
    34: "red_cube",  # 02
    46: "red_gear_1", 48: "red_gear_3",  # 03
    26: "grey_part_1", 25: "grey_part_2", 27: "white_part",  # 04
    1: "banana", 4: "saucer", 5: "plate",  # 05
    18: "saucer", 16: "red_gear_1", 15: "red_gear_2", 17: "red_cube",  # 06
    50: "red_gear", 52: "red_cube",  # 07
    21: "red_gear_1", 20: "red_gear_2", 22: "red_cube",  # 08
    36: "grey_part", 38: "saucer", 39: "red_cube",  # 09
    41: "red_gear_1", 42: "red_gear_2", 40: "red_gear_3",  # 09
    6: "hand", 7: "saucer", 10: "red_gear_1", 9: "red_gear_2",  # 10
}

# What lies on what, from the instructions (masks are modal and never overlap). The mug in
# scene 10 is held in the hand.
ON = {
    "scene_06": {"yellow_gear": "red_gear_2"},
    "scene_07": {"mug": "red_gear"},
    "scene_08": {"red_cube": ["red_gear_1", "red_gear_2"]},
    "scene_09": {"red_cube": ["grey_part", "red_gear_3"]},
    "scene_10": {"mug": "hand"},
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
            char = ord(text[position]) - 48
            value |= (char & 0x1F) << 5 * shift
            more = char & 0x20
            position += 1
            shift += 1
            if not more and char & 0x10:
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
    return flat.reshape(width, height).T  # COCO runs go down columns


def rename(coco: dict) -> dict:
    """Apply ``RENAMES`` and rebuild the category list from the names in use.

    Args:
        coco: the parsed COCO file.

    Returns:
        The same dict, changed in place.
    """
    old_names = {category["id"]: category["name"] for category in coco["categories"]}
    names = {
        annotation["id"]: RENAMES.get(annotation["id"], old_names[annotation["category_id"]])
        for annotation in coco["annotations"]
    }
    root = coco["categories"][0]  # Roboflow's project row, the supercategory of all others
    new_ids = {name: index + 1 for index, name in enumerate(sorted(set(names.values())))}
    coco["categories"] = [root] + [
        {"id": index, "name": name, "supercategory": root["name"]}
        for name, index in new_ids.items()
    ]
    for annotation in coco["annotations"]:
        annotation["category_id"] = new_ids[names[annotation["id"]]]
    return coco


def tray_of(mask: np.ndarray, trays: dict, work_size: tuple[int, int]) -> str:
    """The tray holding most of the object, or "table" if neither holds half of it.

    Args:
        mask: (H, W) bool full-resolution mask.
        trays: output of ``find_trays`` at work scale.
        work_size: (w, h) of the work-scale image.

    Returns:
        "grey_tray", "black_tray" or "table".
    """
    small = cv2.resize(mask.astype(np.uint8), work_size, interpolation=cv2.INTER_NEAREST) > 0
    inside = {name: (tray & small).sum() for name, tray in trays.items()}
    best = max(inside, key=inside.get)
    return best if inside[best] >= small.sum() / 2 else "table"


def scene_graph(scene_name: str, annotations: list[tuple[str, np.ndarray]], perception: dict) -> dict:
    """Scene graph of one scene from its named masks.

    Args:
        scene_name: e.g. "scene_06".
        annotations: (name, (H, W) bool mask) per object.
        perception: the ``perception`` section of the pipeline config.

    Returns:
        ``{"objects": [...]}`` sorted left to right.
    """
    rgb = load_scene(REPO / "dataset" / scene_name).rgb
    small = to_work_scale(rgb, perception["work_scale"])
    trays = find_trays(cv2.cvtColor(small, cv2.COLOR_RGB2HSV), perception)
    on = ON.get(scene_name, {})
    names = [name for name, _ in annotations]
    assert len(set(names)) == len(names), f"{scene_name}: duplicate names {names}"
    for upper, lower in on.items():
        assert {upper, *np.atleast_1d(lower)} <= set(names), f"{scene_name}: unknown {upper}/{lower}"
    objects = []
    for name, mask in annotations:
        u, v = interior_point(np.pad(mask, 1))  # padded: the hand runs off the image edge
        item = {"id": name, "pixel": [u - 1, v - 1]}
        if name in on:
            item["on"] = on[name]
        item["tray"] = tray_of(mask, trays, (small.shape[1], small.shape[0]))
        objects.append(item)
    return {"objects": sorted(objects, key=lambda item: item["pixel"][0])}


def preview(rgb: np.ndarray, annotations: list[tuple[str, np.ndarray]], graph: dict) -> np.ndarray:
    """The photo with every mask tinted, outlined and labelled at its scene-graph pixel.

    Args:
        rgb: (H, W, 3) uint8 photo.
        annotations: (name, (H, W) bool mask) per object.
        graph: output of ``scene_graph`` for the same scene.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pixels = {item["id"]: item["pixel"] for item in graph["objects"]}
    for index, (name, mask) in enumerate(annotations):
        colour = cv2.cvtColor(np.uint8([[[index * 37 % 180, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        image[mask] = (0.6 * image[mask] + 0.4 * colour).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, colour.tolist(), 4)
        for shade, thickness in (((0, 0, 0), 8), ((255, 255, 255), 3)):
            cv2.putText(image, name, pixels[name], cv2.FONT_HERSHEY_SIMPLEX, 1.6, shade, thickness)
    return image


def load_masks(coco: dict) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Decode every annotation, checking each mask against its bounding box.

    Args:
        coco: the parsed COCO file.

    Returns:
        Scene name -> (object name, (H, W) bool mask) per object.
    """
    names = {category["id"]: category["name"] for category in coco["categories"]}
    scenes = {image["id"]: image["file_name"][:8] for image in coco["images"]}  # "scene_06_png..."
    by_scene: dict[str, list] = {}
    for annotation in coco["annotations"]:
        mask = decode_rle(annotation["segmentation"])
        rows, columns = np.nonzero(mask)
        box = [columns.min(), rows.min(), np.ptp(columns) + 1, np.ptp(rows) + 1]
        assert np.allclose(box, annotation["bbox"], atol=2), (annotation["id"], box)
        by_scene.setdefault(scenes[annotation["image_id"]], []).append(
            (names[annotation["category_id"]], mask)
        )
    return by_scene


def perceive_ground_truth(scene: Scene, config: dict, form: str) -> dict:
    """The pipeline's ``perceive_scene`` output, built from the ground-truth scene graph.

    Args:
        scene: loaded scene.
        config: full pipeline config.
        form: "text" (rendered by ``describe``, as the pipeline sends it) or "json" (the scene's
            entry of ``scene_graphs.json`` verbatim).

    Returns:
        Dict with ``objects``, ``text``, ``marked``, ``marked_url`` and ``naming_log``.
    """
    masks = dict(load_masks(json.loads(COCO.read_text()))[scene.name])
    graph = json.loads(OUT.read_text())[scene.name]
    work_scale = config["perception"]["work_scale"]
    height, width = scene.rgb.shape[:2]
    work_size = (round(width * work_scale), round(height * work_scale))
    objects = []
    for item in graph["objects"]:
        lower = item.get("on", [])
        mask = cv2.resize(masks[item["id"]].astype(np.uint8), work_size, interpolation=cv2.INTER_NEAREST)
        name = re.sub(r"_\d+$", "", item["id"]).replace("_", " ")  # "red_gear_2" -> "red gear"
        objects.append(
            {
                "id": item["id"],
                "name": name,
                "aliases": [],
                "pixel": item["pixel"],
                "tray": item["tray"],
                "on_top_of": [lower] if isinstance(lower, str) else lower,
                "is_hand": any(word in name for word in HAND_WORDS),
                "mask": mask.astype(bool),
                "size_mm": size_mm(mask, work_scale, scene),
                "table_xy": table_xy(scene, [item["pixel"]])[0],
                "source_ids": [],
            }
        )
    for item in objects:
        item["under"] = [other["id"] for other in objects if item["id"] in other["on_top_of"]]
    marked = marked_photo(scene.rgb, objects, work_scale)
    text = describe(objects, config["scene_graph"]["too_wide_mm"]) if form == "text" else json.dumps(graph)
    return {
        "objects": objects,
        "text": text,
        "marked": marked,
        "marked_url": image_url(marked, config["decision"]["image_max_side_px"]),
        "naming_log": [],
    }


def main() -> None:
    """Rename the annotations in place and write the scene graphs."""
    coco = rename(json.loads(COCO.read_text()))
    perception = yaml.safe_load((REPO / "configs" / "opencv_apertus.yaml").read_text())["perception"]
    by_scene = load_masks(coco)
    graphs = {name: scene_graph(name, by_scene[name], perception) for name in sorted(by_scene)}
    for image in coco["images"]:
        scene_name = image["file_name"][:8]
        rgb = load_scene(REPO / "dataset" / scene_name).rgb
        drawn = preview(rgb, by_scene[scene_name], graphs[scene_name])
        cv2.imwrite(str(COCO.parent / "annotated" / f"annotated_{image['file_name']}"), drawn)
    COCO.write_text(json.dumps(coco, indent=4))  # as Roboflow wrote it, no final newline
    OUT.write_text(json.dumps(graphs, indent=2) + "\n")
    print(f"renamed {COCO.name}; wrote {OUT.relative_to(REPO)} ({len(graphs)} scenes)")


if __name__ == "__main__":
    main()
