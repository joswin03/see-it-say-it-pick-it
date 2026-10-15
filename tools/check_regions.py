"""Development check: do the OpenCV regions cover the ground-truth objects?

Uses ``dataset/ground_truth/masks`` for evaluation only; the pipeline never reads them.
For every ground-truth object, counts the regions whose interior pixel lies on it, and writes
an overlay image per scene.

    python tools/check_regions.py --data dataset --out runs/region_check
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.perception.regions import find_regions
from src.workcell import load_scene, mm_per_pixel


def overlay(rgb: np.ndarray, found: dict) -> np.ndarray:
    """Draw region outlines and ids on a work-scale copy of the photo.

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        found: output of ``find_regions``.

    Returns:
        (h, w, 3) uint8 BGR image for writing.
    """
    scale = found["work_scale"]
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    for region in found["regions"]:
        contours, _ = cv2.findContours(
            region["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (255, 255, 0), 1)
        u, v = (round(c * scale) for c in region["pixel"])
        cv2.putText(image, str(region["id"]), (u - 6, v + 6), 0, 0.5, (0, 0, 0), 3)
        cv2.putText(image, str(region["id"]), (u - 6, v + 6), 0, 0.5, (0, 255, 255), 1)
    return image


def main() -> None:
    """Run the check over every scene that has a ground-truth mask."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--config", type=Path, default=Path("configs/opencv_apertus.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())["perception"]
    args.out.mkdir(parents=True, exist_ok=True)
    mask_dir = args.data / "ground_truth" / "masks"
    covered, total = 0, 0
    for names_path in sorted(mask_dir.glob("*.json")):
        scene = load_scene(args.data / names_path.stem)
        scale = mm_per_pixel(scene) / config["work_scale"]
        found = find_regions(scene.rgb, config, scale * scale)
        truth = cv2.imread(str(names_path.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
        objects = json.loads(names_path.read_text())["objects"]
        hits = {name: [] for name in objects.values()}
        for region in found["regions"]:
            u, v = region["pixel"]
            key = str(int(truth[v, u]))
            if key in objects:
                hits[objects[key]].append(region["id"])
        summary = ", ".join(f"{name}:{ids or 'MISS'}" for name, ids in hits.items())
        print(f"{scene.name}: {len(found['regions'])} regions | {summary}")
        covered += sum(bool(ids) for ids in hits.values())
        total += len(hits)
        cv2.imwrite(str(args.out / f"{scene.name}.jpg"), overlay(scene.rgb, found))
    print(f"covered {covered} of {total} ground-truth objects")


if __name__ == "__main__":
    main()
