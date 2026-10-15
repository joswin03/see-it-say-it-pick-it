#!/usr/bin/env python3
"""Draw COCO polygon/RLE segmentation annotations on images and save to dataset/coco_annotations/annotated/."""
import json
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mask_utils

base_dir = Path("dataset/coco_annotations")
out_dir = base_dir / "annotated"
out_dir.mkdir(parents=True, exist_ok=True)

with open(base_dir / "_annotations.coco.json") as f:
    coco = json.load(f)

categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
images = {img["id"]: img for img in coco["images"]}

# Group annotations by image
img_to_anns = {}
for ann in coco["annotations"]:
    img_to_anns.setdefault(ann["image_id"], []).append(ann)

# Deterministic colors per category
np.random.seed(42)
colors = {cid: np.random.randint(50, 255, size=3).tolist() for cid in categories}

for img_id, img_info in images.items():
    file_name = img_info["file_name"]
    img_path = base_dir / file_name
    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    overlay = img.copy()

    for ann in img_to_anns.get(img_id, []):
        cid = ann["category_id"]
        color = colors[cid]
        seg = ann["segmentation"]

        # Decode mask (RLE or polygon)
        if isinstance(seg, dict):
            m = mask_utils.decode(seg).astype(bool)
        elif isinstance(seg, list):
            m = np.zeros(img.shape[:2], dtype=np.uint8)
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(m, [pts], 1)
            m = m.astype(bool)
        else:
            continue

        overlay[m] = (overlay[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        contours, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 3)

        # Label near centroid
        ys, xs = np.where(m)
        if len(xs) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            cv2.putText(img, categories.get(cid, str(cid)), (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, categories.get(cid, str(cid)), (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # Blend overlay with boundary image
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    out_path = out_dir / f"annotated_{file_name}"
    cv2.imwrite(str(out_path), img)
    print(f"Saved {out_path.name}")
