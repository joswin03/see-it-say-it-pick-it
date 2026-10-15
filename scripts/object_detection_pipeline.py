#!/usr/bin/env python3
"""Object detection pipeline: isolate objects from CIELAB differences and build proposals."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab

# Workcell boundaries measured from reference edit:
# Shave top, bottom, and left outer border to kill reflections / tray edges
BORDER_TOP = 308
BORDER_BOTTOM = 305  # y > 2048 - 305 = 1743
BORDER_LEFT = 192
BORDER_RIGHT = 0

MIN_OBJECT_AREA = 1500  # pixels

def load_and_preprocess(scene_rgb_path: Path, bg_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute CIELAB mask with borders zeroed and speckle noise removed."""
    im_rgb = np.array(Image.open(scene_rgb_path))
    im_lab = rgb2lab(im_rgb)

    delta_e = np.sqrt(np.sum((im_lab - bg_lab) ** 2, axis=-1))
    delta_ab = np.sqrt((im_lab[..., 1] - bg_lab[..., 1]) ** 2 + (im_lab[..., 2] - bg_lab[..., 2]) ** 2)
    delta_l = im_lab[..., 0] - bg_lab[..., 0]

    # CIELAB shadow suppression
    is_shadow = (delta_l < -4) & (delta_ab < 6)
    mask = (delta_e > 8) & (~is_shadow)

    # Shave borders to zero
    h, w = mask.shape
    mask[:BORDER_TOP, :] = False
    if BORDER_BOTTOM > 0:
        mask[h - BORDER_BOTTOM:, :] = False
    mask[:, :BORDER_LEFT] = False
    if BORDER_RIGHT > 0:
        mask[:, w - BORDER_RIGHT:] = False

    # Connected-components area filtering
    lbl, num = ndimage.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool), im_rgb

    sizes = ndimage.sum(mask, lbl, range(1, num + 1))
    keep = sizes >= MIN_OBJECT_AREA
    keep_padded = np.pad(keep, (1, 0), constant_values=False)
    clean_mask = keep_padded[lbl]

    return clean_mask, im_rgb


def extract_proposals(mask: np.ndarray) -> list[dict]:
    """Extract bounding boxes and centroids for candidate object blobs."""
    lbl, num = ndimage.label(mask)
    proposals = []

    for i in range(1, num + 1):
        blob = (lbl == i)
        area = int(np.sum(blob))
        ys, xs = np.where(blob)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]

        proposals.append({
            "id": i,
            "area": area,
            "centroid": [cx, cy],
            "bbox": bbox,  # [x, y, w, h]
        })

    return proposals


def main():
    bg_path = Path("dataset/background.png")
    if not bg_path.exists():
        raise FileNotFoundError(f"Missing background image: {bg_path}")
    bg_lab = rgb2lab(np.array(Image.open(bg_path)))

    out_dir = Path("dataset/pipeline_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(Path("dataset").glob("scene_*/rgb.png")):
        scene_name = p.parent.name
        mask, im_rgb = load_and_preprocess(p, bg_lab)
        proposals = extract_proposals(mask)

        # Save masked visual preview
        vis = im_rgb.copy()
        vis[~mask] = (vis[~mask] * 0.2).astype(np.uint8)  # dim background

        for prop in proposals:
            x, y, w, h = prop["bbox"]
            cx, cy = prop["centroid"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)

        cv2.imwrite(str(out_dir / f"{scene_name}_proposals.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"{scene_name}: extracted {len(proposals)} candidate object blobs")


if __name__ == "__main__":
    main()
