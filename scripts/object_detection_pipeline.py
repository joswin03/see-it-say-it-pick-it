#!/usr/bin/env python3
"""Object detection pipeline: isolate objects from CIELAB differences and build proposals."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
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
def extract_proposals(mask: np.ndarray, im_rgb: np.ndarray) -> list[dict]:
    """Split merged clusters via Distance Transform peaks + Sobel color edges, then extract proposals."""
    if not np.any(mask):
        return []

    # 1. Color gradient / Sobel edges across the image (boundary topography)
    gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(grad_x, grad_y)

    # 2. Distance transform to locate mass centers of touching objects
    dist = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=145, threshold_rel=0.2, labels=mask)

    if len(coords) == 0:
        return []

    # 3. Label seed markers at mass centers
    markers = np.zeros(dist.shape, dtype=int)
    for i, pt in enumerate(coords, 1):
        markers[pt[0], pt[1]] = i

    # 4. Watershed flooding over the color-edge landscape
    ws_labels = watershed(gradient, markers=markers, mask=mask)

    proposals = []
    for i in range(1, len(coords) + 1):
        blob = (ws_labels == i)
        area = int(np.sum(blob))
        if area < MIN_OBJECT_AREA:
            continue

        ys, xs = np.where(blob)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]

        proposals.append({
            "id": len(proposals) + 1,
            "area": area,
            "centroid": [cx, cy],
            "bbox": bbox,  # [x, y, w, h]
            "mask": blob,
        })

    return proposals


def suppress_by_ios(proposals: list[dict], ios_thresh: float = 0.80) -> list[dict]:
    """Suppress smaller proposals if >= ios_thresh of their bounding box area is contained in a larger one."""
    if not proposals:
        return []

    def bbox_ios(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        min_area = min(w1 * h1, w2 * h2)
        return (inter / min_area) if min_area > 0 else 0

    suppressed_ids = set()
    # Sort descending by bounding box area (larger containers first)
    sorted_props = sorted(proposals, key=lambda p: p["bbox"][2] * p["bbox"][3], reverse=True)

    for i in range(len(sorted_props)):
        for j in range(i + 1, len(sorted_props)):
            if sorted_props[j]["id"] in suppressed_ids:
                continue
            ios = bbox_ios(sorted_props[i]["bbox"], sorted_props[j]["bbox"])
            if ios >= ios_thresh:
                # j is the smaller box (since list is sorted descending)
                suppressed_ids.add(sorted_props[j]["id"])

    return [p for p in proposals if p["id"] not in suppressed_ids]


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
        raw_props = extract_proposals(mask, im_rgb)
        proposals = suppress_by_ios(raw_props, ios_thresh=0.80)

        # High-contrast color palette (BGR)
        palette = [
            (0, 255, 0),     # Bright Green
            (255, 105, 180), # Hot Pink
            (0, 215, 255),   # Gold / Yellow
            (255, 144, 30),  # Dodger Blue
            (50, 205, 50),   # Lime Green
            (147, 20, 255),  # Deep Pink / Purple
            (238, 130, 238), # Violet
            (0, 165, 255),   # Orange
            (205, 90, 106),  # Slate Blue
            (0, 255, 255),   # Cyan
        ]

        vis_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
        vis_bgr[~mask] = (vis_bgr[~mask] * 0.15).astype(np.uint8)  # dim background

        for idx, prop in enumerate(proposals):
            color = palette[idx % len(palette)]
            b_mask = prop["mask"]
            # Tint pixels of this specific object blob
            vis_bgr[b_mask] = (vis_bgr[b_mask] * 0.45 + np.array(color) * 0.55).astype(np.uint8)

            x, y, w, h = prop["bbox"]
            cx, cy = prop["centroid"]
            # Draw bounding box and label in the exact same matching color
            cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), color, 4)
            cv2.circle(vis_bgr, (cx, cy), 9, color, -1)
            cv2.putText(vis_bgr, f"#{prop['id']}", (x + 8, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis_bgr, f"#{prop['id']}", (x + 8, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / f"{scene_name}_proposals.png"), vis_bgr)
        print(f"{scene_name}: extracted {len(proposals)} candidate object blobs")


if __name__ == "__main__":
    main()
