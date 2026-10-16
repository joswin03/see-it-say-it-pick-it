#!/usr/bin/env python3
"""Resolve overlapping bounding boxes by iteratively shaving the larger box along the minimal overlap axis."""
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skimage.color import rgb2lab
from scripts.object_detection_pipeline import load_and_preprocess, extract_proposals, suppress_by_ios

def get_intersection(b1, b2):
    """Return intersection box [xi1, yi1, w, h] or None."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_w = xi2 - xi1
    inter_h = yi2 - yi1
    if inter_w > 0 and inter_h > 0:
        return [xi1, yi1, inter_w, inter_h]
    return None

def resolve_non_overlapping_boxes(proposals: list[dict], min_centroid_margin: int = 30) -> list[dict]:
    """Shave larger box first until reaching 30px from centroid, then shave smaller box."""
    if len(proposals) <= 1:
        return proposals

    props = [{**p, "bbox": list(p["bbox"])} for p in proposals]

    max_iterations = 250
    for _ in range(max_iterations):
        overlap_found = False
        n = len(props)

        for i in range(n):
            for j in range(i + 1, n):
                inter = get_intersection(props[i]["bbox"], props[j]["bbox"])
                if inter is None:
                    continue

                overlap_found = True
                b1, b2 = props[i]["bbox"], props[j]["bbox"]
                c1, c2 = props[i]["centroid"], props[j]["centroid"]
                _, _, inter_w, inter_h = inter

                # Determine larger and smaller box
                a1 = b1[2] * b1[3]
                a2 = b2[2] * b2[3]
                large_box, small_box = (b1, b2) if a1 >= a2 else (b2, b1)
                large_c, small_c = (c1, c2) if a1 >= a2 else (c2, c1)

                # Shave along axis of minimal penetration
                if inter_w <= inter_h:
                    if large_box[0] < small_box[0]:
                        # Large is left of small -> truncate large right edge
                        target_x = small_box[0]
                        limit_x = large_c[0] + min_centroid_margin
                        if target_x >= limit_x:
                            large_box[2] = target_x - large_box[0]
                        else:
                            large_box[2] = limit_x - large_box[0]
                            diff = limit_x - small_box[0]
                            small_box[0] = limit_x
                            small_box[2] = max(10, small_box[2] - diff)
                    else:
                        # Large is right of small -> truncate large left edge
                        target_x = small_box[0] + small_box[2]
                        limit_x = large_c[0] - min_centroid_margin
                        if target_x <= limit_x:
                            diff = target_x - large_box[0]
                            large_box[0] = target_x
                            large_box[2] = max(10, large_box[2] - diff)
                        else:
                            diff = limit_x - large_box[0]
                            large_box[0] = limit_x
                            large_box[2] = max(10, large_box[2] - diff)
                            small_box[2] = max(10, limit_x - small_box[0])
                else:
                    if large_box[1] < small_box[1]:
                        # Large is on top -> truncate large bottom edge
                        target_y = small_box[1]
                        limit_y = large_c[1] + min_centroid_margin
                        if target_y >= limit_y:
                            large_box[3] = target_y - large_box[1]
                        else:
                            large_box[3] = limit_y - large_box[1]
                            diff = limit_y - small_box[1]
                            small_box[1] = limit_y
                            small_box[3] = max(10, small_box[3] - diff)
                    else:
                        # Large is on bottom -> truncate large top edge
                        target_y = small_box[1] + small_box[3]
                        limit_y = large_c[1] - min_centroid_margin
                        if target_y <= limit_y:
                            diff = target_y - large_box[1]
                            large_box[1] = target_y
                            large_box[3] = max(10, large_box[3] - diff)
                        else:
                            diff = limit_y - large_box[1]
                            large_box[1] = limit_y
                            large_box[3] = max(10, large_box[3] - diff)
                            small_box[3] = max(10, limit_y - small_box[1])
        if not overlap_found:
            break

    return props


def main():
    bg_path = Path("dataset/background.png")
    bg_lab = rgb2lab(np.array(Image.open(bg_path)))
    out_dir = Path("dataset/pipeline_output_non_overlapping")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    for p in sorted(Path("dataset").glob("scene_*/rgb.png")):
        scene_name = p.parent.name
        mask, im_rgb = load_and_preprocess(p, bg_lab)
        raw_props = extract_proposals(mask, im_rgb)
        props = suppress_by_ios(raw_props, ios_thresh=0.80)

        # Resolve all overlapping boxes
        resolved_props = resolve_non_overlapping_boxes(props)

        # Verify zero overlaps remaining
        rem_overlaps = 0
        for i in range(len(resolved_props)):
            for j in range(i + 1, len(resolved_props)):
                if get_intersection(resolved_props[i]["bbox"], resolved_props[j]["bbox"]):
                    rem_overlaps += 1

        print(f"{scene_name}: {len(resolved_props)} boxes | Remaining overlaps = {rem_overlaps}")

        # Render preview
        vis_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
        vis_bgr[~mask] = (vis_bgr[~mask] * 0.15).astype(np.uint8)

        for idx, prop in enumerate(resolved_props):
            color = palette[idx % len(palette)]
            x, y, w, h = prop["bbox"]
            cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), color, 4)
            cv2.putText(vis_bgr, f"#{prop['id']}", (x + 8, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis_bgr, f"#{prop['id']}", (x + 8, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / f"{scene_name}_non_overlapping.png"), vis_bgr)


if __name__ == "__main__":
    main()
