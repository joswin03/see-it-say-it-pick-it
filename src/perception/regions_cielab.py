"""Bridge: Use CIELAB background subtraction foreground mask in place of the complex HSV tray floor detector."""
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from skimage.color import rgb2lab
from src.perception.regions import (
    to_work_scale, find_trays, colour_classes, texture_map,
    colour_groups, fill_holes, split_by_texture, split_touching,
    interior_point, distance_to_box_edge, NEUTRAL_WHITE, NEUTRAL_GREY
)
from scripts.object_detection_pipeline import load_and_preprocess

# Load background lab representation once
BG_PATH = Path("dataset/background.png")
BG_LAB = rgb2lab(np.array(Image.open(BG_PATH))) if BG_PATH.exists() else None

def find_regions_cielab(rgb: np.ndarray, config: dict, mm2_per_work_pixel: float, scene_path: Path = None) -> dict:
    """Segment a photo using CIELAB background subtraction for foreground detection,
    then use OpenCV colour and texture splitting from dev/joswin-opencv to build regions."""
    work_scale = config["work_scale"]
    small = to_work_scale(rgb, work_scale)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    trays = find_trays(hsv, config)

    # 1. CIELAB background subtraction mask (replaces the old HSV foreground function)
    if scene_path and scene_path.exists():
        full_mask, _ = load_and_preprocess(scene_path, BG_LAB)
    else:
        # Fallback if raw scene_path not passed: compute on downscaled image
        bg_small = to_work_scale(np.array(Image.open(BG_PATH)), work_scale)
        im_lab = rgb2lab(small)
        bg_lab_small = rgb2lab(bg_small)
        delta_e = np.sqrt(np.sum((im_lab - bg_lab_small)**2, axis=-1))
        delta_ab = np.sqrt((im_lab[..., 1] - bg_lab_small[..., 1])**2 + (im_lab[..., 2] - bg_lab_small[..., 2])**2)
        delta_l = im_lab[..., 0] - bg_lab_small[..., 0]
        is_shadow = (delta_l < -4) & (delta_ab < 6)
        full_mask = (delta_e > 8) & (~is_shadow)

    # Resize mask to work scale
    h_w, w_w = small.shape[:2]
    mask = cv2.resize(full_mask.astype(np.uint8), (w_w, h_w), interpolation=cv2.INTER_NEAREST).astype(bool)

    classes, class_names = colour_classes(hsv, config)
    texture = texture_map(rgb, (small.shape[1], small.shape[0]), config["morph_kernel_px"])
    min_texture_pixels = config["texture_min_piece_mm2"] / mm2_per_work_pixel
    neutral_group = len(colour_groups(class_names)) - 1
    min_pixels = config["min_region_area_mm2"] / mm2_per_work_pixel
    work_pixels_per_mm = 1 / np.sqrt(mm2_per_work_pixel)
    min_depth_pixels = config["min_thickness_mm"] / 2 * work_pixels_per_mm
    edge_margin_pixels = config["tray_edge_margin_mm"] * work_pixels_per_mm
    tray_boxes = {name: cv2.boundingRect(tray.astype(np.uint8)) for name, tray in trays.items()}
    small_kernel = np.ones((3, 3), np.uint8)
    regions = []

    blob_count, blob_labels = cv2.connectedComponents(mask.astype(np.uint8))
    for blob in range(1, blob_count):
        blob_mask = blob_labels == blob
        if blob_mask.sum() < min_pixels:
            continue
        for group_index, group in enumerate(colour_groups(class_names)):
            part = blob_mask & np.isin(classes, group)
            part = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_OPEN, small_kernel)
            part_count, part_labels = cv2.connectedComponents(part)
            for part_index in range(1, part_count):
                part_mask = part_labels == part_index
                if part_mask.sum() < min_pixels:
                    continue
                pieces = [fill_holes(part_mask)]
                if group_index != neutral_group:
                    pieces = split_by_texture(
                        pieces[0],
                        texture,
                        config["smooth_max_texture"],
                        min_texture_pixels,
                        config["morph_kernel_px"],
                    )
                region_masks = [
                    region_mask
                    for piece in pieces
                    for region_mask in split_touching(piece, config["core_fraction"])
                ]
                for region_mask in region_masks:
                    depth = cv2.distanceTransform(region_mask.astype(np.uint8), cv2.DIST_L2, 5)
                    if region_mask.sum() < min_pixels or depth.max() < min_depth_pixels:
                        continue
                    u, v = interior_point(region_mask)
                    tray = max(trays, key=lambda name: (trays[name] & region_mask).sum())
                    if distance_to_box_edge(u, v, tray_boxes[tray]) < edge_margin_pixels:
                        continue
                    majority = np.bincount(classes[region_mask], minlength=len(class_names))
                    regions.append(
                        {
                            "id": len(regions) + 1,
                            "mask": region_mask,
                            "colour": class_names[int(np.argmax(majority))],
                            "colour_group": group_index,
                            "part": (blob, group_index, part_index),
                            "area_mm2": round(float(region_mask.sum() * mm2_per_work_pixel)),
                            "pixel": [round(u / work_scale), round(v / work_scale)],
                            "tray": tray,
                        }
                    )
    return {"trays": trays, "work_scale": work_scale, "regions": regions}
