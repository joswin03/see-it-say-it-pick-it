"""Object proposals from Grounding DINO (open-vocabulary detector) with SAM2 masks.

Grounding DINO is prompted class-agnostically ("object.") so it only says where things are;
Apertus names each proposal afterwards with the same outlined-crop step as the OpenCV path.
SAM2 turns every box into a mask, so pick pixels, sizes and stacking use real outlines.
Models: IDEA-Research/grounding-dino-base (Apache 2.0), facebook/sam2.1-hiera-small (Apache 2.0).
"""

from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_models(detector_id: str, segmenter_id: str) -> tuple:
    """Load both models once per process.

    Args:
        detector_id: Hugging Face id of the Grounding DINO checkpoint.
        segmenter_id: Hugging Face id of the SAM2 checkpoint.

    Returns:
        (detector processor, detector, segmenter processor, segmenter).
    """
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)
    segmenter = Sam2Model.from_pretrained(segmenter_id)
    return (
        AutoProcessor.from_pretrained(detector_id),
        detector.to(DEVICE).eval(),
        Sam2Processor.from_pretrained(segmenter_id),
        segmenter.to(DEVICE).eval(),
    )


@torch.no_grad()
def propose_masks(rgb: np.ndarray, config: dict, mm_per_pixel: float) -> list[np.ndarray]:
    """Detect object boxes and segment each one.

    Boxes wider than ``max_object_mm`` (a whole tray) are dropped, then overlapping duplicates
    are removed with non-maximum suppression before SAM2 runs.

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        config: the ``gdino_sam`` section of the pipeline config.
        mm_per_pixel: table millimetres per full-resolution pixel.

    Returns:
        List of (H, W) bool masks, largest score first.
    """
    detector_processor, detector, segmenter_processor, segmenter = load_models(
        config["detector_id"], config["segmenter_id"]
    )
    image = Image.fromarray(rgb)
    inputs = detector_processor(images=image, text=config["prompt"], return_tensors="pt").to(DEVICE)
    detection = detector_processor.post_process_grounded_object_detection(
        detector(**inputs),
        inputs.input_ids,
        threshold=config["box_threshold"],
        text_threshold=config["text_threshold"],
        target_sizes=[rgb.shape[:2]],
    )[0]
    boxes, scores = detection["boxes"].cpu(), detection["scores"].cpu()
    sides_mm = (boxes[:, 2:] - boxes[:, :2]).max(dim=1).values * mm_per_pixel
    keep = sides_mm <= config["max_object_mm"]
    boxes, scores = boxes[keep], scores[keep]
    boxes = boxes[nms(boxes, scores, config["nms_iou"])]
    if len(boxes) == 0:
        return []
    inputs = segmenter_processor(images=image, input_boxes=[boxes.tolist()], return_tensors="pt")
    outputs = segmenter(**inputs.to(DEVICE), multimask_output=False)
    masks = segmenter_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    return [mask[0].numpy().astype(bool) for mask in masks]


def proposal_regions(
    rgb: np.ndarray, config: dict, mm_per_pixel: float, work_scale: float
) -> list[dict]:
    """Turn proposals into regions shaped like the OpenCV ones, ready for Apertus naming.

    Args:
        rgb: (H, W, 3) uint8 full-resolution photo.
        config: full pipeline config (``gdino_sam`` and ``perception`` sections).
        mm_per_pixel: table millimetres per full-resolution pixel.
        work_scale: work-scale factor for the masks.

    Returns:
        Regions with ``id``, ``mask`` ((h, w) bool at work scale), ``colour``, ``part``,
        ``pixel`` and ``tray``; empty masks are skipped.
    """
    small = to_work_scale(rgb, work_scale)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    trays = find_trays(hsv, config["perception"])
    classes, class_names = colour_classes(hsv, config["perception"])
    regions = []
    for full_mask in propose_masks(rgb, config["gdino_sam"], mm_per_pixel):
        mask = cv2.resize(
            full_mask.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if not mask.any():
            continue
        u, v = interior_point(mask)
        majority = np.bincount(classes[mask], minlength=len(class_names))
        regions.append(
            {
                "id": len(regions) + 1,
                "mask": mask,
                "colour": class_names[int(np.argmax(majority))],
                "part": ("gdino", len(regions) + 1),
                "pixel": [round(u / work_scale), round(v / work_scale)],
                "tray": max(trays, key=lambda name: (trays[name] & mask).sum()),
            }
        )
    return regions
