#!/usr/bin/env python3
"""Run proposal extraction and query Apertus-v1.5-8B for object counts per region."""
import os
import re
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import numpy as np
from openai import OpenAI
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skimage.color import rgb2lab
from scripts.object_detection_pipeline import load_and_preprocess, extract_proposals, suppress_by_ios

API_BASE = "https://api.inference.cscs.ch/v1"
API_KEY = os.environ.get("CSCS_API_KEY", "sk-g233-cdbe531931283179f2028c18d14ac8da")
MODEL_NAME = "swiss-ai/Apertus-v1.5-8B"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)


def encode_crop(crop_img: Image.Image) -> str:
    """Encode PIL crop to JPEG base64."""
    buf = BytesIO()
    crop_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def count_objects_in_crop(crop_img: Image.Image) -> tuple[int, str]:
    """Ask Apertus how many distinct objects are in the crop."""
    b64 = encode_crop(crop_img)
    prompt_text = (
        "You are an industrial robot vision system inspecting an object crop on a table. "
        "How many distinct physical objects do you see? "
        "Rules:\n"
        "- Do NOT count surface patterns, printed graphics, logos, liquid, holes, or sub-parts (like a mug handle or plate decoration) as separate objects.\n"
        "- Only count distinct physical items that are substantially in the frame.\n"
        "Output format: State the integer count first (e.g. \"1\" or \"2\"), followed by a brief description naming each physical object."
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=60,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()

    # Extract first integer number in output
    m = re.search(r"\b(\d+)\b", text)
    count = int(m.group(1)) if m else 1
    return count, text


def main():
    bg_path = Path("dataset/background.png")
    bg_lab = rgb2lab(np.array(Image.open(bg_path)))

    all_results = {}
    multi_object_regions = []

    for p in sorted(Path("dataset").glob("scene_*/rgb.png")):
        scene_name = p.parent.name
        full_im = Image.open(p)

        mask, im_rgb = load_and_preprocess(p, bg_lab)
        raw_props = extract_proposals(mask, im_rgb)
        props = suppress_by_ios(raw_props, ios_thresh=0.80)

        print(f"\n=== {scene_name} ({len(props)} regions) ===")
        scene_counts = []

        for prop in props:
            reg_id = prop["id"]
            x, y, w, h = prop["bbox"]
            crop = full_im.crop((x, y, x + w, y + h))
            crop_arr = np.array(crop).copy()
            crop_mask = prop["mask"][y:y+h, x:x+w]
            # Black out any pixel inside the box not belonging to this object island
            crop_arr[~crop_mask] = 0
            masked_crop = Image.fromarray(crop_arr)

            count, raw_text = count_objects_in_crop(masked_crop)
            print(f"  Region #{reg_id}: {count} object(s) -> \"{raw_text}\"")
            scene_counts.append({"region_id": reg_id, "count": count, "desc": raw_text})

            if count > 1:
                multi_object_regions.append({
                    "scene": scene_name,
                    "region_id": reg_id,
                    "count": count,
                    "bbox": prop["bbox"],
                    "desc": raw_text,
                })

        all_results[scene_name] = scene_counts

    print("\n" + "=" * 50)
    print("REGIONS WITH GREATER THAN 1 OBJECT DETECTED:")
    print("=" * 50)
    if not multi_object_regions:
        print("None. All regions contain <= 1 object.")
    else:
        for r in multi_object_regions:
            print(f"  {r['scene']} - Region #{r['region_id']}: {r['count']} objects (bbox: {r['bbox']})")
            print(f"    Description: {r['desc']}")


if __name__ == "__main__":
    main()
