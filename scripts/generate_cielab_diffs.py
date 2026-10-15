#!/usr/bin/env python3
"""Compute CIELAB difference (Delta E) against dataset/background.png and save to dataset/masked_cielab/."""
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.color import rgb2lab

out_dir = Path("dataset/masked_cielab")
out_dir.mkdir(parents=True, exist_ok=True)

bg_lab = rgb2lab(np.array(Image.open("dataset/background.png")))

for p in sorted(Path("dataset").glob("scene_*/rgb.png")):
    im_lab = rgb2lab(np.array(Image.open(p)))
    delta_e = np.sqrt(np.sum((im_lab - bg_lab) ** 2, axis=-1))
    
    # Scale Delta E (typically 0-100) to 0-255 uint8 for visual inspection
    diff_vis = np.clip(delta_e * 2.55, 0, 255).astype(np.uint8)
    
    out_path = out_dir / f"{p.parent.name}_cielab_diff.png"
    Image.fromarray(diff_vis).save(out_path)
    print(f"Saved {out_path}")
