#!/usr/bin/env python3
"""Subtract dataset/background.png from each scene rgb.png and save diffs to dataset/masked/."""
from pathlib import Path
from PIL import Image
import numpy as np

out_dir = Path("dataset/masked")
out_dir.mkdir(parents=True, exist_ok=True)

bg = np.array(Image.open("dataset/background.png"), dtype=np.int16)

for p in sorted(Path("dataset").glob("scene_*/rgb.png")):
    im = np.array(Image.open(p), dtype=np.int16)
    diff = np.abs(im - bg).astype(np.uint8)
    out_path = out_dir / f"{p.parent.name}_diff.png"
    Image.fromarray(diff).save(out_path)
    print(f"Saved {out_path}")
