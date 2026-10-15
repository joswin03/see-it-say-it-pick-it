#!/usr/bin/env python3
"""Apply 3-step CIELAB shadow-suppressed object isolation and save to dataset/masked_cielab_filtered/."""
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab

out_dir = Path("dataset/masked_cielab_filtered")
out_dir.mkdir(parents=True, exist_ok=True)

bg_lab = rgb2lab(np.array(Image.open("dataset/background.png")))

for p in sorted(Path("dataset").glob("scene_*/rgb.png")):
    im_lab = rgb2lab(np.array(Image.open(p)))

    delta_e = np.sqrt(np.sum((im_lab - bg_lab) ** 2, axis=-1))
    delta_ab = np.sqrt((im_lab[..., 1] - bg_lab[..., 1]) ** 2 + (im_lab[..., 2] - bg_lab[..., 2]) ** 2)
    delta_l = im_lab[..., 0] - bg_lab[..., 0]
    # 1. Cast shadow: strictly darker than background (delta_l < -4) with low chroma (delta_ab < 6)
    is_shadow = (delta_l < -4) & (delta_ab < 6)

    # 2. Real object: delta_e > 8 captures low-contrast grey parts without triggering noise,
    #    as long as we suppress true shadows.
    is_object = (delta_e > 8) & (~is_shadow)

    # 3. Connected-components area filter: drop noise blobs < 1500 px without eroding object edges
    lbl, num = ndimage.label(is_object)
    sizes = ndimage.sum(is_object, lbl, range(1, num + 1))
    keep = sizes >= 1500
    # Fast index lookup (0 is background)
    keep_padded = np.pad(keep, (1, 0), constant_values=False)
    is_object_clean = keep_padded[lbl]

    # Save as binary mask (255 = object, 0 = background/shadow)
    mask_vis = (is_object_clean * 255).astype(np.uint8)

    out_path = out_dir / f"{p.parent.name}_filtered.png"
    Image.fromarray(mask_vis).save(out_path)
    print(f"Saved {out_path}")
