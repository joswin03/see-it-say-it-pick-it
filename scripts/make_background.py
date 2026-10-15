#!/usr/bin/env python3
"""Generate background.png by splicing empty halves of scene_01 and scene_02."""
from PIL import Image
import numpy as np

im1 = np.array(Image.open("dataset/scene_01/rgb.png"))
im2 = np.array(Image.open("dataset/scene_02/rgb.png"))

w = im1.shape[1] // 2
bg = np.concatenate([im2[:, :w], im1[:, w:]], axis=1)

Image.fromarray(bg).save("dataset/background.png")
print("Saved dataset/background.png")
