"""Run perceive_scene with CIELAB background subtraction instead of stock find_regions."""
from pathlib import Path
import numpy as np
from src.workcell import Scene, mm_per_pixel
from src.perception.regions_cielab import find_regions_cielab
from src.pipeline import name_or_mark_unknown, naming_entry, build_scene_graph, marked_photo, describe, image_url, size_mm

def perceive_scene_cielab(scene: Scene, config: dict, endpoint: dict) -> dict:
    perception = config["perception"]
    scale_mm = mm_per_pixel(scene)
    mm_per_work_pixel = scale_mm / perception["work_scale"]
    scene_path = Path("dataset") / scene.name / "rgb.png"
    found = find_regions_cielab(scene.rgb, perception, mm_per_work_pixel**2, scene_path=scene_path)
    naming_log = []
    for region in found["regions"]:
        region["size_mm"] = size_mm(region["mask"], mm_per_work_pixel)
        named = name_or_mark_unknown(scene, region, config, endpoint, scale_mm)
        region["name"], region["is_object"] = named["name"], named["is_object"]
        naming_log.append(naming_entry(region, named))
    objects = build_scene_graph(
        found["regions"], found["work_scale"], mm_per_work_pixel, config["scene_graph"]
    )
    marked = marked_photo(scene.rgb, objects, found["work_scale"])
    return {
        "objects": objects,
        "text": describe(
            objects,
            scale_mm,
            config["scene_graph"]["nearest_count"],
            config["scene_graph"]["too_wide_mm"],
        ),
        "marked": marked,
        "marked_url": image_url(marked, config["decision"]["image_max_side_px"]),
        "naming_log": naming_log,
    }
