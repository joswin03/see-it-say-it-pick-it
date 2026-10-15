"""CPU check for cleaning up the name Apertus gives a region (model call replaced)."""

import numpy as np

from src.reasoning import naming

CONFIG = {
    "crop_padding_fraction": 0.6,
    "crop_min_side_mm": 10,
    "prompt": "{colour} {length_mm} {width_mm} {tray} {examples}",
    "example_names": ["gear"],
    "crop_max_side_px": 64,
    "max_tokens": 10,
    "fixture_words": ["tray"],
}


def name_for(reply_name: str, monkeypatch) -> str:
    reply = {"name": reply_name, "other_names": [], "is_object": True}
    monkeypatch.setattr(naming, "ask_json", lambda *args: (reply, "raw"))
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    region = {"mask": mask, "size_mm": (90, 89), "colour": "red", "tray": "black_tray"}
    rgb = np.zeros((20, 20, 3), np.uint8)
    return naming.name_region(rgb, region, 1.0, 1.0, {}, CONFIG)["name"]


def test_name_without_colour_gets_the_measured_colour(monkeypatch) -> None:
    assert name_for("Gear", monkeypatch) == "red gear"


def test_name_with_a_colour_is_kept(monkeypatch) -> None:
    assert name_for("yellow cup", monkeypatch) == "yellow cup"
