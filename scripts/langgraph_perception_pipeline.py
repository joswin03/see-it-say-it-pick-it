#!/usr/bin/env python3
"""LangGraph perception pipeline: multi-perspective deterministic majority voting on object crops."""
from __future__ import annotations
import os
import re
import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import TypedDict, List
from collections import Counter

from PIL import Image
import numpy as np
from skimage.color import rgb2lab
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.object_detection_pipeline import load_and_preprocess, extract_proposals, suppress_by_ios
from scripts.resolve_overlaps import resolve_non_overlapping_boxes

API_BASE = "https://api.inference.cscs.ch/v1"
API_KEY = os.environ.get("CSCS_API_KEY", "sk-g233-cdbe531931283179f2028c18d14ac8da")
MODEL_NAME = "swiss-ai/Apertus-v1.5-8B"

llm = ChatOpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_tokens=60,
)


class CropVotingState(TypedDict):
    b64_image: str
    vote_direct: int
    desc_direct: str
    vote_itemized: int
    desc_itemized: str
    vote_cluster: int
    desc_cluster: str
    final_count: int
    consensus_desc: str


def encode_crop(crop_img: Image.Image) -> str:
    buf = BytesIO()
    crop_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- LangGraph Nodes (3 deterministic perspectives at temp=0) ---

def node_direct_count(state: CropVotingState) -> dict:
    prompt = (
        "You are an industrial robot vision system inspecting an object crop on a table. "
        "How many distinct physical objects do you see? "
        "Rules: Ignore printed patterns, logos, liquids, holes, or sub-parts (like handles/rims). "
        "Only count complete physical items. "
        "Output format: State the integer count first (e.g. \"1\" or \"2\"), followed by the object name(s)."
    )
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['b64_image']}"}},
    ])
    res = llm.invoke([msg]).content.strip()
    m = re.search(r"\b(\d+)\b", res)
    count = int(m.group(1)) if m else 1
    return {"vote_direct": count, "desc_direct": res}


def node_itemized_list(state: CropVotingState) -> dict:
    prompt = (
        "List each distinct physical object visible on the table surface in this crop as a markdown bullet point (- item). "
        "Do NOT list surface patterns, graphics, holes, or component parts like handles separately. "
        "Only list real separate physical objects."
    )
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['b64_image']}"}},
    ])
    res = llm.invoke([msg]).content.strip()
    bullets = [line.strip("- *").strip() for line in res.splitlines() if line.strip().startswith(("-", "*", "1.", "2.", "3."))]
    count = len(bullets) if bullets else 1
    return {"vote_itemized": count, "desc_itemized": res}


def node_cluster_check(state: CropVotingState) -> dict:
    prompt = (
        "Classify this cropped region:\n"
        "Reply 'SINGLE' if this crop shows exactly one physical object.\n"
        "Reply 'MULTIPLE' if this crop contains two or more distinct physical objects touching or stacked together."
    )
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['b64_image']}"}},
    ])
    res = llm.invoke([msg]).content.strip()
    count = 2 if "MULTIPLE" in res.upper() else 1
    return {"vote_cluster": count, "desc_cluster": res}


def node_majority_vote(state: CropVotingState) -> dict:
    v_dir = state["vote_direct"]
    v_item = state["vote_itemized"]
    v_clust = state["vote_cluster"]

    # Check if consensus indicates multiple objects (>= 2 nodes agree it is multi-object)
    is_multi_flags = [v_dir > 1, v_item > 1, v_clust > 1]
    if sum(is_multi_flags) >= 2:
        # Multiple objects confirmed: take the specific count from itemized or direct
        multi_counts = [v for v in [v_item, v_dir] if v > 1]
        final_count = max(multi_counts) if multi_counts else 2
        desc = state["desc_itemized"] if v_item > 1 else state["desc_direct"]
    else:
        final_count = 1
        desc = state["desc_direct"]

    return {
        "final_count": final_count,
        "consensus_desc": desc,
    }


# Build LangGraph workflow
workflow = StateGraph(CropVotingState)
workflow.add_node("direct_count", node_direct_count)
workflow.add_node("itemized_list", node_itemized_list)
workflow.add_node("cluster_check", node_cluster_check)
workflow.add_node("majority_vote", node_majority_vote)

workflow.add_edge(START, "direct_count")
workflow.add_edge("direct_count", "itemized_list")
workflow.add_edge("itemized_list", "cluster_check")
workflow.add_edge("cluster_check", "majority_vote")
workflow.add_edge("majority_vote", END)

graph = workflow.compile()


def process_scene_proposals(scene_path: Path, bg_lab: np.ndarray) -> list[dict]:
    full_im = Image.open(scene_path)
    mask, im_rgb = load_and_preprocess(scene_path, bg_lab)
    raw_props = extract_proposals(mask, im_rgb)
    ios_props = suppress_by_ios(raw_props, ios_thresh=0.80)
    props = resolve_non_overlapping_boxes(ios_props, min_centroid_margin=30)

    results = []
    for prop in props:
        x, y, w, h = prop["bbox"]
        crop = full_im.crop((x, y, x + w, y + h))
        crop_arr = np.array(crop).copy()
        crop_mask = prop["mask"][y:y+h, x:x+w]
        crop_arr[~crop_mask] = 0  # Black out background/neighboring artifacts
        b64 = encode_crop(Image.fromarray(crop_arr))

        initial_state = {
            "b64_image": b64,
            "vote_direct": 0,
            "desc_direct": "",
            "vote_itemized": 0,
            "desc_itemized": "",
            "vote_cluster": 0,
            "desc_cluster": "",
            "final_count": 0,
            "consensus_desc": "",
        }
        res = graph.invoke(initial_state)

        results.append({
            "region_id": prop["id"],
            "bbox": prop["bbox"],
            "centroid": prop["centroid"],
            "direct": res["vote_direct"],
            "itemized": res["vote_itemized"],
            "cluster": res["vote_cluster"],
            "final_count": res["final_count"],
            "desc": res["consensus_desc"],
        })
    return results


def main():
    bg_path = Path("dataset/background.png")
    bg_lab = rgb2lab(np.array(Image.open(bg_path)))

    multi_count_regions = []

    for p in sorted(Path("dataset").glob("scene_*/rgb.png")):
        scene_name = p.parent.name
        print(f"\n=== Running LangGraph on {scene_name} ===")
        props_res = process_scene_proposals(p, bg_lab)

        for r in props_res:
            votes = f"[Direct:{r['direct']}, Itemized:{r['itemized']}, Cluster:{r['cluster']}]"
            print(f"  Region #{r['region_id']}: Final={r['final_count']} {votes} -> \"{r['desc']}\"")
            if r["final_count"] > 1:
                multi_count_regions.append({"scene": scene_name, **r})

    print("\n" + "=" * 60)
    print("LANGGRAPH MAJORITY VOTE: REGIONS WITH COUNT > 1")
    print("=" * 60)
    for m in multi_count_regions:
        print(f"  {m['scene']} Region #{m['region_id']}: Count={m['final_count']} | BBox={m['bbox']}")
        print(f"    Votes: Direct={m['direct']}, Itemized={m['itemized']}, Cluster={m['cluster']}")
        print(f"    Desc:  {m['desc']}")


if __name__ == "__main__":
    main()
