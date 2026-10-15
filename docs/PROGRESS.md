# Progress log: OpenCV + Apertus pipeline

Branch `dev/joswin-opencv`. Scores are the official script (`starter/evaluate.py`, 65 points) on
the 50 dev instructions. Run records (results, scene graphs, every raw model reply, manifest)
are written to `runs/scratch/<timestamp>-opencv-dataset/` on the Marimo GPU and are not yet
committed. Thresholds were tuned on the dev set, so the held-out score is the honest number.

## Score history

| Version | Commit | Script /65 | Grounding /35 | Safety /30 | Right asks/refusals | False asks/refusals | Main change |
|---|---|---|---|---|---|---|---|
| Baseline (Apertus alone) | `2f24241` | **15.1** | 13.1 (15/40) | 2.0 | 1/10 | 14/40 | organisers' plain prompt |
| v1 | `30b8e01` | 13.9 | 10.5 (12/40) | 3.4 | 3/10 | 25/40 | first end-to-end pipeline |
| v3 | `50b4b66` | 27.8 | 21.0 (24/40) | 6.7 | 3/10 | 10/40 | plan as moves, places listed |
| v5 | `3b16519` | 34.1 | 23.6 (27/40) | 10.5 | 4/10 | 5/40 | safety check, reply recovery |
| v6 | `460bbe3` | 16.3 | 12.2 (14/40) | 4.0 | 3/10 | 22/40 | regression: `names_an_object` asked on nearly everything (problem 10), blob-first merged objects (problem 9) |
| **v7a** | `287e178`, `--set naming.blob_first=false` | **42.0** | 22.8 (26/40) | 19.2 | 8/10 | 8/40 | structured safety screen |

v7a by category: simple 75%, relative 50%, multistep 70%, ambiguous 100%, impossible 75%,
unsafe 75%. A run of v7 with `naming.blob_first=true` (v7b) has not been done yet.

## What was implemented

```
photo -> OpenCV: trays, foreground, colour regions        (src/perception/regions.py)
      -> Apertus names each region from an outlined crop   (src/reasoning/naming.py)
      -> scene graph in code: ids, pixel, size in mm,
         stacking, orderings, distances                     (src/scene_graph.py)
instruction + scene graph
      -> Apertus safety screen (manner, person, contradiction) (src/reasoning/safety.py)
      -> Apertus plan / ask / refuse, constrained JSON       (src/reasoning/decision.py)
      -> code checks the plan, clears stacks, sets pixel     (src/pipeline.py)
run.py --data <folder> --out <results.json> [--set section.key=value]
```

- **Perception (OpenCV only):** the black tray is the largest dark region; the grey tray is the
  band to its left. Objects are pixels clearly brighter, darker or more colourful than their
  tray. Blobs are split by colour class, touching same-colour objects are split at narrow necks
  (distance transform), and slivers on tray walls are dropped. The pick pixel is the deepest
  interior point of the mask. Dev check (`tools/check_regions.py`, ground truth used for
  evaluation only): every one of the 32 target objects gets a region.
- **Naming (Apertus):** one call per region with a crop, the region outlined in cyan and the
  measured size; Set-of-Mark prompting (Yang et al. 2023). Names must be colour + type.
- **Scene graph (code):** object ids, sizes from calibration (0.248 mm/px), stacking from
  outline overlap, nearest distances, left-to-right orderings overall and per name.
- **Safety screen (Apertus):** structured fields (manner of moving, takes from a person's hand,
  contradicts itself); code maps them to refuse.
- **Decision (Apertus):** constrained JSON whose object fields only accept ids in the scene; a
  plan is a list of moves `{object, destination}`.
- **Checks (code):** a plan that moves an object onto itself, picks something too wide, or is
  empty is sent back once with the reason; objects lying on a picked object are moved aside
  first (`decision.enforce_clear_top`).
- **Reproducibility:** prompts, thresholds and decode settings in
  `configs/opencv_apertus.yaml`; every run writes a manifest (commit, dirty flag, config hash,
  overrides, endpoint, wall clock).

## Problems and how they were solved

1. **Grey tray not found in cluttered scenes.** Its floor is hidden by objects. Solved by
   finding the black tray (robust) and taking the band to its left as the grey tray.
2. **Touching gears merged into one region.** Solved by splitting same-colour parts at necks
   with the distance transform; holes (gear hubs, highlights) are filled first so round objects
   are not over-split.
3. **Small off-white part on the grey tray missed.** The floor is blue-grey, the part faintly
   warm: an off-hue rule with low saturation catches it.
4. **Grey parts on the black tray half missing.** The black tray gets lighter towards the bottom;
   a fitted brightness plane replaces the fixed threshold.
5. **Tray-wall reflections and rail edges became objects.** Dropped by a minimum thickness and a
   20 mm margin from the tray edges.
6. **Gripper flag wrong (v1).** Gears measure 88 mm with their teeth but are graspable; flagging
   everything above 85 mm made Apertus refuse every gear. Now only objects above 120 mm (plates,
   saucers) are flagged.
7. **"pick X, place X" plans (v1).** Apertus put the object it had just picked as the place, as
   in the baseline. Solved by the moves format `{object, destination}` and a check that rejects
   moving an object onto itself.
8. **Replies cut off at the token limit.** Under constrained decoding Apertus gets stuck emitting
   whitespace when it wants a token the schema forbids (`grey_tray_1`), when required fields
   follow a finished plan, or when it writes a double quote inside a text field. Solved by
   listing the place names, putting the free-text message before the moves, asking for single
   quotes, and `close_stalled_reply`, which accepts a cut-off reply only if adding the closing
   brackets gives JSON with every required field.
9. **Blob-first naming regressed (v6).** Naming a whole blob first kept two-coloured objects
   (a mug's rim and inside) together, but also merged a mug with the red gear under it. Behind
   the flag `naming.blob_first`; v7a ran with it off.
10. **Unsafe instructions planned anyway.** The planner knew the rules but refused none of the
    unsafe instructions, and an open "is this safe?" check was not reliable. Solved by a
    separate screen with structured fields. A `names_an_object` field was removed after it came
    back false for nearly every instruction.
11. **Names with the wrong word.** "block" for cube, "cup" for mug, saucer-rim pieces as
    "orange". The planner is told names may be synonyms; naming crops are at least 250 mm wide
    for context; same-named fragments within 12 mm are merged.
12. **Prompt examples close to dev instructions.** A compliance check found the examples echoed
    dev wording; they were replaced with generic paraphrases about other objects.

## Open issues

- Same-colour stacks merge into one region (red cube on two red gears).
- A grey part lying in the grey tray is not found.
- Relative instructions (second from the right, farthest from) still fail about half the time.
- Mug interiors are sometimes named as a separate object.
- Run v7b (`naming.blob_first=true`) to settle the flag, then run the held-out scenes.
