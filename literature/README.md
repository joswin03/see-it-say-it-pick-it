# Literature review: how to improve the Apertus pipeline

Side quest, 15.10.2026. Goal: find zero-shot ideas, backed by published work, that fix the
failures we measured. No training on challenge photos, no stored object pictures, Apertus does
all language reasoning, open vision models may only help it see (organiser FAQ).

All papers cited are in `papers/` (40 PDFs, file name = arXiv id + short name). Every arXiv
id was checked against the arXiv API before download. Non-arXiv sources (model cards,
proceedings, GitHub, blogs) are linked at the end.

---

## 1. Where we stand (our own measurements)

| Run | Script /65 | Grounding /35 | Safety /30 | Right ask/refuse | Plans refused by mistake |
|---|---|---|---|---|---|
| Organisers' baseline (Apertus, plain prompt) | 15.1 | 13.1 (15/40) | 2.0 | 1/10 | 14/40 |
| Our graph, judge with scores | 19.2 | 6.1 (7/40) | 13.1 | 7/10 | 15/40 |
| Our graph, judge right/wrong | 18.9 | 5.2 (6/40) | 13.6 | 7/10 | 14/40 |

Three root causes, each measured, not guessed:

1. **Apertus cannot output pixel coordinates.** Synthetic test with coloured dots at known
   positions: asked for `[u, v]`, Apertus lands inside the dot 8% of the time (median error
   20 px across, 50 px down on a 1024 px image). It answers in a fixed ~1024 px frame no
   matter the real image size, and squeezes the vertical axis. A labelled coordinate grid
   makes it worse (0-11%). **Asked to choose a numbered cell instead, it is right 100% of the
   time on 3x3 and 4x3 grids and 94% on 6x5.**
2. **The scene step misses and misnames objects**, and the planner trusts it. Most of the 14
   wrong refusals read "X is not present" (yellow gear, red gears, mug, black tray).
3. **The judge cannot catch these errors.** It sees no more than the actor; it approved 31 of
   50 first answers. Twice the model invented "wider than 85 mm" for objects that fit.

---

## 2. What the literature says about each root cause

### 2.1 VLMs are bad at coordinates, good at multiple choice

- **MOKA** (Liu et al., RSS 2024) states it directly: VLMs "are better at multiple-choice
  problems than directly producing continuous-valued locations", so MOKA overlays marks and
  grid tiles and has the VLM pick tiles; the exact point is then sampled inside the tile.
  This is exactly what our dot test found.
- **Set-of-Mark** (Yang et al., 2023): segment the image with an off-the-shelf model, draw a
  number on every region, ask the VLM for the number. Zero-shot SoM beat fully fine-tuned
  referring-expression models on RefCOCOg.
- **PIVOT** (Nasiriany et al., ICML 2024): draw candidate points as numbered arrows, the VLM
  picks the best few, resample around them, repeat. Turns continuous output into
  "iterative visual question answering", training-free, demonstrated on robot manipulation.
- **Scaffold** (Lei et al., 2024) and **DetToolChain** (Wu et al., ECCV 2024) overlay dot
  matrices and rulers so GPT-4V can read coordinates. **This did not transfer to Apertus-8B in
  our test** (grid made it worse); these tricks seem to need a much stronger VLM.
- **Visual Position Prompt** (2025) explains why: MLLM features favour global context over
  fine spatial detail and lack explicit spatial references.

### 2.2 Small objects in a large photo: zoom

- **V\*** (Wu & Xie, CVPR 2024) and **ZoomEye** (Shen et al., 2024): treat the image as a tree
  of crops and search it, zooming into sub-regions; training-free and model-agnostic
  (ZoomEye).
- **Zoom-Refine** (2025): first answer on the full image, then crop the relevant region at
  high resolution and let the same model re-check and refine its answer. Training-free.
- **LazyMCoT** (2026): zooming everything hurts easy cases; route only uncertain cases to the
  zoom path, using first-token uncertainty.

Our photo is 2448 x 2048 but Apertus sees it at 1024 px; gears and small white parts are a
few dozen pixels there.

### 2.3 Self-judging without new evidence does not work

- **LLMs cannot self-correct reasoning yet** (Huang et al., ICLR 2024): without external
  feedback, self-correction does not help and sometimes hurts. This matches our flat result
  for the judge loop (19.2 vs 18.9).
- **CRITIC** (Gou et al., ICLR 2024): self-correction works when the critic uses **tools**
  that return new evidence (search, code interpreter).
- **Self-Refine** (Madaan et al., 2023) gains mostly on generation quality tasks, where the
  model can see its own mistakes; ours are perception mistakes it cannot see.
- **LLM-as-a-judge** (Zheng et al., 2023): judges show position, verbosity and
  self-enhancement bias; an 8B model judging itself is the weak end of that spectrum.

### 2.4 Spatial relations should be computed, not guessed

- **What's "up" with VLMs** (Kamath et al., EMNLP 2023) and **SpatialVLM** (Chen et al., 2024):
  VLMs are near chance on left/right/above/below between objects. Our "relative" category is
  at 14%.
- **VisProg** (Gupta & Kembhavi, CVPR 2023), **ViperGPT** (Surís et al., ICCV 2023), **Code as
  Policies** (Liang et al., 2022): the language model writes a short program; perception
  calls and geometry run as code. Training-free.
- **SayPlan** (Rana et al., 2023) and **ConceptGraphs** (Gu et al., 2023): plan over an explicit
  scene graph (objects + relations), not over pixels.
- **ReKep** (Huang et al., 2024): the VLM writes constraints over keypoints proposed by vision
  models; a solver does the geometry.

### 2.5 JSON constraints cost reasoning

- **Let Me Speak Freely?** (Tam et al., 2024): reasoning drops significantly under format
  restrictions, more with stricter formats. We also saw the hosted endpoint's JSON-schema
  mode break (alphabetical keys, whitespace loops). Standard fix: reason in free text, then
  a second short call (or a parser) extracts the JSON.
- **Apertus 1.5 model card**: a **thinking mode** exists (`enable_thinking=True`; vLLM
  `--reasoning-parser apertus`), with the advice to budget generously for tokens. CSCS also
  serves `Apertus-v1.5-8B-thinking`.

### 2.6 When to ask, when to refuse

- **KnowNo** (Ren et al., CoRL 2023): the planner proposes options as multiple choice; if
  more than one option stays plausible (conformal prediction over token probabilities), the
  robot asks. No fine-tuning.
- **CLAM** (Kuhn et al., 2022) and **Clarify When Necessary** (Zhang & Choi, NAACL Findings
  2025): first decide *whether* to ask, by estimating how many distinct user intents the
  request admits (intent-sim, entropy over interpretations).
- **Self-consistency** (Wang et al., ICLR 2023): sample several reasoning paths, take the
  majority; disagreement is itself an uncertainty signal.
- **MIGU** (2026): VLM gives semantic priors over candidate objects, fused into a belief;
  the belief decides "proceed" vs "ask".
- **SafeAgentBench** (Yin et al., 2024): LLM safety screens reject over 40% of *safe* tasks
  while rejecting 70% of hazardous ones; over-refusal is the typical failure, as for us.
- **RoboGuard** (Ravichandran et al., 2025): turn general safety rules into context-specific
  specifications first (with chain of thought), then check the plan against them, instead of
  asking "is this safe?" in one shot.
- **ASIMOV / robot constitutions** (Sermanet et al., 2025): short explicit rule sets
  ("constitutions") improve semantic safety judgements of foundation models.

### 2.7 Open vision helpers (allowed by the FAQ)

- **Grounding DINO** (Liu et al., ECCV 2024) and **OWLv2** (Minderer et al., NeurIPS 2023):
  open-vocabulary detectors, boxes from a text query.
- **SAM 2** (Ravi et al., 2024): masks from a box or point; gives object outlines, centre of
  the top surface and stacking (mask overlap).
- **Molmo / Molmo2** (Deitke et al., 2024; 2026), **RoboPoint** (Yuan et al., CoRL 2024): open
  VLMs trained to *point*; output pixel points for a referring expression. Qwen3-VL also
  points, in relative 0-1000 coordinates (its GitHub grounding cookbook).

---

## 3. Ideas, ranked by expected gain for this challenge

Points at stake: grounding 35 (we have ~6), safety 30 (we have ~13), jury 20 for pipeline
logic. Every idea is zero-shot and keeps Apertus as the only language reasoner.

| # | Idea | Fixes | Expected gain | Effort | Compliance |
|---|---|---|---|---|---|
| 1 | Pick-a-cell coarse-to-fine localisation | pixels | high | small | clean, Apertus-only |
| 2 | Detector/segmenter proposes marks, Apertus picks a number (Set-of-Mark) | pixels, missed objects | highest | medium | allowed (vision helper) |
| 3 | Verify by zooming, not by self-judging | judge, pixels, "not present" | medium-high | small | clean |
| 4 | Refuse only with a typed, evidence-backed reason | wrong refusals | high on safety | small | clean |
| 5 | Ambiguity by counting referents | ask_user | medium | small | clean |
| 6 | Geometry in code from the scene graph | relative, multistep, stacking | medium-high | medium | clean if Apertus writes the query |
| 7 | Reason free, then format; try thinking mode | all reasoning steps | medium | small | clean |
| 8 | Tiled, higher-resolution scene description | missed small objects | medium | small | clean |
| 9 | Self-consistency votes for decisions | ask/refuse stability | small-medium | small | needs seeds as arguments |

### Idea 1: pick-a-cell coarse-to-fine localisation (MOKA, PIVOT, ZoomEye)

Never ask Apertus for numbers that are coordinates. After the actor names the target object:

1. Draw a 4x3 grid with cell numbers on the 1024 px photo; ask "which cell holds
   `<object description>`?" (our probe: 100%).
2. Crop that cell **from the full-resolution photo** with a margin of half a cell (objects on
   a border are not lost), redraw a 4x3 grid, ask again.
3. After 3 levels the cell is about 50 x 60 px in the full photo; `target_pixel` is its
   centre. Optional fourth level, or a PIVOT-style last step: draw 5-8 numbered candidate
   dots inside the final crop and let Apertus pick the one on the object's top surface.

Cost: 3-4 short calls per plan. Risk: real objects are less distinct than dots; a cell
holding two objects needs the description from the scene step ("the red gear with nothing
on it").

### Idea 2: Set-of-Mark with an open detector (SoM, Grounding DINO / OWLv2 + SAM 2)

Run an open-vocabulary detector with a generic prompt ("object.") plus SAM 2 for masks, draw
a number on every region, then Apertus names each numbered region and picks the number for
the target. The pick pixel is the deepest interior point of the mask (safe for the robot, see
EVALUATION.md). This removes coordinate output entirely and gives outlines for stacking. The
teammate branch `dev/joswin-gdino` and `dev/joswin-opencv` (44.1/65 with OpenCV regions)
show the direction works here. Alternative helper: Molmo2 pointing for an object name that
Apertus wrote (Apertus decides what to look for; Molmo only finds it).

Compliance note for the report: the detector must stay generic (no dev-scene prompts, no
thresholds fitted to one scene).

### Idea 3: verify by zooming, not by self-judging (CRITIC, Zoom-Refine)

Replace "judge the whole answer on the same photo" with checks that bring **new evidence**:

- crop 2x the object size around `target_pixel` from the full-resolution photo and ask a
  closed question: "Is the centre of this crop on a `<target description>`? yes / no";
- if no, the actor gets that crop as feedback instead of free text.

This is the CRITIC pattern (tool feedback) with a crop as the tool. It also gives the jury a
clear story ("the model checks its own pixel at higher zoom").

### Idea 4: refusals with a typed reason and evidence (RoboGuard, ASIMOV, SafeAgentBench)

Most lost safety points are wrong refusals. Make refusal expensive:

- the refusal reason is an enum: `not_present`, `too_wide`, `person_contact`,
  `uncontrolled_motion`, `does_not_fit`, `contradiction`, each with required evidence fields
  (e.g. `too_wide` must name a scene object whose appearance is plate or saucer;
  `not_present` must come after a targeted search);
- **`not_present` triggers a search first** (V\*): crop the photo into quadrants at higher
  resolution and ask "is there a `<named object>` in this crop?" before refusing;
- a short explicit constitution (ASIMOV) replaces the long rule list, so the 85 mm rule is
  applied only to the objects it names.

### Idea 5: ambiguity by counting referents (KnowNo, CLAM, Clarify When Necessary)

Ask Apertus to list **every** scene object that matches the referring expression
("candidates": ids). One candidate: plan. Several and nothing in the sentence separates them:
`ask_user`, and the question names the candidates. Zero: go to the idea 4 search. This turns
a vague judgement ("is it ambiguous?") into a count, which is the structure KnowNo and
intent-sim use.

### Idea 6: geometry in code from the scene graph (VisProg, SayPlan, Code as Policies)

Apertus translates the instruction into a small structured query over scene ids, e.g.
`select(type=red_gear, relation=closest_to, anchor=mandarin_1)` or
`select(type=red_gear, filter=nothing_on_top)`; code evaluates it with pixel/mask geometry
(left/right as seen in the photo, distances, stacking from `lying_on` or mask overlap). The
plan order (clear the top object first) is also checked in code. Code never reads the
instruction text, only Apertus's query, so the FAQ rule "Apertus does all language
reasoning" holds. Targets the 14% relative and 0% multistep categories.

### Idea 7: reason free, then format; try thinking mode (Let Me Speak Freely, Apertus card)

Two calls per step: (a) free-text reasoning with the photo, (b) "convert this answer to the
JSON template" without the photo. Or one call in Apertus thinking mode, JSON only after the
reasoning block. Cheap A/B test on the dev set; also removes the key-order problem we had to
work around.

### Idea 8: tiled, higher-resolution scene description (ZoomEye, V\*)

Describe the scene from 4 overlapping tiles of the full photo (each sent at 1024 px, i.e.
~2x zoom), then one call merges the four lists into one scene with ids. Small parts and the
black tray region become visible. LazyMCoT suggests zooming only when the first pass is
uncertain, to save calls.

### Idea 9: self-consistency votes (Wang et al.)

For the decision node only (plan / ask / refuse), sample 5 answers with fixed seeds at
temperature > 0 and take the majority; a split vote is itself a reason to ask. Must be
declared in the config and report (CLAUDE.md: temperature 0 unless declared, seeds as
arguments).

---

## 4. Ideas considered and dropped

- **Coordinate scaffolds / rulers** (Scaffold, DetToolChain): measured worse on Apertus-8B.
- **Fine-tuning a pointing model**: allowed only on synthetic/public data, not zero-shot,
  no time before 16.10.
- **Gemini Robotics-ER pointing**: proprietary, forbidden at runtime.
- **Bigger Apertus (70B)**: available on CSCS but does not fit next to a vision model on the
  96 GB GPU for the required local run, and the challenge names the 8B model.

---

## 5. Suggested order of work (time left: until 16.10 12:00)

1. Idea 1 (pick-a-cell) + idea 3 (zoom check) replace the current judge: biggest gain on
   grounding, Apertus-only, a few hours.
2. Idea 4 + idea 5: typed refusals, search before "not present", referent counting: biggest
   gain on safety.
3. Idea 7 A/B (thinking mode) on the dev set.
4. If time: idea 2 or idea 6, which overlap with the teammate's branches; worth merging
   rather than rebuilding.

---

## 6. References

Local PDFs are in `papers/`.

| Paper | Venue / year | File | Source |
|---|---|---|---|
| Apertus: Democratizing Open and Compliant LLMs | tech report 2025 | `2509.14233_apertus.pdf` | arxiv.org/abs/2509.14233 |
| Set-of-Mark Prompting (Yang et al.) | 2023 | `2310.11441_set_of_mark.pdf` | arxiv.org/abs/2310.11441 |
| PIVOT (Nasiriany et al.) | ICML 2024 | `2402.07872_pivot.pdf` | arxiv.org/abs/2402.07872 |
| MOKA (Liu et al.) | RSS 2024 | `2403.03174_moka.pdf` | roboticsproceedings.org/rss20/p062.pdf |
| Scaffolding Coordinates (Lei et al.) | 2024 | `2402.12058_scaffolding_coordinates.pdf` | arxiv.org/abs/2402.12058 |
| DetToolChain (Wu et al.) | ECCV 2024 | `2403.12488_dettoolchain.pdf` | arxiv.org/abs/2403.12488 |
| Visual Position Prompt | 2025 | `2503.15426_visual_position_prompt.pdf` | arxiv.org/abs/2503.15426 |
| V\* guided visual search (Wu & Xie) | CVPR 2024 | `2312.14135_v_star.pdf` | arxiv.org/abs/2312.14135 |
| ZoomEye (Shen et al.) | 2024 | `2411.16044_zoomeye.pdf` | arxiv.org/abs/2411.16044 |
| Zoom-Refine | 2025 | `2506.01663_zoom_refine.pdf` | arxiv.org/abs/2506.01663 |
| LazyMCoT, Focus When Necessary | 2026 | `2606.16158_lazymcot.pdf` | arxiv.org/abs/2606.16158 |
| LLMs Cannot Self-Correct Reasoning Yet (Huang et al.) | ICLR 2024 | `2310.01798_cannot_self_correct.pdf` | arxiv.org/abs/2310.01798 |
| CRITIC (Gou et al.) | ICLR 2024 | `2305.11738_critic.pdf` | arxiv.org/abs/2305.11738 |
| Self-Refine (Madaan et al.) | NeurIPS 2023 | `2303.17651_self_refine.pdf` | arxiv.org/abs/2303.17651 |
| Judging LLM-as-a-Judge (Zheng et al.) | NeurIPS 2023 | `2306.05685_llm_as_judge.pdf` | arxiv.org/abs/2306.05685 |
| What's "up" with VLMs (Kamath et al.) | EMNLP 2023 | `2310.19785_whats_up.pdf` | arxiv.org/abs/2310.19785 |
| SpatialVLM (Chen et al.) | CVPR 2024 | `2401.12168_spatialvlm.pdf` | arxiv.org/abs/2401.12168 |
| VisProg (Gupta & Kembhavi) | CVPR 2023 | `2211.11559_visprog.pdf` | arxiv.org/abs/2211.11559 |
| ViperGPT (Surís et al.) | ICCV 2023 | `2303.08128_vipergpt.pdf` | arxiv.org/abs/2303.08128 |
| Code as Policies (Liang et al.) | ICRA 2023 | `2209.07753_code_as_policies.pdf` | arxiv.org/abs/2209.07753 |
| SayPlan (Rana et al.) | CoRL 2023 | `2307.06135_sayplan.pdf` | arxiv.org/abs/2307.06135 |
| ConceptGraphs (Gu et al.) | ICRA 2024 | `2309.16650_conceptgraphs.pdf` | arxiv.org/abs/2309.16650 |
| ReKep (Huang et al.) | CoRL 2024 | `2409.01652_rekep.pdf` | arxiv.org/abs/2409.01652 |
| SayCan (Ahn et al.) | CoRL 2022 | `2204.01691_saycan.pdf` | arxiv.org/abs/2204.01691 |
| Inner Monologue (Huang et al.) | CoRL 2022 | `2207.05608_inner_monologue.pdf` | arxiv.org/abs/2207.05608 |
| Let Me Speak Freely? (Tam et al.) | EMNLP Industry 2024 | `2408.02442_let_me_speak_freely.pdf` | arxiv.org/abs/2408.02442 |
| Self-Consistency (Wang et al.) | ICLR 2023 | `2203.11171_self_consistency.pdf` | arxiv.org/abs/2203.11171 |
| KnowNo, Robots That Ask For Help (Ren et al.) | CoRL 2023 | `2307.01928_knowno.pdf` | arxiv.org/abs/2307.01928 |
| CLAM (Kuhn et al.) | 2022 | `2212.07769_clam.pdf` | arxiv.org/abs/2212.07769 |
| Clarify When Necessary (Zhang & Choi) | NAACL Findings 2025 | `2311.09469_clarify_when_necessary.pdf` | aclanthology.org/2025.findings-naacl.306.pdf |
| MIGU, instruction grounding under uncertainty | 2026 | `2609.24995_migu.pdf` | arxiv.org/abs/2609.24995 |
| SafeAgentBench (Yin et al.) | 2024 | `2412.13178_safeagentbench.pdf` | arxiv.org/abs/2412.13178 |
| RoboGuard, Safety Guardrails for LLM-Enabled Robots | 2025 | `2503.07885_roboguard.pdf` | arxiv.org/abs/2503.07885 |
| ASIMOV, Robot Constitutions (Sermanet et al.) | 2025 | `2503.08663_asimov_constitutions.pdf` | arxiv.org/abs/2503.08663 |
| Grounding DINO (Liu et al.) | ECCV 2024 | `2303.05499_grounding_dino.pdf` | arxiv.org/abs/2303.05499 |
| OWLv2, Scaling Open-Vocabulary Detection | NeurIPS 2023 | `2306.09683_owlv2.pdf` | arxiv.org/abs/2306.09683 |
| SAM 2 (Ravi et al.) | 2024 | `2408.00714_sam2.pdf` | arxiv.org/abs/2408.00714 |
| Molmo and PixMo (Deitke et al.) | 2024 | `2409.17146_molmo.pdf` | arxiv.org/abs/2409.17146 |
| Molmo2 | CVPR 2026 | `2601.10611_molmo2.pdf` | arxiv.org/abs/2601.10611 |
| RoboPoint (Yuan et al.) | CoRL 2024 | `2406.10721_robopoint.pdf` | arxiv.org/abs/2406.10721 |
| Maestro, orchestrating robot modules with VLMs | 2025 | not available as PDF | arxiv.org/abs/2511.00917 |

Other sources (not papers):

- Apertus v1.5 8B model card, thinking mode and vLLM flags:
  https://huggingface.co/swiss-ai/Apertus-v1.5-8B
- Liip, "Apertus 1.5: 6 ways to try out Switzerland's updated AI model":
  https://www.liip.ch/en/blog/apertus-1-5-6-ways-to-try-out-switzerland-s-updated-ai-model
- Qwen3-VL repository, 2D grounding cookbook (points and boxes, relative 0-1000
  coordinates): https://github.com/qwenlm/qwen3-vl
- Grounding Qwen3-VL detections with SAM 2 (tutorial):
  https://debuggercafe.com/grounding-qwen3-vl-detection-with-sam2/
- MOKA proceedings version: https://www.roboticsproceedings.org/rss20/p062.pdf
- Gemini Robotics (proprietary, context only): https://arxiv.org/abs/2503.20020
