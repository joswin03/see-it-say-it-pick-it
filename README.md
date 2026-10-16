# See It, Say It, Pick It — team `see-it-say-it-pick-it`

**Hack Apertus 2026 · ZHAW challenge.** One workcell photo and one English instruction go in;
one JSON answer comes out: the pixel of the first object to pick plus an ordered pick/place
plan, or exactly one `ask_user` question, or one `refuse` with a reason. Zero-shot, no
training, no robot — the evaluation runs in Python.

Our approach in one sentence: **OpenCV measures the scene, Apertus~1.5-8B does all the
language reasoning, and code validates every move.** Apertus only names each measured region
("red gear", "grey metal part"), screens the instruction for unsafe handling, and plans the
move sequence as schema-constrained JSON; it never has to see precisely, and the code never
parses the instruction text.

## Results (dev set, 10 scenes / 50 instructions)

| Pipeline | Grounding /35 | Safety /30 | Total /65 |
|---|---|---|---|
| Organiser baseline (`baseline/baseline.py`) | 13.1 | 2.0 | 15.1 |
| Apertus marks whole-photo boxes itself (ablation `perception.backend: apertus_boxes`) | 22.8 / 24.5 | 20.4 / 19.8 | 43.2 / 44.3 |
| **Ours: OpenCV regions + Apertus naming** | **32.4** | **23.4** | **55.8** |

- 37 of 40 executable instructions answered with the pixel on the correct object.
- 8 of 10 dangerous instructions handled correctly (all 4 unsafe and both ambiguous ones;
  2 of 4 impossible ones were planned anyway), 1 false alarm in 40 (an `ask_user` where a
  plan existed).
- Three independent re-runs were compared with the submitted held-out answers, per
  instruction: same response type and first-pick pixel 40/40, 40/40 and 37/40; complete
  action lists identical up to question/reason wording 40/40, 40/40 and 34/40; byte-identical
  JSON 40/40, 39/40 and 32/40. The dev set scored 55.8, 52.3 and 55.8 (49/50, 44/50 and
  47/50 decisions identical to the canonical run). The hosted endpoint is not bit-for-bit
  deterministic at temperature 0; local-server equivalence and stability have not been
  measured; plan differences are listed in `outputs/README.md`.
- Every dev number comes from a run in `outputs/` — the baseline from the team's recorded
  baseline run (`outputs/20261015-1151-baseline/`), the rest from the runs listed in
  `outputs/README.md` — each with a manifest (commit, config hash, UTC timestamp) and, for
  scored runs, a `report.json` (the scorer's breakdown).

The short report on where Apertus had difficulty is `docs/report.pdf`; the five slides are
`docs/slides.pdf`.

## How it works

`docs/pipeline.pdf` (source `docs/pipeline.tex`) shows the LangGraph pipeline; the division
of labour is the point:

1. **Once per photo** — OpenCV segments trays, colour regions and parts from geometry and
   colour alone; Apertus names each outlined close-up and says when a region is not an
   object. Code merges the named regions into measured objects (size in mm, position, tray),
   computes stacking from mask overlap and builds the scene-graph text.
2. **Once per instruction** — Apertus screens the instruction for dangerous manner words
   (throw, push off, yank out from under, hands); Apertus then plans `plan`/`ask_user`/
   `refuse` over the measured scene graph. Code validates and repairs the plan (gripper
   width, clear stacks first, retries with feedback) and turns the first pick into
   `target_pixel`.

## Reproduce in under 15 minutes

Python 3.11. The pipeline talks to any OpenAI-compatible chat endpoint; the default config
points at a local server.

```bash
# 1. install (about a minute)
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: uv venv .venv && uv pip install -r requirements.txt

# 2. serve Apertus locally (separate GPU environment: pip install vllm; cached weights)
#    our runs used a hosted Apertus endpoint instead: APERTUS_BASE_URL=...
vllm serve swiss-ai/Apertus-v1.5-8B --chat-template-content-format string

# 3. run the pipeline on any folder of scenes — one command
.venv/bin/python run.py --data dataset --out results_dev.json

# 4. score it (organisers' scorer)
.venv/bin/python starter/evaluate.py results_dev.json --data dataset
```

No GPU or server? `python run.py --data dataset --out results_dev.json --mock` runs the
whole loop with fixed schema-valid replies. (The 15-minute budget assumes the Apertus
weights are already cached; `--mock` checks the pipeline in under a minute.) On servers without a display, install
`opencv-python-headless` instead of `opencv-python`. Our runs took about 3 minutes for the
50 dev instructions and 2 minutes for the held-out scenes on the hosted endpoint.

## Pointing at a different backend

| what | how |
|---|---|
| endpoint | `configs/pipeline.yaml` `base_url` (default `http://127.0.0.1:8000/v1`), overridden by the `APERTUS_BASE_URL` environment variable or `--base-url` |
| model | `configs/pipeline.yaml` `model_id`, overridden by `--model` |
| API key | environment variable named by `api_key_env` (`APERTUS_API_KEY`); leave unset for a local server that needs no auth |
| `.env` | a `.env` file in the repository root is loaded automatically |

Our hackathon runs used the hosted Swiss AI endpoint, i.e.
`APERTUS_BASE_URL=https://api.inference.cscs.ch/v1 APERTUS_API_KEY=… ;`. The pipeline is
provider-agnostic and defaults to a local vLLM server here; equivalence of local and hosted
answers has not been measured.

## Repository layout

| path | what it is |
|---|---|
| `run.py` | the one-command entry point (any folder of scene folders → results JSON) |
| `src/` | pipeline: `graph.py` (LangGraph) and the stages — `regions.py`, `boxes.py`, `naming.py`, `objects.py`, `scene_graph.py`, `safety.py`, `decision.py`, `plans.py`, Apertus client `apertus.py` |
| `configs/pipeline.yaml` | all prompts, thresholds and decode settings in one declared file |
| `schema/result.schema.json` | the answer format (also used for constrained decoding) |
| `starter/` | organisers' scorer and scene loader (`starter/evaluate.py`, `starter/scene_io.py`) |
| `baseline/` | organisers' Apertus baseline (kept for comparison) |
| `examples/` | organisers' example answers for one scene; `EVALUATION.md` is the full scoring rubric |
| `dataset/`, `heldout/` | the 10 dev scenes and the 5 held-out scenes (data CC BY 4.0) |
| `outputs/` | one directory per run: manifest, scorer report, scene graphs — see `outputs/README.md` |
| `figures/` | every figure, produced by the scripts next to them (PDF + PNG) |
| `docs/` | `report.tex`/`report.pdf`, `slides.tex`/`slides.pdf`, pipeline diagram, `DATA.md`, `safety_primer.md` |
| `tests/` | CPU regression checks (`python -m unittest discover -s tests -t .`) |
| `results_dev.json`, `results_heldout.json` | our answers for the dev set and the held-out set (the latter as submitted) |

## Evidence and reproducibility

`run.py` writes `outputs/<timestamp>-<name>/` with `manifest.json` (commit, dirty flag,
config `sha256`, model id, UTC timestamp) and the measured scene graphs.
`.venv/bin/python scripts/evaluate_run.py outputs/<run>` then scores the run with the
organisers' scorer, writes its `report.json` and draws the answers on the photos. The
`scene_graph_comparison.*` files come from an evaluation-only script that is not part of
this release (it consumed dataset-derived annotations, which have been removed). See
`outputs/README.md` for every run and its score; `figures/plot_ablation.py` redraws the
ablation figure.

## Compliance with the challenge rules

- Apertus does all language reasoning; no code parses the instruction text (`src/` reads an
  instruction only to put it into a prompt).
- The runtime path never reads `dataset/ground_truth/` or `dataset/instructions.csv`; the
  scoring scripts do, after a run.
- The default configuration needs no external API call: anything OpenAI-compatible works,
  local vLLM first. Our runs used the hackathon's hosted Swiss AI Apertus endpoint, which
  each manifest records.
- No instruction wording, object ids or scene instances are hardcoded. Fixed workcell facts
  (grey tray left, black tray right) and a generic workcell object vocabulary are stated to
  Apertus; both are documented as borderline points in the report.
- Left/right/top/bottom follow the photo; one object at a time; trays are places only.
  Graspability: the 85 mm opening plus a configured 35 mm mask allowance
  (`objects.grasp_width_margin_mm`) marks objects wider than 120 mm as TOO WIDE; plates
  and saucers are additionally marked unpickable by object class. The allowance is a
  documented engineering deviation, not a validated grasp-width computation — physical
  grasp feasibility between the opening and the allowance is not established.

## Safety

The answers are decision support for a human operator who confirms each action before
execution; the predicted target is meant to be shown on the image first. Nothing here is a
certified safety function: machine-safety rules (ISO 10218, ISO/TS 15066 context) apply to
any post-event execution on the physical cell, which would run under the cell's certified
safety system, reduced speed, with a person present and an e-stop available. See
`docs/safety_primer.md`.

## Licence and citation

Code Apache-2.0, data CC BY 4.0 (ZHAW). This repository builds on the organiser starter at
`github.com/Fatemeh-MA/see-it-say-it-pick-it`; cite the challenge and the ZHAW work listed
in `dataset/README.md`.

Team: Joswin · Vaibhav Prajapati · Macallyster Edmondson.
