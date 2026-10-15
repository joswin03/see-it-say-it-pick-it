# CLAUDE.md

Local only: listed in `.git/info/exclude`, never committed or pushed.

## Context

Hack Apertus 2026, ZHAW challenge "See It, Say It, Pick It". Input: one workcell photo
(2448 x 2048) and one English instruction. Output per instruction: `target_pixel` of the first
object picked plus ordered `pick`/`place` actions, or exactly one `ask_user` / `refuse`.
Scoring is in `EVALUATION.md` (grounding 35, safety 30, jury 35). Apertus
(`swiss-ai/Apertus-v1.5-8B`) must do the language reasoning; open vision models may help it see.
Everything runs locally, no proprietary APIs.

**Two-day hackathon.** Held-out scenes are in `heldout/` (merged from upstream); `results_heldout.json`,
report (1-2 pages) and five slides are due **16.10 at 12:00**. Be precise and quick: apply known
best practice (scene graph from an open detector/VLM, constrained JSON decoding, explicit
ambiguity/feasibility checks), no extensive literature research, no speculative abstractions.

## Organiser FAQ (binding)

- **Zero-shot.** Objects are found from the instruction and the photo alone (e.g. open-vocabulary
  detector / VLM by name). Never store object pictures, and never use crops, masks or labels
  from the dataset photos inside the pipeline. `dataset/ground_truth/` is for scoring only.
- **Fully local at runtime**: own machine or hackathon GPUs, open models only (Apertus + any open
  vision model). Downloading open weights beforehand is fine; no external API calls at runtime.
- Held-out scenes: same workcell, background, camera and object set; new arrangements and new
  instructions of the same six kinds. Top teams may later be tested with new instructions and
  possibly new objects, so nothing may be hardcoded to dev-set instances or ids.
- Fine-tuning an open vision model on synthetic/public data is allowed (never on the challenge
  photos or copies of their objects); if done, it must be reproducible and in the report.
- Writing our own harder test instructions is encouraged.
- **Compliance check before every result that counts.** Every prompt (`configs/*.yaml`) and every
  method in `run.py`/`src/` must satisfy the README "Rules" section and this FAQ: Apertus does all
  language reasoning (code never parses instruction text); other models only help it see;
  nothing at runtime reads `dataset/ground_truth/`, `instructions.csv` or calls an external API;
  prompts contain no dev-set instruction wording, object ids or scene-specific facts (examples
  must be generic paraphrases); perception thresholds are generic colour/geometry rules, never
  fitted to a single dev scene. Re-check whenever a prompt or method changes, and record any
  borderline point in the report.

## Workflow

- Never commit on `main`. Work on `dev/joswin`; merge to `main`
  only when the user asks.
- Code and data move between this machine and the Marimo GPU notebook **only through GitHub**:
  commit + push here, `git pull` in Marimo. Nothing is copied by hand. Results produced on
  Marimo are committed from there (or pushed back) with a manifest.
- Local env: uv, `.venv/` (Python 3.11). Install with `uv pip install -p .venv/bin/python ...`;
  run with `.venv/bin/python`. Local machine is CPU only (mock runs, scoring, LaTeX).
- GPU: Marimo, RTX Pro 6000 Blackwell, 96 GB VRAM. Apertus-8B and a 7-8B open VLM fit
  side by side under vLLM.
- Upstream challenge repo is remote `upstream`; our fork is `origin`.
- One command must run the pipeline on any folder of scenes:
  `python run.py --data <folder> --out <results.json>`.
- Score after every pipeline change: `python starter/evaluate.py <results.json> --data dataset`.

## Research rules (trimmed for the hackathon)

1. **Configs are the declared protocol.** Model names, prompts, thresholds and decode settings
   live in `configs/*.yaml`. CLI flags may override any field; overrides are written into the
   run directory.
2. **No silent cleverness.** Anything that changes a result sits behind a config flag and is
   named in the report.
3. **Every run carries a small manifest**: commit, dirty-tree flag, config hash, model ids,
   timestamp, score. A result without one is not evidence.
4. **Nothing is overwritten in place.** Each run writes to its own `runs/<timestamp>-<name>/`.
5. **Determinism is explicit.** Temperature 0; any seed is an argument.

## Code quality

Read by a jury checking it in under 15 minutes.

- Readable names (`target_pixel`, `scene_graph`), no unexplained abbreviations.
- Modular, not layered: one module one job; no abstraction with one caller.
- Short Google-style docstrings with types/shapes on public functions; type hints on signatures;
  no comments narrating code.
- No magic numbers: config field or named constant.
- Fail loudly: no bare `except`, no silent fallbacks that change a score. A model reply that
  fails to parse is logged and turned into an explicit, schema-valid answer, never dropped.
- `ruff format` + `ruff check` (line length 100). Dead code deleted.
- Quick CPU tests with a mocked model (`--mock`) run before spending GPU time.

## Docs are LaTeX to PDF

**Any document (report, slides, notes meant to be read) is written in LaTeX under `docs/` and
compiled to PDF.** No Markdown/Word deliverables. Compile with tectonic (binary gitignored):

```
./tectonic -X compile docs/<name>.tex --outdir docs
```

Slides use `beamer`. Both `.tex` and `.pdf` are tracked.

- Numbers come from run outputs (`runs/.../score.json`), never typed by hand.
- Every figure is produced by a script in `figures/`, exported as PDF (and PNG 300 dpi),
  title states the finding, each mark labelled with its value.
- Illustrative examples are labelled as such; never a made-up result.
- Superseded docs are deleted, not archived.
