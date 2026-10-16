# Run records

Every run of `python run.py` writes one directory here with `manifest.json` (commit, config
`sha256`, model id, UTC timestamp) and the measured scene graphs. Scoring a run —
`.venv/bin/python scripts/evaluate_run.py outputs/<run>` — invokes the organisers' scorer,
adds `report.json` (the full breakdown), writes the score back into the manifest and draws
the answers.

| run | what it is | dev score |
|---|---|---|
| `20261015-1151-baseline` | the organiser baseline (Apertus alone) recorded on the dev set against a local vLLM server; recovered from the team's earlier `runs/` record | 15.1/65 |
| `20261016-041412-plate-retry-heldout` | the run behind the submitted `results_heldout.json` | — |
| `20261016-043630-final` | the canonical dev run cited by the report and slides | 55.8/65 |
| `20261016-051200-apertus-boxes-a` | perception ablation: Apertus boxes the whole photo (run a) | 43.2/65 |
| `20261016-051702-apertus-boxes-b` | perception ablation (run b) | 44.3/65 |
| `20261016-053146-dev-repro` | re-run of the frozen release — 49/50 decisions identical | 55.8/65 |
| `20261016-053459-heldout-repro` | held-out re-run — submitted answers reproduced byte-for-byte | — |
| `20261016-054521-dev-verify` | second re-run — 44/50 identical (see note) | 52.3/65 |
| `20261016-054815-heldout-verify` | second held-out re-run — 39/40 byte-identical (one refusal reason differs) | — |

Checksums: `results_heldout.json` at the repository root is sha256 `dbb3fc34…` — exactly
what `20261016-053459-heldout-repro` re-produced byte-for-byte. `results_dev.json` is sha256
`4132513b…`, the exact output of `20261016-053146-dev-repro` (55.8/65). The submitted run's
own answer file (`20261016-041412-plate-retry-heldout`) was never committed upstream and is
not retained; the submitted file's identity rests on the checksum above and on byte-identical
independent reproduction from the released pipeline.

The submitted file's originating run used config sha256 `69e9e966…`; the first re-runs used
`0476c067…` (the tip config adds `perception.backend: opencv`), and the released config
differs further only in the endpoint default (local instead of the hosted endpoint).

Note on 55.8 vs 52.3: the hosted Apertus endpoint is not bit-for-bit deterministic at
temperature 0. Between the canonical run and the second verification run, two scored
instructions flipped (one impossible instruction answered with a plan; one false alarm) and
several plans gained or dropped clear-moves; grounding was identical (37/40). Self-hosting
with fixed seeds is the proposed way to control this; it has not been measured.

The released pipeline does not ship the evaluation-only script that produced the
`scene_graph_comparison.*` files, nor the dataset-derived annotations it consumed; the
comparison files stay here as evidence for the report's audit numbers.
