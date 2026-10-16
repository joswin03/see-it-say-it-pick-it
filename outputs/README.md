# Run records

Every run of `python run.py` writes one directory here with `manifest.json` (commit, config
`sha256`, model id, UTC timestamp) and the measured scene graphs, including a comparison
against ground-truth scene graphs (evaluation only). `scripts/evaluate_run.py` adds
`report.json` — the organisers' scorer output with the full breakdown — and writes the score
back into the manifest.

| run | what it is | dev score |
|---|---|---|
| `20261016-041412-plate-retry-heldout` | the run behind the submitted `results_heldout.json` | — |
| `20261016-043630-final` | the canonical dev run the report and slides cite | 55.8/65 |
| `20261016-051200-apertus-boxes-a` | perception ablation: Apertus boxes the whole photo (run a) | 43.2/65 |
| `20261016-051702-apertus-boxes-b` | perception ablation (run b); compare `figures/plot_ablation.py` | 44.3/65 |
| `20261016-053146-dev-repro` | re-run from the released commit — 49/50 decisions identical to canonical | 55.8/65 |
| `20261016-053459-heldout-repro` | re-run on the held-out scenes — submitted answers reproduced 40/40 | — |
| `20261016-054521-dev-verify` | second re-run from the released commit; 44/50 identical, see note | 52.3/65 |
| `20261016-054815-heldout-verify` | second held-out re-run — 40/40 answer-for-answer (one refusal wording differs) | — |

Note on the two 55.8 vs.\ 52.3 dev runs: the hosted Apertus endpoint is not bit-for-bit
deterministic even at temperature 0. Between the canonical run and the second verification
run, two scored instructions flipped (one impossible instruction answered with a plan, one
false alarm) and several plans gained or dropped clear-moves; grounding stayed identical.
The submitted held-out file reproduced answer-for-answer in both independent re-runs.

The answer files at the repository root: `results_dev.json` (the 55.8 dev run) and
`results_heldout.json` (the file as submitted).
