"""Score a finished run with the official script and add the score to its manifest.

Reads ground truth, so it runs after the pipeline and never inside it.

    .venv/bin/python scripts/score_run.py outputs/<run> --data dataset
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
EVALUATE_SCRIPT = REPO_DIR / "starter" / "evaluate.py"
SCORE_PATTERN = {
    "grounding": r"GROUNDING\s+([\d.]+)",
    "safety": r"SAFETY\s+([\d.]+)",
    "total": r"SCRIPT TOTAL\s+([\d.]+)",
}


def main() -> None:
    """Run ``starter/evaluate.py`` on a run directory and record the result.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", type=Path, default=REPO_DIR / "dataset")
    arguments = parser.parse_args()

    completed = subprocess.run(
        [
            sys.executable,
            str(EVALUATE_SCRIPT),
            str(arguments.run_dir / "results.json"),
            "--data",
            str(arguments.data),
            "--report",
            str(arguments.run_dir / "report.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    print(completed.stdout)
    (arguments.run_dir / "score.txt").write_text(completed.stdout)
    score = {
        name: float(re.search(pattern, completed.stdout).group(1))
        for name, pattern in SCORE_PATTERN.items()
    }
    manifest_path = arguments.run_dir / "manifest.json"
    run_manifest = json.loads(manifest_path.read_text())
    run_manifest["score"] = score
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
