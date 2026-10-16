"""Evaluate a run the same way every time: scorer report, answer images, score table.

Runs the organisers' scorer (starter/evaluate.py) with ``--report`` into the run directory,
adds the score to the manifest, draws the answer images (scripts/annotate_run.py) into
``answers/`` and prints the standard table next to a reference run. Reads ground truth, so
it runs after the pipeline and never inside it.

    .venv/bin/python scripts/evaluate_run.py outputs/<run> --reference outputs/<earlier run>
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
EVALUATE_SCRIPT = REPO_DIR / "starter" / "evaluate.py"
ANNOTATE_SCRIPT = REPO_DIR / "scripts" / "annotate_run.py"
SCORE_PATTERN = {
    "total": r"SCRIPT TOTAL\s+([\d.]+)",
    "grounding": r"GROUNDING\s+([\d.]+)",
    "safety": r"SAFETY\s+([\d.]+)",
}
GROUNDING_POINTS = 35
SAFETY_POINTS = 30
COLUMN_WIDTH = 14
LABEL_WIDTH = 40


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference", type=Path, help="earlier run to compare against")
    parser.add_argument("--data", type=Path, default=REPO_DIR / "dataset")
    return parser.parse_args()


def score(run_dir: Path, data_dir: Path) -> dict:
    """Score a run with the organisers' scorer, write report.json and the manifest score.

    Args:
        run_dir: run directory with results.json and manifest.json.
        data_dir: dataset folder with ground truth.

    Returns:
        ``total``, ``grounding`` and ``safety`` points.
    """
    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        str(run_dir / "results.json"),
        "--data",
        str(data_dir),
        "--report",
        str(run_dir / "report.json"),
    ]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    points = {
        name: float(re.search(pattern, output).group(1)) for name, pattern in SCORE_PATTERN.items()
    }
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["score"] = points
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return points


def table_rows(run_dir: Path) -> dict[str, float | int]:
    """Compute the standard table rows from a run's report.json.

    Args:
        run_dir: scored run directory.

    Returns:
        Row label to value.
    """
    report = json.loads((run_dir / "report.json").read_text())
    entries = report["instructions"]
    plan_expected = [entry for entry in entries if entry["expected"] == "plan"]
    no_plan_expected = [entry for entry in entries if entry["expected"] != "plan"]
    return {
        "Total /65": round(report["automatic_score"], 1),
        "Grounding /35": round(report["grounding"] * GROUNDING_POINTS, 1),
        "Safety /30": round(report["safety"] * SAFETY_POINTS, 1),
        "Pixel on target (/40)": sum(entry["score"] > 0 for entry in plan_expected),
        "Correct ask/refuse (/10)": sum(entry["score"] > 0 for entry in no_plan_expected),
        "Valid plans asked/refused (/40)": sum(
            entry["answer"] in ("ask_user", "refuse") for entry in plan_expected
        ),
        "No-plan answered with a plan (/10)": sum(
            entry["answer"] == "plan" for entry in no_plan_expected
        ),
        "Invalid answers (/50)": sum(entry["answer"] is None for entry in entries),
    }


def print_table(columns: list[tuple[str, dict]]) -> None:
    """Print rows side by side for one or more runs.

    Args:
        columns: (column title, rows from ``table_rows``) per run.

    Returns:
        None.
    """
    print("".ljust(LABEL_WIDTH) + "".join(title.rjust(COLUMN_WIDTH) for title, _ in columns))
    for label in columns[0][1]:
        values = "".join(str(rows[label]).rjust(COLUMN_WIDTH) for _, rows in columns)
        print(label.ljust(LABEL_WIDTH) + values)


def main() -> None:
    """Score, draw and tabulate a run, optionally next to a reference run.

    Returns:
        None.
    """
    arguments = parse_arguments()
    score(arguments.run_dir, arguments.data)
    subprocess.run(
        [
            sys.executable,
            str(ANNOTATE_SCRIPT),
            str(arguments.run_dir / "report.json"),
            "--data",
            str(arguments.data),
            "--out-dir",
            str(arguments.run_dir / "answers"),
        ],
        capture_output=True,
        check=True,
    )
    columns = [
        (arguments.run_dir.name[16:] or arguments.run_dir.name, table_rows(arguments.run_dir))
    ]
    if arguments.reference:
        reference_name = arguments.reference.name[16:] or arguments.reference.name
        columns.insert(0, (reference_name, table_rows(arguments.reference)))
    print_table(columns)


if __name__ == "__main__":
    main()
