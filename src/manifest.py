"""Write the manifest that makes a run directory count as evidence."""

import hashlib
import json
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

MANIFEST_NAME = "manifest.json"


def git_state(repo_dir: Path) -> dict:
    """Return the current commit and whether the working tree has uncommitted changes.

    Args:
        repo_dir: any directory inside the git repository.

    Returns:
        Dict with ``commit`` (full hash) and ``dirty`` (bool).
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def write(run_dir: Path, config_path: Path, extra: dict) -> Path:
    """Write ``manifest.json`` into a run directory, atomically.

    Records the commit, a dirty-tree flag, the config hash and contents, the model id, the
    Python version, the host and a UTC timestamp. Values that cannot be read from this process,
    such as the score, are passed in through ``extra``.

    Args:
        run_dir: existing run directory the manifest describes.
        config_path: YAML config the run was produced with.
        extra: additional fields, merged last (e.g. ``score``).

    Returns:
        Path of the written manifest.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest = {
        **git_state(run_dir),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": config,
        "model_id": config["model_id"],
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
    target = run_dir / MANIFEST_NAME
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
    return target
