"""Run the organisers' baseline (baseline/baseline.py, unchanged) against an endpoint with a key.

baseline.py sends a fixed ``Authorization: Bearer none`` header, which a hosted endpoint such
as CSCS rejects. This wrapper only swaps that header for the key in ``.env``; prompt, image
size, decoding and parsing are the organisers' code. Writes a run directory with the results
and a manifest.

    .venv/bin/python scripts/run_organiser_baseline.py --base-url https://api.inference.cscs.ch/v1
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "baseline"))

import baseline
from src import manifest

CONFIG_PATH = REPO_DIR / "configs" / "baseline.yaml"
OUTPUTS_DIR = REPO_DIR / "outputs"
PLACEHOLDER_AUTHORIZATION = "Bearer none"
API_KEY_ENV = "APERTUS_API_KEY"
SERVER_ERROR_ATTEMPTS = 5
SERVER_ERROR_PAUSE_S = 10
FIRST_SERVER_ERROR = 500


def parse_arguments() -> argparse.Namespace:
    """Read the command line.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint")
    parser.add_argument("--data", type=Path, default=REPO_DIR / "dataset")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--name", default="organiser-baseline")
    parser.add_argument("--timeout", type=float, default=180, help="seconds per request")
    return parser.parse_args()


def send_with_key(api_key: str):
    """Wrap ``urllib.request.urlopen``: real key instead of the placeholder, retry on 5xx.

    A server error (HTTP 5xx) is retried with the identical request, so it never changes an
    answer; baseline.py itself would stop the whole run on the first one.

    Args:
        api_key: key for the endpoint.

    Returns:
        Replacement for ``urllib.request.urlopen``.
    """
    original_urlopen = urllib.request.urlopen

    def urlopen(request, *args, **kwargs):
        """Swap the placeholder authorization header, then send the request, retrying 5xx.

        Args:
            request: ``urllib.request.Request`` built by baseline.py.
            *args: passed through to ``urlopen``.
            **kwargs: passed through to ``urlopen``.

        Returns:
            The response of the original ``urlopen``.
        """
        if request.get_header("Authorization") == PLACEHOLDER_AUTHORIZATION:
            request.add_header("Authorization", f"Bearer {api_key}")
        for attempt in range(1, SERVER_ERROR_ATTEMPTS + 1):
            try:
                return original_urlopen(request, *args, **kwargs)
            except urllib.error.HTTPError as error:
                if error.code < FIRST_SERVER_ERROR or attempt == SERVER_ERROR_ATTEMPTS:
                    raise
                print(f"server error {error.code}, retry {attempt} of {SERVER_ERROR_ATTEMPTS - 1}")
                time.sleep(SERVER_ERROR_PAUSE_S)

    return urlopen


def empty_when_missing(ask):
    """Wrap ``baseline.ask`` so a reply without answer text becomes an empty string.

    A thinking model that spends every token on reasoning returns no content; baseline.py
    would crash on it. As an empty string it parses to no answer and is scored as wrong.

    Args:
        ask: the original ``baseline.ask``.

    Returns:
        Replacement for ``baseline.ask``.
    """

    def ask_or_empty(*args, **kwargs) -> str:
        """Call the original ``ask`` and return its text, or an empty string if there is none.

        Args:
            *args: passed through to ``ask``.
            **kwargs: passed through to ``ask``.

        Returns:
            The reply text.
        """
        return ask(*args, **kwargs) or ""

    return ask_or_empty


def main() -> None:
    """Run baseline.py on the dataset with the key injected, then write the manifest.

    Returns:
        None.
    """
    arguments = parse_arguments()
    load_dotenv(REPO_DIR / ".env")
    urllib.request.urlopen = send_with_key(os.environ[API_KEY_ENV])
    baseline.ask = empty_when_missing(baseline.ask)
    run_dir = OUTPUTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{arguments.name}"
    run_dir.mkdir(parents=True)
    started = time.time()
    sys.argv = [
        "baseline.py",
        "--data",
        str(arguments.data),
        "--out",
        str(run_dir / "results.json"),
        "--base-url",
        arguments.base_url,
        "--model",
        arguments.model,
        "--team",
        arguments.name,
        "--timeout",
        str(arguments.timeout),
    ]
    baseline.main()
    manifest.write(
        run_dir,
        CONFIG_PATH,
        {
            "pipeline": "baseline/baseline.py (organisers, unchanged) via run_organiser_baseline",
            "endpoint": arguments.base_url,
            "model_id": arguments.model,
            "data": str(arguments.data),
            "runtime_s": round(time.time() - started),
        },
    )
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
