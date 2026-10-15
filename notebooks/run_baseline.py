"""Run the organisers' baseline (Apertus alone, plain prompt) on the dev set, on a Marimo GPU.

Steps, each behind a button so nothing heavy runs on load:
install -> download weights -> start vLLM -> run baseline -> score -> stop server.
Results are written to the repository root; push them back with git.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess
    import time
    import urllib.request
    from pathlib import Path

    import marimo as mo

    return Path, mo, os, subprocess, time, urllib


@app.cell
def _(Path, os):
    REPO_DIR = Path(__file__).resolve().parent.parent
    MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
    MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/root/models"))
    MODEL_PATH = MODEL_DIR / "Apertus-v1.5-8B"
    VLLM_PORT = 8000
    BASE_URL = f"http://localhost:{VLLM_PORT}/v1"
    GPU_MEMORY_FRACTION = 0.45
    MAX_MODEL_LEN = 8192
    SERVER_STARTUP_TIMEOUT_S = 900
    RESULTS_NAME = "results_baseline"
    return (
        BASE_URL,
        GPU_MEMORY_FRACTION,
        MAX_MODEL_LEN,
        MODEL_DIR,
        MODEL_ID,
        MODEL_PATH,
        REPO_DIR,
        RESULTS_NAME,
        SERVER_STARTUP_TIMEOUT_S,
        VLLM_PORT,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Baseline run: Apertus alone

        Run the steps top to bottom. Set the `MODEL_DIR` environment variable to the disk with
        the most free space (about 20 GB needed) and add `HF_TOKEN` as a secret if the model is
        gated.
        """
    )
    return


@app.function
def shell(command: str, cwd=None) -> str:
    """Run a shell command and return its combined output; raise if it fails.

    Args:
        command: shell command line.
        cwd: working directory, or None for the current one.

    Returns:
        stdout and stderr of the command as one string.
    """
    import subprocess

    completed = subprocess.run(
        command, shell=True, cwd=cwd, capture_output=True, text=True, check=False
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}\n{output}")
    return output


@app.cell
def _(MODEL_DIR, mo):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    environment_report = shell(
        f"df -h {MODEL_DIR} . ; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv;"
        " python --version"
    )
    mo.md(f"## 0. Environment\n```\n{environment_report}\n```")
    return


@app.cell
def _(mo):
    install_button = mo.ui.run_button(label="1. Install vllm + deps")
    install_button
    return (install_button,)


@app.cell
def _(REPO_DIR, install_button, mo):
    mo.stop(not install_button.value)
    install_log = shell(
        "pip install -U vllm huggingface_hub hf_transfer -r requirements.txt", cwd=REPO_DIR
    )
    mo.md(f"```\n{install_log[-3000:]}\n```")
    return


@app.cell
def _(mo):
    download_button = mo.ui.run_button(label="2. Download Apertus weights")
    download_button
    return (download_button,)


@app.cell
def _(MODEL_ID, MODEL_PATH, download_button, mo, os):
    mo.stop(not download_button.value)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download

    downloaded_path = snapshot_download(
        MODEL_ID,
        local_dir=str(MODEL_PATH),
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"],
    )
    mo.md(f"Downloaded to `{downloaded_path}`\n```\n{shell(f'du -sh {MODEL_PATH}')}\n```")
    return


@app.cell
def _(mo):
    server_button = mo.ui.run_button(label="3. Start vLLM server")
    server_button
    return (server_button,)


@app.cell
def _(
    BASE_URL,
    GPU_MEMORY_FRACTION,
    MAX_MODEL_LEN,
    MODEL_ID,
    MODEL_PATH,
    REPO_DIR,
    SERVER_STARTUP_TIMEOUT_S,
    VLLM_PORT,
    mo,
    server_button,
    subprocess,
    time,
    urllib,
):
    mo.stop(not server_button.value)
    server_log_path = REPO_DIR / "vllm_server.log"
    server_process = subprocess.Popen(
        [
            "vllm",
            "serve",
            str(MODEL_PATH),
            "--served-model-name",
            MODEL_ID,
            "--chat-template-content-format",
            "string",
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--gpu-memory-utilization",
            str(GPU_MEMORY_FRACTION),
            "--port",
            str(VLLM_PORT),
        ],
        stdout=server_log_path.open("w"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + SERVER_STARTUP_TIMEOUT_S
    while True:
        if server_process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited; see {server_log_path}:\n{server_log_path.read_text()[-3000:]}"
            )
        try:
            urllib.request.urlopen(f"{BASE_URL}/models", timeout=5)
            break
        except OSError:
            if time.time() > deadline:
                raise TimeoutError(
                    f"vLLM not ready after {SERVER_STARTUP_TIMEOUT_S}s; see {server_log_path}"
                )
            time.sleep(5)
    mo.md(f"Server up at `{BASE_URL}` (pid {server_process.pid}), log: `{server_log_path}`")
    return (server_process,)


@app.cell
def _(mo):
    baseline_button = mo.ui.run_button(label="4. Run baseline + score")
    baseline_button
    return (baseline_button,)


@app.cell
def _(BASE_URL, MODEL_ID, REPO_DIR, RESULTS_NAME, baseline_button, mo):
    mo.stop(not baseline_button.value)
    run_log = shell(
        f"python baseline/baseline.py --data dataset --out {RESULTS_NAME}.json"
        f" --base-url {BASE_URL} --model {MODEL_ID}",
        cwd=REPO_DIR,
    )
    score_log = shell(
        f"python starter/evaluate.py {RESULTS_NAME}.json --data dataset"
        f" --report report_baseline.json",
        cwd=REPO_DIR,
    )
    mo.md(
        f"## Score\n```\n{score_log}\n```\n<details><summary>run log</summary>\n\n```\n{run_log}\n```\n</details>"
    )
    return


@app.cell
def _(mo):
    stop_button = mo.ui.run_button(label="5. Stop vLLM server")
    stop_button
    return (stop_button,)


@app.cell
def _(mo, server_process, stop_button):
    mo.stop(not stop_button.value)
    server_process.terminate()
    server_process.wait(timeout=60)
    mo.md(
        "Server stopped. Push results: `git add -f results_baseline.json results_baseline.raw.jsonl report_baseline.json`"
    )
    return


if __name__ == "__main__":
    app.run()
