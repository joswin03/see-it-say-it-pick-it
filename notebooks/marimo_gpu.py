import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os
    import subprocess
    import sys
    import time
    import urllib.request
    from pathlib import Path

    import marimo as mo
    import yaml

    WORKSPACE = Path("/marimo")
    REPO_DIR = WORKSPACE / "see-it-say-it-pick-it"
    SECRETS_FILE = WORKSPACE / "storage" / ".env"
    CONFIG_PATH = REPO_DIR / "configs" / "pipeline.yaml"

    CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
    VLLM_ENV = Path(CONFIG["vllm_env"])
    MODEL_ID = CONFIG["model_id"]
    MODEL_PATH = Path(CONFIG["model_path"])
    VLLM_PORT = CONFIG["server"]["port"]
    BASE_URL = f"http://localhost:{VLLM_PORT}/v1"
    GPU_MEMORY_FRACTION = CONFIG["server"]["gpu_memory_fraction"]
    MAX_MODEL_LEN = CONFIG["server"]["max_model_len"]
    SERVER_STARTUP_TIMEOUT_S = CONFIG["server"]["startup_timeout_s"]
    SERVER_ENV_OVERRIDES = {
        name: str(value) for name, value in CONFIG["server"]["env_overrides"].items()
    }

    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    for _line in SECRETS_FILE.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _name, _value = _line.split("=", 1)
            os.environ[_name.strip()] = _value.strip()
    missing_secrets = [name for name in ("HF_TOKEN", "GITHUB_TOKEN") if not os.environ.get(name)]
    if missing_secrets:
        raise KeyError(f"missing in {SECRETS_FILE}: {missing_secrets}")

    subprocess.run(
        [
            "git",
            "config",
            "credential.helper",
            "!f() { echo username=x-access-token; echo password=$GITHUB_TOKEN; }; f",
        ],
        cwd=REPO_DIR,
        check=True,
    )
    mo.md(
        f"Config `{CONFIG_PATH.name}` loaded; secrets loaded from `{SECRETS_FILE}`; git push enabled."
    )
    return (
        BASE_URL,
        CONFIG,
        CONFIG_PATH,
        GPU_MEMORY_FRACTION,
        MAX_MODEL_LEN,
        MODEL_ID,
        MODEL_PATH,
        REPO_DIR,
        SERVER_ENV_OVERRIDES,
        SERVER_STARTUP_TIMEOUT_S,
        VLLM_ENV,
        VLLM_PORT,
        WORKSPACE,
        mo,
        os,
        subprocess,
        sys,
        time,
        urllib,
    )


@app.cell
def _(MODEL_ID, MODEL_PATH, mo, os):
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    apertus_path = snapshot_download(
        MODEL_ID,
        local_dir=str(MODEL_PATH),
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
    )
    mo.md(f"Apertus weights in `{apertus_path}`")
    return (apertus_path,)


@app.cell
def _(
    BASE_URL,
    CONFIG,
    GPU_MEMORY_FRACTION,
    MAX_MODEL_LEN,
    MODEL_ID,
    SERVER_ENV_OVERRIDES,
    SERVER_STARTUP_TIMEOUT_S,
    VLLM_ENV,
    VLLM_PORT,
    WORKSPACE,
    apertus_path,
    mo,
    os,
    subprocess,
    time,
    urllib,
):
    def server_is_up() -> bool:
        """Return True if the vLLM OpenAI endpoint answers."""
        try:
            urllib.request.urlopen(f"{BASE_URL}/models", timeout=5)
            return True
        except OSError:
            return False

    server_log_path = WORKSPACE / "vllm_server.log"
    if server_is_up():
        server_process = None
    else:
        server_process = subprocess.Popen(
            [
                str(VLLM_ENV / "bin" / "vllm"),
                "serve",
                str(apertus_path),
                "--served-model-name",
                MODEL_ID,
                "--chat-template-content-format",
                CONFIG["server"]["chat_template_content_format"],
                "--max-model-len",
                str(MAX_MODEL_LEN),
                "--gpu-memory-utilization",
                str(GPU_MEMORY_FRACTION),
                "--port",
                str(VLLM_PORT),
            ],
            stdout=server_log_path.open("w"),
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PATH": f"{VLLM_ENV / 'bin'}:{os.environ['PATH']}",
                **SERVER_ENV_OVERRIDES,
            },
            start_new_session=True,
        )
        _deadline = time.time() + SERVER_STARTUP_TIMEOUT_S
        while not server_is_up():
            if server_process.poll() is not None:
                raise RuntimeError(f"vLLM exited:\n{server_log_path.read_text()[-3000:]}")
            if time.time() > _deadline:
                raise TimeoutError(
                    f"vLLM not up after {SERVER_STARTUP_TIMEOUT_S}s; see {server_log_path}"
                )
            time.sleep(5)
    mo.md(f"Apertus served at `{BASE_URL}` as `{MODEL_ID}`; log `{server_log_path}`")
    return


@app.cell
def _(mo):
    baseline_button = mo.ui.run_button(label="Run baseline + score (creates a new run directory)")
    baseline_button
    return (baseline_button,)


@app.cell
def _(
    BASE_URL,
    CONFIG_PATH,
    MODEL_ID,
    REPO_DIR,
    VLLM_ENV,
    baseline_button,
    mo,
    subprocess,
    sys,
    time,
):
    mo.stop(not baseline_button.value)
    from src.manifest import write as write_manifest

    baseline_run_dir = REPO_DIR / "runs" / f"{time.strftime('%Y%m%d-%H%M')}-baseline"
    baseline_run_dir.mkdir(parents=True)
    baseline_results = baseline_run_dir / "results.json"
    _run = subprocess.run(
        [
            sys.executable,
            "baseline/baseline.py",
            "--data",
            "dataset",
            "--out",
            str(baseline_results),
            "--base-url",
            BASE_URL,
            "--model",
            MODEL_ID,
        ],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    (baseline_run_dir / "run.log").write_text(_run.stdout + _run.stderr)
    _score = subprocess.run(
        [
            sys.executable,
            "starter/evaluate.py",
            str(baseline_results),
            "--data",
            "dataset",
            "--report",
            str(baseline_run_dir / "report.json"),
        ],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    (baseline_run_dir / "score.txt").write_text(_score.stdout)
    _vllm_version = subprocess.run(
        [str(VLLM_ENV / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    write_manifest(
        baseline_run_dir, CONFIG_PATH, {"vllm_version": _vllm_version, "score": _score.stdout}
    )
    mo.md(f"## Baseline score\n```\n{_score.stdout}\n```\nRun dir: `{baseline_run_dir}`")
    return


if __name__ == "__main__":
    app.run()
