#!/usr/bin/env bash
# Install the swiss-ai vLLM + transformers forks that Apertus v1.5 needs into its own venv.
# Mirrors github.com/swiss-ai/model-launch images/vllm_apertus_1.5_release/Dockerfile,
# but uses the precompiled vLLM kernels (no CUDA build). Needs an NVIDIA driver with CUDA >= 13.
#
#   bash scripts/install_vllm_apertus.sh /path/to/venv /path/to/src
set -euo pipefail

VENV_DIR=${1:-/marimo/vllm-env}
SRC_DIR=${2:-/marimo/src}
SWISS_AI_VLLM_COMMIT=a601a9d998ddeb488f0c17e8512874b116aa7658
TRANSFORMERS_COMMIT=3797303dda74844e3d1f8977ff5518bb91f818b4
VLLM_REPO_COMMIT=ae6170f874a7c66512bd24aa714b3a02f8a61838
CUDA_VARIANT=cu130

uv venv --allow-existing --python 3.12 "$VENV_DIR"
PY="$VENV_DIR/bin/python"
uv pip install --python "$PY" --extra-index-url "https://download.pytorch.org/whl/$CUDA_VARIANT" \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 "packaging>=24.2" ninja "cmake>=3.26.1" \
  "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" "jinja2>=3.1.6" setuptools_rust regex wheel \
  "protobuf>=6.33.6" huggingface_hub hf_transfer

mkdir -p "$SRC_DIR"
[ -d "$SRC_DIR/vllm" ] || git clone -q --branch apertus-1-5 https://github.com/swiss-ai/vllm.git "$SRC_DIR/vllm"
git -C "$SRC_DIR/vllm" checkout -q "$SWISS_AI_VLLM_COMMIT"
[ -d "$SRC_DIR/transformers" ] || git clone -q --branch feature/apertus_1p5_pipeline \
  https://github.com/swiss-ai/transformers.git "$SRC_DIR/transformers"
git -C "$SRC_DIR/transformers" checkout -q "$TRANSFORMERS_COMMIT"

uv pip install --python "$PY" --no-deps "$SRC_DIR/transformers"

cd "$SRC_DIR/vllm"
VLLM_USE_PRECOMPILED=1 \
VLLM_MAIN_CUDA_VERSION=13.0 \
VLLM_PRECOMPILED_WHEEL_VARIANT=$CUDA_VARIANT \
VLLM_PRECOMPILED_WHEEL_COMMIT=$VLLM_REPO_COMMIT \
VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${VLLM_REPO_COMMIT}/vllm-0.23.1rc1.dev977%2Bgae6170f87-cp38-abi3-manylinux_2_28_x86_64.whl" \
  uv pip install --python "$PY" --reinstall-package vllm --no-build-isolation ".[audio]" \
  --torch-backend=auto

"$PY" -c "import vllm, transformers; print('vllm', vllm.__version__, 'transformers', transformers.__version__)"
