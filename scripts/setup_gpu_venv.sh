#!/usr/bin/env bash
# GPU ".venv build" for LOCAL INFERENCE (SGLang primary). Run on a GPU host (vast.ai).
# Creates a venv, installs the orchestration package + the shared GPU stack.
#
# CUDA / DRIVER: stack is CUDA 13.0 (torch 2.11+cu130) — REQUIRES NVIDIA driver >= 580.
# Validated on A100-80GB, driver 595.71.05: sglang 0.5.14 + transformers 5.12 serve
# Gemma-4-12B + Qwen3.5-9B. Older-driver fallback (560-579, cu126, no Gemma-4/Qwen3.5):
#   SGLANG_SPEC='sglang[all]<0.5.11' SKIP_TF_UPGRADE=1 bash scripts/setup_gpu_venv.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-.venv-gpu}"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# orchestration layer (harness/gamma/benchmarks/logging/eval) + clients
pip install -e ".[dev]"

# shared GPU inference stack (same installer the Dockerfile uses)
PIP=pip bash scripts/install_gpu_deps.sh

echo "OK: GPU venv '$VENV' ready."
echo "Driver:"; nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
echo "Serve a model:  bash scripts/serve_sglang.sh <hf-model-id> 30000"
echo "Validate:       python -m harness_rl.eval.validate --model <id> --base-url http://localhost:30000/v1"
