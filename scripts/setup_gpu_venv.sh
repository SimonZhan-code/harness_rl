#!/usr/bin/env bash
# GPU ".venv build" for LOCAL INFERENCE (SGLang primary). Run on a GPU host (vast.ai).
# Creates a venv, installs the orchestration package + the shared GPU stack.
#
# DRIVER REQUIREMENT: NVIDIA driver >= 560 (latest SGLang → torch 2.7.1+cu126 / CUDA 12.6).
# Validated on A100, driver 570.133.20. (Dropped the driver-550/cu124 target — newest models
# need recent SGLang, which forces cu126.)
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
