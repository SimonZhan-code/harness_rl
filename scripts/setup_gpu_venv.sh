#!/usr/bin/env bash
# GPU ".venv build" for LOCAL INFERENCE (SGLang primary, vLLM backup) — the non-Docker path.
# Installs the orchestration package + a cu124 GPU stack into a venv. Run on vast.ai (GPU host).
#
# DRIVER COMPATIBILITY: cu124, safe for NVIDIA driver 550.163.01 (supports CUDA <= 12.4).
# Do NOT switch to cu126/cu128 wheels on this driver.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_CUDA="${TORCH_CUDA:-cu124}"
VENV="${1:-.venv-gpu}"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip

# orchestration layer (harness/gamma/benchmarks/logging/eval) + its clients
pip install -e ".[dev]"

# GPU inference stack, pinned to cu124 (driver-550-safe)
pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" "torch==2.5.1"
pip install "sglang[all]>=0.4,<0.5"       # PRIMARY local inference backend
# vLLM (backup) is NOT installed here — it pins torch/flashinfer differently and conflicts
# with SGLang in one venv. Install it in a SEPARATE venv when needed:
#   WITH_VLLM=1 bash scripts/setup_gpu_venv.sh .venv-vllm
if [ "${WITH_VLLM:-0}" = "1" ]; then
  pip install "vllm==0.6.6"               # BACKUP local inference backend (separate venv)
fi

echo "OK: GPU venv '$VENV' ready (cu124 / driver 550.163.01)."
echo "Driver check:"; nvidia-smi --query-gpu=driver_version --format=csv,noheader || true
echo "Torch CUDA check:"; python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
