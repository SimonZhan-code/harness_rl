#!/usr/bin/env bash
# Shared GPU inference-deps installer — used by BOTH scripts/setup_gpu_venv.sh (.venv build)
# and the Dockerfile (single source of truth for the GPU stack).
#
# DRIVER REQUIREMENT: latest SGLang pulls torch 2.7.1+cu126 (CUDA 12.6), which REQUIRES
# NVIDIA driver >= 560. (We dropped the driver-550.163.01 / cu124 target because the newest
# models — Gemma 4, Qwen 3.5 — need a recent SGLang, and recent SGLang forces cu126.)
# Validated on an A100 with driver 570.133.20.
set -euo pipefail
PIP="${PIP:-pip}"

$PIP install --upgrade pip
# PRIMARY local inference backend. sglang[all] brings a matching torch 2.7.x+cu126 + sgl-kernel + flashinfer.
$PIP install "sglang[all]>=0.4.10"

# BACKUP backend (vLLM) — pins torch/flashinfer differently; install in a SEPARATE venv only:
#   WITH_VLLM=1 PIP=pip bash scripts/install_gpu_deps.sh
if [ "${WITH_VLLM:-0}" = "1" ]; then
  $PIP install "vllm>=0.8"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu_avail', torch.cuda.is_available())"
