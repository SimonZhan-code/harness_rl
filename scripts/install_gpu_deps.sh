#!/usr/bin/env bash
# Shared GPU inference-deps installer — used by BOTH scripts/setup_gpu_venv.sh (.venv build)
# and the Dockerfile (single source of truth for the GPU stack).
#
# DRIVER REQUIREMENT: NVIDIA driver >= 580 (CUDA 13).
#   The newest models (Gemma-4 `gemma4_unified`, Qwen3.5) need transformers >= 5.5, which
#   only ships in SGLang >= 0.5.11 — and every SGLang >= 0.5.11 pins torch 2.11.0+cu130
#   (CUDA 13), requiring driver >= 580. Verified: on driver 570 the CUDA-13 torch fails
#   ("driver too old"). Qwen3-14B (older stack) works on 560–579, but Gemma-4 / Qwen3.5 do NOT.
#
#   Older-driver fallback (560–579, cu126): `SGLANG_SPEC='sglang[all]<0.5.11' bash <this>`
#   — gets you Qwen3/Qwen2.5 etc. but NOT Gemma-4 or Qwen3.5.
set -euo pipefail
PIP="${PIP:-pip}"
SGLANG_SPEC="${SGLANG_SPEC:-sglang[all]}"     # default: latest (Gemma-4 / Qwen3.5 capable, driver>=580)

$PIP install --upgrade pip
$PIP install "$SGLANG_SPEC"                   # PRIMARY backend; brings matching torch + transformers

# Gemma-4 (`gemma4_unified`) needs transformers >= 5.12 — NEWER than sglang 0.5.14's pin (5.8.1).
# Pin EXACTLY 5.12.1: it's validated (served Gemma-4-12B + Qwen3.5-9B, driver 595), while newer
# 5.13.x has a model-registration bug ("'qwen3_asr' is already used ..."). Skip with
# SKIP_TF_UPGRADE=1 if you don't need Gemma-4/Qwen3.5.
if [ "${SKIP_TF_UPGRADE:-0}" != "1" ]; then
  $PIP install "transformers==${TF_VERSION:-5.12.1}"
fi

# vLLM (backup) pins torch/flashinfer differently — install in a SEPARATE venv only:
#   WITH_VLLM=1 PIP=pip bash scripts/install_gpu_deps.sh
if [ "${WITH_VLLM:-0}" = "1" ]; then
  $PIP install "vllm"
fi

python -c "import torch,transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'gpu_avail', torch.cuda.is_available())"
