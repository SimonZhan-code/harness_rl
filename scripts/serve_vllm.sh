#!/usr/bin/env bash
# BACKUP local inference backend — vLLM (OpenAI-compatible). Primary is SGLang
# (scripts/serve_sglang.sh); use this only if SGLang can't serve a given model/config.
# Run on vast.ai / a GPU host. The harness talks to this via --base-url http://HOST:8000/v1.
set -euo pipefail

MODEL="${1:-google/gemma-4-12b-it}"     # verify exact HF id
PORT="${2:-8000}"                       # vLLM default OpenAI-compatible port
TP="${VLLM_TP:-1}"                       # tensor-parallel size (set to #GPUs for large models)
CTX="${VLLM_CTX:-32768}"                # max model length (long-horizon → raise as GPU allows)

echo "Serving $MODEL via vLLM (backup) on :$PORT (tp=$TP, ctx=$CTX)"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$CTX"
# Probe against it:
#   hrl-probe --benchmark terminal_bench_2 --model "$MODEL" --base-url http://localhost:$PORT/v1
