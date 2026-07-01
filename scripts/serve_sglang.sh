#!/usr/bin/env bash
# Serve an open-weight model with SGLang (OpenAI-compatible) for Step 1 inference.
# SGLang is the PRIMARY local inference backend (vLLM backup: scripts/serve_vllm.sh).
# Step 2 slime rollouts serve the same way. Run on vast.ai / a GPU host.
# The harness talks to this via --base-url http://HOST:30000/v1.
set -euo pipefail

MODEL="${1:-google/gemma-4-12b-it}"     # verify exact HF id; dense Gemma → --backend fsdp for training
PORT="${2:-30000}"
TP="${SGLANG_TP:-1}"                     # tensor-parallel size (set to #GPUs for large models)
CTX="${SGLANG_CTX:-32768}"              # context length W (long-horizon → raise as GPU allows)

echo "Serving $MODEL via SGLang on :$PORT (tp=$TP, ctx=$CTX)"
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --tp "$TP" \
  --context-length "$CTX"
# Probe against it:
#   hrl-probe --benchmark terminal_bench_2 --model "$MODEL" --base-url http://localhost:$PORT/v1
