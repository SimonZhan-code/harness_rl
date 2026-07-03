#!/usr/bin/env bash
# Validate the env build across: {Gemma-4-12B, Qwen3.5-9B} x {gpu-venv, docker} x each benchmark.
# Serves each model with SGLang, then runs `hrl-validate` (minimal inference per benchmark).
# Run on vast.ai (GPU host, driver >= 580 / CUDA 13 for these models). Prereqs:
#   - GPU venv:  bash scripts/setup_gpu_venv.sh   (creates .venv-gpu)
#   - docker:    bash scripts/build_image.sh      (creates harness-rl-train:latest)
#   - benchmark task registries present (TB2_TASKS_DIR / GAMEDEV_TASKS_DIR / WEBGAME_TASKS_DIR)
#     and HF_TOKEN for the SWE-bench-PRO dataset (gemma-4-12b-it itself is NOT gated).
#
# Validated model ids: google/gemma-4-12b-it, Qwen/Qwen3.5-9B.
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=("${GEMMA_ID:-google/gemma-4-12b-it}" "${QWEN_ID:-Qwen/Qwen3.5-9B}")
PORT="${SGLANG_PORT:-30000}"
IMAGE="${TRAIN_IMAGE:-harness-rl-train:latest}"
BASE_URL="http://localhost:${PORT}/v1"
RESULTS="runs/validate/matrix_$(date +%s 2>/dev/null || echo run).md"
mkdir -p runs/validate

wait_for_endpoint() {  # $1 = url, $2 = timeout_s
  local url="$1" t="${2:-600}" i=0
  until curl -sf "${url%/v1}/health" >/dev/null 2>&1 || curl -sf "$url/models" >/dev/null 2>&1; do
    sleep 5; i=$((i+5)); [ "$i" -ge "$t" ] && return 1
  done
}

serve_sglang_bg() {  # $1 = model ; echoes PID
  python -m sglang.launch_server --model-path "$1" --served-model-name "$1" \
    --host 0.0.0.0 --port "$PORT" --context-length "${SGLANG_CTX:-32768}" \
    >"runs/validate/sglang_$$.log" 2>&1 &
  echo $!
}

run_case() {  # $1 = build label, $2 = model
  local build="$1" model="$2"
  echo "=== [$build] $model ===" | tee -a "$RESULTS"
  if [ "$build" = "gpu-venv" ]; then
    # shellcheck disable=SC1091
    source .venv-gpu/bin/activate
    local pid; pid=$(serve_sglang_bg "$model")
    if wait_for_endpoint "$BASE_URL" 900; then
      python -m harness_rl.eval.validate --model "$model" --base-url "$BASE_URL" | tee -a "$RESULTS"
    else
      echo "SERVE TIMEOUT ($model)" | tee -a "$RESULTS"
    fi
    kill "$pid" 2>/dev/null || true; deactivate || true
  else  # docker
    docker run --rm --gpus all --network host \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v "$PWD/data:/workspace/data" \
      "$IMAGE" -lc "
        python -m sglang.launch_server --model-path '$model' --served-model-name '$model' \
          --host 0.0.0.0 --port $PORT --context-length ${SGLANG_CTX:-32768} >/tmp/sglang.log 2>&1 &
        for i in \$(seq 1 180); do curl -sf $BASE_URL/models >/dev/null && break; sleep 5; done
        python -m harness_rl.eval.validate --model '$model' --base-url $BASE_URL
      " | tee -a "$RESULTS"
  fi
  echo | tee -a "$RESULTS"
}

echo "# Env-build validation matrix ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1))" | tee "$RESULTS"
for m in "${MODELS[@]}"; do
  run_case "gpu-venv" "$m"
  run_case "docker"   "$m"
done
echo "Results written to $RESULTS"
