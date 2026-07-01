#!/usr/bin/env bash
# Step 2 training image build — RUN ON vast.ai (needs GPU + Docker). Deferred locally.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${1:-harness-rl-train:latest}"
echo "Building $IMAGE (vast.ai / GPU host only)..."
docker build -t "$IMAGE" .
echo "OK: $IMAGE built. Launch GRPO with:"
echo "  docker run --gpus all -v /var/run/docker.sock:/var/run/docker.sock $IMAGE \\"
echo "    -c 'python -m harness_rl.rl.train_grpo --model google/gemma-4-12b-it --benchmark terminal_bench_2'"
