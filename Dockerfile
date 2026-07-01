# Step 2 training image (slime + FSDP + SGLang) — built/run on vast.ai, NOT locally.
# Targets dense Gemma 4 (transformers>=5.5) via slime's FSDP backend (SLIME_BACKEND=fsdp)
# and Qwen via Megatron/FSDP.
#
# DRIVER COMPATIBILITY: pinned to CUDA 12.4 (cu124) for NVIDIA driver 550.163.01.
#   driver 550.163.01 supports CUDA <= 12.4 (CUDA 12.4 needs >= 550.54.14; 12.5→555, 12.6→560,
#   12.8→570). DO NOT bump the CUDA base or the cu124 wheels below without also upgrading the
#   driver, or the container will fail with "CUDA driver version is insufficient".
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
ARG TORCH_CUDA=cu124
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git build-essential docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch pinned to the cu124 wheel index (driver-550-safe). Local inference: SGLang primary, vLLM backup.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    "torch==2.5.1" && \
    pip3 install --no-cache-dir "transformers>=5.5" "sglang[all]>=0.4,<0.5" && \
    pip3 install --no-cache-dir "flash-attn>=2.6" --no-build-isolation || true
# vLLM backup local-inference backend — pin a cu124-compatible build (used only if SGLang can't serve).
RUN pip3 install --no-cache-dir "vllm==0.6.6" || true

# slime (THUDM). Pin to a known-good commit on vast. FSDP backend via SLIME_BACKEND=fsdp.
RUN pip3 install --no-cache-dir "git+https://github.com/THUDM/slime.git" || true

# Our orchestration package + its .venv deps (harness/gamma/benchmarks reused in rollouts).
COPY pyproject.toml /workspace/pyproject.toml
COPY harness_rl /workspace/harness_rl
RUN pip3 install --no-cache-dir -e /workspace

# Docker-in-Docker is needed for benchmark task sandboxes during rollouts.
ENTRYPOINT ["/bin/bash"]
