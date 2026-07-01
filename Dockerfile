# Step 2 training image (slime + FSDP + SGLang) — built/run on vast.ai, NOT locally.
# Targets dense Gemma 4 (transformers>=5.5) via slime's FSDP backend (SLIME_BACKEND=fsdp)
# and Qwen via Megatron/FSDP. Pin versions on vast against the actual CUDA/driver.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git build-essential docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# GPU stack (pin on vast). slime uses SGLang for rollouts.
RUN pip3 install --no-cache-dir \
    "torch>=2.4" \
    "transformers>=5.5" \
    "sglang>=0.4" \
    "flash-attn>=2.6" --no-build-isolation || true

# slime (THUDM). Pin to a known-good commit on vast. FSDP backend via SLIME_BACKEND=fsdp.
RUN pip3 install --no-cache-dir "git+https://github.com/THUDM/slime.git" || true

# Our orchestration package + its .venv deps (harness/gamma/benchmarks reused in rollouts).
COPY pyproject.toml /workspace/pyproject.toml
COPY harness_rl /workspace/harness_rl
RUN pip3 install --no-cache-dir -e /workspace

# Docker-in-Docker is needed for benchmark task sandboxes during rollouts.
ENTRYPOINT ["/bin/bash"]
