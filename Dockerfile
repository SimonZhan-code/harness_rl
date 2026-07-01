# Step 2 training image (slime + FSDP + SGLang) — built/run on vast.ai, NOT locally.
# Reuses the SAME GPU-deps installer as the .venv build (scripts/install_gpu_deps.sh) so the
# two environments stay in sync.
#
# DRIVER REQUIREMENT: NVIDIA driver >= 560 (latest SGLang → torch 2.7.1+cu126 / CUDA 12.6).
# Validated target: A100 on driver 570.133.20. (The driver-550/cu124 target was dropped —
# the newest models, Gemma 4 / Qwen 3.5, need recent SGLang, which forces cu126.)
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git build-essential curl docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Orchestration package + shared GPU inference stack (SGLang → torch 2.7.1+cu126).
COPY pyproject.toml /workspace/pyproject.toml
COPY harness_rl /workspace/harness_rl
COPY scripts /workspace/scripts
RUN pip3 install -e /workspace && PIP=pip3 bash scripts/install_gpu_deps.sh
# flash-attn (optional; build against the installed torch)
RUN pip3 install --no-cache-dir "flash-attn>=2.6" --no-build-isolation || true

# slime (THUDM) — Megatron-first, FSDP backup. Pin to a known-good commit on vast.
RUN pip3 install --no-cache-dir "git+https://github.com/THUDM/slime.git" || true

# Docker-in-Docker is needed for benchmark task sandboxes during rollouts (docker-capable host only).
ENTRYPOINT ["/bin/bash"]
