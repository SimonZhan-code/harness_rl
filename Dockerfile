# Step 2 training image (slime + FSDP + SGLang) — built/run on vast.ai, NOT locally.
# Reuses the SAME GPU-deps installer as the .venv build (scripts/install_gpu_deps.sh) so the
# two environments stay in sync.
#
# CUDA / DRIVER: the stack is CUDA 13.0 (torch 2.11+cu130). REQUIRES NVIDIA driver >= 580.
#   Validated on an A100-80GB, driver 595.71.05: sglang 0.5.14 + transformers 5.12 serve
#   Gemma-4-12B + Qwen3.5-9B. torch/sglang/flashinfer ship prebuilt cu130 wheels that BUNDLE
#   their CUDA-13 runtime, so the base image's CUDA version matters mainly for building
#   flash-attn from source; a CUDA 12.8+ base also works at runtime.
ARG CUDA_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04   # verify tag at hub.docker.com/r/nvidia/cuda/tags
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git build-essential curl docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Orchestration package + shared GPU inference stack (SGLang -> torch 2.11+cu130, transformers>=5.12).
COPY pyproject.toml /workspace/pyproject.toml
COPY harness_rl /workspace/harness_rl
COPY scripts /workspace/scripts
RUN pip3 install --break-system-packages -e /workspace && PIP="pip3 --break-system-packages" bash scripts/install_gpu_deps.sh
# flash-attn (optional; sglang uses prebuilt flashinfer, so this is not required)
RUN pip3 install --break-system-packages --no-cache-dir "flash-attn>=2.6" --no-build-isolation || true

# slime (THUDM) — Megatron-first, FSDP backup. Pin to a known-good commit on vast.
RUN pip3 install --break-system-packages --no-cache-dir "git+https://github.com/THUDM/slime.git" || true

# Docker-in-Docker is needed for benchmark task sandboxes during rollouts (docker-capable host only).
ENTRYPOINT ["/bin/bash"]
