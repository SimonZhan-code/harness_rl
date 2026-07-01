# Vast.ai setup — GPU instance

**Default path = host-native (no Docker):** a *plain* GPU box runs model inference (SGLang)
+ the harness driving each benchmark's tasks on a real host shell (`--env-mode local`). No
privileged/Docker needed. Containers can't run on standard vast instances anyway (unprivileged
→ nested-namespace clone blocked; Docker *and* rootless Podman both fail), so host-native is
the norm. Docker/privileged is ONLY needed for the *graded* benchmark verifiers (deferred).

## Requirements

| Need | Value | Why |
|---|---|---|
| VRAM | **≥ 40 GB** (single GPU) | 9–14B models in bf16 (~18–28 GB) + KV cache |
| Arch | **Ampere or newer** (A100 / A6000 / L40S / H100) | real **bf16** (avoid Turing "Q RTX 8000") |
| Driver | **≥ 580 (CUDA 13)** for Gemma-4 / Qwen3.5; ≥ 560 for older (Qwen3-14B) | newest models need SGLang ≥0.5.11 → torch 2.11+cu130 → driver ≥ 580 |
| Disk | **≥ 150 GB** | venv (~10 GB) + model weights (2× ~25 GB) |
| Geo | **US / EU** | reliable Hugging Face downloads (avoid CN — HF often blocked) |
| Privileged | optional (only for *graded* benchmark verifiers, deferred) | Docker-in-Docker |

**Models (verified ids):** `google/gemma-4-12b-it` (not gated), `Qwen/Qwen3.5-9B`,
`Qwen/Qwen3-14B` (older-stack fallback, driver ≥ 560). *Note: `Qwen3.5-*-Instruct` ids don't
exist — the chat model is the bare `Qwen/Qwen3.5-9B`.*

**Recommended offer (driver 580, no Docker needed):** A100 PCIE 40G, California, ~$0.40/hr
(a match at time of writing: id `42971483`). Search: `vastai search offers "cuda_max_good>=13.0 num_gpus=1 gpu_ram>=40 disk_space>=150 rentable=true reliability>0.98" -o 'dph+'`.

## Recommended pick

**A100 PCIE 40 GB, US/EU, ~$0.40/hr** — matches the validated setup; Ampere/bf16; US = fast HF.
(RTX A6000 48 GB is an equally good alt with more long-context headroom.) *Offer IDs churn —
search fresh; a match at time of writing was `42971487`, A100 PCIE 40G, California.*

CLI search that maps to the requirements:
```bash
vastai search offers "gpu_ram>=40 cuda_max_good>=12.6 num_gpus=1 disk_space>=120 rentable=true reliability>0.98" -o 'dph+'
```

## Launch (must use the web console — the CLI can't set Privileged)

1. Pick an offer meeting the table above.
2. **Additional Configuration → enable "Privileged"** ← the whole point (Docker-in-Docker).
3. Image: a CUDA 12.6+ / PyTorch or Ubuntu base. Disk ≥ 120 GB.
4. Add SSH key `vast-ai-simonzhan` (`~/.ssh/id_ed25519.pub`).
5. Set `HF_TOKEN` (env var) — required for gated Gemma weights.

> Note: vast instances are **non-persistent** unless backed by a volume
> (`vast-capabilities | jq '.instance.workspace_is_volume'`). Sync anything important
> off-box (git / HF Hub).

## Verify Docker FIRST (before anything else)

```bash
apt-get install -y docker.io 2>/dev/null; dockerd >/tmp/d.log 2>&1 & sleep 8
docker run --rm hello-world && echo "DOCKER WORKS"
```
If it prints `DOCKER WORKS`, you're fully unblocked. If it fails on `iptables ... Permission
denied`, that host didn't honor privileged → destroy and pick another offer.

## Then: build + validate

```bash
git clone https://github.com/SimonZhan-code/harness_rl.git && cd harness_rl
bash scripts/setup_gpu_venv.sh                     # .venv-gpu (SGLang, cu126; driver ≥ 560)
bash scripts/serve_sglang.sh <hf-model-id> 30000 & # serve a model
python -m harness_rl.eval.validate --model <hf-model-id> --base-url http://localhost:30000/v1
# full matrix (both models x benchmarks) once Docker + task registries are present:
bash scripts/validate_benchmarks.sh
```

Inference-only (no Docker) already validated on an A100 (driver 570.133.20): SGLang serves +
the harness drives multi-turn inference end-to-end. Docker is needed only for the benchmark
task sandboxes.
