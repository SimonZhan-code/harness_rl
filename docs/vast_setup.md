# Vast.ai setup — Docker-capable GPU instance

How to launch an instance that can run **both** model inference (SGLang) **and** the
benchmark Docker sandboxes (Terminal-Bench / SWE-bench-PRO / GameDevBench).

## Requirements

| Need | Value | Why |
|---|---|---|
| VRAM | **≥ 40 GB** (single GPU) | 12–14B models in bf16 (~24–28 GB) + KV cache |
| Arch | **Ampere or newer** (A100 / A6000 / L40 / H100) | real **bf16** (avoid Turing "Q RTX 8000") |
| CUDA / driver | **CUDA ≥ 12.6 → driver ≥ 560** | latest SGLang → `torch 2.7.1+cu126` |
| Disk | **≥ 120 GB** | venv (~8 GB) + model weights + benchmark images |
| Geo | **US / EU** | reliable Hugging Face downloads (avoid CN — HF often blocked) |
| **Privileged** | **ENABLED** | **required** for Docker-in-Docker (benchmark sandboxes) |

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
