# harness-rl

Co-evolving an agent's **context/memory-management policy Γ** with an **RL-tuned policy θ**
on **long-horizon coding**, to disentangle *harness-limited* vs. *capability-limited*
failure. Scope: tools/APIs, skills, and prompts are **fixed**; only **Γ is learnable**.

Full design: `../my-vault/01_Projects/Harness+RL/Drafts/{problem_formulation,implementation_plan}.md`.

## The one architectural invariant

The harness talks to the policy **only** through an OpenAI-compatible `base_url`. That is
the seam that keeps **Step 1 (inference probe)** and **Step 2 (RL)** separate — the same
harness/Γ/logging code runs in both; only `base_url` changes.

```
harness_rl/
  types.py           # shared dataclasses (Message, Trace, GammaSnapshot, ...) — GPU-free
  gamma/             # Γ memory archs: G0_truncate G1_retrieval G2_summarize G3_structured_memory G5_external
  harness/agent.py   # the control loop (fixed bash tool + parser Ω₀) with the Γ seam
  serving/client.py  # ModelClient (LiteLLM) + StubModel (no-GPU fake) — the base_url seam
  benchmarks/        # adapters: terminal_bench, swebench_pro/lite/verified, livecodebench,
                     #           gamedev, gamecraft, webgame (+ LocalShellEnv/DockerEnv/EnvStub)
  logging/           # step-segmented Trace JSONL (logs Γ's kept/dropped/summarized/retrieved)
  eval/              # Step 1 Γ variance probe (GammaProbe, spread_table, `hrl-probe` CLI)
  rl/                # Step 2 ONLY: env.py (neutral rollout), slime_adapter.py, reward.py, train_grpo.py
```

## Environments (dual)

- **`.venv` (CPU) — orchestration / local no-GPU checks:** `bash scripts/setup_venv.sh`
- **`.venv-gpu` — GPU inference (SGLang):** `bash scripts/setup_gpu_venv.sh`
- **Docker — Step 2 training:** `bash scripts/build_image.sh`

Both GPU paths share `scripts/install_gpu_deps.sh` (single source of truth). **CUDA / driver:**
- **CUDA 13.0 (`torch 2.11+cu130`) → driver ≥ 580** for the newest models (**Gemma-4-12B**,
  **Qwen3.5-9B**): sglang ≥0.5.11 + **transformers ≥5.12** (Gemma-4's `gemma4_unified` needs it).
- **CUDA 12.6 → driver ≥ 560** fallback for older models (**Qwen3-14B**) via
  `SGLANG_SPEC='sglang[all]<0.5.11' SKIP_TF_UPGRADE=1`.

**Validated** on an A100-80GB (driver 595.71): sglang 0.5.14 + transformers 5.12 serve both
**Gemma-4-12B** and **Qwen3.5-9B**, harness drives real multi-turn agentic inference on all
benchmarks host-native (no Docker). torch/sglang/flashinfer ship prebuilt cu130 wheels
bundling the CUDA-13 runtime, so no separate CUDA toolkit install is needed.

## Benchmarks (difficulty spread)

| Benchmark | Domain | Reward | Docker? | Role |
|---|---|---|---|---|
| terminal_bench_2 · swebench_pro | hard CLI / SWE | tests | yes | sophisticated-model eval |
| **livecodebench** | competitive programming | hidden tests (subprocess) | **no** | **small-model RL** (real reward, host-native) |
| **swebench_lite** · **swebench_verified** | SWE (300 lite / 500 human-verified) | FAIL/PASS tests | graded=yes | inference host-native, graded (Docker) deferred |
| gamedev · **gamecraft** | Godot games | unit tests / rubric judge | no | gaming eval (gamecraft = eval-only) |
| webgame | browser games | LLM judge | no | eval-only |

Memory architectures (Γ) swept by the probe: **G0** truncate · **G1** retrieval · **G2** summarize ·
**G3** structured-scratchpad · **G5** external/hierarchical (summary + archival retrieval + recency).

> **Driver < 580 (e.g. 570):** the newest models (Gemma-4/Qwen3.5) can't serve — use the small-model
> fallback stack + a small model: `SGLANG_SPEC='sglang[all]<0.5.11' SKIP_TF_UPGRADE=1 bash
> scripts/setup_gpu_venv.sh` then serve e.g. `Qwen/Qwen2.5-Coder-7B-Instruct`. Ideal for the
> livecodebench/swebench_lite small-model Γ comparison.

## Local (no-GPU) checks

```bash
ruff check harness_rl && mypy harness_rl
pytest -q                       # gamma logic, seam/episode, trace roundtrip (stubbed model)
python -m harness_rl.eval.cli --stub   # dry-run the probe pipeline end-to-end, offline
```

## Step 1 — Γ variance probe (the GO/NO-GO gate)

**Local inference: SGLang (primary), vLLM (backup)** — Step 1 and Step 2 serve the policy the
same way; closed models use the provider's own inference via the API. On vast, serve, then
sweep Γ:

```bash
bash scripts/serve_sglang.sh google/gemma-4-12b-it 30000    # primary (vLLM backup: scripts/serve_vllm.sh)
hrl-probe --benchmark terminal_bench_2 --model google/gemma-4-12b-it \
          --base-url http://localhost:30000/v1 \
          --gammas G0_truncate G1_retrieval G2_summarize G3_structured_memory --n-tasks 25
```

Reports `J(θ₀, Γ_i)` per variant and the **spread** `max−min`. Non-trivial spread on
long-horizon tasks ⇒ context management binds ⇒ proceed to Step 2. Negligible ⇒ rescope.

## Step 2 — agentic RL (slime, FSDP, full-parameter GRPO)

Trainable benches only (checkable reward): `terminal_bench_2`, `swebench_pro`, `gamedev`.
`webgame` is eval-only (noisy judge). The harness plugs into slime as a custom rollout
function (`rl/slime_adapter.py`); the policy is served by slime (SGLang) and the harness
points its `base_url` there. Launch on vast:

```bash
python -m harness_rl.rl.train_grpo --model google/gemma-4-12b-it --benchmark terminal_bench_2
# prints the `SLIME_BACKEND=fsdp python -m slime.train ...` command + validates wiring
```

> **Backend — Megatron first, FSDP backup.** Default is Megatron (Qwen + Gemma-4 MoE work).
> **Dense Gemma-4** isn't on Megatron-Bridge (MoE-only), so the launcher **auto-selects the
> FSDP backup** (`SLIME_BACKEND=fsdp`) for it. Validate on vast; if the dense-Gemma FSDP path
> is broken, fall back to Gemma-4 MoE (26B-A4B) or make Qwen3.5 the primary. Force with
> `--backend {megatron,fsdp}`.

## Models

- **Train (FSDP, full-param):** Gemma 4 (`google/gemma-4-*-it`, dense→FSDP) + Qwen3.5
  general, size-matched (~4B, ~12–14B); Qwen3-Coder as comparison. *Verify exact HF ids.*
- **Closed (API eval + grounded judge πref):** OpenAI / Anthropic / **Gemini** (transfer
  target + reference executor). Via LiteLLM. *Verify current ids.*
