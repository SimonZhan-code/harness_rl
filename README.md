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
  gamma/             # Γ context managers: G0_truncate G1_retrieval G2_summarize G3_structured_memory
  harness/agent.py   # the control loop (fixed bash tool + parser Ω₀) with the Γ seam
  serving/client.py  # ModelClient (LiteLLM) + StubModel (no-GPU fake) — the base_url seam
  benchmarks/        # adapters: terminal_bench, swebench_pro, gamedev, webgame (+ EnvStub)
  logging/           # step-segmented Trace JSONL (logs Γ's kept/dropped/summarized/retrieved)
  eval/              # Step 1 Γ variance probe (GammaProbe, spread_table, `hrl-probe` CLI)
  rl/                # Step 2 ONLY: env.py (neutral rollout), slime_adapter.py, reward.py, train_grpo.py
```

## Environments (dual)

- **`.venv` — Step 1 orchestration (GPU-free):** `bash scripts/setup_venv.sh`
- **Docker — Step 2 training (vast.ai/GPU):** `bash scripts/build_image.sh` (deferred)

> **Status:** implementation/code only. **No env build or GPU run yet** — deferred to a
> vast.ai instance for environment testing and validation.

## Local (no-GPU) checks

```bash
ruff check harness_rl && mypy harness_rl
pytest -q                       # gamma logic, seam/episode, trace roundtrip (stubbed model)
python -m harness_rl.eval.cli --stub   # dry-run the probe pipeline end-to-end, offline
```

## Step 1 — Γ variance probe (the GO/NO-GO gate)

On vast, point a served model at the probe and sweep Γ:

```bash
hrl-probe --benchmark terminal_bench_2 --model google/gemma-4-12b-it \
          --base-url http://localhost:8000/v1 \
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

> **Dense Gemma caveat:** dense Gemma-4 is not on slime's Megatron-Bridge path (MoE-only),
> so training uses slime's **FSDP backend** (`SLIME_BACKEND=fsdp`). Validate on vast; if
> broken, fall back to Gemma-4 MoE (26B-A4B) or make Qwen3.5 the primary.

## Models

- **Train (FSDP, full-param):** Gemma 4 (`google/gemma-4-*-it`, dense→FSDP) + Qwen3.5
  general, size-matched (~4B, ~12–14B); Qwen3-Coder as comparison. *Verify exact HF ids.*
- **Closed (API eval + grounded judge πref):** OpenAI / Anthropic / **Gemini** (transfer
  target + reference executor). Via LiteLLM. *Verify current ids.*
