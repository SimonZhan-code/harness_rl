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
  rl/                # Step 2: env.py, slime_adapter.py, reward.py, train_grpo.py
                     #         + SUPO (2510.06727): supo.py, train_supo.py, supo_check.py
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

## SUPO (arXiv 2510.06727) reimplementation

Summarization-augmented Policy Optimization, on our slime infra with the **G2** memory component.
When the working context exceeds a token threshold **L**, the **policy itself** generates a
summary (a trainable action) that compacts history and opens a new **sub-trajectory**. Credit
assignment is **Theorem 3.2**: one training sample per segment (tool-use **and** summary turns),
all sharing the **group-relative outcome advantage**; overlong rollouts are masked.
Target: **Qwen2.5-Coder-3B**, **LiveCodeBench**, **max 40 turns**, `L` configurable (default 4096).

```bash
hrl-supo-check --stub                 # offline: assert compaction fires + Thm-3.2 sample structure
hrl-supo-check --model Qwen/Qwen2.5-Coder-3B-Instruct --base-url http://localhost:30000/v1  # real 3B
python -m harness_rl.rl.train_supo    # prints the slime launch (run on a GPU box)
```
Code: `rl/supo.py` (rollout + segmentation), `rl/reward.py:supo_samples` (Thm 3.2), `rl/train_supo.py`
(launch), `gamma/g2_summarize.py` (`compact_at_tokens=L`, `summary_mode="replace"`).

**Per-category token tracking** (#0/#1): every generated span is tagged **summarization / thinking /
tool_call** (`rl/supo.py:categorize_spans`→`categorize_tokens`, on each `SegmentTurn.category_tokens`);
a batch report (`rl/reward.py:category_advantage_report`) gives per-category token counts, fraction, and
**advantage-weighted mass** (where the GRPO update pressure lands), split by success/fail. Shown by
`hrl-supo-check`. Note: under Thm 3.2 the advantage is uniform within a rollout, so this is
descriptive monitoring, not causal per-category attribution.

**Per-category entropy & KL(π_new‖π_ref)** (`rl/metrics.py`): the training-dynamics dashboard. Since
exact entropy needs the logits and KL needs π_ref, these are computed in the **slime loss step**;
`categorize_spans` emits char-spans so `label_tokens(spans, offsets)` maps the trainer's tokens (HF
`return_offsets_mapping`) to categories, then `bucket_by_category` splits the per-token entropy (from
logits) and KL (`token_kl_k3`, the k3 estimator GRPO already uses) into the three buckets;
`supo_category_metrics` + `merge_category_metrics`/`category_entropy_kl_report` aggregate a step/batch.
Integration point: `rl/train_supo.py:supo_training_hook_note`. **Reads:** summarization-entropy → 0 =
summary collapse (the SUPO failure mode); summary-KL spike = summarizer drifting off π_ref.
`hrl-supo-check` prints the table with synthetic per-token values (mechanics demo — real values come
from the trainer).

### Summary-branching TREE rollout (extension — variance-reduced summary credit)

Flat SUPO gives summary **and** action tokens the *same* outcome advantage (Thm 3.2), so summary
quality and action quality are never disentangled — the highest-leverage decision gets the noisiest
signal. The tree fixes that: at **every** compaction, sample **B** summaries from the *same*
pre-summary context and roll each forward independently (`env.fork()` + `G2.clone()`). A summary's
**subtree value** (mean leaf `u_g`) scores it *against its siblings at a shared anchor* — a
GiGPO-style step group manufactured exactly where LiveCodeBench/SWE states never recur. **Summary
branching only**; action turns stay linear.

Two-level credit (`rl/tree_reward.py`), critic-free (Monte-Carlo subtree backup, λ=1 — no
bootstrapping, so with terminal-only reward the backup *is* the return):

| | advantage |
|---|---|
| action node | `A_macro` |
| summary node | `A_macro + w·A_micro` |

`A_macro = (V(n) − μ)/(σ+ε)` over the group's leaves (GRPO); `A_micro = (V(sᵢ) − baseline_{-i})/(σ_sib+ε)`
over summary siblings (GiGPO, leave-one-out default). `w_micro=0` reduces **exactly** to flat SUPO
(regression guard + ablation baseline). Overlong leaves are excluded from the backup *and* the
samples. Shared-prefix nodes appear once → no gradient double-counting.

**Branching is budgeted by DEPTH, not leaf count.** Every path branches at its first
`branch_depth` compactions, so sibling subtrees are equal-sized by construction (`B**depth` leaves)
and `max_leaves` is only a safety clamp that *lowers the depth*. The earlier leaf-count budget was
spent in DFS order, so the first sibling's subtree absorbed the remainder (`B=2, max_leaves=8` gave
`[7, 1]`) — meaning contrasted summaries had wildly unequal value-estimate precision, which
inflates `σ_sibling` with estimation noise and biases η² upward. Balance is now a regression test.

```bash
hrl-supo-check --tree-stub --branch-factor 3 --max-leaves 6     # offline: tree + macro/micro table
hrl-supo-check --tree --base-url http://localhost:30000/v1 \
               --context-L 1500 --branch-factor 3 --max-leaves 6 --n-tasks 1   # real served model
python -m harness_rl.rl.train_supo --tree --branch-factor 3 --max-leaves 12 --w-micro 1.0
```
Code: `rl/tree_supo.py` (branching rollout), `rl/tree_reward.py` (backup + macro/micro + samples),
`rl/slime_adapter.py:build_tree_supo_generate_fn`, `gamma/g2_summarize.py`
(`pending_compaction`/`apply_summary`/`clone`), `serving/client.py:chat_many`, `Environment.fork()`.
⚠️ Tree samples carry a **precomputed `advantage`** (not a reward) — slime must be configured to use
it instead of re-deriving a group-relative advantage; fallback noted in the adapter docstring.

**Validated on an H100 NVL (driver 580) serving Qwen2.5-Coder-3B-Instruct on LiveCodeBench:**
`chat_many(n=3)` returns 3 distinct summaries in one SGLang request; branching fires at summary
nodes only; `LiveCodeExecEnv.fork()` gives each branch an independent workdir; real `u_g` flows
through the subtree backup into macro/micro advantages.

> **Measured gotcha — `mask_reasons`.** Qwen2.5-Coder-3B **never emits `TASK_COMPLETE`** (0 submits
> in 20 action turns; all outputs parsed fine) even when it *solves* the task (`u_g=1.0`). Under
> faithful SUPO masking (`("budget","max_summaries")`) every rollout is overlong → **zero gradient
> per batch** — for flat SUPO too. So both `TreeConfig` and `SUPORolloutConfig` expose
> **`mask_reasons`**: `max_summaries` is a genuine context-management failure (what SUPO's ablation
> is about), while `budget` just means the agent used its turns — and LiveCodeBench still grades the
> artifact on disk, so `u_g` is meaningful. Set `mask_reasons=("max_summaries",)` (CLI:
> `--grade-turn-exhausted`) to train on turn-exhausted rollouts. Keep it **identical** across flat
> and tree runs or the ablation is unfair. `tree_reward` stats also flag `all_masked` and
> `degenerate_group` (σ=0 → macro teaches nothing; only the micro term can).

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
