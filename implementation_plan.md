---
date: 2026-07-01
tags: [harness-rl, implementation-plan, agentic-rl, benchmarks, coding-agent]
status: in-progress
project: harness-rl
---

# Implementation Plan — Harness+RL: Long-Horizon Coding, Context-Management Co-Evolution

> Code repo lives at `~/Documents/harness-rl/` (separate git repo, not inside the vault).
> Companion to [[problem_formulation]]. Build status tracked in this project's Daily Progress.

## Context

The project (`01_Projects/Harness+RL/`, see `Drafts/problem_formulation.md`) studies
co-evolving an agent's **memory/context-management policy $\Gamma$** with a
**fine-tuned policy $\theta$** (small models → **full-parameter** RL for now; LoRA
optional later) on **long-horizon coding**, to disentangle
harness-limited vs. capability-limited failure. Scope is fixed: tools/APIs, skills, and
prompts are frozen; **only $\Gamma$ is learnable**. This plan gives a
build-it-from-scratch implementation spec, split into two phases with a hard seam so
**Step 1 (inference-only benchmarking + $\Gamma$ variance probe) runs fully independently
of Step 2 (agentic RL)**.

> ⚠️ **Two research findings that shape this plan (verify at implementation time):**
> 1. The two gaming links are **different benchmarks**: `waynchi/gamedevbench` =
>    **GameDevBench** (Godot, deterministic unit tests, arXiv 2602.11103); `2605.17637`
>    = **WebGameBench** (browser games from spec, Playwright+LLM judge). GameDevBench is
>    RL-trainable (checkable reward); WebGameBench is **eval-only** (noisy judge).
> 2. **No coding-agent "Harbor" exists** (only a robot-RL HARBOR, 2606.08610). Replaced
>    with **OpenHands' Condenser system** (a ready-made pluggable-$\Gamma$ mechanism).

---

## Architectural backbone — the OpenAI-compatible seam

The single design decision that makes Step 1 / Step 2 separable:

> **The harness communicates with the policy ONLY through an OpenAI-compatible
> `chat/completions` `base_url`.** $\Gamma$ is a module *inside* the harness that builds
> the message list; the model is a black box behind `base_url`.

- **Step 1:** `base_url` → a local **SGLang** server (primary; **vLLM** backup) for open
  models, *or* a vendor API for closed models (via LiteLLM — provider's own inference, as-is).
- **Step 2:** `base_url` → **slime's SGLang** inference engine serving the policy over an
  OpenAI-compatible endpoint during rollouts.

slime drives the external agent through an OpenAI-compatible endpoint, so **the exact same
harness + $\Gamma$ + logging code is reused unchanged**; only the `base_url` (and a reward
hook) differ. This is the enforced separation. **Local inference standardizes on SGLang**
(vLLM backup) so Step 1 and Step 2 serve identically.

---

## Environments & execution (dual: `.venv` + Docker)

Two environments, matching the Step 1 / Step 2 seam:

- **`.venv` — local inference / orchestration (Step 1).** A Python virtualenv holding the
  harness, `gamma/`, benchmark adapters, serving *clients* (LiteLLM → SGLang/vLLM local
  servers + closed APIs), logging, and the eval runner. GPU-free; the SGLang/vLLM *servers*
  run on the GPU host. This is the layer run locally.
- **Docker — training launch + GPU serving (Step 2).** A pinned CUDA/torch image for
  **SkyRL + FSDP + vLLM** (the RL training launch). Provide a `Dockerfile` +
  `scripts/build_image.sh`. Keep GPU-heavy deps (torch-cuda, vllm, skyrl, flash-attn) in the
  image, out of `.venv`.
- **Note:** the *benchmarks themselves* already run each task in its own Docker sandbox
  (Terminal-Bench, SWE-bench-PRO, GameDevBench), so Docker is also invoked from Step 1 for
  task execution/verification — but per the seam, **local inference uses `.venv`; the
  training launch uses the training Docker image.**

Deliverables per env: `pyproject.toml`/`requirements.txt` + `scripts/setup_venv.sh` for
`.venv`; `Dockerfile` + `scripts/build_image.sh` for the training image; both pinned but
**not built locally** (see below).

## Build/validation deferral (no local GPU)

**Now: implementation & code-writing only.** Write all code, configs, `Dockerfile`, and
venv specs — runnable but **unbuilt**. Do **not** install deps, build images, serve open
models, or run any GPU-dependent step locally (no local GPU). **Later: a vast.ai GPU
instance** will be provisioned for env build, open-model serving, RL training, and
end-to-end validation. Only non-GPU static checks run locally now (imports, type-check,
unit tests with a **stubbed model endpoint**, and the `base_url` seam test against a stub
OpenAI server).

---

## Benchmarks

| Benchmark | Domain | Task | Verifier (checkable?) | Size | Horizon | Role | Repo / paper | License |
|---|---|---|---|---|---|---|---|---|
| **Terminal-Bench 2.0** | CLI/terminal | NL instruction → shell task in Docker | ✅ deterministic test scripts on final container state | 89 (4E/55M/30H) | multi-step, up to 2h tasks | **train + eval** (primary probe) | `laude-institute/terminal-bench-2`, tbench.ai | Apache-2.0 |
| **SWE-bench-PRO** | Repo SWE | Real GitHub issue → patch | ✅ hidden test suite, Pass@1 | 1865 (731 public / 858 held-out / 276 commercial) | long, multi-file (~107 LOC, 4.1 files) | **train + eval** | `scaleapi/SWE-bench_Pro-os`, arXiv 2509.16941 | CC-BY-4.0 |
| **GameDevBench** | Godot game dev | Edit Godot project to add feature | ✅ deterministic Godot unit tests, Pass@1 | 132 (115 base + 17 variants) | ~5 files, ~106 LOC | **train + eval** (gaming; checkable) | `waynchi/gamedevbench`, arXiv 2602.11103 | Apache-2.0 |
| **WebGameBench** | Browser games | Frozen spec → playable web game | ⚠️ Playwright + **LLM judge** (Excellent/Usable/Unusable; ~50% 3-way human agreement) | 111 | ~20–35 turns, 2h timeout | **eval-only** (judge too noisy for RL reward) | arXiv 2605.17637 | UNVERIFIED |

**Reward-checkability rule (ties to the verifiable-reward formulation):** RL (Step 2) uses
only benchmarks with programmatic per-task rewards → **Terminal-Bench 2.0, SWE-bench-PRO,
GameDevBench**. **WebGameBench is inference/eval-only.**

**Practical notes:** the trainable benches are Docker-based; SWE-bench-PRO supports
Modal for distributed eval (default 100 workers) or `--use_local_docker`. GameDevBench
needs **Godot 4.x** and is **multimodal** (editor-screenshot MCP + runtime-video feedback
lifted Sonnet 4.5 from 33.3%→47.7%) — requires a VLM policy to exploit visual feedback;
text-only (GDScript editing) is possible at lower ceiling.

---

## Harnesses

| Harness | Context/memory design | Role here | Repo | License |
|---|---|---|---|---|
| **mini-swe-agent** | Pure **linear message append, NO compaction** (~100 LOC) | **Clean $\Gamma$ baseline** — inject a `ContextManager` where today there is none; primary rollout harness for small open models + Step 2 RL (lightweight, LiteLLM, OpenAI-compatible) | `SWE-agent/mini-swe-agent` | MIT |
| **OpenHands** | **Event-sourced** log + **Condenser registry** (9 pluggable condensers incl. `LLMSummarizingCondenser`; view-level compaction, raw log preserved) | **Ready-made $\Gamma$ search space** for closed/big models + richer $\Gamma$ experiments; its `Condenser` interface *is* our $\Gamma$ abstraction | `OpenHands/OpenHands` | MIT |
| ~~Harbor~~ | — | **Dropped** (no coding-agent Harbor; robot-RL HARBOR is unrelated) | — | — |

**$\Gamma$ instrumentation.** Define one interface, implement it in both harnesses:

```python
class ContextManager(Protocol):
    # Given full interaction state + external memory, produce the bounded message list.
    def build_context(self, task, history, memory_store, budget_tokens) -> list[Message]: ...
    def on_step(self, obs, action) -> None: ...            # update memory / notes
    def snapshot(self) -> dict: ...                         # for logging: what was kept/dropped
```

- In **mini-swe-agent**: replace the linear-append with a `ContextManager` call.
- In **OpenHands**: adapt each `Condenser` to emit our `snapshot()` (kept/dropped/summarized).

**Seed $\Gamma$ variants for Step 1 probe (these become the C1 playbook seed items):**
`G0_truncate` (drop-oldest to fit window) · `G1_retrieval` (retrieve prior edits/decisions
relevant to current step) · `G2_summarize` (periodic LLM compaction of stale history) ·
`G3_structured_memory` (explicit scratchpad of files-touched / decisions / TODOs).

---

## Models

### Open-weight (full-parameter FT in Step 2; inference in Step 1)

Two **size-matched, cross-family** primary training policies (controlled comparison),
plus a coding-specialist comparison. **All trained via SkyRL FSDP** (HF-native) — see
Step 2. Two size bands: **~4B** (fast iteration) and **~12–14B** (headline).

| Model | HF repo id (verify exact ids at pull time) | Sizes | Context | Notes | License | Role |
|---|---|---|---|---|---|---|
| **Gemma 4 (it)** | `google/gemma-4-4b-it`, `google/gemma-4-12b-it` (dense; MoE `26b-a4b` exists) | ~4B, ~12B dense | long (confirm) | multimodal (helps GameDevBench); needs `transformers≥5.5`; **dense → FSDP only** (Megatron-Bridge supports only the 26B-A4B MoE) | Gemma (gated) | **Primary A** (Gemma→Gemini transfer story) |
| **Qwen3.5 (general, it)** | newest `Qwen/Qwen3.5-*` instruct at ~4B & ~14B (confirm ids) | ~4B, ~14B | long | size-matched to Gemma 4 for a clean cross-family comparison | Apache-2.0 | **Primary B** (size-matched control) |
| **Qwen3-Coder** | `Qwen/Qwen3-Coder-*` (coder series) | ~7B/~14B/30B-MoE tier | 256k+ | coding specialist, explicit agentic/tool-use | Apache-2.0 | **Comparison / eval target** |

> Gemma is gated on HF (accept terms). **Re-verify exact Gemma 4, Qwen3.5, and Qwen3-Coder
> repo ids + released sizes before pulling** (ids above are the expected pattern, not yet
> confirmed). Match the ~4B and ~12–14B bands across the Gemma 4 / Qwen3.5 primaries.

### Closed (API, inference-only eval + grounded-judge reference $\pi_{\text{ref}}$)

Call via **LiteLLM** (uniform OpenAI-compatible layer). Closed models use the **provider's
own inference** as-is — no local serving, whatever backend the vendor runs is fine. **Verify
current model IDs at implementation time** — the following were current as of research (July 2026):
`openai/gpt-5.5` (`gpt-5.5-pro`) · `anthropic/claude-sonnet-5` (`claude-opus-4-8`) ·
`gemini/gemini-3.5-flash` (`gemini-3.1-pro`). Gemini doubles as the **transfer target**
(Gemma→Gemini portability) and the **strong reference executor** for the grounded judge.

Minimal request skeleton (OpenAI-compatible; LiteLLM normalizes vendors):
```json
{ "model": "gpt-5.5", "messages": [{"role":"system","content":"..."},
  {"role":"user","content":"..."}], "temperature": 0.2, "max_tokens": 4096 }
```

---

## Step 1 — Inference-only benchmarking + $\Gamma$ variance probe (NO training)

**Goal / gate:** measure the spread of success across $\Gamma$ variants for a *fixed*
model → confirm (or kill) the premise that context management binds for long-horizon
coding. Build the reusable rollout+logging+verifier infra.

**Components**
1. `harness/` — fork mini-swe-agent; add `ContextManager` seam; keep tools fixed (bash;
   plus benchmark-specific fixed tools). Talks to policy via `base_url`.
2. `gamma/` — `G0_truncate`, `G1_retrieval`, `G2_summarize`, `G3_structured_memory`.
3. `benchmarks/` — thin adapters (uniform `run_task`, `verify(task, final_state)->[0,1]`)
   for Terminal-Bench-2, SWE-bench-PRO, GameDevBench, WebGameBench.
4. `serving/` — SGLang (primary) / vLLM (backup) launch scripts for open models;
   LiteLLM for closed (provider inference as-is). `ModelClient.for_sglang/.for_vllm`.
5. `logging/` — **step-segmented trace schema** (see below).
6. `eval/` — runner over a 20–30 task subset per bench; computes $J(\theta_0,\Gamma_i)$.

**Trace schema (critical — log $\Gamma$'s decisions, not just the trajectory):**
```json
{ "task_id","model","gamma_variant","step":[
    {"t","observation","context_built","gamma_snapshot":{"kept","dropped","summarized",
      "retrieved","tokens_in_ctx"},"action","tool_result"}],
  "outcome":{"u_g","passed_tests","cost_tokens","turns"} }
```
This is what makes the later 2×2 attribution and localized RL advantage computable.

**Probe protocol:** fixed model (Gemma-4-12B-it) × {G0,G1,G2,G3} × subset; report
$J(\theta_0,\Gamma_i)$ and spread $\max_i - \min_i$. Eyeball ~10 long-horizon failures:
context-shaped (dropped/buried info) vs. pure capability?

**Deliverables / GO-NO-GO:** reusable harness+logging+verifier; the $\Gamma$-spread table;
qualitative attribution read. **Gate:** if spread is negligible on long-horizon benches →
rescope before Step 2.

**Step 1 does NOT depend on any RL framework** (inference only).

---

## Step 2 — Agentic RL (slime, FSDP, full-parameter, on-policy) — build only after the gate passes

**Framework: slime (THUDM) — decided.** The harness plugs into slime as a **custom rollout
function** (`--rollout-function-path`); the policy is served by slime over an
OpenAI-compatible **SGLang** endpoint, and the harness points its `base_url` there — so the
Step 1 harness/Γ/logging code is reused unchanged. GRPO + full-parameter are native.

**Backend — Megatron first, FSDP backup.** Default is **Megatron** (Qwen3.5 / Qwen3-Coder
and **Gemma-4 MoE** are supported). **Dense Gemma-4** is not on Megatron-Bridge (MoE-only),
so the launcher **auto-selects the FSDP backup** (`SLIME_BACKEND=fsdp`) for it; force with
`--backend`. Roll out with **SGLang** (serves Gemma 4). Small models (~4–14B) →
**full-parameter GRPO** default (natively supported, no PEFT needed); LoRA optional later
(its smaller footprint is a lever for the retention/$G_{\text{ret}}$ study — full-param FT
forgets more, so watch that metric).

> ⚠️ **Validate on vast:** the dense-Gemma FSDP path was undocumented for Gemma when last
> checked. If broken, fall back to **Gemma-4 MoE** (26B-A4B, Megatron) or make **Qwen3.5
> the primary** (Megatron-safe). Qwen is safe on Megatron regardless.

**Integration**
1. `rl/env.py` (`HarnessRollout`) — framework-neutral: runs the **Step 1 harness** for one
   task, returns `(trace, reward)`. `rl/slime_adapter.py` wraps it into slime's custom
   generate-function signature; policy served by slime → harness `base_url` points there
   (only change from Step 1).
2. `rl/reward.py` — outcome $u_g$ from the verifier; optional grounded process bonus
   $\beta q_m$ (Gemini $\pi_{\text{ref}}$ counterfactual), **outcome-gated**; localized
   advantage on the execution span (see `problem_formulation.md` §5/C3).
3. `rl/train_grpo.py` — GRPO, **full-parameter**; on-policy under the deploy harness
   (avoid offline-SFT collapse); train per-benchmark and mixed.
4. Reuse Step 1 logging/verifier verbatim.

**Later (co-evolution + generalization):** wrap $\Gamma$ as the prunable provenance
playbook + $\mathcal{U}$ operator; interleave harness-loop with RL; measure retention
($G_{\text{ret}}$, on held-out non-coding evals) and harness portability ($G_{\text{port}}$,
Gemma-tuned $\Gamma$ dropped onto Gemini + item-level transfer via the playbook).

---

## Proposed repo layout (monorepo)

```
harness-rl/
  harness/            # forked mini-swe-agent + ContextManager seam (shared Step1/Step2)
  gamma/              # G0..G3 context managers  (shared)
  benchmarks/         # adapters: terminal_bench, swebench_pro, gamedev, webgame
  serving/            # vLLM/SGLang + LiteLLM configs (Step 1)
  logging/            # trace schema + writers (shared)
  eval/               # Step 1 probe runner
  rl/                 # Step 2 ONLY: env.py, reward.py, train_grpo.py (SkyRL/slime)
  openhands_gamma/    # OpenHands Condenser adapters (big/closed-model Γ experiments)
  configs/  scripts/  README.md
```

---

## Verification

**Local now (no GPU) — static checks only:**
- imports + type-check (`ruff`/`mypy`); unit tests for `gamma/` (kept/dropped logic),
  logging schema round-trip, benchmark-adapter parsing.
- **Seam test:** run the harness against a **stub OpenAI server** returning canned actions;
  assert identical harness behavior whether `base_url` points at the stub or (later) vLLM.
- dry-run the eval runner with the stubbed endpoint → confirm traces + `gamma_snapshot` +
  `u_g` populate end-to-end (no real model, no GPU).

**On vast.ai (deferred) — GPU / full runs:**
- Step 1: full 20–30 subset × {G0..G3} on Gemma-4-12B via SGLang → the $\Gamma$-spread table;
  sanity: `G0_truncate` underperforms on the longest-horizon tasks.
- Step 2: overfit-one-task (RL drives a single Terminal-Bench task to Pass@1); then a small
  full-parameter GRPO run on a task cluster; verify reward curve rises and retention eval
  does not collapse.

---

## Decisions (resolved)
1. **Gaming:** both, split by role — **GameDevBench** trainable (deterministic Godot tests),
   **WebGameBench** eval-only (noisy judge).
2. **Models:** **Gemma 4** + **Qwen3.5 (general)**, size-matched (~4B and ~12–14B), as
   dual primaries; **Qwen3-Coder** + closed models as comparison/eval. Re-verify exact HF ids.
3. **RL framework:** **slime** (THUDM), **Megatron-first** backend with **FSDP as backup**
   (auto-selected for dense Gemma); **full-parameter** GRPO (small models → no LoRA for now).
   ⚠️ dense-Gemma FSDP path must be validated on vast; fallback = Gemma-4 MoE or Qwen3.5-primary.
4. **Benchmarks:** SWE-bench-Evo **dropped** (too heavy alongside gaming) → Terminal-Bench 2.0,
   SWE-bench-PRO, GameDevBench (trainable) + WebGameBench (eval-only).

## Remaining verification before build (not blockers to approve)
- Exact HF repo ids + released sizes for **Gemma 4**, **Qwen3.5**, and **Qwen3-Coder**.
- Current closed-model API ids (OpenAI / Anthropic / Gemini) at implementation time.
- **WebGameBench license** (unverified) and its deploy/eval harness details.
- GameDevBench multimodal path: whether to run Gemma 4 as a VLM (visual feedback) or
  text-only (GDScript) in v1.
