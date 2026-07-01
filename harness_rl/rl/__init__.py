"""Step 2 — agentic RL (slime, Megatron-first / FSDP-backup, full-parameter GRPO).

Framework: slime (THUDM). The agent acts through our EXTERNAL harness via slime's custom
rollout/generate hook, calling the policy over the OpenAI-compatible endpoint slime serves
(SGLang). Reward = checkable outcome (+ optional grounded, outcome-gated process bonus).

Backend: **Megatron first** (Qwen + Gemma-4 MoE are supported). **FSDP is the backup**,
auto-selected for **dense Gemma 4** (Megatron-Bridge is MoE-only) via `SLIME_BACKEND=fsdp`
— VALIDATE on vast; if the dense-Gemma FSDP path is broken, use Gemma-4 MoE (Megatron) or
make Qwen3.5 the primary.

This subpackage runs only on vast.ai; slime/torch are imported lazily so the `.venv`
orchestration layer stays GPU-free and importable for static checks.
"""
