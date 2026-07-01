"""Step 2 — agentic RL (slime, FSDP for dense Gemma, full-parameter GRPO).

Framework: slime (THUDM). The agent acts through our EXTERNAL harness via slime's custom
rollout/generate hook, calling the policy over the OpenAI-compatible endpoint slime serves
(SGLang). Reward = checkable outcome (+ optional grounded, outcome-gated process bonus).

Backend note: dense Gemma 4 is not on slime's Megatron/Megatron-Bridge path (MoE-only), so
we target slime's **FSDP backend** (`SLIME_BACKEND=fsdp`). VALIDATE THIS ON VAST — if the
FSDP+dense-Gemma path is broken, fall back to Gemma-4 MoE (26B-A4B, Megatron-Bridge OK) or
make Qwen3.5 the primary (fully supported).

This subpackage runs only on vast.ai; slime/torch are imported lazily so the `.venv`
orchestration layer stays GPU-free and importable for static checks.
"""
