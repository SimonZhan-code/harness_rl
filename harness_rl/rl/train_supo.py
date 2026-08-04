"""SUPO training launcher (arXiv 2510.06727) on slime.

Wires `build_supo_generate_fn` (rl/slime_adapter.py) into slime's GRPO trainer for
**Qwen2.5-Coder-3B** on **LiveCodeBench**, **max 40 turns**, with a context threshold `L`. The
credit assignment (Thm 3.2) is realized by the per-segment samples sharing the group-relative
outcome advantage (see rl/reward.py:supo_samples). This validates wiring + prints the slime
launch command; the actual training runs on a GPU box (validate rollout mechanics first — see
rl/supo.py).
"""
from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass

from harness_rl.rl.supo import SUPORolloutConfig
from harness_rl.rl.train_grpo import resolve_backend


@dataclass
class SUPOConfig:
    model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    benchmark: str = "livecodebench"
    # SUPO context management
    context_L: int = 4096          # token threshold L that triggers a policy summarization action
    max_turns: int = 40
    max_summaries: int = 8
    recency_turns: int = 4
    overlong_mask: bool = True     # essential per the SUPO ablation
    # Summary-branching TREE rollout (rl/tree_supo.py); tree=False → flat SUPO
    tree: bool = False
    branch_factor: int = 3         # B summaries sampled per compaction
    branch_depth: int = 1          # branch at the first D compactions (balanced; B**D leaves)
    max_leaves: int = 12           # safety clamp only — lowers D until B**D <= max_leaves
    summary_temperature: float = 0.9
    w_micro: float = 1.0           # weight on the GiGPO sibling-relative summary term
    micro_baseline: str = "loo"    # "loo" | "mean"
    # ("budget","max_summaries") = faithful SUPO masking; ("max_summaries",) also trains on
    # turn-exhausted rollouts (needed when the policy rarely emits TASK_COMPLETE — see tree_supo.py)
    mask_reasons: tuple[str, ...] = ("budget", "max_summaries")
    # GRPO / slime
    group_size: int = 8
    lr: float = 1e-6
    kl_coef: float = 0.001
    served_base_url: str = "http://localhost:30000/v1"   # slime's SGLang endpoint
    rollout_fn: str = "harness_rl.rl.slime_adapter:build_supo_generate_fn"
    backend: str = "megatron"      # Qwen → Megatron; dense Gemma would auto-fall-back to FSDP
    num_gpus: int = 8
    out_dir: str = "./supo_runs"


def rollout_config(cfg: SUPOConfig) -> SUPORolloutConfig:
    return SUPORolloutConfig(context_L=cfg.context_L, max_turns=cfg.max_turns,
                             max_summaries=cfg.max_summaries, recency_turns=cfg.recency_turns)


def tree_config(cfg: SUPOConfig):
    from harness_rl.rl.tree_supo import TreeConfig
    return TreeConfig(context_L=cfg.context_L, max_turns=cfg.max_turns,
                      max_summaries=cfg.max_summaries, recency_turns=cfg.recency_turns,
                      branch_factor=cfg.branch_factor, branch_depth=cfg.branch_depth,
                      max_leaves=cfg.max_leaves, summary_temperature=cfg.summary_temperature,
                      mask_reasons=tuple(cfg.mask_reasons))


def build_generate_fn(cfg: SUPOConfig):
    """Instantiate the SUPO custom-generate function bound to our harness (flat or tree)."""
    from harness_rl.benchmarks import TRAINABLE_BENCHMARKS

    if cfg.benchmark not in TRAINABLE_BENCHMARKS:
        raise ValueError(f"{cfg.benchmark} is eval-only; SUPO needs a checkable reward. "
                         f"Trainable: {sorted(TRAINABLE_BENCHMARKS)}")
    if cfg.tree:
        from harness_rl.rl.slime_adapter import build_tree_supo_generate_fn
        return build_tree_supo_generate_fn(
            benchmark=cfg.benchmark, served_base_url=cfg.served_base_url, model_name=cfg.model,
            cfg=tree_config(cfg), group_size=cfg.group_size, w_micro=cfg.w_micro,
            micro_baseline=cfg.micro_baseline, overlong_mask=cfg.overlong_mask,
        )
    from harness_rl.rl.slime_adapter import build_supo_generate_fn
    return build_supo_generate_fn(
        benchmark=cfg.benchmark, served_base_url=cfg.served_base_url, model_name=cfg.model,
        cfg=rollout_config(cfg), overlong_mask=cfg.overlong_mask,
    )


def supo_training_hook_note() -> str:
    """Where per-category entropy/KL plug into the slime loss step.

    Inside the GRPO loss on a SUPO segment the trainer already forms, per response token: the
    policy logits (→ exact entropy) and pi_ref logprobs (→ KL for the penalty). To split those by
    token category, tokenize the response with `return_offsets_mapping=True` and call, per segment:

        from harness_rl.rl.metrics import (entropy_from_logprobs, token_kl_k3,
                                           supo_category_metrics, merge_category_metrics)
        ent = [entropy_from_logprobs(step_logprobs) for step_logprobs in logits.log_softmax(-1)]
        kl  = [token_kl_k3(lp_new, lp_ref) for lp_new, lp_ref in zip(logp_new_a, logp_ref_a)]
        seg_stats.append(supo_category_metrics(resp_text, is_summary, offsets, entropy=ent, kl=kl))

    then `merge_category_metrics(seg_stats)` → log `entropy/kl by {summarization,thinking,tool_call}`
    each step. Watch summarization-entropy → 0 (summary collapse, the SUPO failure mode) and
    summary-KL spikes (summarizer drift from pi_ref). See `rl/metrics.py`.
    """
    return supo_training_hook_note.__doc__ or ""


def slime_launch_command(cfg: SUPOConfig) -> str:
    backend, _ = resolve_backend(cfg.model, cfg.backend)
    env = "SLIME_BACKEND=fsdp " if backend == "fsdp" else ""
    rollout_fn = ("harness_rl.rl.slime_adapter:build_tree_supo_generate_fn"
                  if cfg.tree else cfg.rollout_fn)
    # tree generate fn owns the whole group + precomputes advantages → group-size 1 at the slime
    # layer (one generate call = one group of trees); flat SUPO uses slime's own grouping.
    return (
        f"{env}python -m slime.train "
        f"--model {shlex.quote(cfg.model)} "
        f"--rl-algorithm grpo --group-size {1 if cfg.tree else cfg.group_size} "
        f"--learning-rate {cfg.lr} --kl-coef {cfg.kl_coef} --full-parameter "
        f"--rollout-function-path {rollout_fn} "
        f"--sglang-base-url {cfg.served_base_url} "
        f"--num-gpus {cfg.num_gpus} --save {cfg.out_dir}"
        + ("  # tree: fn precomputes advantages — configure slime to use them (not reward-grouping)"
           if cfg.tree else "")
    )


def main(cfg: SUPOConfig) -> None:  # pragma: no cover (runs on vast.ai)
    _ = build_generate_fn(cfg)  # fail fast on bad benchmark / import errors
    print(f"SUPO slime launch (model={cfg.model}, L={cfg.context_L}, max_turns={cfg.max_turns}):\n")
    print(slime_launch_command(cfg))
    raise SystemExit(
        "Prints the slime command and validates wiring. Validate rollout mechanics first "
        "(rl/supo.py + hrl-supo-check), then run this on a GPU box with slime installed."
    )


def _parse_args() -> SUPOConfig:
    p = argparse.ArgumentParser(description="SUPO launcher (arXiv 2510.06727) on slime")
    p.add_argument("--model", default=SUPOConfig.model)
    p.add_argument("--benchmark", default=SUPOConfig.benchmark)
    p.add_argument("--context-L", type=int, default=SUPOConfig.context_L)
    p.add_argument("--max-turns", type=int, default=SUPOConfig.max_turns)
    p.add_argument("--max-summaries", type=int, default=SUPOConfig.max_summaries)
    p.add_argument("--num-gpus", type=int, default=SUPOConfig.num_gpus)
    p.add_argument("--served-base-url", default=SUPOConfig.served_base_url)
    p.add_argument("--tree", action="store_true", help="summary-branching tree rollout (rl/tree_supo.py)")
    p.add_argument("--branch-factor", type=int, default=SUPOConfig.branch_factor)
    p.add_argument("--branch-depth", type=int, default=SUPOConfig.branch_depth,
                   help="branch at the first D compactions (balanced tree, B**D leaves)")
    p.add_argument("--max-leaves", type=int, default=SUPOConfig.max_leaves)
    p.add_argument("--w-micro", type=float, default=SUPOConfig.w_micro)
    a = p.parse_args()
    return SUPOConfig(model=a.model, benchmark=a.benchmark, context_L=a.context_L,
                      max_turns=a.max_turns, max_summaries=a.max_summaries,
                      num_gpus=a.num_gpus, served_base_url=a.served_base_url,
                      tree=a.tree, branch_factor=a.branch_factor, branch_depth=a.branch_depth,
                      max_leaves=a.max_leaves, w_micro=a.w_micro)


if __name__ == "__main__":  # pragma: no cover
    main(_parse_args())
