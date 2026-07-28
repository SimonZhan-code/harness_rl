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


def build_generate_fn(cfg: SUPOConfig):
    """Instantiate the SUPO custom-generate function bound to our harness."""
    from harness_rl.benchmarks import TRAINABLE_BENCHMARKS
    from harness_rl.rl.slime_adapter import build_supo_generate_fn

    if cfg.benchmark not in TRAINABLE_BENCHMARKS:
        raise ValueError(f"{cfg.benchmark} is eval-only; SUPO needs a checkable reward. "
                         f"Trainable: {sorted(TRAINABLE_BENCHMARKS)}")
    return build_supo_generate_fn(
        benchmark=cfg.benchmark, served_base_url=cfg.served_base_url, model_name=cfg.model,
        cfg=rollout_config(cfg), overlong_mask=cfg.overlong_mask,
    )


def slime_launch_command(cfg: SUPOConfig) -> str:
    backend, _ = resolve_backend(cfg.model, cfg.backend)
    env = "SLIME_BACKEND=fsdp " if backend == "fsdp" else ""
    return (
        f"{env}python -m slime.train "
        f"--model {shlex.quote(cfg.model)} "
        f"--rl-algorithm grpo --group-size {cfg.group_size} "
        f"--learning-rate {cfg.lr} --kl-coef {cfg.kl_coef} --full-parameter "
        f"--rollout-function-path {cfg.rollout_fn} "
        f"--sglang-base-url {cfg.served_base_url} "
        f"--num-gpus {cfg.num_gpus} --save {cfg.out_dir}"
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
    a = p.parse_args()
    return SUPOConfig(model=a.model, benchmark=a.benchmark, context_L=a.context_L,
                      max_turns=a.max_turns, max_summaries=a.max_summaries,
                      num_gpus=a.num_gpus, served_base_url=a.served_base_url)


if __name__ == "__main__":  # pragma: no cover
    main(_parse_args())
