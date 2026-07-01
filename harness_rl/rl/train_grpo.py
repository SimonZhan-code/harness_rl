"""GRPO training launch (Step 2) — slime, FSDP backend, full-parameter.

This is the launch/config skeleton. Training runs on vast.ai via slime; here we define the
config surface, the env/rollout wiring, and emit the slime launch command. slime/torch are
imported lazily so this file is importable in `.venv`.

Key choices (see implementation_plan.md):
  - framework = slime (THUDM); rollout via a custom generate function (rl/slime_adapter.py)
  - backend = FSDP for dense Gemma (`SLIME_BACKEND=fsdp`); VALIDATE on vast — fallback to
    Gemma-4 MoE (Megatron-Bridge) or Qwen3.5-primary if the dense-Gemma FSDP path is broken
  - rollout/serving via SGLang (OpenAI-compatible); harness points its base_url there
  - full-parameter GRPO (models are small ~4-14B); LoRA is an optional later variant
  - reward = outcome u_g (+ optional grounded, outcome-gated process bonus)
"""
from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    model: str = "google/gemma-4-12b-it"       # verify exact HF id
    benchmark: str = "terminal_bench_2"         # trainable benches only
    gamma_variant: str = "G3_structured_memory"
    backend: str = "megatron"                   # Megatron first; FSDP is the backup (dense Gemma)
    full_parameter: bool = True                 # small models → full FT; LoRA optional later
    lora_rank: int | None = None
    # GRPO / slime
    group_size: int = 8
    lr: float = 1e-6
    kl_coef: float = 0.001
    max_turns: int = 40
    served_base_url: str = "http://localhost:30000/v1"   # slime's SGLang endpoint
    rollout_fn: str = "harness_rl.rl.slime_adapter:build_generate_fn"
    # reward
    lambda_cost: float = 0.0
    beta_process: float = 0.0                   # 0 = pure outcome (conservative default)
    # infra
    num_gpus: int = 8
    out_dir: str = "./rl_runs"
    extra: dict = field(default_factory=dict)


def build_generate_fn(cfg: GRPOConfig):
    """Instantiate slime's custom generate function bound to our harness."""
    from harness_rl.benchmarks import TRAINABLE_BENCHMARKS
    from harness_rl.rl.reward import RewardConfig
    from harness_rl.rl.slime_adapter import build_generate_fn as _build

    if cfg.benchmark not in TRAINABLE_BENCHMARKS:
        raise ValueError(f"{cfg.benchmark} is eval-only; RL requires a checkable reward. "
                         f"Trainable: {sorted(TRAINABLE_BENCHMARKS)}")
    return _build(
        benchmark=cfg.benchmark,
        gamma_variant=cfg.gamma_variant,
        served_base_url=cfg.served_base_url,
        model_name=cfg.model,
        max_turns=cfg.max_turns,
        reward=RewardConfig(lambda_cost=cfg.lambda_cost, beta_process=cfg.beta_process),
    )


def _is_dense_gemma(model: str) -> bool:
    """Dense Gemma variants (4b/12b/27b/31b) — NOT the MoE (…-a4b/…moe)."""
    m = model.lower()
    return "gemma" in m and "a4b" not in m and "moe" not in m


def resolve_backend(model: str, requested: str = "megatron") -> tuple[str, str]:
    """Megatron-first with FSDP backup.

    Honor an explicit non-default request; otherwise default to Megatron but auto-fall-back
    to FSDP for dense Gemma (which Megatron-Bridge cannot convert — MoE-only).
    Returns (backend, note).
    """
    if requested != "megatron":
        return requested, ""
    if _is_dense_gemma(model):
        return "fsdp", "auto: dense Gemma is not on Megatron-Bridge (MoE-only) → FSDP backup"
    return "megatron", ""


def slime_launch_command(cfg: GRPOConfig) -> str:
    """Emit the slime launch command (verify flag names against installed slime on vast)."""
    backend, _ = resolve_backend(cfg.model, cfg.backend)
    peft = "--lora-rank %d" % cfg.lora_rank if cfg.lora_rank else "--full-parameter"
    env = "SLIME_BACKEND=fsdp " if backend == "fsdp" else ""   # Megatron is slime's default
    return (
        f"{env}python -m slime.train "
        f"--model {shlex.quote(cfg.model)} "
        f"--rl-algorithm grpo --group-size {cfg.group_size} "
        f"--learning-rate {cfg.lr} --kl-coef {cfg.kl_coef} {peft} "
        f"--rollout-function-path {cfg.rollout_fn} "
        f"--sglang-base-url {cfg.served_base_url} "
        f"--num-gpus {cfg.num_gpus} --save {cfg.out_dir}"
    )


def main(cfg: GRPOConfig) -> None:  # pragma: no cover (runs on vast.ai)
    """Validate config + print the slime launch. Actual `slime.train` runs on vast."""
    _ = build_generate_fn(cfg)  # fail fast on bad benchmark / import errors
    backend, note = resolve_backend(cfg.model, cfg.backend)
    if note:
        print(f"[backend] {note}")
    print(f"slime launch (backend={backend}; run on vast.ai with slime installed):\n")
    print(slime_launch_command(cfg))
    raise SystemExit(
        "This launcher prints the slime command and validates wiring. Execute the printed "
        "command inside the training Docker image on vast (SLIME_BACKEND=fsdp for dense Gemma)."
    )


def _parse_args() -> GRPOConfig:
    p = argparse.ArgumentParser(description="Step 2 GRPO launcher (slime/FSDP, full-param)")
    p.add_argument("--model", default=GRPOConfig.model)
    p.add_argument("--benchmark", default=GRPOConfig.benchmark)
    p.add_argument("--gamma", dest="gamma_variant", default=GRPOConfig.gamma_variant)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--beta-process", type=float, default=0.0)
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--backend", choices=["megatron", "fsdp"], default="megatron",
                   help="Megatron first; FSDP backup (auto-selected for dense Gemma).")
    p.add_argument("--served-base-url", default=GRPOConfig.served_base_url)
    a = p.parse_args()
    return GRPOConfig(model=a.model, benchmark=a.benchmark, gamma_variant=a.gamma_variant,
                      lora_rank=a.lora_rank, full_parameter=a.lora_rank is None,
                      beta_process=a.beta_process, num_gpus=a.num_gpus, backend=a.backend,
                      served_base_url=a.served_base_url)


if __name__ == "__main__":  # pragma: no cover
    main(_parse_args())
