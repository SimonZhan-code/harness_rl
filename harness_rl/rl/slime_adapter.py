"""slime binding — expose our external harness as a custom rollout/generate function.

slime drives training with a user-supplied rollout function (registered via
`--rollout-function-path` / `--custom-generate-function-path`). That function generates
trajectories by calling the policy slime serves over an OpenAI-compatible endpoint
(SGLang), and returns, per sample, the token/message sequence plus a scalar reward; slime
computes GRPO group-relative advantages from the rewards.

We plug our harness in there: the "generation" for a coding task IS a full multi-turn
harness episode. This module provides `build_generate_fn(...)` returning a callable with
slime's expected signature.

⚠️ The exact slime rollout-function signature and return schema shift across versions.
The signature below reflects the documented pattern; VERIFY against the installed slime
version on vast and adjust field names (`args`, `sample`, `sampling_params`, and the
returned rollout object) accordingly.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from harness_rl.benchmarks import make_benchmark
from harness_rl.rl.env import HarnessRollout, HarnessRolloutConfig
from harness_rl.rl.reward import RewardConfig, localized_advantage_mask
from harness_rl.serving.client import ModelClient
from harness_rl.types import BudgetCaps, TaskSpec


def build_generate_fn(
    benchmark: str,
    gamma_variant: str,
    served_base_url: str,
    model_name: str,
    max_turns: int = 40,
    reward: RewardConfig | None = None,
) -> Callable[[Any, dict, Any], dict]:
    """Return slime's custom-generate function bound to our harness.

    Returned callable signature (slime-style): fn(args, sample, sampling_params) -> dict.
    `sample` carries the task (we expect a `task` payload); the returned dict carries the
    trajectory + reward slime needs for GRPO.
    """
    adapter = make_benchmark(benchmark)
    # The policy under training is served by slime (SGLang) at `served_base_url`.
    model = ModelClient(model=model_name, base_url=served_base_url)
    runner = HarnessRollout(
        adapter,
        HarnessRolloutConfig(benchmark=benchmark, gamma_variant=gamma_variant,
                             budget=BudgetCaps(max_turns=max_turns),
                             reward=reward or RewardConfig()),
        model,
    )

    def generate(args: Any, sample: dict, sampling_params: Any) -> dict:  # pragma: no cover
        task = _sample_to_task(sample, benchmark)
        trace, reward = runner.rollout(task)
        mask = localized_advantage_mask(trace)  # per-step; expand to per-token on vast
        return {
            "reward": reward,
            "trace": trace,                     # our Trace (also written to JSONL via logging)
            "messages": _trace_to_messages(trace),
            "loss_mask_steps": mask,            # localize credit to the execution span
            "outcome_u_g": trace.outcome.u_g if trace.outcome else 0.0,
        }

    return generate


def _sample_to_task(sample: dict, benchmark: str) -> TaskSpec:
    if isinstance(sample.get("task"), TaskSpec):
        return sample["task"]
    return TaskSpec(
        task_id=sample.get("task_id", "unknown"),
        benchmark=benchmark,
        instruction=sample.get("instruction", sample.get("prompt", "")),
        payload=sample.get("payload", {}),
    )


def _trace_to_messages(trace) -> list[dict]:
    """Flatten a trace into an OpenAI-style message list (for slime tokenization).

    On vast, prefer returning the exact token ids slime expects; this text form is the
    portable fallback.
    """
    msgs: list[dict] = []
    for s in trace.steps:
        if s.action is not None:
            msgs.append({"role": "assistant", "content": s.action.raw})
        msgs.append({"role": "tool", "content": s.tool_result})
    return msgs
