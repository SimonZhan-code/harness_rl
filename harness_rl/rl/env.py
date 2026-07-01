"""Framework-neutral rollout wrapper around the Step 1 harness.

`HarnessRollout.rollout(task)` runs one episode of the SAME harness used in Step 1 and
returns `(trace, scalar_reward)`. It is framework-agnostic on purpose: the slime binding
(`rl/slime_adapter.py`) calls this, and any other trainer could too. The only Step-2
change vs. Step 1 is that `ModelClient.base_url` points at the trainer-served policy
(slime's SGLang OpenAI-compatible endpoint).
"""
from __future__ import annotations

from dataclasses import dataclass

from harness_rl.benchmarks.base import BenchmarkAdapter
from harness_rl.harness.agent import run_episode
from harness_rl.rl.reward import RewardConfig, compute_reward
from harness_rl.serving.client import ModelClient
from harness_rl.types import BudgetCaps, TaskSpec, Trace


@dataclass
class HarnessRolloutConfig:
    benchmark: str
    gamma_variant: str = "G3_structured_memory"   # fixed Gamma during the pure-RL phase
    budget: BudgetCaps | None = None
    reward: RewardConfig | None = None


class HarnessRollout:
    def __init__(self, adapter: BenchmarkAdapter, cfg: HarnessRolloutConfig, model: ModelClient):
        self.adapter = adapter
        self.cfg = cfg
        self.model = model
        self.cfg.reward = self.cfg.reward or RewardConfig()

    def rollout(self, task: TaskSpec) -> tuple[Trace, float]:
        """Run one episode; return (trace, scalar_reward). The trace carries per-step
        Gamma snapshots and (once the judge scores it) q_t, from which the localized
        advantage mask is built (see rl/reward.py)."""
        from harness_rl.gamma import make_gamma

        env = self.adapter.make_env(task)
        gamma = make_gamma(self.cfg.gamma_variant)
        tr = run_episode(task, env, gamma, self.model,
                         budget=self.cfg.budget or BudgetCaps(),
                         system_prompt=self.adapter.system_prompt())
        reward = compute_reward(tr, self.cfg.reward)
        return tr, reward
