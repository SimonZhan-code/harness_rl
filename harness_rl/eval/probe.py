"""Step 1 — the Gamma variance probe.

For a FIXED model, sweep the seed Gamma variants over a task subset and report
J(theta0, Gamma_i) = mean u_g, plus the spread max_i - min_i. A non-trivial spread on
long-horizon tasks is the GO signal for Step 2; a negligible spread is the NO-GO that
says rescope.

Fully GPU-free: pass any `ModelClient` (a real vLLM/API client on vast, or a `StubModel`
locally). The heavy Docker envs only start if you pass a Docker-backed benchmark on a
host with Docker; otherwise pass `EnvStub`-backed tasks for a dry run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from harness_rl.benchmarks.base import BenchmarkAdapter, Environment
from harness_rl.gamma import make_gamma
from harness_rl.harness.agent import run_episode
from harness_rl.logging import TraceWriter
from harness_rl.serving.client import ModelClient
from harness_rl.types import BudgetCaps, TaskSpec, Trace


@dataclass
class ProbeResult:
    model: str
    benchmark: str
    per_variant_u_g: dict[str, float] = field(default_factory=dict)   # J(theta0, Gamma_i)
    per_variant_cost: dict[str, float] = field(default_factory=dict)  # mean tokens
    n_tasks: int = 0

    @property
    def spread(self) -> float:
        if not self.per_variant_u_g:
            return 0.0
        vals = list(self.per_variant_u_g.values())
        return max(vals) - min(vals)

    @property
    def best_variant(self) -> str:
        return max(self.per_variant_u_g, key=self.per_variant_u_g.get)


class GammaProbe:
    def __init__(
        self,
        model: ModelClient,
        benchmark: BenchmarkAdapter,
        gamma_variants: list[str],
        out_dir: str | Path = "./runs",
        budget: BudgetCaps | None = None,
        gamma_kwargs: dict | None = None,
    ) -> None:
        self.model = model
        self.benchmark = benchmark
        self.gamma_variants = gamma_variants
        self.out_dir = Path(out_dir)
        self.budget = budget or BudgetCaps()
        self.gamma_kwargs = gamma_kwargs or {}

    def _run_variant(self, variant: str, tasks: list[TaskSpec],
                     env_factory=None) -> tuple[float, float]:
        writer = TraceWriter(self.out_dir / f"{self.benchmark.name}__{variant}.jsonl")
        u_gs, costs = [], []
        for task in tasks:
            env: Environment = env_factory(task) if env_factory else self.benchmark.make_env(task)
            gamma = make_gamma(variant, **self.gamma_kwargs.get(variant, {}))
            tr: Trace = run_episode(task, env, gamma, self.model, budget=self.budget,
                                    system_prompt=self.benchmark.system_prompt())
            writer.write(tr)
            if tr.outcome:
                u_gs.append(tr.outcome.u_g)
                costs.append(tr.outcome.cost_tokens)
        return (mean(u_gs) if u_gs else 0.0, mean(costs) if costs else 0.0)

    def run(self, n_tasks: int = 25, long_horizon: bool = True, env_factory=None) -> ProbeResult:
        tasks = self.benchmark.subset(n_tasks, long_horizon=long_horizon)
        res = ProbeResult(model=self.model.model, benchmark=self.benchmark.name, n_tasks=len(tasks))
        for variant in self.gamma_variants:
            u_g, cost = self._run_variant(variant, tasks, env_factory=env_factory)
            res.per_variant_u_g[variant] = u_g
            res.per_variant_cost[variant] = cost
        return res


def spread_table(results: list[ProbeResult]) -> str:
    """Render a markdown table: rows = benchmark, cols = Gamma variants + spread."""
    if not results:
        return "(no results)"
    variants = list(results[0].per_variant_u_g.keys())
    header = "| benchmark | " + " | ".join(variants) + " | spread | best |"
    sep = "|" + "---|" * (len(variants) + 3)
    rows = [header, sep]
    for r in results:
        cells = " | ".join(f"{r.per_variant_u_g.get(v, 0.0):.3f}" for v in variants)
        rows.append(f"| {r.benchmark} | {cells} | {r.spread:.3f} | {r.best_variant} |")
    return "\n".join(rows)
