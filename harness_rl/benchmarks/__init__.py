"""Benchmark registry.

Trainable (checkable reward → usable for Step 2 RL): terminal_bench_2, swebench_pro, gamedev,
  livecodebench (no-Docker code exec), swebench_lite (graded via Docker, deferred).
Eval-only (soft/noisy judge): webgame, gamecraft (multimodal rubric judge).
"""
from __future__ import annotations

from harness_rl.benchmarks.base import BenchmarkAdapter, Environment, EnvStub
from harness_rl.benchmarks.gamecraft import GameCraftBenchAdapter
from harness_rl.benchmarks.gamedev import GameDevBenchAdapter
from harness_rl.benchmarks.livecodebench import LiveCodeBenchAdapter
from harness_rl.benchmarks.swebench_lite import SWEBenchLiteAdapter
from harness_rl.benchmarks.swebench_pro import SWEBenchProAdapter
from harness_rl.benchmarks.terminal_bench import TerminalBench2Adapter
from harness_rl.benchmarks.webgame import WebGameBenchAdapter

BENCHMARK_REGISTRY = {
    TerminalBench2Adapter.name: TerminalBench2Adapter,
    SWEBenchProAdapter.name: SWEBenchProAdapter,
    SWEBenchLiteAdapter.name: SWEBenchLiteAdapter,
    LiveCodeBenchAdapter.name: LiveCodeBenchAdapter,
    GameDevBenchAdapter.name: GameDevBenchAdapter,
    GameCraftBenchAdapter.name: GameCraftBenchAdapter,
    WebGameBenchAdapter.name: WebGameBenchAdapter,
}

# Benchmarks whose reward is programmatically checkable (safe as RL training signal).
# (swebench_lite is checkable but graded via Docker — deferred on no-Docker hosts.)
TRAINABLE_BENCHMARKS = {"terminal_bench_2", "swebench_pro", "swebench_lite",
                        "livecodebench", "gamedev"}


def make_benchmark(name: str, **kwargs) -> BenchmarkAdapter:
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(f"unknown benchmark {name!r}; known: {list(BENCHMARK_REGISTRY)}")
    return BENCHMARK_REGISTRY[name](**kwargs)


__all__ = [
    "BenchmarkAdapter",
    "Environment",
    "EnvStub",
    "BENCHMARK_REGISTRY",
    "TRAINABLE_BENCHMARKS",
    "make_benchmark",
]
