"""Benchmark registry.

Trainable (checkable reward → usable for Step 2 RL): terminal_bench_2, swebench_pro, gamedev.
Eval-only (noisy judge): webgame.
"""
from __future__ import annotations

from harness_rl.benchmarks.base import BenchmarkAdapter, Environment, EnvStub
from harness_rl.benchmarks.gamedev import GameDevBenchAdapter
from harness_rl.benchmarks.swebench_pro import SWEBenchProAdapter
from harness_rl.benchmarks.terminal_bench import TerminalBench2Adapter
from harness_rl.benchmarks.webgame import WebGameBenchAdapter

BENCHMARK_REGISTRY = {
    TerminalBench2Adapter.name: TerminalBench2Adapter,
    SWEBenchProAdapter.name: SWEBenchProAdapter,
    GameDevBenchAdapter.name: GameDevBenchAdapter,
    WebGameBenchAdapter.name: WebGameBenchAdapter,
}

# Benchmarks whose reward is programmatically checkable (safe as RL training signal).
TRAINABLE_BENCHMARKS = {"terminal_bench_2", "swebench_pro", "gamedev"}


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
