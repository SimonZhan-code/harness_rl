"""SWE-bench-PRO adapter.

Repo: scaleapi/SWE-bench_Pro-os. Dataset: ScaleAI/SWE-bench_Pro (HF). Tasks are real
GitHub issues; the verifier runs the repo's hidden test suite (Pass@1). Docker images:
`jefzda/sweap-images`. Distributed eval via Modal or `--use_local_docker`.

Local: `tasks()`/`subset()` work from the HF dataset (metadata only). `make_env` builds a
DockerEnv on vast. Uses the `public` split (731 instances) by default.
"""
from __future__ import annotations

import os
from typing import Iterator

from harness_rl.benchmarks._docker import DockerEnv
from harness_rl.harness.agent import DEFAULT_SYSTEM_PROMPT
from harness_rl.types import Outcome, TaskSpec


class SWEBenchProAdapter:
    name = "swebench_pro"

    def __init__(self, split: str = "public", hf_id: str = "ScaleAI/SWE-bench_Pro"):
        self.split = split
        self.hf_id = hf_id
        self._cache: list[TaskSpec] | None = None

    def _load(self) -> list[TaskSpec]:
        if self._cache is not None:
            return self._cache
        try:
            from datasets import load_dataset  # lazy .venv dep
        except Exception:
            return []
        ds = load_dataset(self.hf_id, split=self.split)
        specs = []
        for row in ds:
            specs.append(
                TaskSpec(
                    task_id=f"swebp/{row['instance_id']}",
                    benchmark=self.name,
                    instruction=row.get("problem_statement", ""),
                    payload={
                        "instance_id": row["instance_id"],
                        "repo": row.get("repo", ""),
                        "image": f"jefzda/sweap-images:{row['instance_id']}",
                        "test_cmd": row.get("test_cmd", ""),
                    },
                    # LOC-changed is a rough long-horizon proxy when present.
                    horizon_hint=int(row.get("changed_lines", 100)) if row.get("changed_lines") else 3,
                )
            )
        self._cache = specs
        return specs

    def tasks(self) -> Iterator[TaskSpec]:
        yield from self._load()

    def subset(self, n: int, long_horizon: bool = False) -> list[TaskSpec]:
        specs = self._load()
        if long_horizon:
            specs = sorted(specs, key=lambda s: (s.horizon_hint or 0), reverse=True)
        return specs[:n]

    def make_env(self, task: TaskSpec):
        p = task.payload

        def verify_fn(container: str) -> Outcome:
            from harness_rl.benchmarks._docker import subprocess as sp
            proc = sp.run(
                ["docker", "exec", container, "bash", "-lc", p.get("test_cmd", "true")],
                capture_output=True, text=True,
            )
            passed = proc.returncode == 0
            return Outcome(u_g=1.0 if passed else 0.0, passed_tests=int(passed), total_tests=1)

        return DockerEnv(image=p["image"], verify_fn=verify_fn,
                         container_name=f"hrl-swebp-{p['instance_id']}",
                         setup_cmds=["cd /testbed 2>/dev/null || true"])

    def system_prompt(self) -> str:
        return DEFAULT_SYSTEM_PROMPT + (
            "\nYou are resolving a GitHub issue in an existing repository. Inspect the "
            "codebase, implement a fix, and ensure the test suite passes without regressions."
        )
