"""SWE-bench Verified adapter — the 500-task human-verified subset (the standard SWE metric).

HF dataset: `princeton-nlp/SWE-bench_Verified` (split `test`; same schema as Lite/Pro:
`instance_id`, `repo`, `base_commit`, `problem_statement`, `FAIL_TO_PASS`, `PASS_TO_PASS`,
`environment_setup_commit`, `version`).

Two env paths (same as SWE-bench Lite):
  - `make_env` → **host-native `LocalShellEnv`** (clone the repo at `base_commit`, agent edits
    with bash). Validates INFERENCE with no container. `verify()` reports inference-ran.
  - `make_graded_env` → **`DockerEnv`** running FAIL_TO_PASS/PASS_TO_PASS for a real graded
    score — DEFERRED to a Docker-capable host.
"""
from __future__ import annotations

import json
from typing import Iterator

from harness_rl.benchmarks._docker import DockerEnv
from harness_rl.benchmarks.base import LocalShellEnv
from harness_rl.harness.agent import DEFAULT_SYSTEM_PROMPT
from harness_rl.types import Outcome, TaskSpec

_SYSTEM = DEFAULT_SYSTEM_PROMPT + (
    "\nYou are resolving a GitHub issue in the repository checked out in your working directory. "
    "Inspect the code, implement a minimal fix, and make the failing tests pass without breaking "
    "existing ones. Output TASK_COMPLETE when the fix is in place."
)


def _as_list(x) -> list[str]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return [x]
    return []


class SWEBenchVerifiedAdapter:
    name = "swebench_verified"

    def __init__(self, split: str = "test", hf_id: str = "princeton-nlp/SWE-bench_Verified"):
        self.split = split
        self.hf_id = hf_id
        self._cache: list[TaskSpec] | None = None

    def _load(self) -> list[TaskSpec]:
        if self._cache is not None:
            return self._cache
        try:
            from datasets import load_dataset  # lazy
        except Exception:
            return []
        ds = load_dataset(self.hf_id, split=self.split)
        specs = []
        for row in ds:
            specs.append(TaskSpec(
                task_id=f"swebv/{row['instance_id']}",
                benchmark=self.name,
                instruction=row.get("problem_statement", ""),
                payload={
                    "instance_id": row["instance_id"],
                    "repo": row.get("repo", ""),
                    "base_commit": row.get("base_commit", ""),
                    "fail_to_pass": _as_list(row.get("FAIL_TO_PASS")),
                    "pass_to_pass": _as_list(row.get("PASS_TO_PASS")),
                    "env_setup_commit": row.get("environment_setup_commit", ""),
                    "version": row.get("version", ""),
                },
                horizon_hint=2,
            ))
        self._cache = specs
        return specs

    def tasks(self) -> Iterator[TaskSpec]:
        yield from self._load()

    def subset(self, n: int, long_horizon: bool = False) -> list[TaskSpec]:
        return self._load()[:n]

    def make_env(self, task: TaskSpec):
        """Host-native inference: clone the repo at base_commit; agent edits with bash."""
        p = task.payload
        url = f"https://github.com/{p['repo']}.git"
        return LocalShellEnv(setup_cmds=[
            f"git clone --filter=blob:none {url} . 2>&1 | tail -1",
            f"git checkout {p['base_commit']} 2>&1 | tail -1",
        ], exec_timeout_s=300)

    def make_graded_env(self, task: TaskSpec):
        """DEFERRED graded path (Docker). Runs FAIL_TO_PASS/PASS_TO_PASS in the task image."""
        p = task.payload

        def verify_fn(container: str) -> Outcome:
            from harness_rl.benchmarks._docker import subprocess as sp
            tests = " ".join(p["fail_to_pass"] + p["pass_to_pass"])
            proc = sp.run(["docker", "exec", container, "bash", "-lc",
                           f"python -m pytest -q {tests}"], capture_output=True, text=True)
            passed = proc.returncode == 0
            return Outcome(u_g=1.0 if passed else 0.0, passed_tests=int(passed), total_tests=1)

        image = f"swebench/sweb.eval.x86_64.{p['instance_id'].replace('__', '_1776_')}:latest"
        return DockerEnv(image=image, verify_fn=verify_fn,
                         container_name=f"hrl-swebv-{p['instance_id']}")

    def system_prompt(self) -> str:
        return _SYSTEM
