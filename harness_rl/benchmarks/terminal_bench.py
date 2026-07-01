"""Terminal-Bench 2.0 adapter (primary probe benchmark).

Repo: laude-institute/terminal-bench(-2). Tasks live under a `tasks/` directory, one folder
per task. Point `TB2_TASKS_DIR` at that folder (a git checkout of the benchmark).

Known per-task layout (verify field names against the installed benchmark on vast):
    tasks/<task-id>/
        task.yaml            # instruction, difficulty, tags, max_agent_timeout_sec, max_test_timeout_sec
        Dockerfile           # (or docker-compose.yaml) — the task environment
        solution.sh          # oracle (unused by us)
        tests/               # verification (pytest test_outputs.py, run via run-tests.sh)

Flow: build the task image from its Dockerfile → run our harness (fixed `bash` tool) in a
container → after submit, run the task's tests; pass (exit 0) → u_g = 1.0 else 0.0.

Docker work happens on vast; locally only `task.yaml` parsing runs.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterator

from harness_rl.benchmarks._docker import DockerEnv, docker_available
from harness_rl.harness.agent import DEFAULT_SYSTEM_PROMPT
from harness_rl.types import Outcome, TaskSpec

_DIFFICULTY_HORIZON = {"easy": 1, "medium": 2, "hard": 3}

# Convention for running a task's verification suite inside the container.
# TODO(vast): confirm against terminal-bench-2 — it may copy tests/ then `pytest`,
# or provide a `run-tests.sh`. Both are handled below (run-tests.sh preferred if present).
_TEST_CMD = "if [ -f /tests/run-tests.sh ]; then bash /tests/run-tests.sh; " \
            "else python -m pytest -q /tests; fi"


class TerminalBench2Adapter:
    name = "terminal_bench_2"

    def __init__(self, tasks_dir: str | None = None, build_images: bool = True):
        # Point at the benchmark checkout's `tasks/` directory.
        self.tasks_dir = Path(tasks_dir or os.environ.get("TB2_TASKS_DIR", "./data/terminal-bench/tasks"))
        self.build_images = build_images

    # ---- task enumeration (works locally from a checkout) ----
    def _iter_task_dirs(self) -> Iterator[Path]:
        if not self.tasks_dir.exists():
            return iter(())
        return (p for p in sorted(self.tasks_dir.iterdir())
                if p.is_dir() and (p / "task.yaml").exists())

    def _load_spec(self, task_dir: Path) -> TaskSpec:
        import yaml  # lazy; pyyaml is a .venv dep

        meta = yaml.safe_load((task_dir / "task.yaml").read_text()) or {}
        # TODO(vast): the instruction key is `instruction`; some versions use `description`.
        instruction = meta.get("instruction") or meta.get("description") or ""
        difficulty = str(meta.get("difficulty", "medium")).lower()
        return TaskSpec(
            task_id=f"tb2/{task_dir.name}",
            benchmark=self.name,
            instruction=instruction,
            payload={
                "task_dir": str(task_dir),
                "image": f"tb2/{task_dir.name}:latest",
                "test_timeout_s": int(meta.get("max_test_timeout_sec", 600)),
                "agent_timeout_s": int(meta.get("max_agent_timeout_sec", 900)),
                "tags": meta.get("tags", []),
            },
            horizon_hint=_DIFFICULTY_HORIZON.get(difficulty, 2),
        )

    def tasks(self) -> Iterator[TaskSpec]:
        for d in self._iter_task_dirs():
            yield self._load_spec(d)

    def subset(self, n: int, long_horizon: bool = False) -> list[TaskSpec]:
        specs = list(self.tasks())
        if long_horizon:
            specs.sort(key=lambda s: (s.horizon_hint or 0), reverse=True)
        return specs[:n]

    # ---- environment (runs on vast.ai) ----
    def _ensure_image(self, task_dir: Path, image: str) -> None:
        """Build the task image from its Dockerfile if not already present."""
        if not self.build_images or not docker_available():
            return
        exists = subprocess.run(["docker", "image", "inspect", image],
                                capture_output=True).returncode == 0
        if not exists and (task_dir / "Dockerfile").exists():
            subprocess.run(["docker", "build", "-t", image, str(task_dir)],
                           check=True, capture_output=True)

    def make_env(self, task: TaskSpec):
        task_dir = Path(task.payload["task_dir"])
        image = task.payload["image"]
        test_timeout = task.payload.get("test_timeout_s", 600)
        self._ensure_image(task_dir, image)

        def verify_fn(container: str) -> Outcome:
            # Copy the task's tests into the container, then run them; exit 0 == pass.
            subprocess.run(["docker", "cp", str(task_dir / "tests"), f"{container}:/tests"],
                           capture_output=True)
            proc = subprocess.run(
                ["docker", "exec", container, "bash", "-lc", _TEST_CMD],
                capture_output=True, text=True, timeout=test_timeout,
            )
            passed = proc.returncode == 0
            return Outcome(u_g=1.0 if passed else 0.0, passed_tests=int(passed), total_tests=1)

        return DockerEnv(
            image=image,
            verify_fn=verify_fn,
            container_name=f"hrl-tb2-{task_dir.name}",
            setup_cmds=["cd /app 2>/dev/null || cd / "],
            exec_timeout_s=task.payload.get("agent_timeout_s", 900),
        )

    def system_prompt(self) -> str:
        return DEFAULT_SYSTEM_PROMPT
