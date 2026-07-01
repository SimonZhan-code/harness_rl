"""GameDevBench adapter (trainable gaming benchmark).

Repo: waynchi/gamedevbench (arXiv 2602.11103). Edit an existing Godot 4.x project to
implement a feature; verified by DETERMINISTIC Godot unit tests (Pass@1) → a clean
checkable reward, suitable for RL. Multimodal (editor-screenshot MCP + runtime video)
improves scores; v1 here runs TEXT-ONLY (GDScript/scene editing) — flip `multimodal=True`
to attach visual feedback once the VLM path is wired on vast.

Requires Godot 4.x in the task container.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from harness_rl.benchmarks._docker import DockerEnv
from harness_rl.harness.agent import DEFAULT_SYSTEM_PROMPT
from harness_rl.types import Outcome, TaskSpec

_GAMEDEV_SYSTEM = DEFAULT_SYSTEM_PROMPT + (
    "\nYou are editing a Godot 4.x game project (GDScript, scenes, shaders). Implement the "
    "requested feature/fix. The task is verified by Godot unit tests."
)


class GameDevBenchAdapter:
    name = "gamedev"

    def __init__(self, tasks_dir: str | None = None, multimodal: bool = False):
        self.tasks_dir = Path(tasks_dir or os.environ.get("GAMEDEV_TASKS_DIR", "./data/gamedevbench"))
        self.multimodal = multimodal  # v1 default: text-only

    def _iter_task_dirs(self) -> Iterator[Path]:
        if not self.tasks_dir.exists():
            return iter(())
        return (p for p in sorted(self.tasks_dir.iterdir()) if (p / "task.json").exists())

    def tasks(self) -> Iterator[TaskSpec]:
        import json
        for d in self._iter_task_dirs():
            meta = json.loads((d / "task.json").read_text())
            yield TaskSpec(
                task_id=f"gamedev/{d.name}",
                benchmark=self.name,
                instruction=meta.get("instruction", ""),
                payload={"task_dir": str(d), "image": meta.get("image", "godot4-task"),
                         "category": meta.get("category", "")},
                horizon_hint=int(meta.get("files_to_edit", 5)),
            )

    def subset(self, n: int, long_horizon: bool = False) -> list[TaskSpec]:
        specs = list(self.tasks())
        if long_horizon:
            specs.sort(key=lambda s: (s.horizon_hint or 0), reverse=True)
        return specs[:n]

    def make_env(self, task: TaskSpec):
        d = Path(task.payload["task_dir"])

        def verify_fn(container: str) -> Outcome:
            from harness_rl.benchmarks._docker import subprocess as sp
            # Godot headless test run; convention: /task/run_tests.sh emits `PASS n/m`.
            proc = sp.run(
                ["docker", "exec", container, "bash", "-lc", "bash /task/run_tests.sh"],
                capture_output=True, text=True,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            passed = proc.returncode == 0
            return Outcome(u_g=1.0 if passed else 0.0, passed_tests=int(passed), total_tests=1,
                           terminated_reason="submit")

        return DockerEnv(image=task.payload.get("image", "godot4-task"),
                         verify_fn=verify_fn, container_name=f"hrl-gamedev-{d.name}",
                         setup_cmds=["cd /task 2>/dev/null || true"])

    def system_prompt(self) -> str:
        return _GAMEDEV_SYSTEM
