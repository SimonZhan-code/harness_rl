"""GameCraft-Bench adapter — build playable Godot games from a spec (eval-only).

Source: github.com/tongxuluo/gamecraft-bench (arXiv 2606.17861) — 140 Godot 4.6.2 tasks across
15 game families. Official grading: launch the game under xvfb, replay demo traces (xdotool),
and score with a **multimodal rubric judge** (Core Mechanics / Content Depth / Functional
Visuals / Art). The judge is a soft/learned signal → **eval-only** (not an RL reward).

Here we support **host-native inference** (agent writes a Godot/GDScript project with the fixed
bash tool). The full Godot+xvfb+xdotool+judge grader is **DEFERRED** (heavy setup; see the
upstream repo). No HF dataset — task specs come from a checkout dir (`GAMECRAFT_TASKS_DIR`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from harness_rl.benchmarks.base import LocalShellEnv
from harness_rl.harness.agent import DEFAULT_SYSTEM_PROMPT
from harness_rl.types import TaskSpec

_SYSTEM = DEFAULT_SYSTEM_PROMPT + (
    "\nYou are building a complete, playable Godot 4.x game from the given specification. Create "
    "the project files (GDScript scripts, scenes, a project.godot) under your working directory. "
    "Implement the core gameplay loop, rules, and win/loss conditions. Output TASK_COMPLETE when "
    "the project is in place."
)


class GameCraftBenchAdapter:
    name = "gamecraft"
    eval_only = True  # rubric judge → RL trainers must skip this benchmark

    def __init__(self, tasks_dir: str | None = None):
        self.tasks_dir = Path(tasks_dir or os.environ.get("GAMECRAFT_TASKS_DIR",
                                                           "./data/gamecraft-bench/tasks"))

    def _iter_spec_files(self) -> Iterator[Path]:
        if not self.tasks_dir.exists():
            return iter(())
        return (p for p in sorted(self.tasks_dir.iterdir())
                if p.is_file() and p.suffix in {".json", ".md", ".txt", ".yaml"})

    def tasks(self) -> Iterator[TaskSpec]:
        for p in self._iter_spec_files():
            if p.suffix == ".json":
                meta = json.loads(p.read_text())
                instr = meta.get("spec") or meta.get("instruction") or ""
                family = meta.get("family", "")
            else:
                instr = p.read_text()
                family = ""
            yield TaskSpec(
                task_id=f"gamecraft/{p.stem}",
                benchmark=self.name,
                instruction=instr,
                payload={"spec_path": str(p), "family": family},
                horizon_hint=3,  # full games = long-horizon
            )

    def subset(self, n: int, long_horizon: bool = False) -> list[TaskSpec]:
        return list(self.tasks())[:n]

    def make_env(self, task: TaskSpec):
        """Host-native inference: agent writes a Godot project. Graded judge harness deferred."""
        return LocalShellEnv(exec_timeout_s=180)

    def system_prompt(self) -> str:
        return _SYSTEM
