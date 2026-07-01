"""WebGameBench adapter (EVAL-ONLY — noisy LLM judge, not for RL reward).

arXiv 2605.17637. Generate a playable browser-native game from a frozen structured spec;
evaluated by a Playwright browser controller + LLM judge into Excellent/Usable/Unusable
(~50% three-way human agreement). Because the reward is a noisy learned judge, this
benchmark is used ONLY for evaluation, never as an RL training signal.

`verify()` maps the judge label to a scalar for reporting only:
Excellent=1.0, Usable=0.5, Unusable=0.0.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from harness_rl.harness.agent import DEFAULT_SYSTEM_PROMPT
from harness_rl.types import Action, Observation, Outcome, TaskSpec

_LABEL_TO_SCALAR = {"excellent": 1.0, "usable": 0.5, "unusable": 0.0}

_WEBGAME_SYSTEM = DEFAULT_SYSTEM_PROMPT + (
    "\nYou are building a playable browser-native game from a product spec using standard "
    "web tech (HTML/JS/Canvas). Deliver source that runs at a served URL."
)


class WebGameBenchAdapter:
    name = "webgame"
    eval_only = True  # guard: RL trainers must skip this benchmark

    def __init__(self, tasks_dir: str | None = None):
        self.tasks_dir = Path(tasks_dir or os.environ.get("WEBGAME_TASKS_DIR", "./data/webgamebench"))

    def tasks(self) -> Iterator[TaskSpec]:
        import json
        if not self.tasks_dir.exists():
            return
        for p in sorted(self.tasks_dir.glob("*.json")):
            meta = json.loads(p.read_text())
            yield TaskSpec(
                task_id=f"webgame/{p.stem}",
                benchmark=self.name,
                instruction=meta.get("spec", ""),
                payload={"spec_path": str(p), "difficulty": meta.get("difficulty", "D2")},
                horizon_hint=25,
            )

    def subset(self, n: int, long_horizon: bool = False) -> list[TaskSpec]:
        return list(self.tasks())[:n]

    def make_env(self, task: TaskSpec):
        raise NotImplementedError(
            "WebGameBench requires the Playwright + LLM-judge harness (vast.ai). "
            "See judge integration; eval-only, not used in the Step 1 Gamma probe by default."
        )

    def judge_to_outcome(self, label: str) -> Outcome:
        return Outcome(u_g=_LABEL_TO_SCALAR.get(label.lower(), 0.0), terminated_reason="judge")

    def system_prompt(self) -> str:
        return _WEBGAME_SYSTEM
