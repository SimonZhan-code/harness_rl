"""LiveCodeBench adapter — self-contained competitive-programming problems with hidden tests.

Trainable, real reward, **NO Docker** (code runs via subprocess — see `_livecode.py`). Ideal
for small-model RL: fast, self-contained, deterministic checkable reward. Time-windowed
release versions guard against contamination — default `release_v6`.

HF dataset: `livecodebench/code_generation_lite` (trust_remote_code). Fields used:
`question_content`, `starter_code`, `difficulty`, `platform`, `public_test_cases`,
`private_test_cases` (base64+zlib), `metadata` (may carry `func_name` for functional problems).
"""
from __future__ import annotations

import json
import os
from typing import Iterator

from harness_rl.benchmarks._livecode import LiveCodeExecEnv, decode_tests
from harness_rl.types import Outcome, TaskSpec  # noqa: F401 (Outcome re-exported for parity)

_DIFF_HORIZON = {"easy": 1, "medium": 2, "hard": 3}

_LCB_SYSTEM = (
    "You are a competitive-programming agent in a sandboxed shell (no container). Solve the "
    "problem by writing your solution to `solution.py`, then verify it against the provided "
    "example(s). Respond with EXACTLY ONE action per turn:\n"
    "  - run a command:  ```bash\n  <cmd>\n  ```   (e.g. write solution.py with a heredoc, or "
    "run `python solution.py`)\n"
    "  - when the solution passes the examples: output the single line TASK_COMPLETE\n"
    "For stdin problems, read input from standard input and print the answer. For functional "
    "problems, define `class Solution` with the required method."
)


class LiveCodeBenchAdapter:
    name = "livecodebench"

    def __init__(self, version_tag: str = "release_v6",
                 hf_id: str = "livecodebench/code_generation_lite"):
        self.version_tag = version_tag
        self.hf_id = hf_id
        self._cache: list[TaskSpec] | None = None

    def _load(self) -> list[TaskSpec]:
        if self._cache is not None:
            return self._cache
        try:
            from datasets import load_dataset  # lazy .venv dep
        except Exception:
            return []
        ds = load_dataset(self.hf_id, split="test", version_tag=self.version_tag,
                          trust_remote_code=True)
        specs = []
        for row in ds:
            meta = row.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            fn_name = (meta or {}).get("func_name")
            difficulty = str(row.get("difficulty", "medium")).lower()
            starter = row.get("starter_code") or ""
            instr = row["question_content"]
            if starter:
                instr += f"\n\nStarter code:\n{starter}"
            specs.append(TaskSpec(
                task_id=f"lcb/{row.get('question_id', len(specs))}",
                benchmark=self.name,
                instruction=instr,
                payload={
                    "public_raw": row.get("public_test_cases"),
                    "private_raw": row.get("private_test_cases"),
                    "fn_name": fn_name,
                    "difficulty": difficulty,
                    "platform": row.get("platform"),
                },
                horizon_hint=_DIFF_HORIZON.get(difficulty, 2),
            ))
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
        return LiveCodeExecEnv(
            public_tests=decode_tests(p.get("public_raw")),
            private_tests=decode_tests(p.get("private_raw")),
            fn_name=p.get("fn_name"),
        )

    def system_prompt(self) -> str:
        return _LCB_SYSTEM
