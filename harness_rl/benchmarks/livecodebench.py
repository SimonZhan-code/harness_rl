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

# version_tag -> raw JSONL file in the HF dataset repo (datasets>=3 dropped loading scripts,
# so we read the per-version jsonl directly via huggingface_hub).
_VERSION_FILE = {
    "release_v1": "test.jsonl", "release_v2": "test2.jsonl", "release_v3": "test3.jsonl",
    "release_v4": "test4.jsonl", "release_v5": "test5.jsonl", "release_v6": "test6.jsonl",
}

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

# A4(a): the prompt above gates submission on "when the solution passes the examples" — a state a
# small model may never reach, so it never submits (MEASURED: Qwen2.5-Coder-3B emitted TASK_COMPLETE
# in 0/20 turns while 0/20 outputs failed to parse). This variant makes submission unconditional and
# explicitly cheap, so the rollout can terminate normally instead of exhausting its turn budget.
_LCB_SYSTEM_EAGER_SUBMIT = _LCB_SYSTEM.replace(
    "  - when the solution passes the examples: output the single line TASK_COMPLETE\n",
    "  - output the single line TASK_COMPLETE as soon as solution.py holds your best attempt.\n"
    "    ALWAYS finish with TASK_COMPLETE — do not keep iterating once you have a candidate; a\n"
    "    submitted imperfect solution is graded, an unsubmitted one wastes the episode.\n",
)


class LiveCodeBenchAdapter:
    name = "livecodebench"

    def __init__(self, version_tag: str = "release_v6",
                 hf_id: str = "livecodebench/code_generation_lite",
                 cumulative: bool = False, eager_submit: bool = False):
        """`cumulative=True` unions every release up to `version_tag` instead of reading only that
        release's own JSONL.

        MEASURED task counts (release file -> tasks): test.jsonl 400, test2 111, test3 101,
        test4 101, test5 167, **test6 175**. The releases are INCREMENTAL slices, not cumulative, so
        the default (`release_v6` alone) trains on **175 tasks of which 46% are hard and only 25%
        easy** — a tiny, hard-skewed pool. Cumulative gives **1,055** unique problems.
        """
        self.version_tag = version_tag
        self.hf_id = hf_id
        self.cumulative = cumulative
        self.eager_submit = eager_submit
        self._cache: list[TaskSpec] | None = None

    def _version_files(self) -> list[str]:
        order = ["release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_v6"]
        if not self.cumulative:
            return [_VERSION_FILE.get(self.version_tag, "test6.jsonl")]
        upto = order.index(self.version_tag) + 1 if self.version_tag in order else len(order)
        return [_VERSION_FILE[v] for v in order[:upto]]

    def _iter_rows(self):
        """Yield problem dicts — read the per-version JSONL directly (datasets>=3 dropped the
        loading script). Falls back to `load_dataset` for older datasets versions."""
        import json
        seen_ids: set = set()
        for fname in self._version_files():
            for row in self._iter_file(fname):
                qid = row.get("question_id")
                if qid is not None and qid in seen_ids:
                    continue          # dedup across releases
                if qid is not None:
                    seen_ids.add(qid)
                yield row

    def _iter_file(self, fname: str):
        import json
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self.hf_id, fname, repo_type="dataset")
            with open(path) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
            return
        except Exception:
            pass
        try:  # datasets<3 with the loading script
            from datasets import load_dataset
            yield from load_dataset(self.hf_id, split="test", version_tag=self.version_tag,
                                    trust_remote_code=True)
        except Exception:
            return

    def _load(self) -> list[TaskSpec]:
        if self._cache is not None:
            return self._cache
        specs = []
        for row in self._iter_rows():
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

    def subset(self, n: int, long_horizon: bool = False, difficulty: str | None = None,
               shuffle_seed: int | None = None) -> list[TaskSpec]:
        """Select `n` tasks.

        ⚠️ With no `difficulty` and no `shuffle_seed` this returns a DETERMINISTIC FILE-ORDER PREFIX
        (`specs[:n]`) — the historical behaviour, kept only for reproducibility. Since the JSONL is
        ordered by contest, that prefix is a narrow, correlated slice: every batch looks alike, and
        with `long_horizon=True` it is sorted hardest-first. For training, pass `shuffle_seed` (and
        usually `difficulty`) so batches are diverse.

        `difficulty` accepts one tier or a comma-separated list, e.g. "easy,medium".
        """
        specs = self._load()
        if difficulty:
            want = {d.strip().lower() for d in difficulty.split(",") if d.strip()}
            specs = [s for s in specs if str(s.payload.get("difficulty", "")).lower() in want]
        if shuffle_seed is not None:
            import random
            specs = list(specs)
            random.Random(shuffle_seed).shuffle(specs)
        elif long_horizon:
            specs = sorted(specs, key=lambda s: (s.horizon_hint or 0), reverse=True)
        return specs[:n]

    def difficulty_census(self) -> dict:
        """Counts by difficulty / platform / answer format — the pool's diversity, in one call."""
        import collections
        specs = self._load()
        diff = collections.Counter(str(s.payload.get("difficulty", "?")).lower() for s in specs)
        plat = collections.Counter(str(s.payload.get("platform", "?")).lower() for s in specs)
        func = sum(1 for s in specs if s.payload.get("fn_name"))
        return {"total": len(specs), "unique_ids": len({s.task_id for s in specs}),
                "difficulty": dict(diff), "platform": dict(plat),
                "functional": func, "stdin": len(specs) - func,
                "cumulative": self.cumulative, "version_tag": self.version_tag}

    def system_prompt(self) -> str:
        return _LCB_SYSTEM_EAGER_SUBMIT if self.eager_submit else _LCB_SYSTEM

    def make_env(self, task: TaskSpec):
        p = task.payload
        return LiveCodeExecEnv(
            public_tests=decode_tests(p.get("public_raw")),
            private_tests=decode_tests(p.get("private_raw")),
            fn_name=p.get("fn_name"),
        )
