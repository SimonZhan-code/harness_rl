"""Trace (de)serialization + JSONL writer.

The on-disk schema matches implementation_plan.md and is the SHARED artifact between
Step 1 (probe) and Step 2 (RL): it must capture Gamma's per-step decisions so that the
2x2 attribution and the localized RL advantage are computable downstream.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from harness_rl.types import (
    Action,
    GammaSnapshot,
    Outcome,
    Step,
    Trace,
)


def trace_to_dict(tr: Trace) -> dict[str, Any]:
    return {
        "task_id": tr.task_id,
        "benchmark": tr.benchmark,
        "model": tr.model,
        "gamma_variant": tr.gamma_variant,
        "steps": [
            {
                "t": s.t,
                "observation": s.observation,
                "gamma_snapshot": asdict(s.gamma_snapshot),
                "action": (asdict(s.action) if s.action else None),
                "tool_result": s.tool_result,
                "q_t": s.q_t,
            }
            for s in tr.steps
        ],
        "outcome": (asdict(tr.outcome) if tr.outcome else None),
        "extra": tr.extra,
    }


def dict_to_trace(d: dict[str, Any]) -> Trace:
    steps = [
        Step(
            t=s["t"],
            observation=s["observation"],
            gamma_snapshot=GammaSnapshot(**s["gamma_snapshot"]),
            action=(Action(**s["action"]) if s.get("action") else None),
            tool_result=s["tool_result"],
            q_t=s.get("q_t"),
        )
        for s in d.get("steps", [])
    ]
    outcome = Outcome(**d["outcome"]) if d.get("outcome") else None
    return Trace(
        task_id=d["task_id"],
        benchmark=d["benchmark"],
        model=d["model"],
        gamma_variant=d["gamma_variant"],
        steps=steps,
        outcome=outcome,
        extra=d.get("extra", {}),
    )


class TraceWriter:
    """Append-only JSONL writer. One Trace per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, tr: Trace) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(trace_to_dict(tr)) + "\n")

    def read_all(self) -> list[Trace]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [dict_to_trace(json.loads(line)) for line in f if line.strip()]
