"""Benchmark adapter interface.

Each adapter supplies the FIXED environment (Omega_0): initial state s0, action
execution, termination, and the checkable verifier V producing u_g in [0,1]. The tool
set is fixed here; only Gamma varies across runs.

Adapters are thin wrappers over the upstream benchmark's Docker sandboxes. The heavy
Docker/GPU work is deferred to vast.ai; locally these are import-clean and testable with
the in-memory `EnvStub`.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from harness_rl.types import Action, Observation, Outcome, TaskSpec


@runtime_checkable
class Environment(Protocol):
    """One task's live environment. Created by an adapter, driven by the harness."""

    def reset(self) -> Observation:
        """Initialize s0 and return the first observation."""
        ...

    def execute(self, action: Action) -> Observation:
        """Apply an action (fixed tool), return the resulting observation."""
        ...

    def is_done(self, action: Action | None) -> bool:
        """True if the episode should end (e.g. `submit`)."""
        ...

    def verify(self) -> Outcome:
        """Run the checkable verifier V on the final state → u_g in [0,1]."""
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class BenchmarkAdapter(Protocol):
    name: str

    def tasks(self) -> Iterator[TaskSpec]:
        """Enumerate tasks (full set)."""
        ...

    def subset(self, n: int, long_horizon: bool = False) -> list[TaskSpec]:
        """A small slice for the probe; `long_horizon` prefers high horizon_hint."""
        ...

    def make_env(self, task: TaskSpec) -> Environment:
        """Build the live environment for a task."""
        ...

    def system_prompt(self) -> str:
        """Benchmark-specific fixed system prompt (part of frozen Pi_0)."""
        ...


class EnvStub:
    """In-memory environment for local no-GPU tests.

    Executes a scripted sequence: each `execute` returns the next canned observation;
    `verify` returns a preset outcome. Lets us exercise harness+gamma+logging offline.
    """

    def __init__(self, observations: list[str], final_u_g: float = 1.0, done_tool: str = "submit"):
        self._obs = observations
        self._i = 0
        self._final = final_u_g
        self._done_tool = done_tool

    def reset(self) -> Observation:
        self._i = 0
        return Observation(text=self._obs[0] if self._obs else "ready")

    def execute(self, action: Action) -> Observation:
        self._i += 1
        text = self._obs[self._i] if self._i < len(self._obs) else f"(ran: {action.raw[:80]})"
        return Observation(text=text)

    def is_done(self, action: Action | None) -> bool:
        return bool(action and action.tool == self._done_tool)

    def verify(self) -> Outcome:
        return Outcome(u_g=self._final, passed_tests=int(self._final), total_tests=1,
                       terminated_reason="submit")

    def close(self) -> None:
        pass
