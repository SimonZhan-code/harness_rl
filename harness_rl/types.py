"""Core shared types for the Harness+RL project.

These types are deliberately framework-agnostic and are reused *unchanged* between
Step 1 (inference-only probe) and Step 2 (agentic RL). Nothing here imports torch,
vLLM, or any GPU dependency — the orchestration layer lives in `.venv`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """One OpenAI-compatible chat message."""
    role: Role
    content: str
    # Optional structured payload (e.g., multimodal parts for GameDevBench VLM runs).
    parts: list[dict[str, Any]] | None = None

    def to_openai(self) -> dict[str, Any]:
        if self.parts is not None:
            return {"role": self.role.value, "content": self.parts}
        return {"role": self.role.value, "content": self.content}


@dataclass
class TaskSpec:
    """A single benchmark task. `g = (d, s0, V)` in the formulation.

    - `instruction` (d): the natural-language goal.
    - `benchmark`: which adapter owns verification (V) and the initial state (s0).
    - `payload`: adapter-specific handle (container id, repo path, task dir, ...).
    """
    task_id: str
    benchmark: str
    instruction: str
    payload: dict[str, Any] = field(default_factory=dict)
    # Coarse horizon hint used for slicing "long-horizon" subsets in the probe.
    horizon_hint: int | None = None


@dataclass
class Action:
    """A parsed tool call. Tool set is FIXED per project scope."""
    tool: str                      # e.g. "bash", "submit"
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""                  # the raw model text the parser consumed


@dataclass
class Observation:
    """Environment response after an action (tool output, rendered by Omega_0)."""
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GammaSnapshot:
    """What the context-management policy Gamma did at a step.

    Logging THIS (not just the trajectory) is what makes harness-vs-capability
    attribution computable later. See implementation_plan.md trace-schema note.
    """
    kept: list[str] = field(default_factory=list)        # ids/refs kept in-context
    dropped: list[str] = field(default_factory=list)     # dropped this step
    summarized: list[str] = field(default_factory=list)  # compacted refs
    retrieved: list[str] = field(default_factory=list)   # pulled from memory store
    tokens_in_ctx: int = 0
    notes: dict[str, Any] = field(default_factory=dict)  # variant-specific detail


@dataclass
class Step:
    """One turn of the episode."""
    t: int
    observation: str
    gamma_snapshot: GammaSnapshot
    action: Action | None
    tool_result: str
    # Optional per-step process-judge score q_t (filled later by the grounded judge).
    q_t: float | None = None


@dataclass
class Outcome:
    """Terminal result of an episode."""
    u_g: float                     # checkable reward in [0, 1]
    passed_tests: int | None = None
    total_tests: int | None = None
    cost_tokens: int = 0
    turns: int = 0
    terminated_reason: str = ""    # "submit" | "budget" | "error"


@dataclass
class Trace:
    """A full rollout, serialized to one JSONL record. Shared Step1/Step2 artifact."""
    task_id: str
    benchmark: str
    model: str
    gamma_variant: str
    steps: list[Step] = field(default_factory=list)
    outcome: Outcome | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetCaps:
    """Episode budget. Part of the fixed remainder Omega_0."""
    max_turns: int = 40
    max_tokens: int = 200_000
    context_window: int = 32_000   # W: the finite window Gamma must fit into
    wall_clock_s: int | None = None
