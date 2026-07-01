"""The context-management policy Gamma — the ONLY learnable harness component.

Gamma decides, at each step, what enters the finite context window. It is the
harness-side controller (NOT a model-invoked tool), so the action set stays fixed.

Every concrete Gamma must:
  - build_context(...): produce the bounded message list handed to the policy.
  - on_step(...): update any external memory / scratchpad.
  - snapshot(): report what it kept/dropped/summarized/retrieved this step (for logging).

Concrete variants (G0..G3) are the seed items of the C1 provenance playbook.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness_rl.types import Action, GammaSnapshot, Message, Observation, Role, TaskSpec


@runtime_checkable
class ContextManager(Protocol):
    name: str

    def reset(self, task: TaskSpec, system_prompt: str) -> None:
        """Start a new episode; clear per-episode memory."""
        ...

    def build_context(self, budget_tokens: int) -> list[Message]:
        """Assemble the bounded message list for the next model call."""
        ...

    def on_step(self, observation: Observation, action: Action | None) -> None:
        """Record the latest turn into full history / memory."""
        ...

    def snapshot(self) -> GammaSnapshot:
        """Return what Gamma did when it last built context (for the trace)."""
        ...


def count_tokens(messages: list[Message], model: str = "cl100k_base") -> int:
    """Cheap token estimate. Uses tiktoken if available, else a ~4-chars/token heuristic.

    Kept dependency-light so the `.venv` layer needs no model weights.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(m.content or "")) for m in messages)
    except Exception:
        return sum(max(1, len(m.content or "") // 4) for m in messages)


class BaseContextManager:
    """Shared machinery: keeps the full history; subclasses override `build_context`.

    `self.history` is the complete, uncompressed turn list (o_0, a_0, o_1, ...).
    Subclasses read it to produce a *bounded* view.
    """
    name = "base"

    def __init__(self) -> None:
        self.task: TaskSpec | None = None
        self.system_prompt: str = ""
        self.history: list[Message] = []
        self._last_snapshot = GammaSnapshot()

    def reset(self, task: TaskSpec, system_prompt: str) -> None:
        self.task = task
        self.system_prompt = system_prompt
        self.history = [Message(role=Role.USER, content=task.instruction)]
        self._last_snapshot = GammaSnapshot()

    def on_step(self, observation: Observation, action: Action | None) -> None:
        if action is not None:
            self.history.append(Message(role=Role.ASSISTANT, content=action.raw))
        self.history.append(Message(role=Role.TOOL, content=observation.text))

    def snapshot(self) -> GammaSnapshot:
        return self._last_snapshot

    # Subclasses implement build_context and set self._last_snapshot there.
    def build_context(self, budget_tokens: int) -> list[Message]:  # pragma: no cover
        raise NotImplementedError
