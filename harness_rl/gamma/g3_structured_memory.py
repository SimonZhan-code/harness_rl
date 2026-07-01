"""G3 — structured scratchpad memory.

Maintains an explicit, structured working memory that is deterministically updated from
observations/actions (no LLM call): files touched, decisions made, open TODOs, and the
latest error. The context is: system + task + rendered scratchpad + short recency window.

This is the most "interpretable / trimmable" Gamma — each scratchpad field is an atomic
item whose marginal value can be ablated (feeds the C1 provenance playbook cleanly).
"""
from __future__ import annotations

import re

from harness_rl.gamma.base import BaseContextManager, count_tokens
from harness_rl.types import Action, GammaSnapshot, Message, Observation, Role

_FILE_RE = re.compile(r"\b([\w./-]+\.\w{1,6})\b")
_ERROR_RE = re.compile(r"(error|exception|traceback|failed|assert)", re.IGNORECASE)


class G3StructuredMemory(BaseContextManager):
    name = "G3_structured_memory"

    def __init__(self, recency_turns: int = 4, max_files: int = 30, max_todos: int = 20) -> None:
        super().__init__()
        self.recency_turns = recency_turns
        self.max_files = max_files
        self.max_todos = max_todos
        self._files: dict[str, int] = {}     # path -> times touched
        self._decisions: list[str] = []
        self._todos: list[str] = []
        self._last_error: str = ""

    def reset(self, task, system_prompt: str) -> None:  # type: ignore[override]
        super().reset(task, system_prompt)
        self._files, self._decisions, self._todos, self._last_error = {}, [], [], ""

    def on_step(self, observation: Observation, action: Action | None) -> None:
        super().on_step(observation, action)
        text = (action.raw if action else "") + "\n" + observation.text
        for path in set(_FILE_RE.findall(text)):
            self._files[path] = self._files.get(path, 0) + 1
        if _ERROR_RE.search(observation.text):
            self._last_error = observation.text[-800:]
        if action and action.tool == "bash":
            # Treat a shell command as a lightweight "decision" record.
            cmd = action.args.get("cmd", "").strip().splitlines()[0][:120] if action.args else ""
            if cmd:
                self._decisions.append(cmd)

    def _render_scratchpad(self) -> str:
        top_files = sorted(self._files.items(), key=lambda kv: kv[1], reverse=True)[: self.max_files]
        lines = ["## Working memory"]
        if top_files:
            lines.append("Files touched: " + ", ".join(f"{p}(x{n})" for p, n in top_files))
        if self._decisions:
            lines.append("Recent actions:\n  - " + "\n  - ".join(self._decisions[-10:]))
        if self._todos:
            lines.append("Open TODOs:\n  - " + "\n  - ".join(self._todos[: self.max_todos]))
        if self._last_error:
            lines.append(f"Last error:\n{self._last_error}")
        return "\n".join(lines)

    def build_context(self, budget_tokens: int) -> list[Message]:
        head = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        task_msg = self.history[0]
        tail = self.history[1:]
        recent = tail[-self.recency_turns:] if self.recency_turns else []
        recent_start = len(tail) - len(recent)

        scratch = Message(role=Role.USER, content=self._render_scratchpad())
        messages = head + [task_msg, scratch] + list(recent)
        used = count_tokens(messages)

        self._last_snapshot = GammaSnapshot(
            kept=[f"turn:{i}" for i in range(recent_start, len(tail))],
            summarized=["scratchpad"],
            dropped=[f"turn:{i}" for i in range(recent_start)],
            tokens_in_ctx=used,
            notes={
                "n_files": len(self._files),
                "n_decisions": len(self._decisions),
                "has_error": bool(self._last_error),
            },
        )
        return messages
