"""G2 — periodic LLM compaction of stale history.

Keeps a recency window verbatim; older turns are periodically summarized into a rolling
"summary so far" block via a *summarizer* model (an OpenAI-compatible endpoint). This
mirrors OpenHands' LLMSummarizingCondenser.

The summarizer is injected as a callable so the `.venv` layer stays model-free; in tests
we pass a stub summarizer. On vast it points at the same (or a cheaper) served model.
"""
from __future__ import annotations

from typing import Callable

from harness_rl.gamma.base import BaseContextManager, count_tokens
from harness_rl.types import GammaSnapshot, Message, Role

Summarizer = Callable[[list[Message]], str]


def _identity_summarizer(msgs: list[Message]) -> str:
    """Fallback used when no summarizer is provided (concatenate + hard truncate)."""
    joined = "\n".join(f"[{m.role.value}] {m.content}" for m in msgs)
    return joined[:2000]


class G2Summarize(BaseContextManager):
    name = "G2_summarize"

    def __init__(
        self,
        recency_turns: int = 6,
        compact_every: int = 4,
        summarizer: Summarizer | None = None,
    ) -> None:
        super().__init__()
        self.recency_turns = recency_turns
        self.compact_every = compact_every
        self.summarizer = summarizer or _identity_summarizer
        self._summary = ""
        self._summarized_upto = 0  # index into tail that is folded into _summary

    def reset(self, task, system_prompt: str) -> None:  # type: ignore[override]
        super().reset(task, system_prompt)
        self._summary = ""
        self._summarized_upto = 0

    def build_context(self, budget_tokens: int) -> list[Message]:
        head = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        task_msg = self.history[0]
        tail = self.history[1:]

        recent_start = max(self._summarized_upto, len(tail) - self.recency_turns)
        # Fold any newly-stale turns into the rolling summary in chunks.
        newly_stale = tail[self._summarized_upto:recent_start]
        summarized_refs: list[str] = []
        if len(newly_stale) >= self.compact_every or (newly_stale and recent_start == len(tail)):
            chunk_summary = self.summarizer(newly_stale)
            self._summary = (self._summary + "\n" + chunk_summary).strip()
            summarized_refs = [f"turn:{self._summarized_upto + j}" for j in range(len(newly_stale))]
            self._summarized_upto = recent_start

        recent = tail[recent_start:]
        summary_block = (
            [Message(role=Role.USER, content=f"[summary of earlier work]\n{self._summary}")]
            if self._summary
            else []
        )
        messages = head + [task_msg] + summary_block + list(recent)
        used = count_tokens(messages)

        self._last_snapshot = GammaSnapshot(
            kept=[f"turn:{i}" for i in range(recent_start, len(tail))],
            summarized=summarized_refs,
            dropped=[],  # nothing is dropped outright; stale turns are compacted
            tokens_in_ctx=used,
            notes={"summary_chars": len(self._summary), "summarized_upto": self._summarized_upto},
        )
        return messages
