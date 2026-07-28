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
        compact_at_tokens: int | None = None,
        summary_mode: str = "append",
    ) -> None:
        super().__init__()
        self.recency_turns = recency_turns
        self.compact_every = compact_every
        self.summarizer = summarizer or _identity_summarizer
        # SUPO: token-threshold L that also triggers compaction ("summarize when the context is
        # compact"); and summary_mode="replace" so each summary is a fresh consolidated digest of
        # [old summary + newly-stale] (bounded context) rather than an ever-growing append.
        self.compact_at_tokens = compact_at_tokens
        self.summary_mode = summary_mode
        self._summary = ""
        self._summarized_upto = 0  # index into tail that is folded into _summary

    def reset(self, task, system_prompt: str) -> None:  # type: ignore[override]
        super().reset(task, system_prompt)
        self._summary = ""
        self._summarized_upto = 0

    def _summary_block(self) -> list[Message]:
        return ([Message(role=Role.USER, content=f"[summary of earlier work]\n{self._summary}")]
                if self._summary else [])

    def _ctx_tokens(self, head, task_msg, recent_start: int) -> int:
        recent = self.history[1:][recent_start:]
        return count_tokens(head + [task_msg] + self._summary_block() + list(recent))

    def build_context(self, budget_tokens: int) -> list[Message]:
        head = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        task_msg = self.history[0]
        tail = self.history[1:]

        recent_start = max(self._summarized_upto, len(tail) - self.recency_turns)

        # SUPO token trigger: if the assembled context exceeds L, fold older recency turns into
        # the stale set until we're back under L (keeping at least one recent turn).
        over_L = (self.compact_at_tokens is not None
                  and self._ctx_tokens(head, task_msg, recent_start) > self.compact_at_tokens)
        while over_L and recent_start < len(tail) - 1:
            recent_start += 1
            over_L = self._ctx_tokens(head, task_msg, recent_start) > self.compact_at_tokens

        newly_stale = tail[self._summarized_upto:recent_start]
        summarized_refs: list[str] = []
        turn_trigger = len(newly_stale) >= self.compact_every or (newly_stale and recent_start == len(tail))
        if newly_stale and (turn_trigger or over_L):
            if self.summary_mode == "replace":
                to_sum = self._summary_block() + newly_stale       # consolidate everything so far
                self._summary = self.summarizer(to_sum).strip()    # fresh digest replaces old
            else:
                self._summary = (self._summary + "\n" + self.summarizer(newly_stale)).strip()
            summarized_refs = [f"turn:{self._summarized_upto + j}" for j in range(len(newly_stale))]
            self._summarized_upto = recent_start

        recent = tail[recent_start:]
        summary_block = self._summary_block()
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
