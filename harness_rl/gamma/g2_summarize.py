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

    def _plan(self, L: int | None = None) -> tuple[int, list[Message], bool]:
        """Pure (no-mutation) compaction plan: returns (recent_start, newly_stale, should_compact).

        `recent_start` = index into `tail` where the verbatim recency window starts; `newly_stale`
        = tail turns to fold into the summary this step; `should_compact` = whether a summary fires
        (>= compact_every stale turns, all-remaining-stale, or context over the token threshold L).
        Shared by `build_context` (which mutates) and the tree driver's external-compaction hooks.
        """
        head = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        task_msg = self.history[0]
        tail = self.history[1:]
        L = self.compact_at_tokens if L is None else L

        recent_start = max(self._summarized_upto, len(tail) - self.recency_turns)
        over_L = (L is not None and self._ctx_tokens(head, task_msg, recent_start) > L)
        while over_L and recent_start < len(tail) - 1:
            recent_start += 1
            over_L = self._ctx_tokens(head, task_msg, recent_start) > L

        newly_stale = tail[self._summarized_upto:recent_start]
        turn_trigger = (len(newly_stale) >= self.compact_every
                        or (bool(newly_stale) and recent_start == len(tail)))
        return recent_start, newly_stale, bool(newly_stale) and (turn_trigger or over_L)

    def pending_compaction(self, L: int | None = None) -> list[Message] | None:
        """Tree hook — if a summary should fire now, return the message list that WOULD be
        summarized (`[summary block] + newly-stale`, replace-mode), else None. No mutation. The
        tree driver samples B summaries from this and applies each via `apply_summary` on a clone."""
        _, newly_stale, should = self._plan(L)
        if not should:
            return None
        return self._summary_block() + list(newly_stale)

    def apply_summary(self, summary: str, L: int | None = None) -> None:
        """Tree hook — install an externally-sampled summary (replace-mode) and advance the fold
        pointer, so the next `build_context` emits `[prompt + summary + recent]` with no re-summary.
        Recomputes `recent_start` from the same (unchanged) history `pending_compaction` saw."""
        recent_start, _, _ = self._plan(L)
        self._summary = summary.strip()
        self._summarized_upto = recent_start

    def clone(self) -> "G2Summarize":
        """Deep-enough copy for branching: independent history list + summary state; config and the
        (immutable) Message/task objects are shared. `summarizer` carries over but the tree driver
        drives compaction externally, so it is not invoked on a clone."""
        g = G2Summarize(recency_turns=self.recency_turns, compact_every=self.compact_every,
                        summarizer=self.summarizer, compact_at_tokens=self.compact_at_tokens,
                        summary_mode=self.summary_mode)
        g.task = self.task
        g.system_prompt = self.system_prompt
        g.history = list(self.history)
        g._summary = self._summary
        g._summarized_upto = self._summarized_upto
        g._last_snapshot = self._last_snapshot
        return g

    def build_context(self, budget_tokens: int) -> list[Message]:
        head = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        task_msg = self.history[0]
        tail = self.history[1:]

        recent_start, newly_stale, should_compact = self._plan()

        summarized_refs: list[str] = []
        if should_compact:
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
