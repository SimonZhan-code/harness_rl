"""G5 — external / hierarchical memory.

A two-tier external memory that combines summarization AND retrieval (distinct from G1 =
pure retrieval, G2 = pure summary, G3 = deterministic scratchpad):

  - **Tier 1 (rolling summary):** stale turns are periodically folded into a running
    high-level summary (like G2) — the "gist" of everything that scrolled off.
  - **Tier 2 (archival retrieval):** every past turn is kept in an external archival store;
    the top-k most relevant to the current step are retrieved back (BM25, like G1).
  - **Recency window:** the last few turns verbatim.

Context = system + task + [summary] + [retrieved archival entries] + [recency], trimmed to
budget. This models a MemGPT-style external store with hierarchical recall.
"""
from __future__ import annotations

from harness_rl.gamma.base import BaseContextManager, count_tokens
from harness_rl.gamma.g1_retrieval import _bm25_scores
from harness_rl.gamma.g2_summarize import Summarizer, _identity_summarizer
from harness_rl.types import Message, Role


class G5External(BaseContextManager):
    name = "G5_external"

    def __init__(
        self,
        recency_turns: int = 4,
        retrieve_k: int = 3,
        compact_every: int = 4,
        summarizer: Summarizer | None = None,
    ) -> None:
        super().__init__()
        self.recency_turns = recency_turns
        self.retrieve_k = retrieve_k
        self.compact_every = compact_every
        self.summarizer = summarizer or _identity_summarizer
        self._summary = ""
        self._summarized_upto = 0  # index into tail folded into the rolling summary

    def reset(self, task, system_prompt: str) -> None:  # type: ignore[override]
        super().reset(task, system_prompt)
        self._summary = ""
        self._summarized_upto = 0

    def build_context(self, budget_tokens: int) -> list[Message]:
        head = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        task_msg = self.history[0]
        tail = self.history[1:]

        recent_start = max(self._summarized_upto, len(tail) - self.recency_turns)

        # Tier 1: fold newly-stale turns into the rolling summary (in chunks).
        newly_stale = tail[self._summarized_upto:recent_start]
        summarized_refs: list[str] = []
        if len(newly_stale) >= self.compact_every or (newly_stale and recent_start == len(tail)):
            self._summary = (self._summary + "\n" + self.summarizer(newly_stale)).strip()
            summarized_refs = [f"turn:{self._summarized_upto + j}" for j in range(len(newly_stale))]
            self._summarized_upto = recent_start

        recent = tail[recent_start:]

        # Tier 2: retrieve the most relevant archival (stale) turns for the current step.
        archival = tail[:recent_start]
        query = tail[-1].content if tail else task_msg.content
        scores = _bm25_scores(query, [m.content for m in archival])
        ranked = sorted(range(len(archival)), key=lambda i: scores[i], reverse=True)
        retrieved_idx = sorted(i for i in ranked[: self.retrieve_k] if scores[i] > 0)
        retrieved = [
            Message(role=archival[i].role, content=f"[recalled turn {i}]\n{archival[i].content}")
            for i in retrieved_idx
        ]

        summary_block = (
            [Message(role=Role.USER, content=f"[memory summary]\n{self._summary}")]
            if self._summary else []
        )

        messages = head + [task_msg] + summary_block + retrieved + list(recent)
        used = count_tokens(messages)
        # Under budget pressure, drop retrieved entries first (recency + summary win).
        while used > budget_tokens and retrieved:
            retrieved.pop()
            messages = head + [task_msg] + summary_block + retrieved + list(recent)
            used = count_tokens(messages)

        self._last_snapshot = self._snapshot(recent_start, len(tail), summarized_refs,
                                             retrieved_idx[: len(retrieved)], used)
        return messages

    def _snapshot(self, recent_start, n_tail, summarized_refs, retrieved_idx, used):
        from harness_rl.types import GammaSnapshot
        return GammaSnapshot(
            kept=[f"turn:{i}" for i in range(recent_start, n_tail)],
            summarized=summarized_refs,
            retrieved=[f"turn:{i}" for i in retrieved_idx],
            dropped=[f"turn:{i}" for i in range(recent_start)
                     if i not in retrieved_idx and f"turn:{i}" not in summarized_refs],
            tokens_in_ctx=used,
            notes={"summary_chars": len(self._summary), "retrieve_k": self.retrieve_k,
                   "archival_size": recent_start},
        )
