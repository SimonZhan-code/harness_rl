"""G1 — retrieval-augmented context.

Keeps a recency window (like G0) but ALSO retrieves older turns most relevant to the
current situation (last observation) and splices them back in. Retrieval is over an
in-memory index of prior turns. Embeddings are pluggable; the default is a
dependency-light lexical (BM25-ish) scorer so `.venv` needs no model.

This targets the failure mode "the model needed a decision/edit from 30 turns ago that
G0 had already dropped."
"""
from __future__ import annotations

import math
import re
from collections import Counter

from harness_rl.gamma.base import BaseContextManager, count_tokens
from harness_rl.types import GammaSnapshot, Message, Role

_TOKEN = re.compile(r"[A-Za-z0-9_./-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Tiny self-contained BM25. Good enough as a default retriever."""
    q_terms = set(_tokenize(query))
    doc_toks = [_tokenize(d) for d in docs]
    if not doc_toks:
        return []
    avgdl = sum(len(d) for d in doc_toks) / len(doc_toks) or 1.0
    df = Counter()
    for d in doc_toks:
        for term in set(d):
            df[term] += 1
    N = len(doc_toks)
    scores = []
    for d in doc_toks:
        tf = Counter(d)
        s = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(d) / avgdl)
            s += idf * (tf[term] * (k1 + 1)) / denom
        scores.append(s)
    return scores


class G1Retrieval(BaseContextManager):
    name = "G1_retrieval"

    def __init__(self, recency_turns: int = 6, retrieve_k: int = 4) -> None:
        super().__init__()
        self.recency_turns = recency_turns
        self.retrieve_k = retrieve_k

    def build_context(self, budget_tokens: int) -> list[Message]:
        head = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        task_msg = self.history[0]
        tail = self.history[1:]

        recent = tail[-self.recency_turns:] if self.recency_turns else []
        recent_start = len(tail) - len(recent)
        older = tail[:recent_start]

        query = tail[-1].content if tail else task_msg.content
        scores = _bm25_scores(query, [m.content for m in older])
        ranked = sorted(range(len(older)), key=lambda i: scores[i], reverse=True)
        retrieved_idx = [i for i in ranked[: self.retrieve_k] if scores[i] > 0]
        retrieved_idx.sort()  # keep chronological order when splicing

        retrieved_msgs = [
            Message(role=older[i].role, content=f"[retrieved turn {i}]\n{older[i].content}")
            for i in retrieved_idx
        ]

        # Assemble, then trim to budget (recency wins over retrieved if we overflow).
        messages = head + [task_msg] + retrieved_msgs + list(recent)
        used = count_tokens(messages)
        while used > budget_tokens and retrieved_msgs:
            retrieved_msgs.pop()  # drop lowest-priority retrieved first
            messages = head + [task_msg] + retrieved_msgs + list(recent)
            used = count_tokens(messages)

        self._last_snapshot = GammaSnapshot(
            kept=[f"turn:{i}" for i in range(recent_start, len(tail))],
            dropped=[f"turn:{i}" for i in range(recent_start) if i not in retrieved_idx],
            retrieved=[f"turn:{i}" for i in retrieved_idx],
            tokens_in_ctx=used,
            notes={"recency_turns": self.recency_turns, "retrieve_k": self.retrieve_k},
        )
        return messages
