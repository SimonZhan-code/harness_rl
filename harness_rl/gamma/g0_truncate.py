"""G0 — drop-oldest truncation. The naive baseline Gamma.

Keeps the system prompt + task, then as many of the most-recent turns as fit the
budget, dropping the oldest. This is essentially what mini-swe-agent does implicitly
(linear history that eventually overflows). Expected to underperform on the
longest-horizon tasks — the sanity check for the probe.
"""
from __future__ import annotations

from harness_rl.gamma.base import BaseContextManager, count_tokens
from harness_rl.types import GammaSnapshot, Message, Role


class G0Truncate(BaseContextManager):
    name = "G0_truncate"

    def build_context(self, budget_tokens: int) -> list[Message]:
        head = [Message(role=Role.SYSTEM, content=self.system_prompt)]
        # history[0] is the task instruction (user); always keep it.
        task_msg = self.history[0]
        tail = self.history[1:]

        kept: list[Message] = []
        dropped: list[str] = []
        used = count_tokens(head + [task_msg])
        # Walk from most-recent backward, keeping what fits.
        for i in range(len(tail) - 1, -1, -1):
            m = tail[i]
            c = count_tokens([m])
            if used + c <= budget_tokens:
                kept.append(m)
                used += c
            else:
                dropped.append(f"turn:{i}")
        kept.reverse()

        messages = head + [task_msg] + kept
        self._last_snapshot = GammaSnapshot(
            kept=[f"turn:{i}" for i in range(len(tail) - len(kept), len(tail))],
            dropped=dropped,
            tokens_in_ctx=used,
            notes={"strategy": "drop_oldest"},
        )
        return messages
