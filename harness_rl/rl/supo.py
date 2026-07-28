"""SUPO — Summarization augmented Policy Optimization (arXiv 2510.06727).

Reimplements SUPO on our harness using the **G2 memory component**. When the working context
exceeds a token threshold `L`, the **policy itself** generates a summary (an in-context
*action*) that compacts history to `[prompt + summary + recent]` and opens a new
**sub-trajectory**. Summarization is a **trainable action** — its tokens get the policy-gradient
loss (end-to-end). The credit assignment (Thm 3.2) lives in `rl/reward.py:supo_samples`.

Mechanism:
  - Drive a `G2Summarize(compact_at_tokens=L, summary_mode="replace")` whose `summarizer` is a
    policy call that ALSO records the summarization as a `SegmentTurn(is_summary=True)` and
    increments the sub-trajectory counter.
  - Each action turn records a `SegmentTurn(is_summary=False)` on the (bounded) compacted context.
  - Every SegmentTurn is one training sample (bounded context → generated tokens).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from harness_rl.benchmarks.base import Environment
from harness_rl.gamma.base import count_tokens
from harness_rl.gamma.g2_summarize import G2Summarize
from harness_rl.harness.agent import DEFAULT_SYSTEM_PROMPT, parse_action
from harness_rl.serving.client import ModelClient
from harness_rl.types import Message, Observation, Role, TaskSpec

SUMMARIZE_PROMPT = (
    "Your working context is getting long. Summarize your progress SO FAR into a concise digest "
    "that retains EVERYTHING needed to keep solving: the problem, your current best solution, the "
    "test results / errors you have seen, and your current plan. Write ONLY the summary."
)

# Token categories for per-category credit tracking (#0).
CATEGORIES = ("summarization", "thinking", "tool_call")
_BASH_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)
_DONE_RE = re.compile(r"^\s*TASK_COMPLETE\s*$", re.MULTILINE)


def _ntok(text: str) -> int:
    return count_tokens([Message(role=Role.ASSISTANT, content=text)]) if text.strip() else 0


def categorize_spans(output: str, is_summary: bool) -> list[tuple[str, int, int]]:
    """#0 — split a segment's generated text into `(category, start_char, end_char)` spans.

    A summary turn is one `summarization` span. An action turn splits at the parsed action: the
    fenced ```bash``` block / `TASK_COMPLETE` is `tool_call`; text around it is `thinking` (incl.
    any <think>…</think>). Malformed output with no action → all `thinking`. Char spans let the
    trainer label its own tokens (via a tokenizer offset mapping) for per-category entropy/KL.
    """
    if is_summary:
        return [("summarization", 0, len(output))]
    m = _BASH_RE.search(output) or _DONE_RE.search(output)
    if not m:
        return [("thinking", 0, len(output))]
    spans: list[tuple[str, int, int]] = []
    if output[:m.start()].strip():
        spans.append(("thinking", 0, m.start()))
    spans.append(("tool_call", m.start(), m.end()))
    if output[m.end():].strip():
        spans.append(("thinking", m.end(), len(output)))
    return spans


def categorize_tokens(output: str, is_summary: bool) -> dict[str, int]:
    """#0 — per-category (heuristic) token counts, derived from `categorize_spans`."""
    out: dict[str, int] = {}
    for cat, s, e in categorize_spans(output, is_summary):
        out[cat] = out.get(cat, 0) + _ntok(output[s:e])
    return out


@dataclass
class SegmentTurn:
    segment: int              # sub-trajectory index (increments on each summarization)
    is_summary: bool          # True = a summarization action; False = a tool-use action
    context: list[Message]    # bounded input handed to the policy
    output: str               # policy-generated tokens (response or summary)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    category_tokens: dict[str, int] = field(default_factory=dict)  # #0: summarization/thinking/tool_call


@dataclass
class SUPORolloutConfig:
    context_L: int = 4096     # token threshold L that triggers a summarization action
    max_turns: int = 40       # action-turn cap
    max_summaries: int = 8    # summarization-round cap (overlong beyond this)
    recency_turns: int = 4
    max_tokens: int = 2048    # per-call generation cap


@dataclass
class SUPORollout:
    task_id: str
    segments: list[SegmentTurn] = field(default_factory=list)
    outcome_u_g: float = 0.0
    num_summaries: int = 0
    turns: int = 0            # action turns (excludes summary turns)
    hit_limit: bool = False   # exceeded max_turns or max_summaries → overlong (masked)
    terminated_reason: str = ""


def supo_rollout(task: TaskSpec, env: Environment, model: ModelClient,
                 cfg: SUPORolloutConfig, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> SUPORollout:
    roll = SUPORollout(task_id=task.task_id)
    seg = {"id": 0}  # mutable so the summarizer closure can bump the sub-trajectory index

    def policy_summarizer(msgs_to_summarize: list[Message]) -> str:
        """Called by G2 when context > L. The POLICY produces the summary; we record it as a
        trainable summary turn and open a new sub-trajectory."""
        ctx = ([Message(role=Role.SYSTEM, content=system_prompt),
                Message(role=Role.USER, content=f"Task:\n{task.instruction}")]
               + list(msgs_to_summarize)
               + [Message(role=Role.USER, content=SUMMARIZE_PROMPT)])
        out = model.chat(ctx)
        roll.segments.append(SegmentTurn(segment=seg["id"], is_summary=True, context=ctx,
                                         output=out.text, prompt_tokens=out.prompt_tokens,
                                         completion_tokens=out.completion_tokens,
                                         category_tokens=categorize_tokens(out.text, True)))
        roll.num_summaries += 1
        seg["id"] += 1  # subsequent action turns belong to the new (compacted) sub-trajectory
        return out.text

    gamma = G2Summarize(recency_turns=cfg.recency_turns, compact_at_tokens=cfg.context_L,
                        summary_mode="replace", summarizer=policy_summarizer)
    gamma.reset(task, system_prompt)

    obs = env.reset()
    gamma.on_step(obs, action=None)

    reason = "budget"
    hit = True
    for _ in range(cfg.max_turns):
        if roll.num_summaries > cfg.max_summaries:
            reason = "max_summaries"
            break
        # build_context may trigger policy_summarizer (records a summary turn) before returning
        ctx = gamma.build_context(budget_tokens=cfg.context_L)
        out = model.chat(ctx)
        action = parse_action(out.text)
        roll.segments.append(SegmentTurn(segment=seg["id"], is_summary=False, context=ctx,
                                         output=out.text, prompt_tokens=out.prompt_tokens,
                                         completion_tokens=out.completion_tokens,
                                         category_tokens=categorize_tokens(out.text, False)))
        if action is None:
            obs = Observation(text="No valid action found. Emit one ```bash ... ``` block, or TASK_COMPLETE.")
            gamma.on_step(obs, action=None)
        elif env.is_done(action):
            reason = "submit"
            hit = False
            gamma.on_step(obs, action)
            break
        else:
            obs = env.execute(action)
            gamma.on_step(obs, action)

    outcome = env.verify()
    roll.outcome_u_g = outcome.u_g
    roll.turns = sum(1 for s in roll.segments if not s.is_summary)
    roll.hit_limit = hit or reason == "max_summaries"
    roll.terminated_reason = reason
    env.close()
    return roll
