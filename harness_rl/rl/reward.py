"""Reward + advantage shaping for Step 2.

Ground truth is the checkable outcome u_g. An OPTIONAL grounded process bonus q_m (from a
strong reference executor, e.g. Gemini) can densify the signal — but it is OUTCOME-GATED
so it cannot be farmed on failed episodes, and it is grounded in real reference success
(not a surface rubric), per problem_formulation.md §C2/§C3.

    R_tilde(tau) = u_g(tau) - lambda * cost(tau)
                   + beta * q_m(tau) * 1[u_g > 0 or grounded]

The localized advantage mask restricts credit to the execution span t >= t_star (the
on-track prefix length), so gradient is spent where the failure was localized.
"""
from __future__ import annotations

from dataclasses import dataclass

from harness_rl.types import Trace


@dataclass
class RewardConfig:
    lambda_cost: float = 0.0        # cost penalty coefficient (tokens)
    beta_process: float = 0.0       # process-bonus weight; 0.0 = pure outcome (conservative default)
    cost_norm: float = 100_000.0    # normalizer for cost term
    outcome_gate: bool = True       # gate process bonus by outcome success


def compute_reward(trace: Trace, cfg: RewardConfig) -> float:
    o = trace.outcome
    if o is None:
        return 0.0
    r = o.u_g
    if cfg.lambda_cost:
        r -= cfg.lambda_cost * (o.cost_tokens / cfg.cost_norm)
    if cfg.beta_process:
        q_m = _terminal_q(trace)
        if q_m is not None and (not cfg.outcome_gate or o.u_g > 0):
            r += cfg.beta_process * q_m
    return r


def _terminal_q(trace: Trace) -> float | None:
    """q_m = the last available per-step process-judge score (filled by the grounded judge)."""
    for step in reversed(trace.steps):
        if step.q_t is not None:
            return step.q_t
    return None


def on_track_prefix_len(trace: Trace, kappa: float = 0.5) -> int:
    """t_star = largest t with q_t >= kappa (on-track prefix). Falls back to full length
    if the judge has not scored the trace yet."""
    t_star = 0
    for i, step in enumerate(trace.steps):
        if step.q_t is not None and step.q_t >= kappa:
            t_star = i + 1
    return t_star if t_star else len(trace.steps)


def localized_advantage_mask(trace: Trace, kappa: float = 0.5) -> list[bool]:
    """Per-step mask: True where gradient should apply (execution span t >= t_star)."""
    t_star = on_track_prefix_len(trace, kappa)
    return [i >= t_star for i in range(len(trace.steps))]


def supo_samples(rollout, overlong_mask: bool = True) -> list[dict]:
    """SUPO credit assignment — Theorem 3.2 (arXiv 2510.06727).

    Emit ONE training sample per sub-trajectory segment (tool-use AND summary turns), each a
    `(bounded context, generated response)` pair carrying the rollout's **outcome** reward R and
    a shared `group_key`. slime GRPO then groups by `group_key`, computes the group-relative
    advantage `A = (R - mu_G)/(sigma_G+eps)`, and applies it to ALL response tokens of every
    sample. Result:
      grad = sum_j sum_{t in T_j} grad log pi(a_t|s_t) * A       (all sub-trajectories share A)
    i.e. every token across every summarized sub-trajectory — **including summary tokens** —
    is trained with the same outcome advantage. `rollout` is a `rl.supo.SUPORollout` (duck-typed).

    Overlong masking (essential per the SUPO ablation — without it summarization collapses):
    rollouts that hit the turn/summary limit are dropped (no gradient).
    """
    if overlong_mask and getattr(rollout, "hit_limit", False):
        return []
    R = float(rollout.outcome_u_g)
    return [
        {
            "group_key": rollout.task_id,
            "messages": [m.to_openai() for m in s.context],
            "response": s.output,
            "reward": R,                 # shared across all segments → shared group-relative advantage
            "is_summary": s.is_summary,
            "segment": s.segment,
            "category_tokens": dict(getattr(s, "category_tokens", {}) or {}),  # #0 summarization/thinking/tool_call
        }
        for s in rollout.segments
    ]


def category_advantage_report(rollouts: list, eps: float = 1e-6) -> dict:
    """#1 — observational per-category token-mass, advantage-weighted (arXiv 2510.06727 context).

    Given a batch/group of SUPORollouts, compute each rollout's group-relative advantage
    `A = (R - mu)/(sigma+eps)` over the batch (GRPO/SUPO), then aggregate by token category
    (summarization / thinking / tool_call):
      - `tokens[c]`     : total generated tokens in category c
      - `adv_mass[c]`   : sum over turns of (category tokens) * A  — where the update PRESSURE lands
      - `tokens_success[c]` / `tokens_fail[c]` : token split by outcome (u_g>0 vs u_g==0)

    This is DESCRIPTIVE monitoring (under Thm 3.2 the advantage is uniform within a rollout, so
    this shows where the gradient mass falls, not causal per-category contribution — that's #3).
    """
    from harness_rl.rl.supo import CATEGORIES  # lazy (avoids import cycle)

    rewards = [float(r.outcome_u_g) for r in rollouts]
    n = len(rewards)
    mu = sum(rewards) / n if n else 0.0
    sigma = (sum((x - mu) ** 2 for x in rewards) / n) ** 0.5 if n else 0.0

    tokens = {c: 0 for c in CATEGORIES}
    adv_mass = {c: 0.0 for c in CATEGORIES}
    tok_succ = {c: 0 for c in CATEGORIES}
    tok_fail = {c: 0 for c in CATEGORIES}
    for r in rollouts:
        A = (float(r.outcome_u_g) - mu) / (sigma + eps)
        bucket = tok_succ if float(r.outcome_u_g) > 0 else tok_fail
        for s in r.segments:
            for c, ntok in (getattr(s, "category_tokens", {}) or {}).items():
                tokens[c] += ntok
                adv_mass[c] += ntok * A
                bucket[c] += ntok
    total = sum(tokens.values()) or 1
    return {
        "n_rollouts": n,
        "reward_mean": mu,
        "reward_std": sigma,
        "tokens": tokens,
        "token_frac": {c: tokens[c] / total for c in CATEGORIES},
        "adv_mass": adv_mass,
        "tokens_success": tok_succ,
        "tokens_fail": tok_fail,
    }
