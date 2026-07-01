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
