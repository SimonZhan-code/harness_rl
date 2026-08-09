"""Per-category training diagnostics for SUPO — entropy and KL(pi_new || pi_ref).

These are TRAINING-time signals: exact per-token entropy needs the full logits, and KL needs the
reference policy — both live in the slime training step, not in a rollout. This module supplies
the glue so the trainer can bucket its per-token entropy/KL by token category (summarization /
thinking / tool_call):

  1. `label_tokens(spans, offsets)`  — map each response token (via its char-offset from the
     tokenizer's `return_offsets_mapping`) to the category span it falls in.
  2. `bucket_by_category(labels, values)` — aggregate any per-token metric by category.
  3. reference `entropy_from_logprobs` / `token_kl_k3` (Schulman k3, matches GRPO's KL penalty).
  4. `supo_category_metrics(...)` + `merge_category_metrics(...)` — the trainer hook + batch merge.

What they tell you (per category):
  - **entropy** — uncertainty/diversity; watch summarization-entropy → 0 = summary collapse.
  - **KL(pi_new||pi_ref)** — where the update MOVES behavior; watch summary-KL for summarizer drift.
Both are distributional/training-dynamics (correlational), NOT causal contribution (that's #3).
"""
from __future__ import annotations

import math


def label_tokens(spans: list[tuple[str, int, int]],
                 offsets: list[tuple[int, int]]) -> list[str | None]:
    """Assign each response token a category by which char-span its midpoint falls in.

    `spans` = `[(category, start_char, end_char), ...]` from `supo.categorize_spans`.
    `offsets` = per-token `(start_char, end_char)` for the RESPONSE text (HF
    `return_offsets_mapping=True`). Tokens in gaps (whitespace between spans) → None.
    """
    labels: list[str | None] = []
    for s, e in offsets:
        if e <= s:            # special/empty tokens
            labels.append(None)
            continue
        mid = (s + e) / 2.0
        cat = None
        for c, cs, ce in spans:
            if cs <= mid < ce:
                cat = c
                break
        labels.append(cat)
    return labels


def bucket_by_category(labels: list[str | None], values: list[float]) -> dict[str, dict]:
    """Aggregate a per-token metric by category → {cat: {count, sum, mean}}."""
    out: dict[str, dict] = {}
    for lab, v in zip(labels, values):
        if lab is None:
            continue
        d = out.setdefault(lab, {"count": 0, "sum": 0.0})
        d["count"] += 1
        d["sum"] += float(v)
    for d in out.values():
        d["mean"] = d["sum"] / d["count"] if d["count"] else 0.0
    return out


def entropy_from_logprobs(logprobs) -> float:
    """Shannon entropy (nats) of ONE token from its full/top-k log-prob vector: H = -sum p·logp.

    Full vocab logprobs → exact (what the trainer has from logits). Top-k (e.g. from the SGLang
    OpenAI API) → an approximation/lower bound; renormalize the top-k first if you use it."""
    return -sum(math.exp(lp) * lp for lp in logprobs)


def token_kl_k3(logp_new: float, logp_ref: float) -> float:
    """Schulman k3 unbiased, non-negative per-token KL estimator (the one GRPO uses for its KL
    penalty), from the sampled token's new/ref logprobs. log_r = logp_ref - logp_new."""
    log_r = logp_ref - logp_new
    return math.exp(log_r) - log_r - 1.0


def supo_category_metrics(output: str, is_summary: bool, offsets: list[tuple[int, int]],
                          entropy: list[float] | None = None,
                          kl: list[float] | None = None) -> dict:
    """Trainer hook for ONE segment: bucket its per-token entropy/KL by token category.

    Call inside the slime loss step after the forward pass, with the response `offsets` and the
    per-token `entropy`/`kl` arrays it computed. Returns {"tokens": {cat:n}, "entropy": {cat:...},
    "kl": {cat:...}}.
    """
    from harness_rl.rl.supo import categorize_spans

    labels = label_tokens(categorize_spans(output, is_summary), offsets)
    res: dict = {"tokens": {}}
    for lab in labels:
        if lab:
            res["tokens"][lab] = res["tokens"].get(lab, 0) + 1
    if entropy is not None:
        res["entropy"] = bucket_by_category(labels, entropy)
    if kl is not None:
        res["kl"] = bucket_by_category(labels, kl)
    return res


def merge_category_metrics(metrics: list[dict]) -> dict:
    """Merge per-segment metrics (from `supo_category_metrics`) into a batch report with
    count-weighted per-category means for entropy and KL."""
    tokens: dict[str, int] = {}
    acc: dict[str, dict[str, dict]] = {"entropy": {}, "kl": {}}
    for m in metrics:
        for c, n in m.get("tokens", {}).items():
            tokens[c] = tokens.get(c, 0) + n
        for metric in ("entropy", "kl"):
            for c, d in m.get(metric, {}).items():
                a = acc[metric].setdefault(c, {"count": 0, "sum": 0.0})
                a["count"] += d["count"]
                a["sum"] += d["sum"]
    report = {"tokens": tokens}
    for metric in ("entropy", "kl"):
        report[metric] = {c: {"count": d["count"], "mean": (d["sum"] / d["count"] if d["count"] else 0.0)}
                          for c, d in acc[metric].items()}
    return report


def category_shares(report: dict) -> dict[str, float]:
    """B1 — per-category token share s_c (fractions summing to 1) from a merged report."""
    tokens = report.get("tokens", {})
    total = sum(tokens.values()) or 1
    return {c: n / total for c, n in tokens.items()}


def decompose_change(report_t0: dict, report_t1: dict, metric: str = "entropy") -> dict:
    """B2 — split an aggregate entropy/KL change into WITHIN-category and MIX (composition) parts.

    In a multi-category agent the aggregate can move with **no per-token change at all**, purely
    because the episode's token composition shifted (e.g. proportionally more summarization). So
    "entropy rose" is uninterpretable until you separate:

        d_agg = SUM_c  s_bar_c * (H_c(t1) - H_c(t0))     <- within: the policy really changed
              + SUM_c  H_bar_c * (s_c(t1) - s_c(t0))     <- mix:    composition changed

    using midpoint weights (s_bar, H_bar), which makes the two terms sum EXACTLY to the aggregate
    change (a Törnqvist/mean-value decomposition — no interaction residual left over).

    Returns per-category contributions plus the totals; `residual` is ~0 and is reported only as a
    numerical check.
    """
    s0, s1 = category_shares(report_t0), category_shares(report_t1)
    m0 = {c: d.get("mean", 0.0) for c, d in (report_t0.get(metric) or {}).items()}
    m1 = {c: d.get("mean", 0.0) for c, d in (report_t1.get(metric) or {}).items()}
    cats = sorted(set(s0) | set(s1) | set(m0) | set(m1))

    within, mix = {}, {}
    for c in cats:
        s_bar = (s0.get(c, 0.0) + s1.get(c, 0.0)) / 2.0
        m_bar = (m0.get(c, 0.0) + m1.get(c, 0.0)) / 2.0
        within[c] = s_bar * (m1.get(c, 0.0) - m0.get(c, 0.0))
        mix[c] = m_bar * (s1.get(c, 0.0) - s0.get(c, 0.0))

    agg0 = sum(s0.get(c, 0.0) * m0.get(c, 0.0) for c in cats)
    agg1 = sum(s1.get(c, 0.0) * m1.get(c, 0.0) for c in cats)
    total = agg1 - agg0
    w, x = sum(within.values()), sum(mix.values())
    return {"metric": metric, "aggregate_t0": agg0, "aggregate_t1": agg1, "aggregate_change": total,
            "within_total": w, "mix_total": x, "residual": total - (w + x),
            "within": within, "mix": mix, "shares_t0": s0, "shares_t1": s1}


def format_decomposition(d: dict) -> str:
    L = [f"{d['metric']} aggregate {d['aggregate_t0']:.4f} -> {d['aggregate_t1']:.4f} "
         f"(change {d['aggregate_change']:+.4f})",
         f"  within-category (policy changed): {d['within_total']:+.4f}",
         f"  mix (composition changed):        {d['mix_total']:+.4f}",
         f"  {'category':16} {'within':>10} {'mix':>10} {'share t0':>9} {'share t1':>9}"]
    for c in sorted(set(d["within"]) | set(d["mix"])):
        L.append(f"  {c:16} {d['within'].get(c, 0.0):+10.4f} {d['mix'].get(c, 0.0):+10.4f} "
                 f"{d['shares_t0'].get(c, 0.0):9.3f} {d['shares_t1'].get(c, 0.0):9.3f}")
    if abs(d["mix_total"]) > abs(d["within_total"]):
        L.append("  ==> the aggregate moved MOSTLY because token composition shifted, not because "
                 "the policy's per-token distribution changed. Do not read it as exploration.")
    return "\n".join(L)


def category_entropy_kl_report(segment_stats: list[dict]) -> dict:
    """Batch per-category entropy + KL report — the SUPO training dashboard companion to
    `reward.category_advantage_report` (#1).

    `segment_stats` = one dict per trained segment, each with the response text and the trainer's
    per-token arrays:
        {"output": str, "is_summary": bool, "offsets": [(s,e),...],
         "entropy": [float,...], "kl": [float,...]}   # entropy/kl optional
    `offsets` are the response tokenizer offsets (HF `return_offsets_mapping=True`); `entropy`/`kl`
    are the per-token values the slime loss step computes (exact H from logits; k1/k3 KL vs pi_ref).
    Returns {"tokens": {cat:n}, "entropy": {cat:{count,mean}}, "kl": {cat:{count,mean}}}.
    """
    per_seg = [supo_category_metrics(s["output"], s["is_summary"], s["offsets"],
                                     entropy=s.get("entropy"), kl=s.get("kl"))
               for s in segment_stats]
    return merge_category_metrics(per_seg)
