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
