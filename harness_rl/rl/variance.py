"""Summary-leverage effect size (eta^2) for the M x K GiGPO design — with honest error bars.

Answers "how much of outcome variance is attributable to the CHOICE OF SUMMARY (macro, m) rather
than downstream execution luck (micro, k)?" via a one-way ANOVA on each task's reward matrix
`R[m][k]`, pooled across tasks.

Deliberately dependency-free: the only input is `list[list[float]]` per task (exactly the
`reward_matrix` the production GiGPO rollout already builds), so this drops into any fork.

THREE TRAPS THIS MODULE EXISTS TO AVOID
---------------------------------------
1. **eta^2 is badly biased upward at small M, K.** Under the null of NO true summary effect,
       E[eta^2 | H0] = df_between / (df_between + df_within)
   For a single M=2, K=2 task that is 1/3 = 0.333 — so an observed eta^2 of ~0.33 is exactly what
   pure noise produces, not evidence that summaries matter. `omega_squared` is the bias-corrected
   version (E[omega^2 | H0] ~ 0); always read it alongside eta^2, and compare both against the
   reported `null_baseline`.

2. **eta^2 is undefined at K=1.** With one trajectory per summary there is no within-summary
   variance (SS_within = 0, df_within = 0), so eta^2 is identically 1.0 — a degenerate artifact,
   not a measurement. K >= 2 is REQUIRED to estimate summary leverage at all.

3. **A single task carries ~no information.** M=2, K=2 gives 1 between-df and 2 within-df. Pool
   across tasks (this module does) so df scales with the number of tasks, and read the bootstrap
   CI rather than the point estimate.

Also provides `macro_advantage_spread`, which empirically confirms the N=2 standardization
degeneracy: with M=2 the group-relative advantage (R - mu)/sigma collapses to +/-1 exactly,
carrying sign but NO magnitude information.
"""
from __future__ import annotations

import json
import random
import sys

Matrix = list[list[float]]      # R[m][k]; rows may be ragged (len(row) = K_m)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def task_ss(matrix: Matrix) -> dict:
    """One task's ANOVA sums of squares. Rows may be ragged (variable K per summary).

    Returns ss_between (summary choice), ss_within (execution noise), df's, and the shape.
    Rows with no trajectories are dropped; the grand mean is the task's own mean, so between-task
    difficulty never leaks into the summary effect.
    """
    rows = [list(r) for r in matrix if len(r) > 0]
    m = len(rows)
    n = sum(len(r) for r in rows)
    if m == 0 or n == 0:
        return {"ss_between": 0.0, "ss_within": 0.0, "df_between": 0, "df_within": 0, "m": 0, "n": 0}
    grand = _mean([x for r in rows for x in r])
    ss_b = sum(len(r) * (_mean(r) - grand) ** 2 for r in rows)
    ss_w = sum((x - _mean(r)) ** 2 for r in rows for x in r)
    return {"ss_between": ss_b, "ss_within": ss_w,
            "df_between": m - 1, "df_within": n - m, "m": m, "n": n}


def summary_leverage(matrices: list[Matrix], n_boot: int = 2000, seed: int = 0,
                     ci: float = 0.95) -> dict:
    """Pooled summary-leverage effect size across tasks, with a bootstrap CI over tasks.

    eta^2  = SS_between / SS_total                                  (biased upward; see module doc)
    omega^2 = (SS_between - df_b * MS_within) / (SS_total + MS_within)   (bias-corrected)
    F       = MS_between / MS_within  on (df_b, df_w) degrees of freedom

    `null_baseline` = E[eta^2 | no true summary effect]. **Compare eta^2 against this, not 0.**
    Returns `degenerate=True` when every task has K=1 (df_within = 0), where eta^2 is meaningless.
    """
    per_task = [task_ss(mx) for mx in matrices]
    kept = [t for t in per_task if t["n"] > 0]
    ss_b = sum(t["ss_between"] for t in kept)
    ss_w = sum(t["ss_within"] for t in kept)
    df_b = sum(t["df_between"] for t in kept)
    df_w = sum(t["df_within"] for t in kept)
    ss_tot = ss_b + ss_w

    out: dict = {
        "n_tasks": len(kept), "ss_between": ss_b, "ss_within": ss_w,
        "df_between": df_b, "df_within": df_w,
        "mean_m": _mean([t["m"] for t in kept]) if kept else 0.0,
        "mean_k": _mean([t["n"] / t["m"] for t in kept if t["m"]]) if kept else 0.0,
        "null_baseline": (df_b / (df_b + df_w)) if (df_b + df_w) else None,
    }
    if df_w == 0:
        out.update({"degenerate": True, "eta2": None, "omega2": None, "F": None,
                    "reason": "K=1 for every task: no within-summary variance, so eta^2 is "
                              "undefined (identically 1.0). Re-run with K>=2 to measure leverage."})
        return out
    if ss_tot == 0:
        out.update({"degenerate": True, "eta2": None, "omega2": None, "F": None,
                    "reason": "zero total variance: every leaf scored identically, so no method "
                              "can extract signal from this batch."})
        return out

    ms_w = ss_w / df_w
    ms_b = (ss_b / df_b) if df_b else 0.0
    out.update({
        "degenerate": False,
        "eta2": ss_b / ss_tot,
        "omega2": (ss_b - df_b * ms_w) / (ss_tot + ms_w),
        "epsilon2": (ss_b - df_b * ms_w) / ss_tot,
        "F": (ms_b / ms_w) if ms_w > 0 else None,
        "ms_between": ms_b, "ms_within": ms_w,
    })

    # bootstrap over TASKS (the independent unit) — resample tasks, repool
    rng = random.Random(seed)
    boots: list[float] = []
    if len(kept) > 1 and n_boot > 0:
        for _ in range(n_boot):
            samp = [kept[rng.randrange(len(kept))] for _ in range(len(kept))]
            b = sum(t["ss_between"] for t in samp)
            w = sum(t["ss_within"] for t in samp)
            if b + w > 0:
                boots.append(b / (b + w))
    if boots:
        boots.sort()
        lo = boots[int((1 - ci) / 2 * (len(boots) - 1))]
        hi = boots[int((1 + ci) / 2 * (len(boots) - 1))]
        out["eta2_ci"] = (lo, hi)
        out["ci_level"] = ci
    return out


def macro_advantage_spread(matrices: list[Matrix], eps: float = 1e-6) -> dict:
    """Empirically confirm (or refute) the N=2 standardization degeneracy.

    Computes the group-relative macro advantage A(m) = (Rbar(m) - mu) / (sigma + eps) over each
    task's M summary means. At M=2 the magnitude cancels algebraically and A collapses to +/-1 —
    sign only, NO information about HOW much better one summary was. Reports the number of
    distinct |A| values seen; `sign_only=True` means the advantage carries one bit per summary.
    """
    mags: list[float] = []
    vals: list[float] = []
    for mx in matrices:
        rows = [list(r) for r in mx if len(r) > 0]
        if len(rows) < 2:
            continue
        means = [_mean(r) for r in rows]
        mu = _mean(means)
        var = _mean([(x - mu) ** 2 for x in means])
        sigma = var ** 0.5
        if sigma <= 0:
            continue
        for x in means:
            a = (x - mu) / (sigma + eps)
            vals.append(a)
            mags.append(abs(a))
    if not mags:
        return {"n": 0, "sign_only": None, "distinct_magnitudes": 0,
                "reason": "no task had >=2 summaries with non-zero spread"}
    mean_mag = _mean(mags)
    # RELATIVE spread, not exact equality: the `eps` regularizer perturbs |A| by ~eps/sigma, which
    # differs per task, so distinct float values do NOT imply graded magnitude. Sign-only means all
    # |A| are the same up to that perturbation.
    rel_spread = (max(mags) - min(mags)) / mean_mag if mean_mag > 0 else 0.0
    sign_only = rel_spread < 1e-3
    return {"n": len(vals), "distinct_magnitudes": len({round(x, 3) for x in mags}),
            "magnitude_mean": mean_mag, "magnitude_min": min(mags), "magnitude_max": max(mags),
            "magnitude_rel_spread": rel_spread, "sign_only": sign_only,
            "note": ("|A| is constant → advantage is a pure SIGN function (algebraically forced at "
                     "M=2): it says WHICH summary was better, never BY HOW MUCH"
                     if sign_only else
                     "advantage carries graded magnitude across summaries")}


def format_report(lev: dict, spread: dict | None = None) -> str:
    """Human-readable one-screen summary, safe to paste into a run log."""
    L = [f"summary leverage over {lev['n_tasks']} tasks "
         f"(mean M={lev['mean_m']:.2f}, mean K={lev['mean_k']:.2f}; "
         f"df_between={lev['df_between']}, df_within={lev['df_within']})"]
    if lev.get("degenerate"):
        L.append(f"  DEGENERATE — {lev['reason']}")
    else:
        nb = lev["null_baseline"]
        L.append(f"  eta^2   = {lev['eta2']:.4f}"
                 + (f"   95% CI [{lev['eta2_ci'][0]:.4f}, {lev['eta2_ci'][1]:.4f}]"
                    if "eta2_ci" in lev else ""))
        L.append(f"  omega^2 = {lev['omega2']:.4f}   (bias-corrected; ~0 under the null)")
        L.append(f"  F({lev['df_between']},{lev['df_within']}) = {lev['F']:.3f}")
        L.append(f"  null baseline E[eta^2 | no summary effect] = {nb:.4f}"
                 f"  <-- compare eta^2 against THIS, not 0")
        if lev["eta2"] <= nb:
            L.append("  ==> eta^2 at/below the null baseline: NO evidence summaries matter here.")
        elif lev["omega2"] <= 0:
            L.append("  ==> omega^2 <= 0: the apparent effect vanishes after bias correction.")
    if spread and spread.get("n"):
        L.append(f"  macro advantage: {spread['distinct_magnitudes']} distinct |A| over "
                 f"{spread['n']} values (mean |A|={spread['magnitude_mean']:.3f})")
        L.append(f"    {spread['note']}")
    return "\n".join(L)


def main() -> None:  # pragma: no cover
    """`python -m harness_rl.rl.variance rewards.json` where the file is a list of M x K matrices,
    e.g. [[[1.0, 0.5], [0.0, 0.25]], ...]  (one matrix per task). Reads stdin if no path given."""
    raw = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    matrices = json.loads(raw)
    lev = summary_leverage(matrices)
    print(format_report(lev, macro_advantage_spread(matrices)))


if __name__ == "__main__":  # pragma: no cover
    main()
