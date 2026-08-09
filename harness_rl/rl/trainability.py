"""hrl-trainability — does this (model, benchmark, config) produce a LEARNING SIGNAL at all?

Inference-only. Run this BEFORE comparing credit-assignment schemes or scaling the model, because
if the reward is structurally zero — or has zero variance within a task group — then flat SUPO,
M=4xK=1 and M=2xK=2 all receive the same near-zero gradient and their curves MUST coincide. Two
arms looking identical is then a statement about the data, not about the algorithms.

The audit answers, in order (stop at the first failure):

  A0  total_tests > 0            — is there a reward function at all? `total_tests == 0` means the
                                   LCB payload never reached the env, so u_g == 0 by construction.
  A2  effective sample yield     — what fraction of rollouts survive `mask_reasons`? A policy that
                                   never emits TASK_COMPLETE terminates on "budget"; under faithful
                                   SUPO masking that discards EVERY rollout (measured: the 3B
                                   submits 0/20) and the batch yields no gradient.
  A2  reward variance            — GRPO consumes the WITHIN-GROUP spread. sigma == 0 (all-solved or
                                   all-failed) gives a zero advantage for every token.
  A1  batch diversity            — unique tasks, difficulty/format mix. LiveCodeBench release_v6 is
                                   only 175 problems (46% hard), and `subset()` without a
                                   shuffle_seed returns a deterministic file-order prefix, so
                                   batches can be near-duplicates of one contest slice.
  A3  trainable difficulty band  — u_g mean AND variance per tier; the band where the model is
                                   neither at 0 nor at ceiling is the only one that can teach.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter

from harness_rl.harness.agent import parse_action
from harness_rl.rl.supo import SUPORolloutConfig, supo_rollout
from harness_rl.serving.client import ModelClient
from harness_rl.types import TaskSpec


def audit_rollouts(rollouts: list, mask_reasons: tuple[str, ...] = ("budget", "max_summaries"),
                   outcomes: list | None = None) -> dict:
    """Aggregate one task's rollouts (`rl.supo.SUPORollout`) into the trainability numbers.

    `outcomes` (optional, parallel to `rollouts`) supplies `Outcome` objects so `total_tests` can be
    reported; without them the A0 column is omitted rather than guessed.
    """
    n = len(rollouts)
    reasons = Counter(getattr(r, "terminated_reason", "?") for r in rollouts)
    u_gs = [float(r.outcome_u_g) for r in rollouts]

    acts = [s for r in rollouts for s in r.segments if not s.is_summary]
    parsed = [parse_action(s.output) for s in acts]
    unparseable = sum(1 for a in parsed if a is None)
    submits = sum(1 for a in parsed if a is not None and a.tool == "submit")

    kept = [r for r in rollouts if getattr(r, "terminated_reason", "") not in mask_reasons]
    sigma = statistics.pstdev(u_gs) if len(u_gs) > 1 else 0.0
    kept_u = [float(r.outcome_u_g) for r in kept]
    out = {
        "n_rollouts": n,
        "termination": dict(reasons),
        "submit_rate": submits / len(acts) if acts else 0.0,
        "parse_fail_rate": unparseable / len(acts) if acts else 0.0,
        "u_g_mean": statistics.fmean(u_gs) if u_gs else 0.0,
        "u_g_std": sigma,
        "u_g_zero_frac": (sum(1 for u in u_gs if u == 0.0) / n) if n else 0.0,
        # THE number: fraction of rollouts that actually reach the loss
        "effective_yield": len(kept) / n if n else 0.0,
        # sigma over the rollouts that survive masking — what GRPO really sees
        "kept_u_g_std": statistics.pstdev(kept_u) if len(kept_u) > 1 else 0.0,
        "degenerate_group": sigma == 0.0,
        "summaries_per_episode": statistics.fmean([r.num_summaries for r in rollouts]) if n else 0.0,
        "turns_per_episode": statistics.fmean([r.turns for r in rollouts]) if n else 0.0,
        "tokens_per_episode": statistics.fmean(
            [sum(s.completion_tokens for s in r.segments) for r in rollouts]) if n else 0.0,
    }
    if outcomes:
        totals = [getattr(o, "total_tests", 0) or 0 for o in outcomes]
        out["total_tests_mean"] = statistics.fmean(totals) if totals else 0.0
        out["has_tests_frac"] = sum(1 for t in totals if t > 0) / len(totals) if totals else 0.0
    return out


def pool_report(adapter, n_tasks: int, difficulty: str | None, shuffle_seed: int | None) -> dict:
    """A1 — what the sampler will actually hand the trainer."""
    census = adapter.difficulty_census() if hasattr(adapter, "difficulty_census") else {}
    try:
        picked = adapter.subset(n_tasks, difficulty=difficulty, shuffle_seed=shuffle_seed)
    except TypeError:                       # adapters without the new signature
        picked = adapter.subset(n_tasks)
    diff = Counter(str(t.payload.get("difficulty", "?")).lower() for t in picked)
    plat = Counter(str(t.payload.get("platform", "?")).lower() for t in picked)
    return {"pool": census, "selected": len(picked),
            "unique_selected": len({t.task_id for t in picked}),
            "selected_difficulty": dict(diff), "selected_platform": dict(plat),
            "shuffled": shuffle_seed is not None}


def format_report(rows: list[tuple[str, dict]], pool: dict | None = None) -> str:
    L: list[str] = []
    if pool:
        p = pool.get("pool") or {}
        if p:
            L.append(f"POOL  total={p.get('total')} unique={p.get('unique_ids')} "
                     f"cumulative={p.get('cumulative')} tag={p.get('version_tag')}")
            L.append(f"      difficulty={p.get('difficulty')}  platform={p.get('platform')}")
            L.append(f"      functional={p.get('functional')} stdin={p.get('stdin')}")
        L.append(f"BATCH selected={pool.get('selected')} unique={pool.get('unique_selected')} "
                 f"shuffled={pool.get('shuffled')} mix={pool.get('selected_difficulty')}")
        if not pool.get("shuffled"):
            L.append("      ⚠️  NOT shuffled — deterministic file-order prefix; batches will be a "
                     "narrow, correlated contest slice. Pass --shuffle-seed.")
        L.append("")
    hdr = (f"  {'task':22} {'n':>3} {'u_g':>6} {'sigma':>6} {'yield':>6} {'submit':>7} "
           f"{'parse!':>7} {'tests':>6} {'summ':>5}")
    L.append(hdr)
    for name, a in rows:
        L.append(f"  {name[:22]:22} {a['n_rollouts']:3d} {a['u_g_mean']:6.3f} {a['u_g_std']:6.3f} "
                 f"{a['effective_yield']:6.2f} {a['submit_rate']:7.2f} {a['parse_fail_rate']:7.2f} "
                 f"{a.get('has_tests_frac', float('nan')):6.2f} {a['summaries_per_episode']:5.1f}")
    if rows:
        n = len(rows)
        agg = {k: statistics.fmean([a[k] for _, a in rows]) for k in
               ("u_g_mean", "u_g_std", "effective_yield", "submit_rate", "parse_fail_rate")}
        degen = sum(1 for _, a in rows if a["degenerate_group"]) / n
        no_tests = sum(1 for _, a in rows if a.get("has_tests_frac", 1.0) < 1.0) / n
        L += ["", "VERDICT",
              f"  mean u_g={agg['u_g_mean']:.3f}  mean within-task sigma={agg['u_g_std']:.3f}",
              f"  effective sample yield = {agg['effective_yield']:.2f}"
              f"   (fraction of rollouts that reach the loss)",
              f"  submit rate = {agg['submit_rate']:.2f}   parse-failure rate = "
              f"{agg['parse_fail_rate']:.2f}",
              f"  degenerate groups (sigma=0) = {degen:.2f} of tasks"]
        if no_tests > 0:
            L.append(f"  ⛔ A0 FAIL: {no_tests:.0%} of tasks had total_tests=0 — reward is "
                     f"structurally 0. Fix the payload path before anything else.")
        if agg["effective_yield"] < 0.1:
            L.append("  ⛔ effective yield ≈ 0 — masking discards nearly every rollout. Try "
                     "--grade-turn-exhausted and/or --eager-submit; NO credit scheme can learn here.")
        if degen > 0.9:
            L.append("  ⛔ nearly every group is degenerate (sigma=0) — no advantage for any token. "
                     "Change the difficulty band (--difficulty) or the model, not the algorithm.")
        if agg["parse_fail_rate"] > 0.2:
            L.append("  ⚠️  high parse-failure rate — THEN the action space is implicated.")
        elif agg["submit_rate"] < 0.02:
            L.append("  ⚠️  the policy essentially never submits, yet parsing succeeds: the "
                     "TERMINATION PROTOCOL is the bottleneck, not the action space.")
    return "\n".join(L)


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(prog="hrl-trainability")
    p.add_argument("--stub", action="store_true", help="offline mechanics check (no GPU/server)")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--base-url", default=None)
    p.add_argument("--benchmark", default="livecodebench")
    p.add_argument("--n-tasks", type=int, default=10)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--context-L", type=int, default=8192)
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--difficulty", default=None, help='e.g. "easy,medium"')
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--cumulative", action="store_true", help="union all releases (1055 tasks)")
    p.add_argument("--eager-submit", action="store_true", help="A4(a) unconditional-submit prompt")
    p.add_argument("--grade-turn-exhausted", action="store_true",
                   help="A4(c) mask only max_summaries, so turn-exhausted rollouts still train")
    a = p.parse_args()

    mask = ("max_summaries",) if a.grade_turn_exhausted else ("budget", "max_summaries")

    if a.stub:
        from harness_rl.benchmarks.base import EnvStub
        from harness_rl.serving.client import StubModel

        def never_submits(msgs):
            if msgs and "Summarize your progress" in (msgs[-1].content or ""):
                return "SUMMARY"
            return "```bash\nls\n```"              # the measured 3B behaviour: never TASK_COMPLETE

        cfg = SUPORolloutConfig(context_L=600, max_turns=8, recency_turns=3, mask_reasons=mask)
        task = TaskSpec(task_id="stub/0", benchmark="stub", instruction="x")
        rolls = [supo_rollout(task,
                              EnvStub(observations=["obs " + ("tok " * 300)] * 30,
                                      final_u_g=1.0 if i else 0.0),
                              StubModel(never_submits), cfg, system_prompt="SYS")
                 for i in range(a.group_size)]
        print(format_report([("stub/0", audit_rollouts(rolls, mask))]))
        return

    from harness_rl.benchmarks import make_benchmark
    adapter = make_benchmark(a.benchmark)
    for attr, val in (("cumulative", a.cumulative), ("eager_submit", a.eager_submit)):
        if hasattr(adapter, attr) and val:
            setattr(adapter, attr, val)
            adapter._cache = None
    model = (ModelClient.for_sglang(a.model, base_url=a.base_url) if a.base_url
             else ModelClient(a.model))
    pool = pool_report(adapter, a.n_tasks, a.difficulty, a.shuffle_seed)
    try:
        tasks = adapter.subset(a.n_tasks, difficulty=a.difficulty, shuffle_seed=a.shuffle_seed)
    except TypeError:
        tasks = adapter.subset(a.n_tasks)

    cfg = SUPORolloutConfig(context_L=a.context_L, max_turns=a.max_turns, mask_reasons=mask)
    rows = []
    for t in tasks:
        rolls, outs = [], []
        for _ in range(a.group_size):
            env = adapter.make_env(t)
            r = supo_rollout(t, env, model, cfg, system_prompt=adapter.system_prompt())
            rolls.append(r)
            outs.append(env.verify())
        rows.append((t.task_id, audit_rollouts(rolls, mask, outcomes=outs)))
        print(f"  [{len(rows)}/{len(tasks)}] {t.task_id} u_g={rows[-1][1]['u_g_mean']:.3f} "
              f"yield={rows[-1][1]['effective_yield']:.2f}", flush=True)
    print()
    print(format_report(rows, pool))


if __name__ == "__main__":  # pragma: no cover
    main()
