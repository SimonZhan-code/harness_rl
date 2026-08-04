"""hrl-supo-check — validate SUPO rollout mechanics (arXiv 2510.06727), no training.

  --stub   : offline (no GPU). Drives supo_rollout with a StubModel over long observations and a
             tiny context_L so compaction MUST fire → asserts sub-trajectories form, summary turns
             are recorded as trainable, and supo_samples has the Thm 3.2 structure (one sample per
             segment incl. summaries, all sharing the outcome reward + group_key).
  (default): real — point --base-url at a served model (e.g. Qwen2.5-Coder-3B on SGLang) and run
             on LiveCodeBench; confirms summaries actually fire under the 3B model.
"""
from __future__ import annotations

import argparse
import re

from harness_rl.rl.metrics import category_entropy_kl_report
from harness_rl.rl.reward import category_advantage_report, supo_samples
from harness_rl.rl.supo import CATEGORIES, SUPORolloutConfig, supo_rollout
from harness_rl.serving.client import ModelClient, StubModel
from harness_rl.types import TaskSpec


def _report(rollout, samples) -> None:
    print(f"turns={rollout.turns} num_summaries={rollout.num_summaries} "
          f"segments={len(rollout.segments)} u_g={rollout.outcome_u_g} "
          f"hit_limit={rollout.hit_limit} reason={rollout.terminated_reason}")
    print(f"training samples (Thm 3.2, one per sub-trajectory segment): {len(samples)}")
    if samples:
        print(f"  shared group_key={samples[0]['group_key']!r} shared reward={samples[0]['reward']}")
        print(f"  summary samples (trainable): {sum(1 for s in samples if s['is_summary'])}/{len(samples)}")
    print(f"  sub-trajectory ids: {sorted({s.segment for s in rollout.segments})}")
    cat = {c: 0 for c in CATEGORIES}
    for s in rollout.segments:
        for c, n in (s.category_tokens or {}).items():
            cat[c] += n
    print("  #0 category tokens: " + "  ".join(f"{c}={cat[c]}" for c in CATEGORIES))


def _print_category_report(rep: dict) -> None:
    print(f"\n#1 per-category advantage-weighted token mass "
          f"(n={rep['n_rollouts']}, reward mean={rep['reward_mean']:.2f} std={rep['reward_std']:.2f}):")
    print(f"  {'category':14} {'tokens':>8} {'frac':>6} {'adv_mass':>10} {'tok(succ)':>10} {'tok(fail)':>10}")
    for c in CATEGORIES:
        print(f"  {c:14} {rep['tokens'][c]:8d} {rep['token_frac'][c]:6.2f} "
              f"{rep['adv_mass'][c]:10.2f} {rep['tokens_success'][c]:10d} {rep['tokens_fail'][c]:10d}")


def _word_offsets(text: str) -> list[tuple[int, int]]:
    """A stand-in tokenizer offset mapping (whitespace tokens) for the offline entropy/KL demo.
    At training time these come from the real tokenizer via `return_offsets_mapping=True`."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _synthetic_segment_stats(rollouts) -> list[dict]:
    """Build the trainer-hook input from stub rollouts with SYNTHETIC per-token entropy/KL
    (no logits offline). Deterministic so the demo is reproducible; at training time `entropy`
    comes from the policy logits and `kl` from k1/k3 vs pi_ref (see rl/metrics.py)."""
    stats = []
    for r in rollouts:
        for s in r.segments:
            offs = _word_offsets(s.output)
            # deterministic stand-in values; summary turns given a lower-entropy signature so the
            # dashboard visibly separates categories (purely illustrative, not real model stats)
            base = 0.6 if s.is_summary else 1.2
            ent = [base + 0.05 * (i % 4) for i in range(len(offs))]
            kl = [0.03 + 0.01 * (i % 3) for i in range(len(offs))]
            stats.append({"output": s.output, "is_summary": s.is_summary,
                          "offsets": offs, "entropy": ent, "kl": kl})
    return stats


def _print_entropy_kl_report(rep: dict) -> None:
    print("\nper-category entropy & KL(pi_new||pi_ref)  [SYNTHETIC per-token values — "
          "training-hook mechanics demo; real values come from the slime loss step]:")
    print(f"  {'category':14} {'tokens':>8} {'entropy':>10} {'kl':>10}")
    for c in CATEGORIES:
        n = rep["tokens"].get(c, 0)
        e = rep["entropy"].get(c, {}).get("mean", 0.0)
        k = rep["kl"].get(c, {}).get("mean", 0.0)
        print(f"  {c:14} {n:8d} {e:10.3f} {k:10.4f}")


def _print_tree(t) -> None:
    print(f"tree: nodes={len(t.nodes)} leaves={len(t.leaves)} summary_nodes="
          f"{sum(1 for n in t.nodes if n.is_summary)} num_summaries={t.num_summaries}")
    for n in t.nodes:
        kind = "SUMMARY" if n.is_summary else "action "
        leaf = f" u_g={n.u_g}{' OVERLONG' if n.hit_limit else ''}" if n.u_g is not None else ""
        val = f" V={n.value:.3f}" if n.value is not None else " V=None"
        print(f"  [{n.id:3d}] parent={str(n.parent):>4} {kind}{val}{leaf} kids={n.children}")


def _print_tree_advantages(trees, w_micro: float) -> None:
    from harness_rl.rl.tree_reward import tree_samples
    samples, stats = tree_samples(trees, w_micro=w_micro)
    print(f"\ntwo-level advantage (macro GRPO + w={w_micro}·micro GiGPO): "
          f"leaves={stats['n_leaves']}/{stats['n_leaves_total']} (overlong-masked="
          f"{stats['n_overlong_leaves']}) mu={stats['mu']:.3f} sigma={stats['sigma']:.3f} "
          f"nodes={stats['n_nodes']} scored={stats['n_scored']}")
    if stats.get("all_masked"):
        print("  ⚠️  ALL leaves masked → ZERO gradient this batch. Raise --max-leaves/turn caps, "
              "lower --context-L, or pass --grade-turn-exhausted.")
    if stats.get("degenerate_group"):
        print("  ⚠️  degenerate group (sigma=0): every leaf scored the same → macro carries no "
              "signal; only the sibling-relative micro term can teach this batch.")
    print(f"  {'node':>5} {'kind':8} {'advantage':>10}")
    for s in samples:
        print(f"  {s['node_id']:5d} {'SUMMARY' if s['is_summary'] else 'action':8} "
              f"{s['advantage']:10.3f}")
    print(f"  samples={len(samples)} (summary={sum(1 for s in samples if s['is_summary'])})")


def _tree_stub_check(branch_factor: int, max_leaves: int, w_micro: float,
                     branch_depth: int = 1) -> None:
    """Offline mechanics check for the summary-branching tree + macro/micro advantages."""
    from harness_rl.benchmarks.base import EnvStub
    from harness_rl.rl.tree_reward import tree_advantages
    from harness_rl.rl.tree_supo import TreeConfig, tree_supo_rollout

    def make_responder():
        st = {"s": 0, "a": 0}

        def r(messages):
            if messages and "Summarize your progress" in (messages[-1].content or ""):
                st["s"] += 1
                return f"SUMMARY variant {st['s']}: explored, ran tests, plan noted."
            st["a"] += 1
            return "```bash\nls\n```" if st["a"] % 4 else "TASK_COMPLETE"
        return r

    # distinct per-leaf outcomes so macro/micro are non-degenerate (verify() runs once per leaf)
    seq = [1.0, 0.0, 0.5, 1.0, 0.0, 0.25, 0.75, 1.0]
    cnt = {"n": 0}

    def verify_hook():
        cnt["n"] += 1
        return seq[(cnt["n"] - 1) % len(seq)]

    cfg = TreeConfig(context_L=600, max_turns=14, max_summaries=6, recency_turns=3,
                     branch_factor=branch_factor, branch_depth=branch_depth,
                     max_leaves=max_leaves, summary_temperature=0.9)
    task = TaskSpec(task_id="stub/tree", benchmark="stub", instruction="demo task")
    env = EnvStub(observations=["obs " + ("token " * 300)] * 40, verify_hook=verify_hook)
    t = tree_supo_rollout(task, env, StubModel(make_responder()), cfg, system_prompt="SYS")

    from harness_rl.rl.tree_reward import tree_backup
    tree_backup(t)            # fill node.value so the tree print shows the subtree backup
    _print_tree(t)
    _print_tree_advantages([t], w_micro)

    assert t.num_summaries > 0, "compaction must fire (tiny L + long observations)"
    assert len(t.leaves) > 1, "tree must branch into multiple leaves"
    assert len(t.leaves) <= max_leaves, f"leaf budget violated: {len(t.leaves)} > {max_leaves}"
    assert any(n.is_summary for n in t.nodes), "summary nodes must be recorded"
    # micro is zero-sum across a sibling group; w_micro=0 must reduce to flat (macro-only)
    a_full, _ = tree_advantages([t], w_micro=w_micro)
    a_flat, _ = tree_advantages([t], w_micro=0.0)
    for n in t.nodes:
        if not n.is_summary and (0, n.id) in a_full:
            assert abs(a_full[(0, n.id)] - a_flat[(0, n.id)]) < 1e-9, \
                "action nodes must be macro-only (unaffected by w_micro)"
    print("\nSTUB TREE OK — branched at summaries only; subtree backup + macro/micro advantages "
          "computed; action nodes macro-only; leaf budget respected.")


def _make_stub_responder():
    # count OUR action calls (compaction removes context messages, so we can't count those)
    state = {"n": 0}

    def responder(messages):
        last = messages[-1].content if messages else ""
        if "Summarize your progress" in (last or ""):
            return "SUMMARY: explored the repo, ran a test, current plan noted."
        state["n"] += 1
        return "```bash\nls\n```" if state["n"] < 8 else "TASK_COMPLETE"

    return responder


def main() -> None:
    p = argparse.ArgumentParser(prog="hrl-supo-check")
    p.add_argument("--stub", action="store_true", help="offline mechanics check (no GPU/server)")
    p.add_argument("--tree-stub", action="store_true",
                   help="offline check of the summary-branching tree + macro/micro advantages")
    p.add_argument("--tree", action="store_true", help="real run using the tree rollout")
    p.add_argument("--branch-factor", type=int, default=3, help="B summaries per compaction")
    p.add_argument("--branch-depth", type=int, default=1,
                   help="branch at the first D compactions (balanced; B**D leaves)")
    p.add_argument("--max-leaves", type=int, default=4, help="leaf budget per tree")
    p.add_argument("--w-micro", type=float, default=1.0, help="weight on the sibling-relative term")
    p.add_argument("--group-size", type=int, default=2, help="trees per task (tree mode)")
    p.add_argument("--grade-turn-exhausted", action="store_true",
                   help="mask ONLY max_summaries (train on turn-exhausted rollouts too) — needed "
                        "for policies that rarely emit TASK_COMPLETE")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--base-url", default=None, help="SGLang endpoint for a real 3B run")
    p.add_argument("--benchmark", default="livecodebench")
    p.add_argument("--context-L", type=int, default=4096)
    p.add_argument("--n-tasks", type=int, default=2)
    a = p.parse_args()

    if a.tree_stub:
        _tree_stub_check(a.branch_factor, a.max_leaves, a.w_micro, a.branch_depth)
        return

    if a.stub:
        from harness_rl.benchmarks.base import EnvStub
        cfg = SUPORolloutConfig(context_L=600, max_turns=15, recency_turns=3, max_summaries=20)
        rollouts = []  # a small GROUP with mixed success so the advantage report is non-trivial
        for i in range(4):
            task = TaskSpec(task_id=f"stub/{i}", benchmark="stub", instruction="demo task")
            env = EnvStub(observations=["obs " + ("token " * 300)] * 30,
                          final_u_g=1.0 if i % 2 == 0 else 0.0)
            rollouts.append(supo_rollout(task, env, StubModel(_make_stub_responder()), cfg,
                                         system_prompt="You are an agent."))
        r0 = rollouts[0]
        samples = supo_samples(r0)
        _report(r0, samples)
        _print_category_report(category_advantage_report(rollouts))
        ek = category_entropy_kl_report(_synthetic_segment_stats(rollouts))
        _print_entropy_kl_report(ek)
        assert r0.num_summaries > 0, "compaction should fire with tiny L + long observations"
        assert any(s["is_summary"] for s in samples), "summary turns must be trainable samples"
        assert all(s["reward"] == samples[0]["reward"] for s in samples), "all segments share the advantage"
        assert set(ek["entropy"]) and "summarization" in ek["tokens"], "entropy/KL by category wired"
        print("\nSTUB SUPO OK — compaction fired; summaries trainable; per-category token mass, "
              "entropy & KL reported.")
        return

    from harness_rl.benchmarks import make_benchmark
    model = (ModelClient.for_sglang(a.model, base_url=a.base_url) if a.base_url
             else ModelClient(a.model))
    bench = make_benchmark(a.benchmark)

    if a.tree:   # real summary-branching tree run against a served model
        from harness_rl.rl.tree_supo import TreeConfig, tree_supo_rollout
        tcfg = TreeConfig(context_L=a.context_L, max_turns=40, branch_factor=a.branch_factor,
                          branch_depth=a.branch_depth, max_leaves=a.max_leaves,
                          mask_reasons=("max_summaries",) if a.grade_turn_exhausted
                          else ("budget", "max_summaries"))
        for task in bench.subset(a.n_tasks, long_horizon=False):
            trees = [tree_supo_rollout(task, bench.make_env(task), model, tcfg,
                                       system_prompt=bench.system_prompt())
                     for _ in range(a.group_size)]
            print(f"\n== {task.task_id} (group of {len(trees)} trees) ==")
            for t in trees:
                _print_tree(t)
            _print_tree_advantages(trees, a.w_micro)
            tot = sum(t.num_summaries for t in trees)
            print(f"summaries across group: {tot} "
                  f"({'branching fired' if tot else 'NO compaction — lower --context-L'})")
        return

    cfg = SUPORolloutConfig(context_L=a.context_L, max_turns=40)
    rollouts = []
    for task in bench.subset(a.n_tasks, long_horizon=False):
        env = bench.make_env(task)
        rollout = supo_rollout(task, env, model, cfg, system_prompt=bench.system_prompt())
        rollouts.append(rollout)
        print(f"\n== {task.task_id} ==")
        _report(rollout, supo_samples(rollout))
    _print_category_report(category_advantage_report(rollouts))
    _print_entropy_kl_report(category_entropy_kl_report(_synthetic_segment_stats(rollouts)))
    total_sum = sum(r.num_summaries for r in rollouts)
    print(f"\ntotal summaries across {a.n_tasks} tasks: {total_sum} "
          f"({'compaction fired' if total_sum else 'NO compaction — lower --context-L'})")


if __name__ == "__main__":
    main()
