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
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--base-url", default=None, help="SGLang endpoint for a real 3B run")
    p.add_argument("--benchmark", default="livecodebench")
    p.add_argument("--context-L", type=int, default=4096)
    p.add_argument("--n-tasks", type=int, default=2)
    a = p.parse_args()

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
        assert r0.num_summaries > 0, "compaction should fire with tiny L + long observations"
        assert any(s["is_summary"] for s in samples), "summary turns must be trainable samples"
        assert all(s["reward"] == samples[0]["reward"] for s in samples), "all segments share the advantage"
        print("\nSTUB SUPO OK — compaction fired; summaries trainable; per-category token mass reported.")
        return

    from harness_rl.benchmarks import make_benchmark
    model = (ModelClient.for_sglang(a.model, base_url=a.base_url) if a.base_url
             else ModelClient(a.model))
    bench = make_benchmark(a.benchmark)
    cfg = SUPORolloutConfig(context_L=a.context_L, max_turns=40)
    rollouts = []
    for task in bench.subset(a.n_tasks, long_horizon=False):
        env = bench.make_env(task)
        rollout = supo_rollout(task, env, model, cfg, system_prompt=bench.system_prompt())
        rollouts.append(rollout)
        print(f"\n== {task.task_id} ==")
        _report(rollout, supo_samples(rollout))
    _print_category_report(category_advantage_report(rollouts))
    total_sum = sum(r.num_summaries for r in rollouts)
    print(f"\ntotal summaries across {a.n_tasks} tasks: {total_sum} "
          f"({'compaction fired' if total_sum else 'NO compaction — lower --context-L'})")


if __name__ == "__main__":
    main()
