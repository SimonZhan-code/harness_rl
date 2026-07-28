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

from harness_rl.rl.reward import supo_samples
from harness_rl.rl.supo import SUPORolloutConfig, supo_rollout
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
        model = StubModel(_make_stub_responder())
        cfg = SUPORolloutConfig(context_L=600, max_turns=15, recency_turns=3, max_summaries=20)
        task = TaskSpec(task_id="stub/1", benchmark="stub", instruction="demo task")
        env = EnvStub(observations=["obs " + ("token " * 300)] * 30, final_u_g=1.0)
        rollout = supo_rollout(task, env, model, cfg, system_prompt="You are an agent.")
        samples = supo_samples(rollout)
        _report(rollout, samples)
        assert rollout.num_summaries > 0, "compaction should fire with tiny L + long observations"
        assert any(s["is_summary"] for s in samples), "summary turns must be trainable samples"
        assert all(s["reward"] == samples[0]["reward"] for s in samples), "all segments share the advantage"
        print("STUB SUPO OK — compaction fired; summaries recorded as trainable, shared-advantage segments.")
        return

    from harness_rl.benchmarks import make_benchmark
    model = (ModelClient.for_sglang(a.model, base_url=a.base_url) if a.base_url
             else ModelClient(a.model))
    bench = make_benchmark(a.benchmark)
    cfg = SUPORolloutConfig(context_L=a.context_L, max_turns=40)
    total_sum = 0
    for task in bench.subset(a.n_tasks, long_horizon=False):
        env = bench.make_env(task)
        rollout = supo_rollout(task, env, model, cfg, system_prompt=bench.system_prompt())
        total_sum += rollout.num_summaries
        print(f"\n== {task.task_id} ==")
        _report(rollout, supo_samples(rollout))
    print(f"\ntotal summaries across {a.n_tasks} tasks: {total_sum} "
          f"({'compaction fired' if total_sum else 'NO compaction — lower --context-L'})")


if __name__ == "__main__":
    main()
