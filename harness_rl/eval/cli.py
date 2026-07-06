"""`hrl-probe` CLI — run the Step 1 Gamma variance probe.

Two modes:
  --stub        : no-GPU dry run. Uses a canned StubModel + EnvStub tasks so the whole
                  harness→gamma→logging path runs offline (for local validation).
  (default)     : real run. Point --base-url at an SGLang server (open model) or omit it and
                  pass a vendor model id via LiteLLM (closed). Runs on vast.ai / a GPU host.
"""
from __future__ import annotations

import argparse

from harness_rl.eval.probe import GammaProbe, spread_table
from harness_rl.serving.client import ModelClient, StubModel


def _stub_responder(messages):
    # Minimal canned policy: inspect, then submit. Enough to exercise the pipeline.
    turn = sum(1 for m in messages if m.role.value == "assistant")
    if turn == 0:
        return "Let me look around.\n```bash\nls -la\n```"
    if turn == 1:
        return "```bash\ncat README.md\n```"
    return "Done.\nTASK_COMPLETE"


def main() -> None:
    p = argparse.ArgumentParser(prog="hrl-probe")
    p.add_argument("--benchmark", default="terminal_bench_2")
    p.add_argument("--model", default="google/gemma-4-12b-it")
    p.add_argument("--base-url", default=None,
                   help="SGLang OpenAI-compatible endpoint (open models); omit for closed API models")
    p.add_argument("--gammas", nargs="+",
                   default=["G0_truncate", "G1_retrieval", "G2_summarize",
                            "G3_structured_memory", "G5_external"])
    p.add_argument("--n-tasks", type=int, default=25)
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--stub", action="store_true", help="no-GPU dry run with a canned model+env")
    args = p.parse_args()

    if args.stub:
        from harness_rl.benchmarks.base import EnvStub
        from harness_rl.types import TaskSpec

        model: ModelClient = StubModel(_stub_responder)

        class _StubBench:
            name = "stub"

            def subset(self, n, long_horizon=False):
                return [TaskSpec(task_id=f"stub/{i}", benchmark="stub",
                                 instruction="demo task", horizon_hint=3) for i in range(n)]

            def make_env(self, task):
                return EnvStub(observations=["repo root", "file contents"], final_u_g=1.0)

            def system_prompt(self):
                return "You are a demo agent."

        probe = GammaProbe(model, _StubBench(), args.gammas, out_dir=args.out_dir)
        res = probe.run(n_tasks=min(args.n_tasks, 5), long_horizon=False)
    else:
        from harness_rl.benchmarks import make_benchmark

        # Open model → SGLang server behind --base-url; closed model → vendor API (no base_url).
        if args.base_url:
            model = ModelClient.for_sglang(served_model=args.model, base_url=args.base_url)
        else:
            model = ModelClient(model=args.model)
        bench = make_benchmark(args.benchmark)
        probe = GammaProbe(model, bench, args.gammas, out_dir=args.out_dir)
        res = probe.run(n_tasks=args.n_tasks, long_horizon=True)

    print(spread_table([res]))
    print(f"\nspread = {res.spread:.3f}  (GO if non-trivial on long-horizon tasks)")


if __name__ == "__main__":
    main()
