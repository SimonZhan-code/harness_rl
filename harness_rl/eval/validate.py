"""Env-build validation: confirm a served model can drive inference on each benchmark.

For each benchmark, run a MINIMAL inference check (not a correctness eval):
  - trainable benches (terminal_bench_2, swebench_pro, gamedev): run 1 task through the
    harness with G0 and confirm an episode completes with a defined outcome u_g.
  - webgame (eval-only, no harness env): a single chat completion under its system prompt,
    confirming the served model responds.

Prints a PASS/FAIL matrix + latency. This is the check `scripts/validate_benchmarks.sh`
runs for each (model × build). Use `--stub` to exercise the harness path offline (no GPU).
"""
from __future__ import annotations

import argparse
import time
import traceback

from harness_rl.eval.probe import GammaProbe
from harness_rl.serving.client import ModelClient, StubModel
from harness_rl.types import BudgetCaps, Message, Role, TaskSpec

TRAINABLE = ["terminal_bench_2", "swebench_pro", "gamedev"]
ALL_BENCHES = TRAINABLE + ["webgame"]


def _check_trainable(bench_name: str, model: ModelClient) -> tuple[bool, str]:
    from harness_rl.benchmarks import make_benchmark

    bench = make_benchmark(bench_name)
    tasks = bench.subset(1, long_horizon=False)
    if not tasks:
        return False, "no tasks found (dataset/registry not present)"
    probe = GammaProbe(model, bench, ["G0_truncate"], out_dir="./runs/validate",
                       budget=BudgetCaps(max_turns=6))
    res = probe.run(n_tasks=1, long_horizon=False)
    ok = "G0_truncate" in res.per_variant_u_g
    return ok, f"u_g={res.per_variant_u_g.get('G0_truncate')}"


def _check_webgame(model: ModelClient) -> tuple[bool, str]:
    from harness_rl.benchmarks import make_benchmark

    bench = make_benchmark("webgame")
    msgs = [
        Message(role=Role.SYSTEM, content=bench.system_prompt()),
        Message(role=Role.USER, content="Spec: a minimal HTML/JS clicker game. Reply with a plan."),
    ]
    out = model.chat(msgs)
    return bool(out.text.strip()), f"completion_tokens={out.completion_tokens}"


def validate(model: ModelClient, benchmarks: list[str]) -> dict[str, tuple[bool, str, float]]:
    results: dict[str, tuple[bool, str, float]] = {}
    for b in benchmarks:
        t0 = time.time()
        try:
            ok, detail = _check_webgame(model) if b == "webgame" else _check_trainable(b, model)
        except Exception as e:  # noqa: BLE001 — validation must report, not crash
            ok, detail = False, f"ERROR: {e.__class__.__name__}: {e}"
            traceback.print_exc()
        results[b] = (ok, detail, time.time() - t0)
    return results


def _stub_responder(messages):
    turn = sum(1 for m in messages if m.role.value == "assistant")
    return "look\n```bash\nls\n```" if turn == 0 else "TASK_COMPLETE"


def main() -> None:
    p = argparse.ArgumentParser(prog="hrl-validate")
    p.add_argument("--model", default="google/gemma-4-9b-it")
    p.add_argument("--base-url", default=None, help="SGLang endpoint; omit for closed API model")
    p.add_argument("--benchmarks", nargs="+", default=ALL_BENCHES)
    p.add_argument("--stub", action="store_true", help="offline harness-path check (no GPU/server)")
    a = p.parse_args()

    if a.stub:
        model: ModelClient = StubModel(_stub_responder, model=a.model)
    elif a.base_url:
        model = ModelClient.for_sglang(served_model=a.model, base_url=a.base_url)
    else:
        model = ModelClient(model=a.model)

    print(f"# validating model={a.model} base_url={a.base_url or '(vendor API)'}")
    results = validate(model, a.benchmarks)
    print(f"\n| benchmark | result | detail | secs |\n|---|---|---|---|")
    all_ok = True
    for b, (ok, detail, secs) in results.items():
        all_ok &= ok
        print(f"| {b} | {'PASS' if ok else 'FAIL'} | {detail} | {secs:.1f} |")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
