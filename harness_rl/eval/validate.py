"""Env-build validation: confirm a served model can drive inference on each benchmark.

Two env modes:
  --env-mode local  (default): HOST-NATIVE. Runs the harness with a real host `bash` shell
     (LocalShellEnv, NO container) against a representative task per benchmark. Validates that
     the model does coherent multi-turn tool-use inference in each benchmark's style. Works on
     boxes where Docker/Podman can't run (e.g. unprivileged vast.ai instances). Reports whether
     inference ran + completed, NOT a graded benchmark score.
  --env-mode docker : runs the real benchmark task sandbox + verifier (needs Docker + datasets).

webgame is generation-only → validated by a single completion under its system prompt.
Use `--stub` to exercise the harness path offline (no GPU/server).
"""
from __future__ import annotations

import argparse
import time
import traceback

from harness_rl.gamma import make_gamma
from harness_rl.harness.agent import run_episode
from harness_rl.serving.client import ModelClient, StubModel
from harness_rl.types import BudgetCaps, Message, Role, TaskSpec

TRAINABLE = ["terminal_bench_2", "swebench_pro", "gamedev"]
ALL_BENCHES = TRAINABLE + ["webgame"]

# Representative host-native tasks (no dataset / no container needed) — exercise each
# benchmark's tool-use style so we can validate model inference on the box.
REPRESENTATIVE_TASKS = {
    "terminal_bench_2":
        "Write the number of lines in /etc/hostname into a file called count.txt, "
        "verify it with cat, then output TASK_COMPLETE.",
    "swebench_pro":
        "Create math_utils.py with an add(a,b) function and test_math.py that asserts "
        "add(2,3)==5. Run `python -m pytest -q test_math.py`. If it passes, output TASK_COMPLETE.",
    "gamedev":
        "Create a GDScript file player.gd with a `func _ready(): print(\"hello\")`. "
        "Show it with cat, then output TASK_COMPLETE.",
}


def _check_local(bench_name: str, model: ModelClient) -> tuple[bool, str]:
    """Host-native harness run against a representative task (real bash, no container)."""
    from harness_rl.benchmarks import make_benchmark
    from harness_rl.benchmarks.base import LocalShellEnv

    bench = make_benchmark(bench_name)
    task = TaskSpec(task_id=f"{bench_name}/local", benchmark=bench_name,
                    instruction=REPRESENTATIVE_TASKS[bench_name])
    env = LocalShellEnv()
    tr = run_episode(task, env, make_gamma("G3_structured_memory"), model,
                     budget=BudgetCaps(max_turns=8), system_prompt=bench.system_prompt())
    o = tr.outcome
    ran = bool(o and o.cost_tokens > 0 and o.turns > 0)
    return ran, f"turns={o.turns} tokens={o.cost_tokens} reason={o.terminated_reason}"


def _check_docker(bench_name: str, model: ModelClient) -> tuple[bool, str]:
    from harness_rl.benchmarks import make_benchmark
    from harness_rl.eval.probe import GammaProbe

    bench = make_benchmark(bench_name)
    if not bench.subset(1):
        return False, "no tasks found (dataset/registry not present)"
    probe = GammaProbe(model, bench, ["G0_truncate"], out_dir="./runs/validate",
                       budget=BudgetCaps(max_turns=6))
    res = probe.run(n_tasks=1, long_horizon=False)
    return ("G0_truncate" in res.per_variant_u_g,
            f"u_g={res.per_variant_u_g.get('G0_truncate')}")


def _check_webgame(model: ModelClient) -> tuple[bool, str]:
    from harness_rl.benchmarks import make_benchmark

    bench = make_benchmark("webgame")
    out = model.chat([
        Message(role=Role.SYSTEM, content=bench.system_prompt()),
        Message(role=Role.USER, content="Spec: a minimal HTML/JS clicker game. Reply with a short plan."),
    ])
    return bool(out.text.strip()), f"completion_tokens={out.completion_tokens}"


def validate(model: ModelClient, benchmarks: list[str], env_mode: str = "local") -> dict:
    check = _check_local if env_mode == "local" else _check_docker
    results: dict[str, tuple[bool, str, float]] = {}
    for b in benchmarks:
        t0 = time.time()
        try:
            ok, detail = _check_webgame(model) if b == "webgame" else check(b, model)
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
    p.add_argument("--model", default="google/gemma-4-12b-it")
    p.add_argument("--base-url", default=None, help="SGLang endpoint; omit for closed API model")
    p.add_argument("--benchmarks", nargs="+", default=ALL_BENCHES)
    p.add_argument("--env-mode", choices=["local", "docker"], default="local",
                   help="local = host bash (no container); docker = real sandbox (needs Docker)")
    p.add_argument("--stub", action="store_true", help="offline harness-path check (no GPU/server)")
    a = p.parse_args()

    if a.stub:
        model: ModelClient = StubModel(_stub_responder, model=a.model)
    elif a.base_url:
        model = ModelClient.for_sglang(served_model=a.model, base_url=a.base_url)
    else:
        model = ModelClient(model=a.model)

    print(f"# validating model={a.model} base_url={a.base_url or '(vendor API)'} env-mode={a.env_mode}")
    results = validate(model, a.benchmarks, env_mode=a.env_mode)
    print("\n| benchmark | result | detail | secs |\n|---|---|---|---|")
    all_ok = True
    for b, (ok, detail, secs) in results.items():
        all_ok &= ok
        print(f"| {b} | {'PASS' if ok else 'FAIL'} | {detail} | {secs:.1f} |")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
