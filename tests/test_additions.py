"""Tests for the added benchmarks (LiveCodeBench/SWE-Lite/GameCraft) + G5 memory arch."""
from __future__ import annotations

import os
import tempfile

from harness_rl.benchmarks import BENCHMARK_REGISTRY, TRAINABLE_BENCHMARKS
from harness_rl.benchmarks._livecode import decode_tests, run_tests
from harness_rl.gamma import GAMMA_REGISTRY, make_gamma
from harness_rl.types import Action, Observation, TaskSpec


def test_new_benchmarks_registered():
    for b in ("livecodebench", "swebench_lite", "swebench_verified", "gamecraft"):
        assert b in BENCHMARK_REGISTRY
    assert {"livecodebench", "swebench_lite", "swebench_verified"} <= TRAINABLE_BENCHMARKS
    assert "gamecraft" not in TRAINABLE_BENCHMARKS  # eval-only (rubric judge)


def test_g5_registered_and_recalls():
    assert "G5_external" in GAMMA_REGISTRY
    g = make_gamma("G5_external", recency_turns=2, retrieve_k=2, compact_every=2)
    g.reset(TaskSpec(task_id="t", benchmark="x", instruction="goal"), "SYS")
    g.on_step(Observation(text="created config at special_widget.cfg"), None)
    for i in range(6):
        g.build_context(10_000)
        g.on_step(Observation(text=f"noise {i}"), Action(tool="bash", args={}, raw="x"))
    g.on_step(Observation(text="need special_widget.cfg now"), None)
    g.build_context(10_000)
    s = g.snapshot()
    assert s.summarized, "stale turns should be folded into the rolling summary"
    assert s.retrieved, "the distinctive early turn should be recalled from archival memory"


def test_livecode_stdin_grading():
    wd = tempfile.mkdtemp()
    open(os.path.join(wd, "solution.py"), "w").write("a,b=map(int,input().split())\nprint(a+b)\n")
    passed, total = run_tests(wd, [{"input": "2 3", "output": "5"},
                                   {"input": "10 20", "output": "30"}], None)
    assert (passed, total) == (2, 2)


def test_livecode_functional_grading():
    wd = tempfile.mkdtemp()
    open(os.path.join(wd, "solution.py"), "w").write(
        "class Solution:\n def add(self,a,b): return a+b\n")
    passed, total = run_tests(wd, [{"input": "2\n3", "output": "5"}], "add")
    assert (passed, total) == (1, 1)


def test_livecode_wrong_solution_scores_zero():
    wd = tempfile.mkdtemp()
    open(os.path.join(wd, "solution.py"), "w").write("print(0)\n")
    passed, total = run_tests(wd, [{"input": "2 3", "output": "5"}], None)
    assert passed == 0 and total == 1


def test_decode_tests_roundtrip():
    import base64, json, zlib
    tests = [{"input": "1", "output": "1"}]
    enc = base64.b64encode(zlib.compress(json.dumps(tests).encode())).decode()
    assert decode_tests(enc) == tests
    assert decode_tests(json.dumps(tests)) == tests   # plain-json fallback


# --- SUPO (arXiv 2510.06727) ---

def test_g2_token_trigger_compacts():
    g = make_gamma("G2_summarize", recency_turns=2, compact_at_tokens=200,
                   summary_mode="replace", summarizer=lambda msgs: "SUMMARY")
    g.reset(TaskSpec(task_id="t", benchmark="x", instruction="go"), "SYS")
    fired = False
    for _ in range(8):
        g.build_context(10_000)
        fired = fired or bool(g.snapshot().summarized)   # compaction may fire on any turn
        g.on_step(Observation(text="obs " + ("word " * 80)), Action(tool="bash", args={}, raw="x"))
    assert fired, "token-threshold L should trigger compaction"


def _supo_stub_rollout():
    from harness_rl.benchmarks.base import EnvStub
    from harness_rl.rl.supo import SUPORolloutConfig, supo_rollout
    from harness_rl.serving.client import StubModel

    # count OUR action calls (not context messages — compaction removes those) to decide submit
    state = {"n": 0}

    def r(messages):
        if messages and "Summarize your progress" in (messages[-1].content or ""):
            return "SUMMARY of progress"
        state["n"] += 1
        return "```bash\nls\n```" if state["n"] < 6 else "TASK_COMPLETE"

    cfg = SUPORolloutConfig(context_L=600, max_turns=12, recency_turns=3, max_summaries=20)
    task = TaskSpec(task_id="stub/1", benchmark="stub", instruction="demo")
    env = EnvStub(observations=["obs " + ("token " * 300)] * 30, final_u_g=1.0)
    return supo_rollout(task, env, StubModel(r), cfg, system_prompt="SYS")


def test_supo_rollout_compacts_and_records_summary_turns():
    roll = _supo_stub_rollout()
    assert roll.num_summaries > 0, "tiny L + long obs must trigger policy summarization"
    assert any(s.is_summary for s in roll.segments), "summary turns must be recorded"
    assert sorted({s.segment for s in roll.segments}) == list(range(roll.num_summaries + 1))


def test_supo_samples_thm32_shared_advantage():
    from harness_rl.rl.reward import supo_samples
    roll = _supo_stub_rollout()
    samples = supo_samples(roll)
    assert len(samples) == len(roll.segments)              # one sample per sub-trajectory segment
    assert any(s["is_summary"] for s in samples)           # summary tokens are trainable
    assert len({s["reward"] for s in samples}) == 1        # all segments share the advantage (Thm 3.2)
    assert all(s["group_key"] == "stub/1" for s in samples)


def test_supo_overlong_masking():
    from harness_rl.rl.reward import supo_samples

    class _R:  # duck-typed SUPORollout that hit the limit
        hit_limit = True
        outcome_u_g = 1.0
        task_id = "t"
        segments: list = []

    assert supo_samples(_R(), overlong_mask=True) == []    # overlong → no gradient
