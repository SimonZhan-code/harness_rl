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
