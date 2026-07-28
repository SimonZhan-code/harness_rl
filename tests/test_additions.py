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


def test_supo_token_categorization():   # #0
    from harness_rl.rl.supo import categorize_tokens
    d = categorize_tokens("Let me look around.\n```bash\nls -la\n```", is_summary=False)
    assert d.get("thinking", 0) > 0 and d.get("tool_call", 0) > 0
    assert "tool_call" in categorize_tokens("Done.\nTASK_COMPLETE", is_summary=False)
    assert categorize_tokens("free-form reasoning, no action", is_summary=False).keys() == {"thinking"}
    s = categorize_tokens("SUMMARY of progress so far", is_summary=True)
    assert set(s) == {"summarization"} and s["summarization"] > 0


def test_supo_category_advantage_report():   # #1
    from harness_rl.rl.reward import category_advantage_report
    from harness_rl.rl.supo import CATEGORIES
    r_succ = _supo_stub_rollout()
    r_fail = _supo_stub_rollout()
    r_fail.outcome_u_g = 0.0
    rep = category_advantage_report([r_succ, r_fail])
    assert set(rep["tokens"]) == set(CATEGORIES) and rep["n_rollouts"] == 2
    assert sum(rep["tokens"].values()) > 0
    assert sum(rep["tokens_success"].values()) > 0 and sum(rep["tokens_fail"].values()) > 0
    assert abs(sum(rep["token_frac"].values()) - 1.0) < 1e-6


# --- per-category entropy + KL (training-hook mechanics) ---

def _word_offsets(text):
    import re
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def test_categorize_spans_action_split():
    from harness_rl.rl.supo import categorize_spans
    out = "Let me look.\n```bash\nls -la\n```\nok"
    spans = categorize_spans(out, is_summary=False)
    cats = [c for c, _, _ in spans]
    assert "thinking" in cats and "tool_call" in cats
    # tool_call span must cover the fenced block exactly
    tc = next((s, e) for c, s, e in spans if c == "tool_call")
    assert out[tc[0]:tc[1]].startswith("```bash") and out[tc[0]:tc[1]].endswith("```")
    assert categorize_spans("SUMMARY", is_summary=True) == [("summarization", 0, len("SUMMARY"))]


def test_label_tokens_and_bucket():
    from harness_rl.rl.metrics import bucket_by_category, label_tokens
    from harness_rl.rl.supo import categorize_spans
    out = "think hard\n```bash\nls\n```"
    offs = _word_offsets(out)
    labels = label_tokens(categorize_spans(out, is_summary=False), offs)
    assert "thinking" in labels and "tool_call" in labels
    vals = [1.0] * len(offs)
    b = bucket_by_category(labels, vals)
    assert b["thinking"]["count"] + b["tool_call"]["count"] == sum(1 for x in labels if x)
    assert all(d["mean"] == 1.0 for d in b.values())


def test_token_kl_k3_nonnegative_and_zero_at_equal():
    from harness_rl.rl.metrics import token_kl_k3
    assert abs(token_kl_k3(-1.3, -1.3)) < 1e-12          # identical policies → 0
    assert token_kl_k3(-2.0, -1.0) > 0 and token_kl_k3(-1.0, -2.0) > 0   # k3 is always >= 0


def test_entropy_from_logprobs_matches_uniform():
    import math

    from harness_rl.rl.metrics import entropy_from_logprobs
    k = 4
    uniform = [math.log(1.0 / k)] * k
    assert abs(entropy_from_logprobs(uniform) - math.log(k)) < 1e-9   # H(uniform) = log k
    peaked = [math.log(0.97)] + [math.log(0.01)] * 3
    assert entropy_from_logprobs(peaked) < entropy_from_logprobs(uniform)


def test_supo_category_metrics_and_merge():
    from harness_rl.rl.metrics import merge_category_metrics, supo_category_metrics
    a = "reason\n```bash\nls\n```"
    m1 = supo_category_metrics(a, False, _word_offsets(a),
                               entropy=[1.0] * len(_word_offsets(a)),
                               kl=[0.1] * len(_word_offsets(a)))
    s = "a concise summary digest"
    m2 = supo_category_metrics(s, True, _word_offsets(s),
                               entropy=[0.5] * len(_word_offsets(s)),
                               kl=[0.2] * len(_word_offsets(s)))
    rep = merge_category_metrics([m1, m2])
    assert rep["tokens"]["summarization"] == len(_word_offsets(s))
    assert abs(rep["entropy"]["summarization"]["mean"] - 0.5) < 1e-9
    assert abs(rep["kl"]["summarization"]["mean"] - 0.2) < 1e-9
    assert "thinking" in rep["entropy"] or "tool_call" in rep["entropy"]
