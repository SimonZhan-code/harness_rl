"""End-to-end no-GPU test of the harness → gamma → logging path, and the base_url seam.

Proves the harness is model-agnostic behind the OpenAI-compatible seam: swapping the
StubModel responder is the only change needed to drive the whole loop (same as swapping
base_url between vLLM and a SkyRL-served policy in Step 2).
"""
from __future__ import annotations

from harness_rl.benchmarks.base import EnvStub
from harness_rl.gamma import make_gamma
from harness_rl.harness.agent import parse_action, run_episode
from harness_rl.logging import dict_to_trace, trace_to_dict
from harness_rl.serving.client import StubModel
from harness_rl.types import BudgetCaps, TaskSpec

TASK = TaskSpec(task_id="t/seam", benchmark="stub", instruction="inspect then submit")


def _responder(messages):
    turn = sum(1 for m in messages if m.role.value == "assistant")
    if turn == 0:
        return "look\n```bash\nls\n```"
    return "TASK_COMPLETE"


def test_parse_action():
    assert parse_action("```bash\nls -la\n```").tool == "bash"
    assert parse_action("TASK_COMPLETE").tool == "submit"
    assert parse_action("no action here") is None


def test_run_episode_populates_trace_and_gamma_snapshots():
    model = StubModel(_responder)
    env = EnvStub(observations=["root", "listing"], final_u_g=1.0)
    gamma = make_gamma("G3_structured_memory")
    tr = run_episode(TASK, env, gamma, model, budget=BudgetCaps(max_turns=5))

    assert tr.outcome is not None and tr.outcome.u_g == 1.0
    assert tr.outcome.terminated_reason == "submit"
    assert len(tr.steps) >= 1
    # the critical invariant: every step logs Gamma's decisions
    for s in tr.steps:
        assert s.gamma_snapshot is not None
        assert s.gamma_snapshot.tokens_in_ctx >= 0
    assert tr.outcome.cost_tokens > 0


def test_seam_is_model_agnostic():
    """Two different 'models' (responders) drive the identical harness code."""
    env1 = EnvStub(observations=["a", "b"], final_u_g=1.0)
    env2 = EnvStub(observations=["a", "b"], final_u_g=0.0)
    g1, g2 = make_gamma("G0_truncate"), make_gamma("G0_truncate")
    tr_pass = run_episode(TASK, env1, g1, StubModel(_responder), budget=BudgetCaps(max_turns=5))
    tr_fail = run_episode(TASK, env2, g2, StubModel(lambda m: "```bash\nsleep 1\n```"),
                          budget=BudgetCaps(max_turns=3))
    assert tr_pass.outcome.u_g == 1.0
    assert tr_fail.outcome.u_g == 0.0
    assert tr_fail.outcome.terminated_reason == "budget"  # never submitted


def test_trace_roundtrip():
    model = StubModel(_responder)
    env = EnvStub(observations=["root", "listing"], final_u_g=1.0)
    tr = run_episode(TASK, env, make_gamma("G1_retrieval"), model, budget=BudgetCaps(max_turns=5))
    restored = dict_to_trace(trace_to_dict(tr))
    assert restored.task_id == tr.task_id
    assert len(restored.steps) == len(tr.steps)
    assert restored.outcome.u_g == tr.outcome.u_g
