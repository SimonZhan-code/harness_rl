"""Unit tests for the Gamma context managers (no GPU, no network)."""
from __future__ import annotations

from harness_rl.gamma import make_gamma
from harness_rl.types import Action, Observation, TaskSpec

TASK = TaskSpec(task_id="t/1", benchmark="stub", instruction="do the thing")


def _drive(gamma, n_steps: int):
    gamma.reset(TASK, system_prompt="SYS")
    for i in range(n_steps):
        gamma.build_context(budget_tokens=200)
        gamma.on_step(Observation(text=f"observation {i} touching file_{i}.py"),
                      Action(tool="bash", args={"cmd": f"echo {i}"}, raw=f"```bash\necho {i}\n```"))
    return gamma


def test_registry_has_four_variants():
    from harness_rl.gamma import GAMMA_REGISTRY
    assert set(GAMMA_REGISTRY) == {
        "G0_truncate", "G1_retrieval", "G2_summarize", "G3_structured_memory"}


def test_g0_truncate_respects_budget_and_reports_dropped():
    g = _drive(make_gamma("G0_truncate"), n_steps=20)
    msgs = g.build_context(budget_tokens=80)
    snap = g.snapshot()
    assert snap.tokens_in_ctx <= 80 or len(msgs) <= 3  # fits budget (or only head+task)
    assert snap.dropped, "long history under a tight budget should drop old turns"


def test_g1_retrieval_pulls_relevant_old_turn():
    g = make_gamma("G1_retrieval", recency_turns=2, retrieve_k=2)
    g.reset(TASK, "SYS")
    # early distinctive turn, then filler, then a query referencing the early turn
    g.on_step(Observation(text="created config at special_widget.cfg"), None)
    for i in range(8):
        g.on_step(Observation(text=f"noise step {i}"),
                  Action(tool="bash", args={}, raw="```bash\nnoop\n```"))
    g.on_step(Observation(text="need to update special_widget.cfg now"), None)
    g.build_context(budget_tokens=10_000)
    snap = g.snapshot()
    assert snap.retrieved, "should retrieve the distinctive early turn by relevance"


def test_g2_summarize_compacts_without_hard_drop():
    g = _drive(make_gamma("G2_summarize", recency_turns=2, compact_every=2), n_steps=10)
    g.build_context(budget_tokens=10_000)
    snap = g.snapshot()
    assert snap.summarized, "stale turns should be summarized, not dropped outright"
    assert not snap.dropped


def test_g3_structured_memory_tracks_files_and_errors():
    g = make_gamma("G3_structured_memory", recency_turns=2)
    g.reset(TASK, "SYS")
    g.on_step(Observation(text="edited game.gd"), Action(tool="bash", args={"cmd": "vim game.gd"},
                                                         raw="```bash\nvim game.gd\n```"))
    g.on_step(Observation(text="Traceback: assertion failed in game.gd"), None)
    g.build_context(budget_tokens=10_000)
    snap = g.snapshot()
    assert snap.notes["n_files"] >= 1
    assert snap.notes["has_error"] is True
