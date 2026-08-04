"""Tests for the added benchmarks (LiveCodeBench/SWE-Lite/GameCraft) + G5 memory arch."""
from __future__ import annotations

import os
import tempfile

from harness_rl.benchmarks import BENCHMARK_REGISTRY, TRAINABLE_BENCHMARKS
from harness_rl.benchmarks._livecode import decode_tests, run_tests
from harness_rl.gamma import GAMMA_REGISTRY, make_gamma
from harness_rl.types import Action, Message, Observation, Role, TaskSpec


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


# --- summary-branching TREE rollout + macro/micro advantages ---

def test_env_fork_isolates_state():
    from harness_rl.benchmarks._livecode import LiveCodeExecEnv
    from harness_rl.benchmarks.base import EnvStub
    env = LiveCodeExecEnv(public_tests=[], private_tests=[{"input": "", "output": ""}])
    env.reset()
    open(os.path.join(env.workdir, "solution.py"), "w").write("print(1)\n")
    f = env.fork()
    assert f.workdir != env.workdir
    assert open(os.path.join(f.workdir, "solution.py")).read() == "print(1)\n"   # copied
    open(os.path.join(f.workdir, "solution.py"), "w").write("print(2)\n")        # diverge
    assert open(os.path.join(env.workdir, "solution.py")).read() == "print(1)\n"  # isolated
    s = EnvStub(observations=["a", "b", "c"])
    s.reset()
    s.execute(Action(tool="bash", args={}, raw="x"))
    assert s.fork()._i == s._i


def test_g2_external_compaction_hooks_and_clone():
    g = make_gamma("G2_summarize", recency_turns=2, compact_at_tokens=200,
                   summary_mode="replace", summarizer=lambda m: "SHOULD_NOT_FIRE")
    g.reset(TaskSpec(task_id="t", benchmark="x", instruction="go"), "SYS")
    for _ in range(6):
        g.on_step(Observation(text="obs " + ("word " * 80)), Action(tool="bash", args={}, raw="x"))
    to_sum = g.pending_compaction(200)
    assert to_sum, "pending_compaction must detect the over-L condition"
    assert g._summary == "", "pending_compaction must NOT mutate state"
    a, b = g.clone(), g.clone()
    a.apply_summary("SUMMARY A", 200)
    b.apply_summary("SUMMARY B", 200)
    assert (a._summary, b._summary) == ("SUMMARY A", "SUMMARY B")   # branches independent
    assert g._summary == ""                                          # parent untouched
    assert a.pending_compaction(200) is None, "after apply_summary no compaction is pending"
    a.on_step(Observation(text="new"), Action(tool="bash", args={}, raw="y"))
    assert len(a.history) != len(b.history), "cloned history lists must be independent"


def _tree_stub(branch_factor=2, max_leaves=4, u_seq=(1.0, 0.0, 0.5, 1.0)):
    from harness_rl.benchmarks.base import EnvStub
    from harness_rl.rl.tree_supo import TreeConfig, tree_supo_rollout
    from harness_rl.serving.client import StubModel

    st = {"s": 0, "a": 0}

    def r(messages):
        if messages and "Summarize your progress" in (messages[-1].content or ""):
            st["s"] += 1
            return f"SUMMARY {st['s']}"
        st["a"] += 1
        return "```bash\nls\n```" if st["a"] % 4 else "TASK_COMPLETE"

    cnt = {"n": 0}

    def hook():
        cnt["n"] += 1
        return u_seq[(cnt["n"] - 1) % len(u_seq)]

    cfg = TreeConfig(context_L=600, max_turns=14, max_summaries=6, recency_turns=3,
                     branch_factor=branch_factor, max_leaves=max_leaves)
    task = TaskSpec(task_id="stub/tree", benchmark="stub", instruction="demo")
    env = EnvStub(observations=["obs " + ("token " * 300)] * 40, verify_hook=hook)
    return tree_supo_rollout(task, env, StubModel(r), cfg, system_prompt="SYS")


def test_tree_branches_only_at_summaries_within_budget():
    t = _tree_stub(branch_factor=2, max_leaves=4)
    assert t.num_summaries > 0 and len(t.leaves) > 1
    assert len(t.leaves) <= 4, "max_leaves budget must bound the tree"
    for n in t.nodes:
        if len(n.children) > 1:                      # only summary fan-out creates >1 child
            assert all(t.nodes[c].is_summary for c in n.children), \
                "branching children must all be summary nodes (no action branching)"
    # every summary node's siblings share the identical pre-summary context (the GiGPO anchor)
    for n in t.nodes:
        sibs = [t.nodes[c] for c in n.children if t.nodes[c].is_summary]
        if len(sibs) > 1:
            ctx0 = [(m.role, m.content) for m in sibs[0].seg.context]
            for s in sibs[1:]:
                assert [(m.role, m.content) for m in s.seg.context] == ctx0


def test_tree_backup_is_subtree_leaf_mean():
    from harness_rl.rl.tree_reward import tree_backup
    t = _tree_stub()
    tree_backup(t)
    for n in t.nodes:
        leaves = [t.nodes[i].u_g for i in t.leaves
                  if not t.nodes[i].hit_limit and _is_desc(t, i, n.id)]
        if leaves:
            assert abs(n.value - sum(leaves) / len(leaves)) < 1e-9
    root = t.nodes[0]
    assert abs(root.value - sum(t.leaf_u_gs()) / len(t.leaf_u_gs())) < 1e-9


def _is_desc(t, node_id, anc_id):
    cur = node_id
    while cur is not None:
        if cur == anc_id:
            return True
        cur = t.nodes[cur].parent
    return False


def test_tree_advantages_macro_micro_split():
    from harness_rl.rl.tree_reward import tree_advantages
    t = _tree_stub()
    full, stats = tree_advantages([t], w_micro=1.0)
    flat, _ = tree_advantages([t], w_micro=0.0)
    # action nodes are macro-only → unaffected by w_micro; w_micro=0 reduces to flat SUPO
    for n in t.nodes:
        if (0, n.id) in full and not n.is_summary:
            assert abs(full[(0, n.id)] - flat[(0, n.id)]) < 1e-9
    # at least one summary node differs (its micro term is non-zero)
    assert any(abs(full[(0, n.id)] - flat[(0, n.id)]) > 1e-9
               for n in t.nodes if n.is_summary and (0, n.id) in full)
    # micro is (leave-one-out) zero-sum in sign across a sibling pair
    for n in t.nodes:
        sibs = [t.nodes[c] for c in n.children if t.nodes[c].is_summary and t.nodes[c].value is not None]
        if len(sibs) == 2:
            m = [full[(0, s.id)] - flat[(0, s.id)] for s in sibs]
            assert abs(m[0] + m[1]) < 1e-9, "sibling micro terms must cancel"
    assert stats["n_leaves"] == len(t.leaf_u_gs())


def test_tree_samples_carry_precomputed_advantage():
    from harness_rl.rl.tree_reward import tree_samples
    t = _tree_stub()
    samples, stats = tree_samples([t], w_micro=1.0)
    assert samples and len(samples) == stats["n_scored"]
    for s in samples:
        assert "advantage" in s and "reward" not in s      # precomputed advantage, not a reward
        assert s["group_key"] == "stub/tree" and s["messages"] and isinstance(s["response"], str)
        assert set(s["category_tokens"]) <= set(("summarization", "thinking", "tool_call"))
    assert any(s["is_summary"] for s in samples), "summary nodes must be trainable samples"
    ids = [s["node_id"] for s in samples]
    assert len(ids) == len(set(ids)), "one sample per node (no shared-prefix double-counting)"


def test_tree_overlong_leaf_masked():
    from harness_rl.rl.tree_reward import tree_backup, tree_samples
    t = _tree_stub()
    leaf = t.nodes[t.leaves[0]]
    leaf.hit_limit = True                      # simulate an overlong (turn/summary-cap) leaf
    tree_backup(t)
    assert leaf.value is None, "overlong leaf is excluded from the backup"
    samples, _ = tree_samples([t], w_micro=1.0)
    assert leaf.id not in [s["node_id"] for s in samples], "overlong leaf gets no gradient"


def test_tree_mask_reasons_separates_turn_cap_from_summary_cap():
    """max_summaries (real SUPO overlong) vs budget (turn cap, artifact still gradable)."""
    from harness_rl.rl.tree_reward import tree_samples
    from harness_rl.rl.tree_supo import TreeConfig, tree_supo_rollout
    from harness_rl.benchmarks.base import EnvStub
    from harness_rl.serving.client import StubModel

    def never_submits(messages):
        if messages and "Summarize your progress" in (messages[-1].content or ""):
            return "SUMMARY"
        return "```bash\nls\n```"          # never emits TASK_COMPLETE (like the real 3B)

    def build(mask_reasons):
        cfg = TreeConfig(context_L=600, max_turns=6, max_summaries=20, recency_turns=3,
                         branch_factor=2, max_leaves=2, mask_reasons=mask_reasons)
        env = EnvStub(observations=["obs " + ("token " * 300)] * 40, final_u_g=1.0)
        return tree_supo_rollout(TaskSpec(task_id="t", benchmark="s", instruction="i"),
                                 env, StubModel(never_submits), cfg, system_prompt="SYS")

    t_faithful = build(("budget", "max_summaries"))
    assert all(t_faithful.nodes[i].terminated_reason == "budget" for i in t_faithful.leaves)
    assert all(t_faithful.nodes[i].hit_limit for i in t_faithful.leaves)
    s_faithful, st_faithful = tree_samples([t_faithful])
    assert s_faithful == [] and st_faithful["all_masked"], "faithful masking → zero gradient"

    t_graded = build(("max_summaries",))
    assert not any(t_graded.nodes[i].hit_limit for i in t_graded.leaves)
    s_graded, st_graded = tree_samples([t_graded])
    assert s_graded and not st_graded["all_masked"], "turn-exhausted rollouts become trainable"
    assert st_graded["n_leaves"] == len(t_graded.leaves)


# --- summary leverage (eta^2) diagnostics ---

def test_eta2_undefined_when_k_is_one():
    from harness_rl.rl.variance import summary_leverage
    lev = summary_leverage([[[1.0], [0.0]], [[0.5], [1.0]]], n_boot=0)   # K=1 everywhere
    assert lev["degenerate"] and lev["eta2"] is None
    assert "K=1" in lev["reason"]


def test_eta2_degenerate_when_no_variance():
    from harness_rl.rl.variance import summary_leverage
    lev = summary_leverage([[[1.0, 1.0], [1.0, 1.0]]], n_boot=0)
    assert lev["degenerate"] and "zero total variance" in lev["reason"]


def test_eta2_null_baseline_matches_theory_for_m2k2():
    """E[eta^2 | H0] = df_b/(df_b+df_w); for M=2,K=2 pooled that is 1/3 — NOT 0."""
    from harness_rl.rl.variance import summary_leverage
    import random as _r
    rng = _r.Random(7)
    mats = [[[rng.gauss(0, 1), rng.gauss(0, 1)], [rng.gauss(0, 1), rng.gauss(0, 1)]]
            for _ in range(400)]                      # NO true summary effect
    lev = summary_leverage(mats, n_boot=200, seed=1)
    assert abs(lev["null_baseline"] - 1 / 3) < 1e-9
    assert abs(lev["eta2"] - 1 / 3) < 0.05, "biased eta^2 should sit at the null baseline"
    assert abs(lev["omega2"]) < 0.05, "bias-corrected omega^2 must be ~0 under the null"
    assert lev["eta2_ci"][0] < lev["eta2"] < lev["eta2_ci"][1]


def test_eta2_detects_a_real_summary_effect():
    from harness_rl.rl.variance import summary_leverage
    import random as _r
    rng = _r.Random(11)
    # summary 0 is genuinely better than summary 1; execution noise is small
    mats = [[[1.0 + rng.gauss(0, 0.05), 1.0 + rng.gauss(0, 0.05)],
             [0.0 + rng.gauss(0, 0.05), 0.0 + rng.gauss(0, 0.05)]] for _ in range(50)]
    lev = summary_leverage(mats, n_boot=200, seed=2)
    assert lev["eta2"] > 0.95 and lev["omega2"] > 0.9
    assert lev["eta2"] > lev["null_baseline"]
    assert lev["F"] > 50


def test_macro_advantage_is_sign_only_at_m2_but_graded_at_m4():
    from harness_rl.rl.variance import macro_advantage_spread
    two = macro_advantage_spread([[[1.0, 1.0], [0.0, 0.0]], [[0.9, 0.9], [0.1, 0.1]]])
    assert two["sign_only"] and two["distinct_magnitudes"] == 1
    assert abs(two["magnitude_mean"] - 1.0) < 1e-4          # collapses to exactly +/-1
    four = macro_advantage_spread([[[1.0], [0.9], [0.2], [0.0]]])
    assert four["sign_only"] is False and four["distinct_magnitudes"] > 1


def test_reward_matrices_from_trees_feed_eta2():
    from harness_rl.rl.tree_reward import reward_matrices
    from harness_rl.rl.variance import summary_leverage
    t = _tree_stub(branch_factor=2, max_leaves=4)
    mats = reward_matrices([t])
    assert mats, "a branched tree must yield at least one M x K matrix"
    assert all(len(rows) >= 2 for rows in mats), "each anchor contributes >=2 sibling rows"
    summary_leverage(mats, n_boot=0)                        # must not raise on ragged rows


def test_client_chat_many_stub_varies():
    from harness_rl.serving.client import StubModel
    st = {"n": 0}

    def r(_m):
        st["n"] += 1
        return f"variant {st['n']}"
    outs = StubModel(r).chat_many([Message(role=Role.USER, content="x")], n=3, temperature=0.9)
    assert [o.text for o in outs] == ["variant 1", "variant 2", "variant 3"]


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
