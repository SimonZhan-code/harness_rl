"""SUPO with SUMMARY-BRANCHING tree rollout (variance-reduced summary/action credit).

Flat SUPO (rl/supo.py) gives every token — summary and action — the same outcome advantage
(Thm 3.2), conflating summary quality with action quality. Here, at each compaction point we
sample **B** summaries from the *same* pre-summary context and roll each forward independently.
A summary's **subtree value** (mean outcome of the leaves under it) scores it *relative to its
siblings at the same anchor* — a GiGPO-style step group manufactured exactly where the domain
(LiveCodeBench/SWE states rarely recur) would give none, plus a critic-free tree-backup baseline.

Scope: **summary branching only** — action turns stay linear within a branch. Branching is budgeted
by **DEPTH** (`branch_depth`): every path branches at its first `branch_depth` compactions, so the
policy is symmetric and sibling subtrees are equal-sized by construction (B**depth leaves).
`max_leaves` is only a safety clamp that lowers the depth. The advantage math (macro GRPO +
micro GiGPO) lives in
`rl/tree_reward.py`. Env forking (`env.fork()`) and G2 cloning (`gamma.clone()` +
`pending_compaction`/`apply_summary`) make independent branches possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from harness_rl.benchmarks.base import Environment
from harness_rl.gamma.g2_summarize import G2Summarize
from harness_rl.harness.agent import DEFAULT_SYSTEM_PROMPT, parse_action
from harness_rl.rl.supo import SUMMARIZE_PROMPT, SegmentTurn, categorize_tokens
from harness_rl.serving.client import ModelClient
from harness_rl.types import Message, Observation, Role, TaskSpec

_MALFORMED = "No valid action found. Emit one ```bash ... ``` block, or TASK_COMPLETE."


def _summarizer_guard(_msgs):  # pragma: no cover - defensive
    raise RuntimeError("tree rollout drives compaction externally; G2.summarizer must not fire")


def fork_env(env: Environment) -> Environment:
    """Duplicate a live env for a branch. Requires the env to implement `fork()`."""
    fork = getattr(env, "fork", None)
    if fork is None:
        raise TypeError(f"{type(env).__name__} has no fork(); tree rollout needs a forkable env")
    return fork()


@dataclass
class TreeNode:
    id: int
    parent: int | None
    is_summary: bool
    seg: SegmentTurn                 # (bounded context, generated tokens) → one training sample
    children: list[int] = field(default_factory=list)
    u_g: float | None = None         # set on leaves (terminal action nodes)
    value: float | None = None       # tree-backup: mean subtree leaf u_g (filled by tree_reward)
    hit_limit: bool = False           # leaf should be MASKED (per cfg.mask_reasons)
    terminated_reason: str = ""       # "submit" | "budget" (turn cap) | "max_summaries"


@dataclass
class TreeConfig:
    context_L: int = 4096            # token threshold L that triggers a summarization action
    max_turns: int = 40              # action-turn cap PER root→leaf path
    max_summaries: int = 8           # summarization cap per path (overlong beyond)
    recency_turns: int = 4
    branch_factor: int = 3           # B summaries sampled per compaction
    # BUDGET BY DEPTH, NOT LEAF COUNT. Branch at the first `branch_depth` compactions along every
    # path, so the policy is symmetric and sibling subtrees are the same size by construction
    # (B**branch_depth leaves when every path compacts that often).
    #
    # The old leaf-count budget ("branch while committed_leaves + B-1 <= max_leaves") was spent in
    # DFS order, so the FIRST sibling's subtree absorbed the remainder and the rest got one leaf
    # each — e.g. B=2, max_leaves=8 gave [7, 1]. That made V(s_i) estimates wildly unequal in
    # precision across siblings being contrasted, inflating sigma_sibling with estimation noise and
    # biasing eta^2 upward. Depth-budgeting removes that artifact entirely; any residual imbalance
    # is then real (a branch that submitted early genuinely has fewer leaves), not traversal order.
    branch_depth: int = 1
    max_leaves: int = 12             # SAFETY CLAMP only: depth is reduced until B**depth <= this
    summary_temperature: float = 0.9  # higher temp → diverse sibling summaries
    max_tokens: int = 2048

    def effective_depth(self) -> int:
        """Branch depth after the `max_leaves` safety clamp. Clamping reduces DEPTH (keeping the
        tree balanced) rather than truncating breadth mid-traversal."""
        if self.branch_factor < 2 or self.branch_depth < 1:
            return 0
        d, leaves = 0, 1
        while d < self.branch_depth and leaves * self.branch_factor <= self.max_leaves:
            leaves *= self.branch_factor
            d += 1
        return d
    # Which terminations get overlong-MASKED. SUPO's ablation masks trajectories that don't finish,
    # so ("budget","max_summaries") is the faithful default. But the two differ in kind:
    #   max_summaries = a real context-management failure (what SUPO's masking is about);
    #   budget        = the agent merely used all its turns — on LiveCodeBench the verifier still
    #                   grades the artifact on disk, so u_g is perfectly meaningful.
    # Set to ("max_summaries",) to TRAIN on turn-exhausted rollouts — necessary for policies that
    # rarely emit TASK_COMPLETE (measured: Qwen2.5-Coder-3B submits in 0/20 turns), where masking
    # both reasons discards every rollout and the batch yields zero gradient.
    mask_reasons: tuple[str, ...] = ("budget", "max_summaries")


@dataclass
class TreeRollout:
    task_id: str
    nodes: list[TreeNode] = field(default_factory=list)
    leaves: list[int] = field(default_factory=list)   # ids of terminal action nodes
    num_summaries: int = 0

    def leaf_u_gs(self, include_overlong: bool = False) -> list[float]:
        return [self.nodes[i].u_g for i in self.leaves
                if self.nodes[i].u_g is not None and (include_overlong or not self.nodes[i].hit_limit)]


class _TreeBuilder:
    """DFS builder: a linear run of action turns until a compaction (→ branch B) or termination
    (→ leaf), recursing into each branch with a forked env + cloned G2."""

    def __init__(self, task: TaskSpec, model: ModelClient, cfg: TreeConfig, system_prompt: str):
        self.task = task
        self.model = model
        self.cfg = cfg
        self.sp = system_prompt
        self.roll = TreeRollout(task_id=task.task_id)
        self.eff_depth = cfg.effective_depth()  # branch at the first `eff_depth` compactions

    # --- tree bookkeeping -------------------------------------------------
    def _new(self, parent: int | None, is_summary: bool, seg: SegmentTurn) -> int:
        nid = len(self.roll.nodes)
        self.roll.nodes.append(TreeNode(id=nid, parent=parent, is_summary=is_summary, seg=seg))
        if parent is not None:
            self.roll.nodes[parent].children.append(nid)
        return nid

    def _summ_ctx(self, to_sum: list[Message]) -> list[Message]:
        return ([Message(role=Role.SYSTEM, content=self.sp),
                 Message(role=Role.USER, content=f"Task:\n{self.task.instruction}")]
                + list(to_sum) + [Message(role=Role.USER, content=SUMMARIZE_PROMPT)])

    def _can_branch(self, depth: int) -> bool:
        """Depth-symmetric: EVERY path branches at its first `eff_depth` compactions. No global
        counter, so traversal order cannot decide which sibling gets the remaining budget."""
        return depth < self.eff_depth

    def _record_summary(self, gamma: G2Summarize, parent: int | None, seg_id: int,
                        to_sum: list[Message], chat) -> int:
        sid = self._new(parent, True, SegmentTurn(
            segment=seg_id, is_summary=True, context=self._summ_ctx(to_sum), output=chat.text,
            prompt_tokens=chat.prompt_tokens, completion_tokens=chat.completion_tokens,
            category_tokens=categorize_tokens(chat.text, True)))
        gamma.apply_summary(chat.text, self.cfg.context_L)
        self.roll.num_summaries += 1
        return sid

    def _finish_leaf(self, env: Environment, node_id: int | None, reason: str) -> None:
        if node_id is not None:
            n = self.roll.nodes[node_id]
            n.u_g = env.verify().u_g
            n.terminated_reason = reason
            n.hit_limit = reason in self.cfg.mask_reasons
            self.roll.leaves.append(node_id)
        env.close()

    # --- the recursion ----------------------------------------------------
    def run(self, gamma: G2Summarize, env: Environment) -> TreeRollout:
        gamma.reset(self.task, self.sp)
        obs = env.reset()
        gamma.on_step(obs, action=None)  # seed with the initial observation
        self._segment(gamma, env, seg_id=0, parent_id=None, turns=0, summaries=0, depth=0)
        return self.roll

    def _segment(self, gamma: G2Summarize, env: Environment, seg_id: int, parent_id: int | None,
                 turns: int, summaries: int, depth: int) -> None:
        last = parent_id
        while True:
            if summaries > self.cfg.max_summaries:
                self._finish_leaf(env, last, "max_summaries")     # true SUPO overlong
                return
            if turns >= self.cfg.max_turns:
                self._finish_leaf(env, last, "budget")            # turn cap (gradable artifact)
                return

            to_sum = gamma.pending_compaction(self.cfg.context_L)
            if to_sum is not None:
                if self._can_branch(depth):
                    self._branch(gamma, env, seg_id, last, to_sum, turns, summaries, depth)
                    return                                        # children own continuation + env
                # past branch depth → single summary, continue this path linearly (B=1)
                chat = self.model.chat_many(self._summ_ctx(to_sum), n=1,
                                            temperature=self.cfg.summary_temperature)[0]
                last = self._record_summary(gamma, last, seg_id + 1, to_sum, chat)
                seg_id += 1
                summaries += 1
                continue

            ctx = gamma.build_context(self.cfg.context_L)         # no pending compaction here
            out = self.model.chat(ctx)
            action = parse_action(out.text)
            aid = self._new(last, False, SegmentTurn(
                segment=seg_id, is_summary=False, context=ctx, output=out.text,
                prompt_tokens=out.prompt_tokens, completion_tokens=out.completion_tokens,
                category_tokens=categorize_tokens(out.text, False)))
            last = aid
            turns += 1
            if action is None:
                gamma.on_step(Observation(text=_MALFORMED), None)
            elif env.is_done(action):
                self._finish_leaf(env, aid, "submit")
                return
            else:
                obs = env.execute(action)
                gamma.on_step(obs, action)

    def _branch(self, gamma: G2Summarize, env: Environment, seg_id: int, parent_id: int | None,
                to_sum: list[Message], turns: int, summaries: int, depth: int) -> None:
        summs = self.model.chat_many(self._summ_ctx(to_sum), n=self.cfg.branch_factor,
                                     temperature=self.cfg.summary_temperature)
        for chat in summs:                       # every sibling recurses at the SAME depth+1,
            g2 = gamma.clone()                   # so none can absorb another's share of the budget
            e2 = fork_env(env)
            sid = self._record_summary(g2, parent_id, seg_id + 1, to_sum, chat)
            self._segment(g2, e2, seg_id + 1, sid, turns, summaries + 1, depth + 1)
        env.close()                                               # parent env done; forks live on


def tree_supo_rollout(task: TaskSpec, env: Environment, model: ModelClient, cfg: TreeConfig,
                      system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> TreeRollout:
    """Build one summary-branching rollout tree. Branches only at summarization nodes; each branch
    forks the env and clones G2 so continuations are independent. Advantages: `rl/tree_reward.py`."""
    gamma = G2Summarize(recency_turns=cfg.recency_turns, compact_at_tokens=cfg.context_L,
                        summary_mode="replace", summarizer=_summarizer_guard)
    return _TreeBuilder(task, model, cfg, system_prompt).run(gamma, env)
