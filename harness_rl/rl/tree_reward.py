"""Two-level credit assignment for the summary-branching tree (rl/tree_supo.py).

Given a GROUP of rollout trees for one task (the G GRPO samples), assign every generated node an
advantage that separates summary credit from action credit:

  * **backup**  V(node) = mean u_g over the non-overlong leaves in its subtree (leaf: V = u_g).
                A critic-free Monte-Carlo value; with terminal-only reward this is the return, so
                the "GAE" here is the λ=1 / no-bootstrap advantage V(node) − baseline.
  * **macro** (GRPO / outcome, across the group's leaves):  A_macro(n) = (V(n) − μ)/(σ+ε).
  * **micro** (GiGPO / step, across summary siblings at a shared anchor):
                A_micro(sᵢ) = (V(sᵢ) − baseline_{-i})/(σ_sib+ε), default leave-one-out baseline.
  * **combined**:  action node → A_macro ;  summary node → A_macro + w·A_micro.

Overlong leaves (turn/summary cap) are excluded from the backup AND the samples (SUPO masking);
a node whose whole subtree is overlong gets V = None and is dropped. Shared-prefix nodes appear
once, so gradient is never double-counted (context lives in `messages`, only `response` trains).
"""
from __future__ import annotations

from harness_rl.rl.tree_supo import TreeRollout


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def tree_backup(tree: TreeRollout, overlong_mask: bool = True) -> None:
    """Fill `node.value` = mean u_g over valid leaves in each node's subtree (None if none).

    Post-order via reverse id (every node's id exceeds its parent's, by construction)."""
    agg: dict[int, tuple[float, int]] = {}   # node id → (sum u_g, count) over valid subtree leaves
    for nid in sorted((n.id for n in tree.nodes), reverse=True):
        n = tree.nodes[nid]
        if not n.children:                    # leaf
            valid = (n.u_g is not None) and (not overlong_mask or not n.hit_limit)
            agg[nid] = (float(n.u_g), 1) if valid else (0.0, 0)
        else:
            s = sum(agg[c][0] for c in n.children)
            c = sum(agg[c][1] for c in n.children)
            agg[nid] = (s, c)
        s, c = agg[nid]
        n.value = (s / c) if c else None


def _micro(tree: TreeRollout, node, baseline: str, eps: float) -> float:
    """Sibling-relative advantage for a summary node, over the summary siblings sharing its parent."""
    if node.parent is None:
        return 0.0
    sibs = [tree.nodes[c] for c in tree.nodes[node.parent].children
            if tree.nodes[c].is_summary and tree.nodes[c].value is not None]
    vals = [s.value for s in sibs]
    if len(vals) < 2:
        return 0.0
    if baseline == "loo":
        base = _mean([s.value for s in sibs if s.id != node.id])
    else:
        base = _mean(vals)
    return (node.value - base) / (_std(vals) + eps)


def tree_advantages(trees: list[TreeRollout], w_micro: float = 1.0, micro_baseline: str = "loo",
                    overlong_mask: bool = True, eps: float = 1e-6) -> tuple[dict, dict]:
    """Compute per-node advantages for a group of trees. Returns (adv, stats) where
    adv[(tree_idx, node_id)] = A_macro (+ w·A_micro for summary nodes)."""
    for t in trees:
        tree_backup(t, overlong_mask=overlong_mask)

    leaf_vals = [v for t in trees for v in t.leaf_u_gs(include_overlong=not overlong_mask)]
    mu, sigma = _mean(leaf_vals), _std(leaf_vals)

    adv: dict[tuple[int, int], float] = {}
    for ti, t in enumerate(trees):
        for n in t.nodes:
            if n.value is None:               # all-overlong subtree → no training signal
                continue
            a = (n.value - mu) / (sigma + eps)
            if n.is_summary:
                a += w_micro * _micro(t, n, micro_baseline, eps)
            adv[(ti, n.id)] = a

    n_leaves_total = sum(len(t.leaves) for t in trees)
    n_overlong = sum(1 for t in trees for i in t.leaves if t.nodes[i].hit_limit)
    stats = {"mu": mu, "sigma": sigma, "n_leaves": len(leaf_vals),
             "n_leaves_total": n_leaves_total, "n_overlong_leaves": n_overlong,
             "n_nodes": sum(len(t.nodes) for t in trees), "n_scored": len(adv),
             # loud diagnostics: silently emitting 0 samples (or a degenerate group) into training
             # burns a batch. all_masked → every leaf hit the turn/summary cap (raise the caps or L);
             # degenerate_group → every leaf scored the same, so macro carries no signal (σ=0) and
             # only the sibling-relative micro term can teach anything this batch.
             "all_masked": n_leaves_total > 0 and len(leaf_vals) == 0,
             "degenerate_group": len(leaf_vals) > 0 and sigma == 0.0}
    return adv, stats


def _leaves_under(tree: TreeRollout, node_id: int, overlong_mask: bool = True) -> list[float]:
    """u_g of every (non-masked) leaf in the subtree rooted at `node_id`."""
    out, stack = [], [node_id]
    while stack:
        n = tree.nodes[stack.pop()]
        if not n.children:
            if n.u_g is not None and (not overlong_mask or not n.hit_limit):
                out.append(float(n.u_g))
        else:
            stack.extend(n.children)
    return out


def reward_matrices(trees: list[TreeRollout], overlong_mask: bool = True) -> list[list[list[float]]]:
    """Extract one `M x K` reward matrix per BRANCH ANCHOR, for `rl/variance.py`.

    Each anchor (a node whose children are >1 sibling summaries) becomes a matrix whose row m is
    the list of leaf outcomes under sibling summary m. Under depth budgeting rows are equal-length
    by construction; they can still go ragged when a branch terminates early (submit), which is
    real signal rather than a traversal artifact. `variance.task_ss` handles ragged rows either way.
    """
    out: list[list[list[float]]] = []
    for t in trees:
        for n in t.nodes:
            sibs = [c for c in n.children if t.nodes[c].is_summary]
            if len(sibs) > 1:
                rows = [_leaves_under(t, c, overlong_mask) for c in sibs]
                if sum(len(r) for r in rows) > 0:
                    out.append(rows)
    return out


def tree_samples(trees: list[TreeRollout], w_micro: float = 1.0, micro_baseline: str = "loo",
                 overlong_mask: bool = True, eps: float = 1e-6) -> tuple[list[dict], dict]:
    """Emit one training sample per scored node, carrying a PRECOMPUTED `advantage` (not a reward).

    Each: {group_key=task_id, messages (bounded context), response (generated tokens), advantage,
    is_summary, node_id, category_tokens}. slime applies the given advantage to the response tokens
    (it does NOT recompute a group-relative advantage — that's the integration point to verify)."""
    adv, stats = tree_advantages(trees, w_micro=w_micro, micro_baseline=micro_baseline,
                                 overlong_mask=overlong_mask, eps=eps)
    samples: list[dict] = []
    for ti, t in enumerate(trees):
        for n in t.nodes:
            if (ti, n.id) not in adv:
                continue
            if overlong_mask and n.hit_limit:     # subsumed by value-None, kept explicit
                continue
            samples.append({
                "group_key": t.task_id,
                "messages": [m.to_openai() for m in n.seg.context],
                "response": n.seg.output,
                "advantage": adv[(ti, n.id)],
                "is_summary": n.is_summary,
                "node_id": n.id,
                "category_tokens": dict(n.seg.category_tokens or {}),
            })
    return samples, stats
