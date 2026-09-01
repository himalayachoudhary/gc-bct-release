"""
Graph-valid chronological histories underlying the higher-order network
construction (dissertation Section 3.4.1, shift operator sh(u,w), Eq 3.52).

Orientation note: contexts elsewhere in the codebase are most-recent-first
(ctx[0] = current node). Histories here are the opposite: chronological,
history[0] oldest, history[-1] current (tau(u) = history[-1]). The two
conventions meet in exactly one place, `selected_context` below, which
converts a chronological History into a most-recent-first Context for
keying into theta. Everywhere else in this module and in hon.py, histories
stay chronological.
"""
from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

from common import Context, Node

History = Tuple[Node, ...]  # chronological: history[0] oldest, history[-1] current


def is_graph_valid_history(history: History, out_nbrs: Dict[Node, Set[Node]]) -> bool:
    """u in S_k(G): (u_i, u_{i+1}) in E for i = 1, ..., k-1. Trivially
    true for length <=1 histories."""
    return all(
        history[i + 1] in out_nbrs.get(history[i], ())
        for i in range(len(history) - 1)
    )


def shift_history(history: History, next_node: Node) -> History:
    """sh(u, w) = (u2, ..., uD, w), Eq 3.52."""
    return history[1:] + (next_node,)


def enumerate_valid_histories(
    out_nbrs: Dict[Node, Set[Node]],
    nodes: Iterable[Node],
    D: int,
) -> Set[History]:
    """
    S_D(G), built by forward extension (Algorithm 1, Appendix C.1.1):
    start from every physical node and extend one step at a time through
    valid out-edges. No max-reach filtering needed -- a chain that can't
    reach length D just produces no depth-D descendants and drops out.
    """
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")
    level: Set[History] = {(v,) for v in nodes}
    for _ in range(D - 1):
        level = {u + (w,) for u in level for w in out_nbrs.get(u[-1], ())}
    return level


def enumerate_reachable_histories(
    out_nbrs: Dict[Node, Set[Node]],
    D: int,
    initial_histories: Iterable[History],
) -> Set[History]:
    """
    Reach(I): forward-shift closure of a seed set I subseteq S_D(G) --
    every history reachable from I, without building the full S_D(G).
    `initial_histories` must already be graph-valid; shifting only
    extends forward, so it can't repair an invalid seed.
    """
    frontier = set(initial_histories)
    reached = set(frontier)
    while frontier:
        nxt = {shift_history(u, w) for u in frontier for w in out_nbrs.get(u[-1], ())}
        nxt -= reached
        reached |= nxt
        frontier = nxt
    return reached


def selected_context(history: History, selected_forest: Iterable[Context]) -> Context:
    """
    C_T(u): the selected suffix of a chronological history, returned as
    a most-recent-first Context -- the one orientation conversion point
    in this module (see module docstring).

    Longest-to-shortest suffix search, O(D) per call. Pass a pre-built
    set/frozenset for `selected_forest` if doing many lookups, to avoid
    re-converting it each call.
    """
    forest = selected_forest if isinstance(selected_forest, (set, frozenset)) else set(selected_forest)
    for length in range(len(history), 0, -1):
        ctx = history[-length:][::-1]
        if ctx in forest:
            return ctx
    raise ValueError(f"no selected context matches history {history!r}")
