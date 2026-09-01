"""
Builds N+(v)/N-(v) adjacency, the predictive alphabet A_s = N+(cur(s))
(Eq 3.4), and graph-valid children Ch_G(s) (Eq 3.7-3.8). Section 3.1.2.

A child sq is only kept if its oldest node q can still extend backward
D-(k+1) more steps; otherwise it's a dead end.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, Hashable, Iterable, List, Set, Tuple

Node = Hashable
Context = Tuple[Node, ...]  # most-recent-first (Section 3.1.1)


def build_adjacency(edges: Iterable[Tuple[Node, Node]]):
    """Build out- and in-neighbour dicts and the node set from an edge list."""
    out_nbrs: Dict[Node, Set[Node]] = defaultdict(set)
    in_nbrs: Dict[Node, Set[Node]] = defaultdict(set)
    nodes: Set[Node] = set()
    for u, v in edges:
        out_nbrs[u].add(v)
        in_nbrs[v].add(u)
        nodes.add(u)
        nodes.add(v)
    for n in nodes:
        out_nbrs.setdefault(n, set())
        in_nbrs.setdefault(n, set())
    return dict(out_nbrs), dict(in_nbrs), nodes


def compute_max_backward_reach(in_nbrs: Dict[Node, Set[Node]], nodes: Set[Node], D: int) -> Dict[Node, int]:
    """
    reach[v] = length (capped at D) of the longest backward walk
    v = v_0 <- v_1 <- ... <- v_t, v_i in in_nbrs[v_{i-1}].

    r_t[v] := "v admits a backward walk of length >= t" is monotone
    non-increasing in t (a walk of length t truncates to one of length
    t' < t), so a single forward pass t = 1..D, each using only the
    previous pass's values, suffices; no fixed-point iteration is needed
    despite possible cycles in G.
    """
    reach: Dict[Node, int] = {v: 0 for v in nodes}
    r_prev = {v: True for v in nodes}  # r_0[v] = True trivially (empty walk)
    for t in range(1, D + 1):
        r_cur = {}
        any_true = False
        for v in nodes:
            ok = any(r_prev.get(q, False) for q in in_nbrs.get(v, ()))
            r_cur[v] = ok
            if ok:
                reach[v] = t
                any_true = True
        if not any_true:
            break
        r_prev = r_cur
    return reach


def alphabet(node: Node, out_nbrs: Dict[Node, Set[Node]]) -> FrozenSet[Node]:
    """A_s = N+(cur(s)), Eq 3.4."""
    return frozenset(out_nbrs.get(node, ()))


def active_roots(
    nodes: Iterable[Node],
    out_nbrs: Dict[Node, Set[Node]],
    max_reach: Dict[Node, int],
    D: int,
) -> List[Tuple[Node]]:
    """Root contexts of length 1 (Section 3.1.2).

    Graph-determined, not data-determined: include every valid root even
    with zero counts, since GC-CTW (Theorem 3.2.1) and GC-BCT
    (Theorem 3.2.2) both range over all roots, not just the ones with data.
    """
    return sorted(
        [(v,) for v in nodes if out_nbrs.get(v) and max_reach.get(v, 0) >= D - 1],
        key=repr,
    )


def children_of(
    ctx: Context,
    out_nbrs: Dict[Node, Set[Node]],
    in_nbrs: Dict[Node, Set[Node]],
    max_reach: Dict[Node, int],
    D: int,
) -> List[Context]:
    """Ch_G(s), Eq 3.7-3.8, filtered to contexts that extend to depth D."""
    k = len(ctx)
    if k >= D:
        return []
    oldest = ctx[-1]
    needed = D - (k + 1)  # further backward steps required after appending q
    return [ctx + (q,) for q in in_nbrs.get(oldest, ()) if max_reach.get(q, 0) >= needed]


def children_of_unfiltered(
    ctx: Context,
    in_nbrs: Dict[Node, Set[Node]],
    D: int,
) -> List[Context]:
    """Same as children_of but without the depth-D filter. Kept only for
    test_gc_bct.py, to show why that filter matters (see
    test_max_reach_filter_matters)."""
    k = len(ctx)
    if k >= D:
        return []
    oldest = ctx[-1]
    return [ctx + (q,) for q in in_nbrs.get(oldest, ())]
