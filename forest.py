"""
Builds count vectors n_s(w) at every observed potential context from raw
path data. 
"""
from collections import defaultdict
from typing import Dict, Hashable, Iterable, Sequence, Tuple

Node = Hashable
Context = Tuple[Node, ...]


def build_counts_from_paths(
    paths: Iterable[Sequence[Node]],
    D: int,
) -> Dict[Context, Dict[Node, int]]:
    """
    counts[ctx] = {next_node: count} for every observed context ctx.

    A path shorter than D+1 nodes contributes nothing (Section 3.1.1).
    """
    counts: Dict[Context, Dict[Node, int]] = defaultdict(dict)
    for path in paths:
        path = list(path)
        L = len(path) - 1
        if L < D:
            continue
        for i in range(D, L + 1):
            # s^{(D)}_{r,i} = (x_{i-1}, x_{i-2}, ..., x_{i-D}) -- most recent first
            ctx_full = tuple(path[i - 1 - j] for j in range(D))
            nxt = path[i]
            for k in range(1, D + 1):
                prefix = ctx_full[:k]
                d = counts[prefix]
                d[nxt] = d.get(nxt, 0) + 1
    return dict(counts)


def build_counts_from_paths_variable_boundary(
    paths: Iterable[Sequence[Node]],
    D: int,
) -> Dict[Context, Dict[Node, int]]:
    
    counts: Dict[Context, Dict[Node, int]] = defaultdict(dict)
    for path in paths:
        path = list(path)
        L = len(path) - 1
        for i in range(1, L + 1):
            k_max = min(i, D)
            ctx_full = tuple(path[i - 1 - j] for j in range(k_max))  # most-recent-first
            nxt = path[i]
            for k in range(1, k_max + 1):
                prefix = ctx_full[:k]
                d = counts[prefix]
                d[nxt] = d.get(nxt, 0) + 1
    return dict(counts)


def validate_paths_against_graph(paths: Iterable[Sequence[Node]], out_nbrs) -> None:
    """Optional sanity check: raises if any path uses a graph-invalid edge."""
    for path in paths:
        path = list(path)
        for a, b in zip(path[:-1], path[1:]):
            if b not in out_nbrs.get(a, ()):
                raise ValueError(f"path contains invalid edge ({a!r} -> {b!r}) not in G")
