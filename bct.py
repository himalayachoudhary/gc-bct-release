"""
GC-BCT: the graph-constrained Bayesian context-tree algorithm.

Zero-count subtrees (Appendix B): fast path (beta constant >= 1/2) sets
Q_{m,s} = beta_s in O(1) per node; exact only in that regime. Otherwise
falls back to the general recursion, walking the unobserved subtree down
to depth D, which is exact everywhere but can be slow for large D.

ChG(s) must be filtered by max_reach (graph_utils.children_of): a node
with no graph-valid extendable child below depth D is a forced leaf
(Q_m = 1), not an empty-product 'expand' alternative.
"""
import math
import sys
import warnings
from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

from common import BetaSpec, Context, Node, beta_fn
from evidence import log_Pe
from forest import build_counts_from_paths
from graph_utils import active_roots, alphabet, build_adjacency, children_of, compute_max_backward_reach


class GCBCTResult:
    """Output of run_gc_bct."""

    def __init__(self, log_pm_lambda, map_leaves, log_pm, decision, counts, roots, beta, eta):
        self.log_pm_lambda = log_pm_lambda  #: log P(D_D, T_MAP | G, D), Theorem 3.2.2
        self.map_leaves = map_leaves  #: the reconstructed T_MAP, as a list of contexts
        self.log_pm = log_pm  #: dict: (materialised) context -> log P^G_{m,s}
        self.decision = decision  #: dict: (materialised) context -> 0 (stop) / 1 (expand)
        self.counts = counts
        self.roots = roots
        self.beta = beta  #: the BetaSpec (float or callable) this result was fitted with
        self.eta = eta  #: the Dirichlet concentration this result was fitted with

    @property
    def pm_lambda(self) -> float:
        return math.exp(self.log_pm_lambda)


def _log_Qm(ctx, out_nbrs, in_nbrs, max_reach, D, b_fn, cache):
    """Exact Q_{m,s} for an entirely-unobserved subtree"""
    if ctx in cache:
        return cache[ctx]
    if len(ctx) == D:
        cache[ctx] = 0.0
        return 0.0
    kids = children_of(ctx, out_nbrs, in_nbrs, max_reach, D)
    if not kids:
        raise RuntimeError(
            f"context {ctx!r} has no graph-valid extendable child below depth {D}; "
            "this is impossible for a genuine member of S+_{<=D}(G) and indicates "
            "children_of()'s max_reach filtering was bypassed or is inconsistent"
        )
    b = b_fn(ctx)
    log_stop = math.log(b)
    log_expand = math.log(1.0 - b) + sum(
        _log_Qm(u, out_nbrs, in_nbrs, max_reach, D, b_fn, cache) for u in kids
    )
    val = max(log_stop, log_expand)
    cache[ctx] = val
    return val


def run_gc_bct(
    paths: Iterable[Sequence[Node]],
    D: int,
    edges: Iterable[Tuple[Node, Node]],
    beta: BetaSpec,
    eta: float = 0.5,
    exact_zero_count: bool = False,
    counts: Dict[Context, Dict[Node, int]] = None,
) -> GCBCTResult:
    """
    Parameters are as in ctw.run_gc_ctw, plus:

    exact_zero_count : force the general zero-count recursion (Eq
        3.33-3.35) even when beta >= 1/2 would allow the O(1) fast
        path. Useful for cross-checking the two code paths.

    counts : optional pre-computed count dict, bypassing
        forest.build_counts_from_paths. Lets a caller swap in an
        alternative counting/boundary convention without touching the
        MAP-fitting logic. None (default) preserves normal behaviour.
    """
    out_nbrs, in_nbrs, nodes = build_adjacency(edges)
    max_reach = compute_max_backward_reach(in_nbrs, nodes, D)
    if counts is None:
        counts = build_counts_from_paths(paths, D)
    b_fn = beta_fn(beta)

    const_beta = beta if not callable(beta) else None
    fast_path = (not exact_zero_count) and (const_beta is not None) and (const_beta >= 0.5)
    if not fast_path and not exact_zero_count and const_beta is not None:
        warnings.warn(
            f"beta={const_beta} < 0.5: the O(1) zero-count shortcut (Q_m = beta_s) is "
            "only exact for beta_s >= 1/2 (see Appendix B). Falling back to the "
            "exact recursion (Eq 3.34), which walks the unobserved subtree and can "
            "be expensive for large D / high branching.",
            stacklevel=2,
        )

    qm_cache: Dict[Context, float] = {}

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 4 * D + 100))
    try:
        def log_Qm_for(u):
            if len(u) == D:  # base case: Q_m = 1, no stop/expand choice at all
                return 0.0
            if fast_path:
                return math.log(b_fn(u))
            return _log_Qm(u, out_nbrs, in_nbrs, max_reach, D, b_fn, qm_cache)

        log_pe: Dict[Context, float] = {
            ctx: log_Pe(c, len(alphabet(ctx[0], out_nbrs)), eta) for ctx, c in counts.items()
        }

        by_depth = defaultdict(list)
        for ctx in counts:
            by_depth[len(ctx)].append(ctx)

        log_pm: Dict[Context, float] = {}
        decision: Dict[Context, int] = {}
        for ctx in by_depth[D]:
            log_pm[ctx] = log_pe[ctx]  # base case, depth D
            decision[ctx] = 0

        for depth in range(D - 1, 0, -1):
            for ctx in by_depth[depth]:
                kids = children_of(ctx, out_nbrs, in_nbrs, max_reach, D)
                b = b_fn(ctx)
                log_stop = math.log(b) + log_pe[ctx]  # stop term
                child_sum = 0.0
                for u in kids:
                    child_sum += log_pm[u] if u in counts else log_Qm_for(u)
                log_expand = math.log(1.0 - b) + child_sum  # expand term
                if log_stop >= log_expand:  # stop-on-tie, Eq 3.35
                    log_pm[ctx] = log_stop
                    decision[ctx] = 0
                else:
                    log_pm[ctx] = log_expand
                    decision[ctx] = 1

        roots = sorted(by_depth[1], key=repr)  # materialised roots only
        log_pm_lambda = sum(log_pm[r] for r in roots)  # Theorem 3.2.2, materialised part

        # Unlike GC-CTW (where an unobserved root contributes exactly 1),
        # an unobserved root here still needs its own Q_m (generally < 1) --
        # same zero-count treatment as unobserved descendants (Appendix B).
        
        all_active_roots = active_roots(nodes, out_nbrs, max_reach, D)
        unobserved_roots = [r for r in all_active_roots if r not in counts]
        log_pm_lambda += sum(log_Qm_for(r) for r in unobserved_roots)  # Appendix B

        # ---- top-down reconstruction ----
        map_leaves: List[Context] = []

        def reconstruct_zero_count(ctx):
            if len(ctx) == D:
                map_leaves.append(ctx)
                return
            if fast_path:
                map_leaves.append(ctx)  # stop is maximising when beta_s>=0.5 throughout
                return
            kids = children_of(ctx, out_nbrs, in_nbrs, max_reach, D)
            b = b_fn(ctx)
            log_stop = math.log(b)
            log_expand = math.log(1.0 - b) + sum(qm_cache[u] for u in kids)
            if log_stop >= log_expand:
                map_leaves.append(ctx)
            else:
                for u in kids:
                    reconstruct_zero_count(u)

        def reconstruct(ctx):
            if decision.get(ctx, 0) == 0 or len(ctx) == D:
                map_leaves.append(ctx)
            else:
                for u in children_of(ctx, out_nbrs, in_nbrs, max_reach, D):
                    if u in counts:
                        reconstruct(u)
                    else:
                        reconstruct_zero_count(u)

        for r in roots:
            reconstruct(r)
        for r in unobserved_roots:
            reconstruct_zero_count(r)
    finally:
        sys.setrecursionlimit(old_limit)

    return GCBCTResult(log_pm_lambda, map_leaves, log_pm, decision, counts, roots, beta, eta)
