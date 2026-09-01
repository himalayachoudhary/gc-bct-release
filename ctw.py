"""
GC-CTW: the graph-constrained context-tree weighting algorithm.
Computes

    P_{w,lambda} = P(D_D | G, D)

exactly, in the log domain, via the backward weighted-probability
recursion.

"""
import math
from collections import defaultdict
from typing import Dict, Hashable, Iterable, Sequence, Tuple

from scipy.special import logsumexp

from common import BetaSpec, Context, Node, beta_fn
from evidence import log_Pe
from forest import build_counts_from_paths
from graph_utils import alphabet, build_adjacency, children_of, compute_max_backward_reach


class GCCTWResult:

    def __init__(self, log_pw_lambda, log_pw, log_pe, counts, roots, out_nbrs, in_nbrs, max_reach, D, beta, eta):
        self.log_pw_lambda = log_pw_lambda  #: log P(D_D | G, D), Theorem 3.2.1
        self.log_pw = log_pw  #: dict: context -> log P^G_{w,s}, Eq 3.29-3.30
        self.log_pe = log_pe  #: dict: context -> log P_e^G(a_s), Lemma 3.1.2 (Eq 3.25)
        self.counts = counts  #: dict: context -> {next_node: count}
        self.roots = roots  #: the materialised (data-touching) length-1 contexts
        self.out_nbrs = out_nbrs
        self.in_nbrs = in_nbrs
        self.max_reach = max_reach
        self.D = D
        self.beta = beta  #: the BetaSpec (float or callable) this result was fitted with
        self.eta = eta  #: the Dirichlet concentration this result was fitted with

    @property
    def pw_lambda(self) -> float:
        return math.exp(self.log_pw_lambda)


def run_gc_ctw(
    paths: Iterable[Sequence[Node]],
    D: int,
    edges: Iterable[Tuple[Node, Node]],
    beta: BetaSpec,
    eta: float = 0.5,        #Jeffrey's prior
) -> GCCTWResult:
    
    out_nbrs, in_nbrs, nodes = build_adjacency(edges)
    max_reach = compute_max_backward_reach(in_nbrs, nodes, D)
    counts = build_counts_from_paths(paths, D)
    b_fn = beta_fn(beta)

    log_pe: Dict[Context, float] = {
        ctx: log_Pe(c, len(alphabet(ctx[0], out_nbrs)), eta) for ctx, c in counts.items()
    }

    by_depth = defaultdict(list)
    for ctx in counts:
        by_depth[len(ctx)].append(ctx)

    log_pw: Dict[Context, float] = {}
    for ctx in by_depth[D]:
        log_pw[ctx] = log_pe[ctx]  # base case, depth D

    for depth in range(D - 1, 0, -1):
        for ctx in by_depth[depth]:
            kids = children_of(ctx, out_nbrs, in_nbrs, max_reach, D)
            b = b_fn(ctx)
            log_stop = math.log(b) + log_pe[ctx]  # stop term
            child_sum = 0.0
            for u in kids:
                child_sum += log_pw[u] if u in counts else 0.0  # unobserved: P_w = 1 (Appendix B)
            log_expand = math.log(1.0 - b) + child_sum  # expand term
            log_pw[ctx] = float(logsumexp([log_stop, log_expand]))  # Eq 3.29-3.30

    roots = sorted(by_depth[1], key=repr)
    log_pw_lambda = sum(log_pw[r] for r in roots)  # Theorem 3.2.1
    return GCCTWResult(log_pw_lambda, log_pw, log_pe, counts, roots, out_nbrs, in_nbrs, max_reach, D, beta, eta)
