"""
Posterior inference for the graph-constrained Bayesian context-tree
model, built on top of already-fitted GC-CTW (ctw.py) and GC-BCT
(bct.py) results. Implements dissertation Section 3.3 ("Posterior
inference, sampling, and prediction").

gamma_s is the probability of STOPPING at a reached node s (Eq 3.40);
1 - gamma_s is the probability of expanding (Eq 3.41).
"""

import math
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from common import BetaSpec, Context, Node, beta_fn
from evidence import log_Pe
from graph_utils import alphabet, children_of



# Bundled result object and main entry point


class GCPosteriorResult:
    

    def __init__(self, ctw_result, bct_result, beta, eta, gamma, xi, r, q, log_map_posterior):
        self.ctw_result = ctw_result
        self.bct_result = bct_result
        self.beta = beta
        self.eta = eta
        self.gamma = gamma  #: dict: context -> gamma_s (stop prob.), Eq 3.40
        self.xi = xi  #: dict: context -> xi_s (expand prob.), Eq 3.41
        self.r = r  #: dict: context -> r_s (reach),       Eq 3.42
        self.q = q  #: dict: context -> q_s (selection),   Eq 3.43
        self.log_map_posterior = log_map_posterior  #: Eq 3.37-3.38, MAP forest

    @property
    def map_posterior_probability(self) -> float:
        """pi_D(T_MAP | D_D; beta), Eq 3.37-3.38."""
        return math.exp(self.log_map_posterior)


def run_gc_posterior(ctw_result, bct_result, beta: BetaSpec, eta: Optional[float] = None) -> GCPosteriorResult:
    
    if eta is None:
        eta = ctw_result.eta
    elif eta != ctw_result.eta:
        raise ValueError(
            f"eta={eta} does not match eta={ctw_result.eta} used to produce "
            "ctw_result; posterior quantities require the same eta throughout. "
            "Omit eta here to reuse ctw_result.eta automatically."
        )
    if not callable(beta) and not callable(ctw_result.beta) and beta != ctw_result.beta:
        raise ValueError(
            f"beta={beta} does not match the constant beta={ctw_result.beta} "
            "used to produce ctw_result."
        )
    if ctw_result.counts.keys() != bct_result.counts.keys():
        raise ValueError(
            "ctw_result and bct_result have different materialised context "
            "sets -- they must come from the same paths, D, and edges."
        )

    gamma, xi = local_posterior(ctw_result, beta)
    r, q = reach_and_selection_probabilities(ctw_result, beta, gamma, xi)
    log_map = bct_result.log_pm_lambda - ctw_result.log_pw_lambda  # Eq 3.37-3.38
    return GCPosteriorResult(ctw_result, bct_result, beta, eta, gamma, xi, r, q, log_map)



# 1. MAP posterior probability

# The scalar itself is just bct_result.log_pm_lambda minus
# ctw_result.log_pw_lambda (see run_gc_posterior above / the
# .log_map_posterior / .map_posterior_probability attributes).
# log_forest_posterior below is the same formula for an ARBITRARY
# forest, not just the MAP one; the enumeration tests use it to check
# that the MAP log-posterior matches the GC-BCT MAP theorem (Thm 3.2.2).

def log_forest_posterior(leaves: Sequence[Context], post: GCPosteriorResult) -> float:
   
    ctw_result = post.ctw_result
    D = ctw_result.D
    b_fn = beta_fn(post.beta)
    leaves = [tuple(s) for s in leaves]

    # Proper ancestors of the leaves, each internal node counted once
    # even if several leaves share it.
    internal = set()
    for leaf in leaves:
        for k in range(1, len(leaf)):
            internal.add(leaf[:k])

    log_prior = 0.0  # tree structural prior, Section 3.1.4
    for s in leaves:
        if len(s) < D:
            log_prior += math.log(b_fn(s))
    for s in internal:
        log_prior += math.log(1.0 - b_fn(s))

    log_lik = 0.0
    for s in leaves:
        if s in ctw_result.counts:
            log_lik += ctw_result.log_pe[s]
        else:
            m_s = len(alphabet(s[0], ctw_result.out_nbrs))
            log_lik += log_Pe({}, m_s, post.eta)  # = 0.0 for an unobserved node

    return log_prior + log_lik - ctw_result.log_pw_lambda



# 2. Local posterior probabilities gamma_s, xi_s 



def zero_count_local_posterior(ctx: Context, beta: BetaSpec) -> Tuple[float, float]:
    
    b = beta_fn(beta)(ctx)
    return b, 1.0 - b


def local_posterior(ctw_result, beta: BetaSpec) -> Tuple[Dict[Context, float], Dict[Context, float]]:
    
    b_fn = beta_fn(beta)
    D = ctw_result.D
    counts, log_pe, log_pw = ctw_result.counts, ctw_result.log_pe, ctw_result.log_pw
    out_nbrs, in_nbrs, max_reach = ctw_result.out_nbrs, ctw_result.in_nbrs, ctw_result.max_reach

    gamma: Dict[Context, float] = {}
    xi: Dict[Context, float] = {}
    for ctx in counts:
        if len(ctx) == D:
            gamma[ctx], xi[ctx] = 1.0, 0.0
            continue
        b = b_fn(ctx)
        log_g = math.log(b) + log_pe[ctx] - log_pw[ctx]  # Eq 3.40, log form
        kids = children_of(ctx, out_nbrs, in_nbrs, max_reach, D)
        child_sum = sum(log_pw[u] if u in counts else 0.0 for u in kids)  # unobserved child contributes 0
        log_x = math.log(1.0 - b) + child_sum - log_pw[ctx]  # Eq 3.41, log form
        g, x = math.exp(log_g), math.exp(log_x)
        total = g + x
        gamma[ctx], xi[ctx] = g / total, x / total
    return gamma, xi



# 3. Reach r_s, selection q_s, local-memory distribution 


def reach_and_selection_probabilities(ctw_result, beta: BetaSpec, gamma=None, xi=None):
    
    if gamma is None or xi is None:
        gamma, xi = local_posterior(ctw_result, beta)
    D = ctw_result.D
    counts = ctw_result.counts
    out_nbrs, in_nbrs, max_reach = ctw_result.out_nbrs, ctw_result.in_nbrs, ctw_result.max_reach

    by_depth = defaultdict(list)
    for ctx in counts:
        by_depth[len(ctx)].append(ctx)

    r: Dict[Context, float] = {ctx: 1.0 for ctx in ctw_result.roots}  # Eq 3.42, root case
    q: Dict[Context, float] = {}
    for depth in range(1, D + 1):
        for ctx in by_depth[depth]:
            q[ctx] = r[ctx] if depth == D else r[ctx] * gamma[ctx]  # Eq 3.43
            if depth < D:
                for u in children_of(ctx, out_nbrs, in_nbrs, max_reach, D):
                    if u in counts:
                        r[u] = r[ctx] * xi[ctx]  # Eq 3.42, recursive case
    return r, q


def active_roots(ctw_result) -> List[Context]:
    
    D = ctw_result.D
    return [
        (v,) for v in ctw_result.out_nbrs
        if ctw_result.out_nbrs[v] and ctw_result.max_reach.get(v, 0) >= D - 1
    ]


def local_memory_distribution(u: Sequence[Node], post: GCPosteriorResult) -> List[float]:
    
    ctw_result = post.ctw_result
    D = ctw_result.D
    u = tuple(u)
    if len(u) != D:
        raise ValueError(f"expected a length-{D} history, got length {len(u)}")
    counts = ctw_result.counts
    probs: List[float] = []
    r = 1.0
    for k in range(1, D + 1):
        s = u[:k]
        if k == D:
            g, x = 1.0, 0.0
        elif s in counts:
            g, x = post.gamma[s], post.xi[s]
        else:
            g, x = zero_count_local_posterior(s, post.beta)
        probs.append(r * g)
        r *= x
    return probs



# 4. Exact posterior forest sampler


def sample_forest(post: GCPosteriorResult, rng: Optional[np.random.Generator] = None) -> List[Context]:
    
    if rng is None:
        rng = np.random.default_rng()
    ctw_result = post.ctw_result
    D = ctw_result.D
    b_fn = beta_fn(post.beta)
    counts = ctw_result.counts
    out_nbrs, in_nbrs, max_reach = ctw_result.out_nbrs, ctw_result.in_nbrs, ctw_result.max_reach

    leaves: List[Context] = []
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 4 * D + 100))  # mirrors bct.py's guard, same rationale
    try:
        def sample_node(ctx):
            if len(ctx) == D:
                leaves.append(ctx)
                return
            g = post.gamma[ctx] if ctx in counts else b_fn(ctx)
            if rng.random() < g:
                leaves.append(ctx)
                return
            for child in children_of(ctx, out_nbrs, in_nbrs, max_reach, D):
                sample_node(child)

        for root in active_roots(ctw_result):
            sample_node(root)
    finally:
        sys.setrecursionlimit(old_limit)
    return leaves


def sample_joint(post: GCPosteriorResult, rng: Optional[np.random.Generator] = None):
    
    if rng is None:
        rng = np.random.default_rng()
    T = sample_forest(post, rng)
    theta = {}
    for s in T:
        tp = transition_posterior(s, post)
        draw = rng.dirichlet(list(tp["alpha"].values()))
        theta[s] = dict(zip(tp["alphabet"], draw))
    return T, theta



# 5. Transition-parameter posterior 


def transition_posterior(ctx: Context, post: GCPosteriorResult, n_samples: int = 0,
                          rng: Optional[np.random.Generator] = None) -> dict:
    
    ctw_result = post.ctw_result
    out_nbrs = ctw_result.out_nbrs
    A_s = sorted(alphabet(ctx[0], out_nbrs), key=repr)  # canonical order, robust to non-orderable nodes
    m_s = len(A_s)
    if m_s == 0:
        raise ValueError(f"context {ctx!r} has empty predictive alphabet")
    a_s = ctw_result.counts.get(ctx, {})
    alpha = np.array([a_s.get(j, 0) + post.eta for j in A_s], dtype=float)  # posterior Dirichlet parameters
    alpha0 = float(alpha.sum())
    mean = alpha / alpha0  # posterior mean
    var = alpha * (alpha0 - alpha) / (alpha0 ** 2 * (alpha0 + 1))  # posterior variance
    cov = -np.outer(alpha, alpha) / (alpha0 ** 2 * (alpha0 + 1))  # posterior covariance
    np.fill_diagonal(cov, var)

    out = {
        "alphabet": A_s,
        "alpha": dict(zip(A_s, alpha.tolist())),
        "alpha0": alpha0,
        "mean": dict(zip(A_s, mean.tolist())),
        "variance": dict(zip(A_s, var.tolist())),
        "covariance": cov,
    }
    if n_samples > 0:
        if rng is None:
            rng = np.random.default_rng()
        out["samples"] = rng.dirichlet(alpha, size=n_samples)  # columns follow `alphabet` order
    return out


# 6. Posterior predictive distribution 


def predictive_mixture(u: Sequence[Node], post: GCPosteriorResult) -> Dict[Node, float]:
    
    ctw_result = post.ctw_result
    D = ctw_result.D
    u = tuple(u)
    A_u = sorted(alphabet(u[0], ctw_result.out_nbrs), key=repr)
    q_branch = local_memory_distribution(u, post)
    probs = {j: 0.0 for j in A_u}
    for k in range(1, D + 1):
        q_s = q_branch[k - 1]
        if q_s == 0.0:
            continue
        s = u[:k]
        a_s = ctw_result.counts.get(s, {})
        denom = sum(a_s.values()) + len(A_u) * post.eta
        for j in A_u:
            probs[j] += q_s * (a_s.get(j, 0) + post.eta) / denom  # mixture over branch contexts
    return probs


def predictive_evidence_ratio(u: Sequence[Node], j: Node, post: GCPosteriorResult) -> float:
    
    ctw_result = post.ctw_result
    D = ctw_result.D
    u = tuple(u)
    out_nbrs, in_nbrs, max_reach = ctw_result.out_nbrs, ctw_result.in_nbrs, ctw_result.max_reach
    A_u = alphabet(u[0], out_nbrs)
    if j not in A_u:
        return 0.0
    b_fn = beta_fn(post.beta)
    counts, log_pw = ctw_result.counts, ctw_result.log_pw
    eta = post.eta

    updated = None  # updated log P_w at the deepest-processed prefix so far
    for k in range(D, 0, -1):
        s = u[:k]
        a_s = dict(counts.get(s, {}))
        a_s[j] = a_s.get(j, 0) + 1
        m_s = len(alphabet(s[0], out_nbrs))
        ell_e_updated = log_Pe(a_s, m_s, eta)
        if k == D:
            updated = ell_e_updated  # local likelihood at the leaf
            continue
        b = b_fn(s)
        on_path = u[:k + 1]
        child_sum = 0.0
        for kid in children_of(s, out_nbrs, in_nbrs, max_reach, D):
            child_sum += updated if kid == on_path else (log_pw[kid] if kid in counts else 0.0)
        ell_stop = math.log(b) + ell_e_updated
        ell_expand = math.log(1.0 - b) + child_sum
        updated = float(logsumexp([ell_stop, ell_expand]))  # GC-CTW recursion, Eq 3.29-3.30

    root = (u[0],)
    original_root = log_pw[root] if root in counts else 0.0
    return math.exp(updated - original_root)
