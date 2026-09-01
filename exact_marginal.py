"""
Exact (non-simulated) first-order marginal transition probabilities of a
graph-constrained variable-order process, via the stationary distribution
of the underlying history-state Markov chain.

Method: build the transition matrix P over graph-valid length-D
histories, find its stationary distribution pi (pi P = pi), then
marginalise over histories sharing the same current physical node:

    P(w | v) = [sum_{h: tau(h)=v} pi(h) * theta_{C(h)}(w)]
               / [sum_{h: tau(h)=v} pi(h)]

"""

from typing import Dict, Tuple

import numpy as np

from common import Context, Node
from graph_utils import build_adjacency
from history import History, enumerate_valid_histories, selected_context, shift_history

Theta = Dict[Context, Dict[Node, float]]


def _build_history_transition_matrix(edges, forest, theta: Theta, D: int):
    out_nbrs, in_nbrs, nodes = build_adjacency(edges)
    forest = frozenset(forest)
    histories = sorted(enumerate_valid_histories(out_nbrs, nodes, D), key=repr)
    idx = {h: i for i, h in enumerate(histories)}
    n = len(histories)

    P = np.zeros((n, n))
    for i, h in enumerate(histories):
        v = h[-1]
        c = selected_context(h, forest)
        if c not in theta:
            raise ValueError(f"theta has no entry for selected context {c!r} (history {h!r})")
        theta_c = theta[c]
        for w in out_nbrs.get(v, ()):
            if w not in theta_c:
                raise ValueError(f"theta[{c!r}] has no entry for graph-valid successor {w!r}")
            h2 = shift_history(h, w)
            P[i, idx[h2]] += theta_c[w]

    row_sums = P.sum(axis=1)
    bad = np.where(np.abs(row_sums - 1.0) > 1e-8)[0]
    if len(bad):
        raise AssertionError(
            f"transition matrix is not row-stochastic at {len(bad)} history index/indices "
            f"(e.g. row {bad[0]}={histories[bad[0]]!r}, sum={row_sums[bad[0]]!r}) -- theta must "
            "assign probability to every graph-valid successor of every selected context"
        )
    return P, histories, out_nbrs, nodes


def exact_history_stationary_distribution(edges, forest, theta: Theta, D: int) -> Dict[History, float]:
    P, histories, out_nbrs, nodes = _build_history_transition_matrix(edges, forest, theta, D)
    n = len(histories)

    eigvals, eigvecs = np.linalg.eig(P.T)
    i_star = int(np.argmin(np.abs(eigvals - 1.0)))
    if abs(eigvals[i_star] - 1.0) > 1e-6:
        raise RuntimeError(
            f"no eigenvalue of P^T is close to 1 (closest: {eigvals[i_star]!r}); P may not be a "
            "valid row-stochastic matrix, or the chain may be reducible"
        )
    vec = np.real(eigvecs[:, i_star])
    if vec.sum() < 0:
        vec = -vec
    if np.any(vec < -1e-9):
        raise RuntimeError(
            f"stationary eigenvector has entries below -1e-9 (min={vec.min()!r}) -- chain may be "
            "reducible or periodic in an unexpected way"
        )
    vec = np.clip(vec, 0, None)
    pi = vec / vec.sum()

    if not np.allclose(pi @ P, pi, atol=1e-6):
        resid = np.abs(pi @ P - pi).max()
        raise AssertionError(f"pi @ P != pi (max residual {resid!r}) -- stationary solve failed")

    return {h: float(pi[i]) for i, h in enumerate(histories)}


def exact_first_order_marginal(edges, forest, theta: Theta, D: int) -> Dict[Node, Dict[Node, float]]:
    """{v: {w: P(w|v)}} under the exact stationary distribution of the
    length-D history chain, for every physical node v with out-edges."""
    pi = exact_history_stationary_distribution(edges, forest, theta, D)
    out_nbrs, in_nbrs, nodes = build_adjacency(edges)
    forest = frozenset(forest)

    mass_v: Dict[Node, float] = {v: 0.0 for v in nodes}
    mass_vw: Dict[Node, Dict[Node, float]] = {v: {w: 0.0 for w in out_nbrs.get(v, ())} for v in nodes}
    for h, p in pi.items():
        v = h[-1]
        c = selected_context(h, forest)
        theta_c = theta[c]
        mass_v[v] += p
        for w, pw in theta_c.items():
            mass_vw[v][w] += p * pw

    marginal: Dict[Node, Dict[Node, float]] = {}
    for v in nodes:
        if not out_nbrs.get(v) or mass_v[v] <= 0:
            continue
        marginal[v] = {w: mass_vw[v][w] / mass_v[v] for w in out_nbrs[v]}
    return marginal


def max_first_order_deviation(edges, marginal_a, marginal_b) -> float:
    worst = 0.0
    for v, w in edges:
        if v in marginal_a and v in marginal_b and w in marginal_a[v] and w in marginal_b[v]:
            worst = max(worst, abs(marginal_a[v][w] - marginal_b[v][w]))
    return worst
