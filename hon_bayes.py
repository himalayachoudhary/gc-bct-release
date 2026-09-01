"""
Bayesian wrapper procedures (Section 3.4.4) connecting hon.py's
posterior-agnostic HON construction to the existing GC-CTW/GC-BCT/
posterior pipeline (ctw.py, bct.py, posterior.py).

Applies the deterministic map Phi_{G,D}(T, theta_T) to posterior draws
of (T, theta_T): the selected forest comes from bct_result.map_leaves,
posterior-mean parameters from posterior.transition_posterior, and full
joint (T, theta) draws from posterior.sample_joint.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from hon import RefinedHON, build_exact_hon, build_refined_hon, refine_routing_partition
from posterior import GCPosteriorResult, sample_joint, transition_posterior


def build_map_hon(post: GCPosteriorResult, D: Optional[int] = None) -> RefinedHON:
    """
    Conditional posterior-mean HON based on the MAP forest T_hat_MAP:
    apply Phi_{G,D} to (T_hat_MAP, Theta_hat), with Theta_hat_c the
    posterior mean of theta_c for each c in T_hat_MAP. A useful graph
    summary, not necessarily the mode of the induced graph-valued
    posterior.
    """
    D = D if D is not None else post.ctw_result.D
    selected_forest = post.bct_result.map_leaves
    theta = {c: transition_posterior(c, post)["mean"] for c in selected_forest}

    exact = build_exact_hon(post.ctw_result.out_nbrs, D, selected_forest, theta)
    partition = refine_routing_partition(exact)
    return build_refined_hon(exact, partition, theta)


def sample_posterior_hon(
    post: GCPosteriorResult,
    rng: Optional[np.random.Generator] = None,
    D: Optional[int] = None,
) -> RefinedHON:
    """
    One draw H^(m) = Phi_{G,D}(T^(m), Theta^(m)) from the posterior over
    BCT-HONs (Section 3.4.4): sample (T, Theta) jointly via
    posterior.sample_joint, then run the same construction build_map_hon
    uses.
    """
    if rng is None:
        rng = np.random.default_rng()
    D = D if D is not None else post.ctw_result.D

    T, theta = sample_joint(post, rng)
    exact = build_exact_hon(post.ctw_result.out_nbrs, D, T, theta)
    partition = refine_routing_partition(exact)
    return build_refined_hon(exact, partition, theta)


def sample_posterior_hons(
    post: GCPosteriorResult,
    n: int,
    rng: Optional[np.random.Generator] = None,
    D: Optional[int] = None,
) -> List[RefinedHON]:
    """
    n independent posterior BCT-HON draws H^(1), ..., H^(n) (Section
    3.4.4). Each draw resamples both T^(m) and Theta^(m) independently
    and reruns the full construction -- no reuse across draws even when
    two happen to share the same sampled forest.
    """
    if rng is None:
        rng = np.random.default_rng()
    D = D if D is not None else post.ctw_result.D
    return [sample_posterior_hon(post, rng=rng, D=D) for _ in range(n)]
