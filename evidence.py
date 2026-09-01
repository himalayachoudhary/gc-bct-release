"""
Local evidence P_e,s (Lemma 3.1.2, Eq 3.25), computed in the log domain
via scipy.special.gammaln:

    log Pe(a_s) = log Gamma(m_s*eta) - log Gamma(M_s + m_s*eta)
                  + sum_j [ log Gamma(a_s(j) + eta) - log Gamma(eta) ]

Zero-count symbols contribute 0 to the sum, so we only need the sparse
counts dict and alphabet size m_s, not the full alphabet.
"""
from typing import Dict, Hashable

from scipy.special import gammaln

Node = Hashable


def log_Pe(counts: Dict[Node, int], alphabet_size: int, eta: float = 0.5) -> float:
    """
    log P_e,s for a context with predictive-alphabet size m_s =
    `alphabet_size` and next-state counts `counts` (missing keys = 0).

    Returns 0.0 when there's no data or m_s = 1 (deterministic row) --
    no special-casing needed.
    """
    if alphabet_size <= 0:
        raise ValueError(
            "context has empty predictive alphabet (a physical sink); this "
            "should be unreachable if S+_{<=D}(G) filtering upstream is correct"
        )
    total = sum(counts.values())
    ll = float(gammaln(alphabet_size * eta) - gammaln(total + alphabet_size * eta))
    for a_j in counts.values():
        ll += float(gammaln(a_j + eta) - gammaln(eta))
    return ll
