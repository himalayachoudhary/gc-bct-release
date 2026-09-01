"""
Local evidence P_e,s computed in the log domain
via scipy.special.gammaln:

    log Pe(a_s) = log Gamma(m_s*eta) - log Gamma(M_s + m_s*eta)
                  + sum_j [ log Gamma(a_s(j) + eta) - log Gamma(eta) ]
"""
from typing import Dict, Hashable

from scipy.special import gammaln

Node = Hashable


def log_Pe(counts: Dict[Node, int], alphabet_size: int, eta: float = 0.5) -> float:
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
