"""Small shared helpers used by both ctw.py and bct.py."""
from typing import Callable, Hashable, Tuple, Union

Node = Hashable
Context = Tuple[Node, ...]
BetaSpec = Union[float, Callable[[Context], float]]


def beta_fn(beta: BetaSpec) -> Callable[[Context], float]:
    """Normalise a constant or per-context beta_s into a callable."""
    return beta if callable(beta) else (lambda ctx: beta)
