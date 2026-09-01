"""
Construction of a Bayesian higher-order network (HON) from a fixed
graph-constrained context forest T and transition parameters
Theta = {theta_c : c in T}.

Implements the two algorithms of dissertation Appendix C, operating on a
given (T, Theta) with no dependency on ctw.py/bct.py/posterior.py:

  build_exact_hon          Algorithm 1 (Appendix C.1.1) -- exact
                            history-state HON (Section 3.4.1, Eq 3.52-3.53)
  refine_routing_partition Algorithm 2, first half (Appendix C.1.2) --
                            routing-consistent partition refinement
                            (Section 3.4.2, Theorem 3.4.2)
  build_refined_hon        Algorithm 2, second half (Appendix C.1.2) --
                            BCT-HON construction (Section 3.4.3, Theorem 3.4.3)

hon_bayes.py supplies T and Theta from posterior output (Section 3.4.4).

Histories throughout are `history.History`, chronological tuples (oldest
first, current last) -- see history.py's module docstring for why this
differs from the most-recent-first `Context` type used elsewhere, and for
the one place the two meet (`history.selected_context`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Set, Tuple

from common import Context, Node
from history import History, enumerate_valid_histories, selected_context, shift_history

Theta = Dict[Context, Dict[Node, float]]  # theta_c(w), c in T, w in N+(tau(c))


# ======================================================================
# Algorithm 1 (Appendix C.1.1, Section 3.4.1): exact history-state HON
# ======================================================================

@dataclass
class ExactHON:
    """
    Output of build_exact_hon: the uncompressed HON on the full (or
    reachable-restricted) history-state set U.

    histories               U (= S_D(G), or a caller-supplied subset)
    physical_node           pi(u) = tau(u),                per history
    context_label           lambda_T(u) = C_T(u),           per history
    next_history            Next[u, w] = sh(u, w)
    transition_probability  q[u, w] = theta_{lambda_T(u)}(w)
    """

    histories: FrozenSet[History]
    physical_node: Dict[History, Node]
    context_label: Dict[History, Context]
    next_history: Dict[Tuple[History, Node], History]
    transition_probability: Dict[Tuple[History, Node], float]


def build_exact_hon(
    out_nbrs: Dict[Node, Set[Node]],
    D: int,
    selected_forest: Iterable[Context],
    theta: Theta,
    histories: Optional[Set[History]] = None,
) -> ExactHON:
    """
    Algorithm 1: exact, uncompressed history-state HON on a fixed context
    forest T and parameters Theta (Section 3.4.1).

    `histories` defaults to the full S_D(G); pass
    `history.enumerate_reachable_histories` instead to restrict to states
    reachable from a seed set. Construction is otherwise identical.
    """
    forest = frozenset(selected_forest)
    missing_theta = forest - theta.keys()
    if missing_theta:
        raise ValueError(
            f"theta is missing entries for selected context(s) {sorted(missing_theta, key=repr)}"
        )

    if histories is None:
        histories = enumerate_valid_histories(out_nbrs, set(out_nbrs.keys()), D)

    physical_node: Dict[History, Node] = {}
    context_label: Dict[History, Context] = {}
    next_history: Dict[Tuple[History, Node], History] = {}
    transition_probability: Dict[Tuple[History, Node], float] = {}

    for u in histories:
        if len(u) != D:
            raise ValueError(f"history {u!r} does not have length D={D}")
        pi_u = u[-1]  # tau(u)
        physical_node[u] = pi_u

        if not out_nbrs.get(pi_u):
            # pi_u is a graph sink: no successors, and no context in the
            # forest has pi_u as its current node, so selected_context
            # would fail here. Give it its own absorbing context label
            # (pi_u,) instead, with no outgoing transitions.
            context_label[u] = (pi_u,)
            continue

        c = selected_context(u, forest)  # C_T(u); the one History->Context conversion
        context_label[u] = c
        theta_c = theta[c]
        for w in out_nbrs.get(pi_u, ()):
            v = shift_history(u, w)
            if v not in histories:
                raise RuntimeError(
                    f"shift({u!r}, {w!r}) = {v!r} is not in the supplied history set; "
                    "`histories` must be closed under every graph-valid shift "
                    "(Algorithm 1's own 'assert v in U') -- did you pass a manually "
                    "restricted set that isn't actually shift-closed? "
                    "history.enumerate_reachable_histories produces a closed set."
                )
            if w not in theta_c:
                raise ValueError(
                    f"theta[{c!r}] has no entry for successor {w!r} needed by history "
                    f"{u!r}; theta_c must cover every w in N+(tau(c))"
                )
            next_history[(u, w)] = v
            transition_probability[(u, w)] = theta_c[w]

    return ExactHON(
        histories=frozenset(histories),
        physical_node=physical_node,
        context_label=context_label,
        next_history=next_history,
        transition_probability=transition_probability,
    )


# ======================================================================
# Algorithm 2, first half (Appendix C.1.2, Section 3.4.2): routing-
# consistent partition refinement
# ======================================================================

@dataclass
class RoutingPartition:
    """
    Output of refine_routing_partition: the stable partition P*_T
    (Theorem 3.4.2).

    blocks            the stable blocks, each a frozenset of histories
    block_of_history  history -> index into `blocks`
    iterations        number of sweeps of the repeat-loop that ran,
                       including the final non-splitting confirmation
                       sweep (a forest already routing-consistent gives
                       iterations == 1)
    """

    blocks: List[FrozenSet[History]]
    block_of_history: Dict[History, int]
    iterations: int


def _successors_by_history(exact_hon: ExactHON) -> Dict[History, List[Node]]:
    """N+(pi(u)) as represented in `exact_hon`, per history, in a fixed
    canonical order -- shared by refine_routing_partition (signature
    construction) and build_refined_hon (edge construction)."""
    successors_of: Dict[History, List[Node]] = {}
    for (u, w) in exact_hon.next_history:
        successors_of.setdefault(u, []).append(w)
    for succs in successors_of.values():
        succs.sort(key=repr)
    return successors_of


def refine_routing_partition(exact_hon: ExactHON) -> RoutingPartition:
    """
    Algorithm 2, first half (Section 3.4.2): coarsest routing-consistent
    refinement P*_T of the predictive partition P_T (Theorem 3.4.2).

    Starts with one block per selected context, then within each block
    buckets histories by their destination-block signature (Eq 3.57)
    under a fixed successor order, splitting wherever members disagree,
    repeating until a full sweep produces no split. Blocks only ever
    split, never merge across contexts, so every final block sits inside
    exactly one initial context-labelled block.
    """
    successors_of = _successors_by_history(exact_hon)

    initial: Dict[Context, Set[History]] = {}
    for u, c in exact_hon.context_label.items():
        initial.setdefault(c, set()).add(u)
    blocks: List[FrozenSet[History]] = [frozenset(members) for members in initial.values()]

    iterations = 0
    while True:
        iterations += 1
        block_of: Dict[History, int] = {u: i for i, b in enumerate(blocks) for u in b}

        new_blocks: List[FrozenSet[History]] = []
        changed = False
        for b in blocks:
            buckets: Dict[Tuple, List[History]] = {}
            for u in b:
                sig = tuple(
                    (w, block_of[exact_hon.next_history[(u, w)]])
                    for w in successors_of.get(u, [])
                )
                buckets.setdefault(sig, []).append(u)
            if len(buckets) > 1:
                changed = True
            new_blocks.extend(frozenset(members) for members in buckets.values())
        blocks = new_blocks
        if not changed:
            break

    block_of_history = {u: i for i, b in enumerate(blocks) for u in b}
    return RoutingPartition(blocks=blocks, block_of_history=block_of_history, iterations=iterations)


# ======================================================================
# Algorithm 2, second half (Appendix C.1.2, Section 3.4.3): BCT-HON
# construction
# ======================================================================

class HONState(NamedTuple):
    """One row of the HON states output table."""

    physical_node: Node
    context: Context
    context_depth: int
    label: str  # display only -- membership is the state's real definition


class HONEdge(NamedTuple):
    """One row of the HON edges output table."""

    source: int
    target: int
    next_physical_node: Node
    probability: float


@dataclass
class RefinedHON:
    """
    Output of build_refined_hon: H*_T = (P*_T, E*_T, Q*_T, pi, lambda_T),
    the BCT-HON (Eq 3.61).

    states / membership / edges     the three primary output tables
    ties                            context -> state ids sharing it
    block_of_history                history -> state id
    transition                      Q*_T as a sparse {source: {target:
                                     probability}} cache derived from
                                     `edges`, for O(1) lookups
    """

    states: Dict[int, HONState]
    membership: Dict[int, FrozenSet[History]]
    edges: List[HONEdge]
    ties: Dict[Context, FrozenSet[int]]
    block_of_history: Dict[History, int]
    transition: Dict[int, Dict[int, float]]


def _display_label(context: Context, D: int, representatives: List[Tuple[int, History]]) -> Dict[int, str]:
    """
    Human-readable label per block sharing `context` as their predictive-
    context label. Uses "current | rest" when the context alone
    disambiguates, otherwise extends with more of a representative
    history's older tokens until labels in the group are unique, falling
    back to a neutral "label[i]" suffix if even the full history doesn't
    disambiguate (in practice this never triggers, since distinct blocks
    always have distinct representative histories).

    Labels are cosmetic only -- `RefinedHON.membership` is the real
    definition of a state.
    """
    base = str(context[0]) if len(context) == 1 else f"{context[0]}|{','.join(str(x) for x in context[1:])}"
    if len(representatives) == 1:
        return {representatives[0][0]: base}

    depth = len(context)
    stop = D - depth
    for extra_len in range(1, stop + 1):
        start = stop - extra_len
        labels: Dict[int, str] = {}
        seen: Set[str] = set()
        ok = True
        for block_id, rep in representatives:
            extra = ",".join(str(x) for x in reversed(rep[start:stop]))
            lbl = f"{base}|{extra}"
            if lbl in seen:
                ok = False
                break
            seen.add(lbl)
            labels[block_id] = lbl
        if ok:
            return labels

    return {block_id: f"{base}[{i}]" for i, (block_id, _rep) in enumerate(representatives)}


def build_refined_hon(
    exact_hon: ExactHON,
    routing_partition: RoutingPartition,
    theta: Theta,
    tolerance: float = 1e-9,
) -> RefinedHON:
    """
    Algorithm 2, second half (Section 3.4.3): converts the stable blocks
    of P*_T into the BCT-HON's state nodes and edges (Theorem 3.4.3).

    Checks block homogeneity, unique routing, and row-stochasticity at
    runtime -- these only hold if `routing_partition` was produced by
    `refine_routing_partition` on this same `exact_hon`.

    Block ids are assigned in a canonical (repr-sorted) order rather than
    RoutingPartition.blocks' incidental order, so output is reproducible.
    """
    if not exact_hon.histories:
        raise ValueError("exact_hon has no histories to build states from")
    D = len(next(iter(exact_hon.histories)))

    def block_sort_key(block: FrozenSet[History]):
        rep = next(iter(block))
        return (
            repr(exact_hon.physical_node[rep]),
            repr(exact_hon.context_label[rep]),
            tuple(sorted(repr(u) for u in block)),
        )

    ordered_blocks = sorted(routing_partition.blocks, key=block_sort_key)
    representative_of: Dict[int, History] = {}
    block_of_history: Dict[History, int] = {}
    for block_id, block in enumerate(ordered_blocks):
        rep = next(iter(block))
        representative_of[block_id] = rep
        for u in block:
            if exact_hon.physical_node[u] != exact_hon.physical_node[rep]:
                raise AssertionError(
                    f"block homogeneity violated in block {block_id}: {u!r} has physical node "
                    f"{exact_hon.physical_node[u]!r}, expected {exact_hon.physical_node[rep]!r} "
                    "(Algorithm 2's 'assert pi(u) = pi(B)')"
                )
            if exact_hon.context_label[u] != exact_hon.context_label[rep]:
                raise AssertionError(
                    f"block homogeneity violated in block {block_id}: {u!r} has context "
                    f"{exact_hon.context_label[u]!r}, expected {exact_hon.context_label[rep]!r} "
                    "(Algorithm 2's 'assert lambda_T(u) = lambda_T(B)')"
                )
            block_of_history[u] = block_id

    by_context: Dict[Context, List[Tuple[int, History]]] = {}
    for block_id in range(len(ordered_blocks)):
        rep = representative_of[block_id]
        by_context.setdefault(exact_hon.context_label[rep], []).append((block_id, rep))
    labels: Dict[int, str] = {}
    for context, reps in by_context.items():
        labels.update(_display_label(context, D, reps))

    states: Dict[int, HONState] = {}
    membership: Dict[int, FrozenSet[History]] = {}
    ties: Dict[Context, Set[int]] = {}
    for block_id, block in enumerate(ordered_blocks):
        rep = representative_of[block_id]
        lambda_b = exact_hon.context_label[rep]
        states[block_id] = HONState(
            physical_node=exact_hon.physical_node[rep],
            context=lambda_b,
            context_depth=len(lambda_b),
            label=labels[block_id],
        )
        membership[block_id] = block
        ties.setdefault(lambda_b, set()).add(block_id)

    successors_of = _successors_by_history(exact_hon)
    edges: List[HONEdge] = []
    transition: Dict[int, Dict[int, float]] = {bid: {} for bid in states}
    for block_id, block in enumerate(ordered_blocks):
        rep = representative_of[block_id]
        lambda_b = exact_hon.context_label[rep]
        succs = successors_of.get(rep, [])
        if not succs:
            # Sink/absorbing state (build_exact_hon's (pi_u,) case): no
            # successors, no theta row to look up, so skip.
            continue
        theta_b = theta[lambda_b]
        for w in succs:
            target_id = block_of_history[exact_hon.next_history[(rep, w)]]
            for u in block:
                actual_target = block_of_history[exact_hon.next_history[(u, w)]]
                if actual_target != target_id:
                    raise AssertionError(
                        f"routing consistency violated at block {block_id} ({lambda_b!r}), successor "
                        f"{w!r}: history {u!r} routes to state {actual_target}, representative {rep!r} "
                        f"routes to state {target_id} (Algorithm 2's 'assert BlockOf[Next[u,w]] = B''); "
                        "this should be impossible if routing_partition was produced by "
                        "refine_routing_partition(exact_hon) on this exact_hon"
                    )
            p = theta_b[w]
            edges.append(HONEdge(source=block_id, target=target_id, next_physical_node=w, probability=p))
            transition[block_id][target_id] = transition[block_id].get(target_id, 0.0) + p

    for block_id in states:
        total = sum(transition[block_id].values())
        if not transition[block_id]:
            # Sink/absorbing state: no outgoing edges, so nothing to
            # normalise -- zero successors here is expected, not a bug.
            continue
        if abs(total - 1.0) >= tolerance:
            raise AssertionError(
                f"row-stochasticity violated at block {block_id} ({states[block_id]!r}): outgoing "
                f"probabilities sum to {total!r}, not 1 (Algorithm 2's final assertion). This state DOES "
                "have some successors recorded, so this is a real partial-coverage bug -- not the genuine "
                "no-successors-at-all sink case (handled separately above)."
            )

    return RefinedHON(
        states=states,
        membership=membership,
        edges=edges,
        ties={c: frozenset(ids) for c, ids in ties.items()},
        block_of_history=block_of_history,
        transition=transition,
    )
