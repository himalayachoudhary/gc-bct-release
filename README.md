# Graph-Constrained Bayesian Context Trees (GC-CTW / GC-BCT)

Core inference engine accompanying the MSc dissertation *"Bayesian Learning
of Higher-Order Networks from Path Data"* (Himalaya Choudhary, Linacre
College, MSc in Mathematical Modelling and Scientific Computing, University
of Oxford, Trinity Term 2026, supervised by Prof. Renaud Lambiotte).

The dissertation extends the Bayesian Context Trees (BCT) framework of
Kontoyiannis, Mertzanis, Panotopoulou, Papageorgiou & Skoularidou (*JRSSB*,
2022) with an explicit graph constraint, giving every node in a network its
own locally-inferred memory order rather than a single global order for the
whole process. This repository contains the exact graph-constrained
context-tree weighting (GC-CTW) and Bayesian context-tree (GC-BCT)
algorithms, the posterior-inference layer built on top of them, and the
construction of a Bayesian higher-order network (BCT-HON) from the fitted
posterior — the two original contributions of the dissertation (Chapter 3;
proofs in Appendix A).

**Scope note.** This repository holds the core inference engine only.
Baseline-method implementations used for comparison (FON, the Bayesian
global-order model, BuildHON+, SDVOM), the experiment/data-generation
scripts, and the datasets are not included here.

## Module map

| File | Implements |
|---|---|
| `common.py` | Shared types (`Node`, `Context`, `BetaSpec`) used by `ctw.py` and `bct.py` |
| `graph_utils.py` | Graph primitives underlying the potential context forest: adjacency, predictive alphabet, graph-valid children (Section 3.1.2) |
| `forest.py` | Count-vector construction at every genuinely-observed potential context, from raw path data (Section 3.1.3) |
| `evidence.py` | Local parameter-integrated evidence under the symmetric Dirichlet prior (Section 3.1.5) |
| `ctw.py` | GC-CTW: the graph-constrained context-tree weighting recursion, exact prior-predictive likelihood (Section 3.2.1) |
| `bct.py` | GC-BCT: the graph-constrained Bayesian context-tree algorithm, exact MAP context tree (Section 3.2.2; proof in Appendix A.1) |
| `posterior.py` | Posterior inference on top of the fitted GC-CTW/GC-BCT results: posterior transition parameters, local memory probabilities, exact posterior sampling, posterior prediction (Section 3.3) |
| `history.py` | Graph-valid chronological histories underlying the higher-order network construction (Section 3.4.1) |
| `hon.py` | Construction of a Bayesian higher-order network (BCT-HON) from a fixed context forest and transition parameters, including the coarsest routing-consistent refinement (Section 3.4, Appendix C) |
| `hon_bayes.py` | Bayesian wrapper connecting `hon.py`'s posterior-agnostic procedures to the GC-CTW/GC-BCT/posterior pipeline (Section 3.4.4) |
| `exact_marginal.py` | Exact (non-simulated) first-order marginal transition probabilities of a graph-constrained variable-order process, via the stationary distribution of the underlying history-state Markov chain — a validation utility, not a numbered dissertation result |

Each module's own docstring cites the specific dissertation section, theorem,
or equation it implements.

## Requirements

- Python 3
- NumPy
- SciPy (`scipy.special.gammaln`, `scipy.special.logsumexp`)

No other third-party dependencies.

## Citation

If you use this code, please cite the dissertation:

```
Himalaya Choudhary. "Bayesian Learning of Higher-Order Networks from Path
Data." MSc Dissertation, Mathematical Modelling and Scientific Computing,
University of Oxford, 2026.
```

## License

MIT — see `LICENSE`.
