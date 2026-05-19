# Reproducing the validation results

This document maps each numerical / experimental claim in
(*Permutation Routing on Ramanujan Hypergraphs*) and (*Block
Permutation Routing on Ramanujan Hypergraphs for Fault-Tolerant Quantum
Architectures*) back to the script in this repository that produces it.

> The README has the high-level layout, per-script runtimes, and dependency
> notes. This document is to aid data reproducibility

## Quickstart

```bash
# Reproducible environment (exact versions used by the authors)
python -m venv .venv
. .venv/Scripts/activate                   # Windows PowerShell
# or: source .venv/bin/activate            # POSIX
pip install -r requirements-lock.txt
pip install -e . --no-deps                 # makes routing_lib importable

# Sanity check — the test suite (~10 s)
pytest tests/

# Run the fast experiments first (each <1 min)
python experiments/paper2/regime_condition_table.py
python experiments/paper1/derandomization.py
python experiments/paper1/hierarchical_routing.py
python experiments/paper1/covering_tower.py
```

On Windows, set `PYTHONIOENCODING=utf-8` before running scripts that print
Greek/math symbols.

## Determinism

Every active script seeds `numpy.random` with `np.random.seed(42)` at the
start of `main()`. Reported values in the papers correspond to seed 42 on
NumPy / SciPy / NetworkX versions pinned in `requirements-lock.txt`.

The exception is `experiments/paper1/aom_3d_routing.py`, which reports
Monte-Carlo medians without seeding and is intended to be read as a
distribution, not a point estimate.

## Paper II — Block Routing on Ramanujan Hypergraphs

| Claim                                                            | Script                                          | Approx. runtime |
| ---                                                              | ---                                             | --- |
| Theorem 1.1: $\mathrm{rt}_B = \Theta(d_C \log N_L)$              | `experiments/paper2/block_routing.py`           | ~2 min |
| Theorem 4.8: $\beta_Q < 1$ inheritance on the quotient graph     | `experiments/paper2/block_routing.py`           | included above |
| Lemma 5.2 (block Valiant) + Lemma 5.4 (block LMR) + Lemma 5.7 (serialization) | `experiments/paper2/block_routing.py` | included above |
| Lemma 5.5 / 5.5b: König chromatic-index bound on block-translation bipartite multigraph | `tests/test_v8_standalone.py` | ~1 s |
| Experiments F2.1 / F2.2 / F3.1 / F5.1                            | `experiments/paper2/block_routing.py`           | included above |
| Remark 4.6: regime practicality under three intra-block-degree bounds | `experiments/paper2/regime_condition_table.py` | <1 s |
| Issue-9 remediation: rerun $d_C = 7$ rows at $d' = 200$ and $d'$ sweep at $(d_C, N_L) = (7, 32)$ | `experiments/paper2/regime_validation_sweep.py` | ~5 min |

## Paper I — Permutation Routing on Ramanujan Hypergraphs

| Section | Claim                                                                                              | Script                                            | Approx. runtime |
| ---     | ---                                                                                                | ---                                               | --- |
| §3.5, §11 | Model A (2D AOD) and Model B (3D AOL) hypergraph constructions: spectra, diameter, routing bounds | `experiments/paper1/aom_3d_routing.py`           | ~1 min |
| §4      | Theorem 4.1: constructive derandomization to $O(\log N)$ congestion via conditional expectations  | `experiments/paper1/derandomization.py`          | ~30 s |
| §5      | Theorem 5.1 (overlay routing depth $O(\log N / (1-\beta))$), Lemma 5.2 ($\beta_{\rm union}$), Prop 5.4 (crosstalk) | `experiments/paper1/multilayer_aol.py`     | ~3 min |
| §5      | Capacity-depth tradeoff for 2D / 3D / overlay (selective-transfer capacity $k$ -> depth)          | `experiments/paper1/aol_constrained_routing.py`  | ~5 min |
| §5.3, §5.4 | Theorem 5.3 (capacity independence), Theorem 5.4 (sparse overlay with partial matchings)       | `experiments/paper1/sparse_overlay_routing.py`   | ~3 min |
| §6      | Theorem 6.1 (abelian Alon–Boppana barrier), Theorem 6.2 (affine derandomization, 15–30% reduction) | `experiments/paper1/algebraic_overlay.py`        | ~2 min |
| §7      | Theorem 7.1: covering-tower routing $\mathrm{rt}(H_L) = O(L \log k + \log N_0)$; 94% Ramanujan at Fano $k=2$ | `experiments/paper1/covering_tower.py`     | ~30 s |
| §8      | Theorem 8.1 (teleportation depth), Theorem 8.2 (Bell-pair distribution), Cor 8.3 ($R_{\rm break} \approx 4$) | `experiments/paper1/entanglement_routing.py` | ~1 min |
| §9      | Theorem 9.3 (greedy monotonicity), Theorem 9.4 (greedy stall), Theorem 9.5 (MW regret); $\Phi_{\rm stall}/\Phi_0 \to 0.17$ | `experiments/paper1/dynamic_adaptive.py` | ~30 s |
| §9 (Assumption ass:concentration) | Tail $\mathbb{P}[\Delta\Phi_t < 0.5\,\mathbb{E}[\Delta\Phi_t]] \le 0.51$ for $N \in \{64, 100, 144, 196\}$ | `tests/test_concentration.py`   | ~5 s |
| §9 (Theorem 9.4 extended) | Greedy-stall claim extended to $N = 256$                                                | `tests/test_greedy_stall_n256.py`                 | ~3 s |
| §10     | Theorem 10.1 (hierarchical $T = O(\log^2 N / \log b)$), Lemma 10.2 (capacity invariance), Theorem 10.3 (boundary-only) | `experiments/paper1/hierarchical_routing.py` | ~10 s |

## Where the values live

- **Console output.** Every script prints its key numbers; no JSON output
  files are generated. Reviewers can diff console output against the values
  cited in the manuscripts.
- **Tests have assertions.** The three pytest modules (`test_concentration`,
  `test_greedy_stall_n256`, `test_v8_standalone`) make the paper claims
  into pass/fail assertions. CI runs these on every push.
- **Tests are dual-mode.** Run `pytest tests/<file>.py` for assertion-style
  output, or `python tests/<file>.py` for the original print-and-PASS style.

## Re-running a single paper section

```bash
# Paper II, all-in-one
python experiments/paper2/block_routing.py
python experiments/paper2/regime_condition_table.py
python experiments/paper2/regime_validation_sweep.py
pytest tests/test_v8_standalone.py

# Paper I §9 (greedy / MW)
python experiments/paper1/dynamic_adaptive.py
pytest tests/test_concentration.py tests/test_greedy_stall_n256.py
```

## Compute envelope

A complete reproduction of every experiment at the full parameter grid runs
in approximately 25 minutes wall-clock on a single modern x86 core.
(per the per-script runtimes in the table above; most scripts are
single-threaded). Peak memory remains under 2 GB; the matching-union
graph constructors at $N \ge 30{,}000$ should use `sparse=True` (the API
exposes it).

These scripts are not optimized for memory or time, being apparent as they are written in Python.
They are designed to validate single claims at a time rather than cumulatively 
and can be optimized for computational resources significantly.
We invite the reader to optimize these scripts prior to running to ensure 
reasonable runtimes in the absence of computational clusters.

The results as presented in the manuscripts were calculated on a standard
Linux Mint 22.3 - Cinnamon 64-bit OS with an Intel Core i7-8665U CPU @ 1.90GHz x 4.