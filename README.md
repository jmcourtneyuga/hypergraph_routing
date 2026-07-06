# Code: Validation Scripts

Validation scripts accompanying:

- *Permutation Routing on Ramanujan Hypergraphs with Applications to Neutral
  Atom Quantum Architectures* (Paper I)
- *Block Permutation Routing on Ramanujan Hypergraphs for Fault-Tolerant
  Quantum Architectures* (Paper II)
- *Using Tanner Spectral Reduction to Improve Multi-Layer Optical Lattice Routing for 
   Hypergraph-Product and Bivariate Bicycle qLDPC Codes* (Paper III)

Each experiment script maps to specific theorems or claims in the manuscripts.

This README and accompanying documentation was constructed from the code base using Claude Opus.
Comments were inserted into the scripts using Claude Opus, impacting no functionality. 

The author claims ownership of the scripts and their functionality.

## Layout

```
code/
├── routing_lib/                    Shared primitives (graphs, spectral, routing,
│                                   blocks, greedy). Imported by every experiment.
├── experiments/
│   ├── paper1/                     Paper I scripts (10 files)
│   └── paper2/                     Paper II scripts (4 files)
├── tests/                          Pytest suite (4 files)
├── pyproject.toml                  Package metadata + pytest config
├── requirements.txt                Pinned dependency versions
```

`routing_lib/` is the package that consolidates primitives previously
duplicated across the standalone scripts:

| Submodule | Contents |
|-----------|----------|
| `routing_lib.graphs` | random regular / configuration-model / matching-union / networkx-based / best-of variants; 2D grid clique expansions (open + torus); 3D AOL; Margulis; Cayley(Z_n²); voltage coverings of Fano plane and PG(2,3); circulant cliques; affine/GL(2) helpers. |
| `routing_lib.spectral` | Three β conventions kept distinct (`spectral_params_lam2_over_d`, `spectral_params_lambda_star`, `spectral_params_abs_eigs`, `spectral_params_covering_tower`, `spectral_params_minimal`); Ramanujan bounds; theoretical routing bound; sparse spectrum / diameter helpers. |
| `routing_lib.routing` | BFS variants; Valiant routing variants (one per call site); König edge-coloring; block routing pipeline (`block_valiant_route`, `block_lmr_schedule`, `simulate_block_routing`); derandomization via method of conditional expectations. |
| `routing_lib.blocks` | `Block`, `BlockConfig`, two block-placement variants (`random_block_config`, `place_blocks_bfs`), two quotient-graph variants (`quotient_graph` raises on disconnection; `build_quotient_graph` returns None). |
| `routing_lib.greedy` | Open vs torus grid distance, displacement energy, greedy/random matchings (Paper I §9). |

Several primitives appear in multiple variants under distinct names because
of deviations on β normalization, eigvalsh source library,
torus-vs-open grid distance, and odd-degree handling. Each function's
docstring names which scripts originally used it.

## Active scripts

### Paper II — Block Routing on Ramanujan Hypergraphs

| Script | Tests |
|--------|-------|
| `experiments/paper2/block_routing.py` | Theorem 1.1 ($\rt_B = \Theta(d_C \log N_L)$), Theorem 4.8 ($\beta_Q < 1$ inheritance), Lemma 5.2 (block Valiant), Lemma 5.4 (block LMR), Lemma 5.7 (serialization). Experiments F2.1–F5.1. |
| `experiments/paper2/regime_condition_table.py` | Recompute Remark 4.6 (regime practicality) under three intra-block-degree bounds. |
| `experiments/paper2/regime_validation_sweep.py` | Issue-9 remediation: rerun $d_C=7$ at $d'=200$ and $d'$ sweep at $(d_C, N_L) = (7, 32)$. |

### Paper I — Permutation Routing on Ramanujan Hypergraphs

| Script | Paper section | Tests |
|--------|---------------|-------|
| `experiments/paper1/derandomization.py` | §4 | Theorem 4.1 (constructive routing in $O(N^2 d')$ time) |
| `experiments/paper1/multilayer_aol.py` | §5 | Theorem 5.1 (overlay theorem), Lemma 5.2 (multi-layer spectral gain), Proposition 5.4 (crosstalk) |
| `experiments/paper1/sparse_overlay_routing.py` | §5.3, §5.4 | Theorem 5.3 (capacity independence), Theorem 5.4 (sparse overlay with partial matchings) |
| `experiments/paper1/aol_constrained_routing.py` | §5 | Capacity-depth tradeoff; spectral scaling of grid vs overlay |
| `experiments/paper1/aom_3d_routing.py` | §3.5, §11 | Model A (2D AOD) and Model B (3D AOL) hypergraph constructions |
| `experiments/paper1/algebraic_overlay.py` | §6 | Theorem 6.1 (abelian Alon–Boppana barrier), Theorem 6.2 (affine derandomization) |
| `experiments/paper1/covering_tower.py` | §7 | Theorem 7.1 (covering tower routing); 94% Ramanujan fraction at Fano plane $k=2$ |
| `experiments/paper1/entanglement_routing.py` | §8 | Theorem 8.1 (teleportation routing depth), Corollary 8.3 ($R_{\rm break} \approx 4$) |
| `experiments/paper1/dynamic_adaptive.py` | §9 | Theorem 9.3 (greedy monotonicity), Theorem 9.4 (greedy stall), Theorem 9.5 (MW regret) |
| `experiments/paper1/hierarchical_routing.py` | §10 | Theorem 10.1 (hierarchical routing), Lemma 10.2 (capacity invariance), Theorem 10.3 (boundary-only) |

## Tests

```
pytest tests/ -v
```

| Test file | Validates |
|-----------|-----------|
| `tests/test_v8_standalone.py` | König edge-coloring chromatic index for parallel + pessimistic block translations |
| `tests/test_concentration.py` | Direct test of Assumption ass:concentration: tail $\mathbb{P}[\Delta\Phi_t < 0.5\,\mathbb{E}[\Delta\Phi_t]] \le 0.51$ for $N \in \{64, 100, 144, 196\}$ |
| `tests/test_greedy_stall_n256.py` | Greedy-stall test extended to $N=256$ (paper claim: $\Phi_{\rm stall}/\Phi_0 \to 0.17$ for $N \ge 64$) |

Each test file is dual-mode: `pytest tests/<file>.py` runs assertion-based
tests; `python tests/<file>.py` runs the original print/PASS-style script.

## Installation

For exact reproduction of the published numbers, use the pinned lock file:

```
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

For a flexible install within the supported version ranges:

```
pip install -r requirements.txt
pip install -e .            # makes routing_lib importable from anywhere
```

Editable install is required for the experiment scripts to find
`routing_lib` when invoked as `python experiments/paper1/<script>.py`.

For pytest:

```
pip install -e .[dev]
```

See [REPRODUCING.md](./REPRODUCING.md) for the claim → script → invocation
index that maps each theorem / experiment in the manuscripts to the
specific script that produces its numbers.

## Running

Each experiment script is self-contained and runs as
`python experiments/<paper>/<script>.py`. Output is plain text. Typical
runtimes (Python 3.12, single core):

| Script | Typical runtime |
|--------|----------------|
| `experiments/paper2/regime_condition_table.py` | <1s |
| `tests/test_v8_standalone.py` | ~1s |
| `tests/test_concentration.py` | ~5s |
| `tests/test_greedy_stall_n256.py` | ~3s |
| `experiments/paper1/derandomization.py` | ~30s |
| `experiments/paper1/aom_3d_routing.py` | ~1m |
| `experiments/paper1/hierarchical_routing.py` | ~10s |
| `experiments/paper2/block_routing.py` | ~2m |
| `experiments/paper1/dynamic_adaptive.py` | ~30s |
| `experiments/paper1/multilayer_aol.py` | ~3m |
| `experiments/paper1/entanglement_routing.py` | ~1m |
| `experiments/paper1/sparse_overlay_routing.py` | ~3m |
| `experiments/paper1/aol_constrained_routing.py` | ~5m |
| `experiments/paper1/algebraic_overlay.py` | ~2m |
| `experiments/paper1/covering_tower.py` | ~30s |
| `experiments/paper2/regime_validation_sweep.py` | ~5m |

## Reproducibility

Each script seeds `numpy` with `np.random.seed(42)` at the top of `main()`.
Reported numbers in the papers correspond to this seed. Scripts that do NOT
seed (`aom_3d_routing.py`) report Monte-Carlo medians; sample-by-sample
values may vary across runs.

On Windows, set `PYTHONIOENCODING=utf-8` before running scripts that print
Greek/math symbols (`β`, `Σ`, `Φ`, `─`):

```
$env:PYTHONIOENCODING = "utf-8"     # PowerShell
set PYTHONIOENCODING=utf-8          # cmd.exe
```

## Dependencies

```
numpy >= 1.26
scipy >= 1.11
networkx >= 3.0
pytest >= 7    (for the test suite)
```

See `pyproject.toml` and `requirements.txt` for exact bounds.

## Paper III — Multi-Layer Optical Lattice Routing for HGP/LP and BB qLDPC Codes

Validation scripts accompanying *Multi-Layer Optical Lattice Routing for
Hypergraph-Product and Bivariate Bicycle qLDPC Codes via Tanner Spectral
Reduction* (Paper III).

Paper III reuses the shared `routing_lib` primitives and adds a `routing_lib.qldpc`
subpackage for qLDPC code construction (HGP/LP Tanner graphs and the Bravyi
bivariate-bicycle codes).

### New library submodule: `routing_lib.qldpc`

| Module | Contents |
| --- | --- |
| `routing_lib.qldpc.codes` | Classical (3,4)-biregular check-matrix sampler; classical Tanner graph; **HGP (Tillich–Zémor) Tanner graph** `hgp_tanner_graph`; circulant **LP** check matrices `lp_check_matrix_circulant`, `lp_code`. |
| `routing_lib.qldpc.published_codes` | The four **Bravyi BB codes** (`BRAVYI_BB_CODES`) with builders `bb_check_matrices` / `bb_tanner_graph` / `get_bravyi_bb_tanner`; the Xu LP family (`XU_LP_CODES`, `get_xu_lp_tanner`). |
| `routing_lib.qldpc.xu_scrambling` | Xu et al. Algorithm 3 single-layer-AOD baseline model. |

### Active scripts — `experiments/paper3/`

| Script | Validates / produces |
| --- | --- |
| `lp_tanner_spectrum.py` | Prop. 3.1 spectrum decomposition; Thm. 3.2 β_HGP = (1+β_base)/2 |
| `bb_beta_feasibility_probe.py` | Thm. 3.5 BB Fourier reduction; the "no closed form for β_BB" result |
| `bb_code_comparison.py` | Table `tab:bb-codes` (Bravyi BB β_BB, χ′, per-cycle) |
| `direct_simulation.py` | Tables `tab:head-to-head` (HGP), `tab:asymptotic` per-cycle counts |
| `lp_direct_simulation.py` | Table `tab:head-to-head` (LP rows); constructive König check |
| `empirical_routing_constant.py` | Table `tab:k-emp` small-N block (k_emp ≈ 0.5) |
| `kemp_large_n.py` | Table `tab:k-emp` engineering-N block (to 100,000) |
| `konig_optimal_test.py` | Table `tab:konig-tight` (χ′ = Δ = 8 to N = 122,500) |
| `nonqc_simulation.py` | Table `tab:nonqc` (QC vs non-QC wall-clock) |
| `lp_random_voltage_sweep.py` | Table `tab:ram-frac` (Ramanujan fraction, point estimate) |
| `lp_voltage_bands.py` | Table `tab:ram-frac` cross-seed mean ± std bands |
| `crossover_boundary.py` | Amortization crossover R\* behind `tab:recommendations` |
| `published_codes_comparison.py` | Cross-check on exact published [[N,K,d]] parameters |
| `shifted_cayley_barrier.py` | Future-hardware N^{1/4} motion-barrier sweep (see note in `REPRODUCING.md`) |

### Tests — `tests/`

```
pytest tests/test_paper3_spectral.py tests/test_paper3_konig.py -v
```

| Test file | Validates |
| --- | --- |
| `tests/test_paper3_spectral.py` | Thm. 3.2 (HGP closed form) and Thm. 3.5 (BB Fourier reduction) vs direct diagonalization |
| `tests/test_paper3_konig.py` | Δ(G_T) = 8 (HGP) / 6 (BB) and König-tight explicit edge coloring (χ′ = Δ) |
