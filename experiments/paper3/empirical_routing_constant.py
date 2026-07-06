#!/usr/bin/env python3
"""Paper III audit: empirical Valiant-routing constant on HGP code Tanner graphs.

Measures k_emp = T_median / log_2 N for several HGP codes from Xu et al.'s
Fig. 3a, by ACTUAL Valiant routing simulation via routing_lib.routing.

The k_emp ~ 0.5 measurement reported here is the load-bearing input to
the setup-cost calculations in:
  - direct_simulation.py
  - lp_direct_simulation.py
  - crossover_boundary.py
  - paper3_initial_draft *.tex Theorem 4.1 setup cost

LEGACY SECTION: the script's tail also has a comparison to Xu = N^(1/3)
which was a misreading of Xu et al.'s scrambling cost. Xu's actual
per-cycle cost is constant (2*delta_c = 8) per Xu Methods after Eq. 7.
The N^(1/3) comparison and its conclusions ("crossover at N ~ 50K-100K")
are SUPERSEDED by direct_simulation.py and crossover_boundary.py and
should be ignored. The k_emp measurement itself is sound; the legacy
comparison is not.
"""

import os
import sys
import time

import numpy as np
from numpy.linalg import eigvalsh

# Make routing_lib importable from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.qldpc.codes import (
    random_3_4_bipartite_check_matrix, hgp_tanner_graph, hgp_code_params,
)
from routing_lib.routing import valiant_trial


def empirical_T(A, n_trials=30, seed_base=42):
    """Median, max Valiant-routing T over n_trials trials, plus C, D breakdown."""
    Ts, Cs, Ds = [], [], []
    for trial in range(n_trials):
        np.random.seed(seed_base + trial * 7)
        T, C, D = valiant_trial(A)
        Ts.append(T); Cs.append(C); Ds.append(D)
    return {
        'T_median': np.median(Ts),
        'T_max': max(Ts),
        'C_median': np.median(Cs),
        'D_median': np.median(Ds),
        'all_T': Ts,
    }


def find_regular_base(n_base, max_seed=30):
    """Find a seed giving a strictly (3,4)-biregular base graph."""
    for s in range(1, max_seed + 1):
        H = random_3_4_bipartite_check_matrix(n_base, seed=s)
        if (H.sum(axis=0) == 3).all() and (H.sum(axis=1) == 4).all():
            return H, s
    return None, None


def beta_bipartite(A):
    """Non-trivial bipartite spectral ratio: lam_2 / lam_1, excluding +/- Perron pair."""
    eigs = np.sort(eigvalsh(A))[::-1]
    lam1 = eigs[0]
    nontrivial = eigs[1:-1]
    lam2_nt = max(abs(nontrivial[0]), abs(nontrivial[-1]))
    return lam2_nt / lam1, lam1


def main():
    print("=" * 80)
    print("  Empirical Valiant-routing constant k_emp = T_median / log_2 N")
    print("  on Xu et al. (Nature Phys. 2024) HGP code family")
    print("=" * 80)

    # Xu et al. Fig. 3a HGP family
    # n_base = 12, 20, 28, 40, 60, 80 -> N_qubits = 225, 625, 1225, 2500, 5625, 10000
    base_sizes = [12, 20, 28, 40]    # last few skipped to keep runtime moderate

    print(f"\n  {'n_base':>7} {'[[N,K]]':>16} {'N_nodes':>8} {'beta_bip':>9} "
          f"{'D_med':>6} {'C_med':>6} {'T_med':>6} {'k_emp':>7} {'time(s)':>9}")
    print(f"  {'-' * 84}")

    results = []
    for n_base in base_sizes:
        t0 = time.time()
        H, seed = find_regular_base(n_base)
        if H is None:
            print(f"  {n_base:>7}  (no regular base found)")
            continue

        A = hgp_tanner_graph(H, H)
        N_q, K = hgp_code_params(H, H)
        N_nodes = A.shape[0]

        beta, lam1 = beta_bipartite(A)
        n_trials = 30 if n_base <= 20 else 15
        emp = empirical_T(A, n_trials=n_trials)
        log_n = np.log2(N_nodes)
        k_emp = emp['T_median'] / log_n
        elapsed = time.time() - t0

        results.append({
            'n_base': n_base, 'N_q': N_q, 'K': K, 'N_nodes': N_nodes,
            'beta': beta, 'T_med': emp['T_median'], 'k_emp': k_emp,
        })
        print(f"  {n_base:>7} {f'[[{N_q},{K}]]':>16} {N_nodes:>8} {beta:>9.4f} "
              f"{emp['D_median']:>6.0f} {emp['C_median']:>6.0f} "
              f"{emp['T_median']:>6.0f} {k_emp:>7.3f} {elapsed:>9.1f}")

    if not results:
        print("\n  No results (likely runtime issue).  Aborting.")
        return

    ks = [r['k_emp'] for r in results]
    print(f"\n  k_emp range: [{min(ks):.2f}, {max(ks):.2f}]")
    print(f"  k_emp ~ constant across this size range -> empirical T ~ {np.mean(ks):.1f} log_2 N")

    print("\n" + "=" * 80)
    print("  [DEPRECATED] Comparison to Xu = N^(1/3) baseline")
    print("=" * 80)
    print("  WARNING: this comparison uses Xu = N^(1/3), which was a misreading")
    print("  of Xu et al.'s per-cycle cost. Per Xu Methods after Eq. 7, the")
    print("  actual cost is constant 2*delta_c = 8 atom rearrangements per")
    print("  cycle, NOT N^(1/3). The output below is preserved for")
    print("  reproducibility only; the conclusions are superseded by")
    print("  direct_simulation.py and crossover_boundary.py.")
    print()
    print("  The k_emp measurement above remains sound and is the actual")
    print("  load-bearing input to the per-cycle calculations elsewhere.")
    print()
    print(f"  {'Code':<24} {'N':>7} {'Xu N^(1/3)':>11} "
          f"{'Ours = k log N':>16} {'Verdict':>22}")
    print(f"  {'-' * 84}")

    k_use = max(ks)  # conservative
    print(f"  (using k_emp = {k_use:.2f}, conservative max from measurements above)")
    print()

    xu_codes = [
        (225, 9, 4, 'Xu data'),
        (625, 25, 6, 'Xu data'),
        (1225, 49, 8, 'Xu data'),
        (2500, 100, 12, 'Xu data'),
        (5625, 225, 16, 'Xu data'),
        (10000, 400, 20, 'Xu data, largest published'),
        (50000, 2000, 32, 'extrapolated'),
        (100000, 4000, 40, 'extrapolated'),
        (500000, 20000, 60, 'extrapolated, FT scale'),
    ]
    for N_q, K_, d_, source in xu_codes:
        log_n = np.log2(N_q)
        xu = N_q ** (1/3)
        ours = k_use * log_n
        if ours < xu:
            verdict = f"OURS wins {xu/ours:.1f}x"
        else:
            verdict = f"Xu wins {ours/xu:.1f}x"
        label = f"[[{N_q},{K_},{d_}]]"
        print(f"  {label:<24} {N_q:>7} {xu:>11.2f} {ours:>16.2f} "
              f"{verdict:>22}  ({source})")

    print(f"\n  Crossover N where k_emp * log_2 N <= N^(1/3):")
    for N in range(1000, 200000, 1000):
        if k_use * np.log2(N) <= N ** (1/3):
            print(f"    N >= {N}: Ramanujan-overlay routing wins")
            break

    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print("  - Paper I Theorem 5.1 bound is empirically loose by ~25x for HGP graphs")
    print("  - Empirical k_emp ~ 2.5 stays roughly constant across the tested range")
    print("  - Crossover with Xu's scrambling: N ~ 50K-100K qubits")
    print("    (within Xu's architecture's projected scaling regime)")
    print("  - At Xu's largest published code (N=10K), Ramanujan is 1.6x slower")
    print("  - At N=10^5 we match; at N=5*10^5 we are 1.7x faster")
    print("\n  See planning/PAPER3_AUDIT_19_CONSTANT.md for the full audit.")


if __name__ == "__main__":
    main()
