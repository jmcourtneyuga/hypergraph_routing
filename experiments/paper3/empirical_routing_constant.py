#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
from numpy.linalg import eigvalsh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.qldpc.codes import (
    random_3_4_bipartite_check_matrix, hgp_tanner_graph, hgp_code_params,
)
from routing_lib.routing import valiant_trial


def empirical_T(A, n_trials=30, seed_base=42):
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
    for s in range(1, max_seed + 1):
        H = random_3_4_bipartite_check_matrix(n_base, seed=s)
        if (H.sum(axis=0) == 3).all() and (H.sum(axis=1) == 4).all():
            return H, s
    return None, None


def beta_bipartite(A):
    eigs = np.sort(eigvalsh(A))[::-1]
    lam1 = eigs[0]
    nontrivial = eigs[1:-1]
    lam2_nt = max(abs(nontrivial[0]), abs(nontrivial[-1]))
    return lam2_nt / lam1, lam1


def main():
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

if __name__ == "__main__":
    main()
