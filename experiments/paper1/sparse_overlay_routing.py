#!/usr/bin/env python3
import numpy as np
from numpy.linalg import eigvalsh
from itertools import combinations
from math import ceil, log2
import time

np.random.seed(42)


from routing_lib.graphs import (
    random_regular_matching_union as random_regular_graph,
    build_2d_grid_clique as build_2d_grid,
)
from routing_lib.spectral import (
    spectral_params_lam2_over_d as spectral_params,
    ramanujan_bound,
)
from routing_lib.routing import (
    bfs_path,
    bfs_distance_matrix,
    compute_edge_congestion,
    valiant_routing_trial_grid as valiant_routing_trial,
)


def main():
    t_start = time.time()

    print("  DIRECTION 6: APPROXIMATE ROUTING AND SPARSE OVERLAYS")

    print("  PART 1: SPARSE EXPANDERS — RAMANUJAN CONDITION")
    print("  Random d-regular graphs with d = O(log N)")

    print(f"\n  {'N':>5} {'d':>4} | {'d_prime':>8} {'lam2':>8} {'beta':>8} "
      f"{'1-beta':>8} | {'Ram bound':>9} {'Ramanujan?':>10}")
    print(f"  {'-' * 75}")

    results_part1 = []
    for N in [64, 100, 144, 256, 400]:
        logN = max(2, int(round(log2(N))))
        degrees = sorted(set([4, logN, 2 * logN, 8, min(N // 4, 32)]))
        degrees = [d if d % 2 == 0 else d + 1 for d in degrees]
        degrees = sorted(set(degrees))

        for d in degrees:
            if d >= N:
                continue
            n_trials = 5
            betas = []
            lam2s = []
            d_primes = []
            for _ in range(n_trials):
                A = random_regular_graph(N, d)
                sp = spectral_params(A)
                betas.append(sp['beta'])
                lam2s.append(sp['lambda_2'])
                d_primes.append(sp['d_prime'])

            beta_mean = np.mean(betas)
            lam2_mean = np.mean(lam2s)
            d_prime_mean = np.mean(d_primes)
            rb = ramanujan_bound(d_prime_mean)
            is_ram = "YES" if lam2_mean <= rb else "no"

            tag = ""
            if d == logN or d == logN + 1:
                tag = " <-- d=log(N)"
            elif d == 2 * logN or d == 2 * logN + 1:
                tag = " <-- d=2log(N)"

            print(f"  {N:>5} {d:>4} | {d_prime_mean:>8.1f} {lam2_mean:>8.3f} "
              f"{beta_mean:>8.4f} {1 - beta_mean:>8.4f} | {rb:>9.3f} "
              f"{'':>2}{is_ram}{tag}")

            results_part1.append({
                'N': N, 'd': d, 'beta': beta_mean, 'lam2': lam2_mean,
                'd_prime': d_prime_mean, 'ramanujan': is_ram == "YES"
            })
        print()

    print("  PART 2: ROUTING DEPTH — SPARSE vs DENSE OVERLAYS")
    print("  Valiant routing on random d-regular graphs")

    n_trials = 20

    for n in [8, 10, 12]:
        N = n * n
        logN = log2(N)
        A_grid = build_2d_grid(n)
        sp_grid = spectral_params(A_grid)

        print(f"\n  --- N = {N} ({n}x{n} grid), log2(N) = {logN:.1f} ---")
        print(f"  {'Model':<30} {'d':>4} {'beta':>7} {'T_med':>6} {'C_med':>6} "
          f"{'D_med':>6} {'T/logN':>7} {'T/log2N':>8}")
        print(f"  {'-' * 80}")

        # Grid baseline
        grid_Ts = []
        grid_Cs = []
        grid_Ds = []
        for _ in range(n_trials):
            T, C, D = valiant_routing_trial(A_grid)
            grid_Ts.append(T); grid_Cs.append(C); grid_Ds.append(D)

        T_med = int(np.median(grid_Ts))
        C_med = int(np.median(grid_Cs))
        D_med = int(np.median(grid_Ds))
        print(f"  {'2D grid (baseline)':<30} {sp_grid['d_prime']:>4.0f} "
          f"{sp_grid['beta']:>7.4f} {T_med:>6} {C_med:>6} {D_med:>6} "
          f"{T_med / logN:>7.2f} {T_med / logN**2:>8.3f}")

        # Sparse overlays: d = 4, log(N), 2*log(N), 8, 16, 32
        logN_int = max(4, int(round(logN)))
        if logN_int % 2 == 1:
            logN_int += 1
        degrees = sorted(set([4, logN_int, 2 * logN_int, 8, 16, 32]))
        degrees = [d if d % 2 == 0 else d + 1 for d in degrees]
        degrees = sorted(set(d for d in degrees if d < N))

        for d in degrees:
            Ts = []; Cs = []; Ds = []
            betas_trial = []
            for _ in range(n_trials):
                A_overlay = random_regular_graph(N, d)
                sp = spectral_params(A_overlay)
                betas_trial.append(sp['beta'])
                T, C, D = valiant_routing_trial(A_overlay)
                Ts.append(T); Cs.append(C); Ds.append(D)

            T_med = int(np.median(Ts))
            C_med = int(np.median(Cs))
            D_med = int(np.median(Ds))
            beta_mean = np.mean(betas_trial)

            tag = ""
            if d == logN_int:
                tag = " [d~logN]"
            elif d == 2 * logN_int:
                tag = " [d~2logN]"

            print(f"  {f'd={d} random overlay':<30} {d:>4} "
              f"{beta_mean:>7.4f} {T_med:>6} {C_med:>6} {D_med:>6} "
              f"{T_med / logN:>7.2f} {T_med / logN**2:>8.3f}{tag}")

        # Grid + sparse overlay hybrid
        print()
        print(f"  Grid + overlay hybrids:")
        print(f"  {'Model':<30} {'d_eff':>4} {'beta':>7} {'T_med':>6} {'C_med':>6} "
          f"{'D_med':>6} {'T/logN':>7} {'T/log2N':>8}")
        print(f"  {'-' * 80}")
        for d_overlay in [4, logN_int, 2 * logN_int]:
            Ts = []; Cs = []; Ds = []
            betas_trial = []
            for _ in range(n_trials):
                A_combo = A_grid + random_regular_graph(N, d_overlay)
                sp = spectral_params(A_combo)
                betas_trial.append(sp['beta'])
                T, C, D = valiant_routing_trial(A_combo)
                Ts.append(T); Cs.append(C); Ds.append(D)

            T_med = int(np.median(Ts))
            C_med = int(np.median(Cs))
            D_med = int(np.median(Ds))
            beta_mean = np.mean(betas_trial)
            d_eff = sp_grid['d_prime'] + d_overlay

            print(f"  {f'grid + d={d_overlay} overlay':<30} {d_eff:>4.0f} "
              f"{beta_mean:>7.4f} {T_med:>6} {C_med:>6} {D_med:>6} "
              f"{T_med / logN:>7.2f} {T_med / logN**2:>8.3f}")


    print("  PART 3: CAPACITY-DEPTH TRADEOFF — SPARSE vs DENSE")
    print("  T_total = T_overlay * ceil(N / (2*k))")


    print(f"  {'N':>6} {'d_overlay':>9} {'beta':>7} {'T_ovrl':>7} {'match_sz':>9} "
      f"{'k_need':>7} {'T_total(k=N/2)':>14} {'T_total(k=N/logN)':>18}")
    print(f"  {'-' * 90}")

    for N in [64, 256, 1024, 4096]:
        logN = log2(N)
        logN_int = max(4, int(round(logN)))
        if logN_int % 2 == 1:
            logN_int += 1

        for d in [logN_int, 8, 16, 32]:
            if d >= N:
                continue
            # Predicted beta from Friedman: lambda_2 ~ 2*sqrt(d-1), so beta ~ 2*sqrt(d-1)/d
            beta_pred = 2 * np.sqrt(d - 1) / d
            T_overlay = logN / (1 - beta_pred) if beta_pred < 1 else float('inf')

            # Matching size is always N/2 for a perfect matching on any graph
            match_size = N // 2

            # With capacity k
            k_full = N // 2
            k_sparse = max(1, int(N / logN))

            substeps_full = max(1, ceil(match_size / k_full))
            substeps_sparse = max(1, ceil(match_size / k_sparse))

            T_total_full = T_overlay * substeps_full
            T_total_sparse = T_overlay * substeps_sparse

            tag = " <--" if d == logN_int else ""
            print(f"  {N:>6} {d:>9} {beta_pred:>7.4f} {T_overlay:>7.1f} "
              f"{match_size:>9} {match_size:>7} "
              f"{T_total_full:>14.1f} {T_total_sparse:>18.1f}{tag}")
        print()

    # Simulation: Compare routing with limited matching size
    print(f"  Simulated routing with partial matchings (N=100):\n")
    N = 100
    n = 10
    logN = log2(N)
    n_trials = 15

    print(f"  {'Overlay':>15} {'d':>4} {'beta':>7} | {'k=N/2':>6} {'k=N/4':>6} "
      f"{'k=N/8':>6} {'k=N/logN':>8} | {'D_med':>6}")
    print(f"  {'-' * 75}")

    for d in [4, 8, 12, 16, 32]:
        trial_data = {k_frac: [] for k_frac in ['N/2', 'N/4', 'N/8', 'N/logN']}
        D_meds = []
        betas = []

        for trial in range(n_trials):
            A = random_regular_graph(N, d)
            sp = spectral_params(A)
            betas.append(sp['beta'])

            pi = np.random.permutation(N)
            sigma = np.random.permutation(N)
            adj = [np.where(A[i] > 0)[0].tolist() for i in range(N)]

            # Compute all paths for Valiant routing
            scatter_lengths = []
            gather_lengths = []
            for v in range(N):
                p1 = bfs_path(adj, v, sigma[v])
                p2 = bfs_path(adj, sigma[v], pi[v])
                scatter_lengths.append(len(p1) - 1)
                gather_lengths.append(len(p2) - 1)

            D = max(max(scatter_lengths), max(gather_lengths))
            D_meds.append(D)

            # Total demand = sum of all path lengths (both phases)
            total_demand = sum(scatter_lengths) + sum(gather_lengths)

            # With capacity k: T = ceil(total_demand / k) + D
            for k_label, k_val in [('N/2', N // 2), ('N/4', N // 4),
                                    ('N/8', N // 8), ('N/logN', max(1, int(N / logN)))]:
                # Each step we can handle k atoms moving one hop
                # Total hops = total_demand, spread over k per step
                C_eff = ceil(total_demand / max(k_val, 1))
                T = C_eff + D
                trial_data[k_label].append(T)

        beta_mean = np.mean(betas)
        D_med = int(np.median(D_meds))
        row = f"  {f'd={d} overlay':>15} {d:>4} {beta_mean:>7.4f} |"
        for k_label in ['N/2', 'N/4', 'N/8', 'N/logN']:
            T_med = int(np.median(trial_data[k_label]))
            row += f" {T_med:>6}"
        row += f"  | {D_med:>6}"
        print(row)

    print("\n\n" + "=" * 90)
    print("  PART 5: CROSSOVER — SPARSE vs DENSE OVERLAY")
    print("  At what capacity k does sparse become preferable?")
    print("=" * 90)

    print(f"\n  Comparing d=8 (dense) vs d=logN (sparse) overlays")
    print(f"  T_total = ceil(sum_path_lengths / k) + D\n")

    for n in [8, 10, 12, 16]:
        N = n * n
        logN = log2(N)
        d_sparse = max(4, int(round(logN)))
        if d_sparse % 2 == 1:
            d_sparse += 1
        d_dense = max(8, d_sparse)  # ensure dense >= sparse
        if d_dense <= d_sparse:
            d_dense = d_sparse + 4
        if d_dense % 2 == 1:
            d_dense += 1

        print(f"  --- N = {N}, d_sparse = {d_sparse}, d_dense = {d_dense} ---")
        print(f"  {'k':>8} | {'T_sparse':>9} {'T_dense':>9} {'winner':>8} {'ratio':>7}")
        print(f"  {'-' * 50}")

        # Average over a few trials
        n_trials_xover = 10
        for k_frac_name, k_frac in [('N/2', 0.5), ('N/4', 0.25), ('N/8', 0.125),
                                      ('N/logN', 1.0 / logN), ('N/log2N', 1.0 / logN**2)]:
            k = max(1, int(N * k_frac))

            T_sparse_trials = []
            T_dense_trials = []

            for _ in range(n_trials_xover):
                pi = np.random.permutation(N)
                sigma = np.random.permutation(N)

                # Sparse overlay
                A_s = random_regular_graph(N, d_sparse)
                adj_s = [np.where(A_s[i] > 0)[0].tolist() for i in range(N)]
                demand_s = 0; D_s = 0
                for v in range(N):
                    p1 = bfs_path(adj_s, v, sigma[v])
                    p2 = bfs_path(adj_s, sigma[v], pi[v])
                    demand_s += (len(p1) - 1) + (len(p2) - 1)
                    D_s = max(D_s, len(p1) - 1, len(p2) - 1)
                T_s = ceil(demand_s / max(k, 1)) + D_s
                T_sparse_trials.append(T_s)

                # Dense overlay
                A_d = random_regular_graph(N, d_dense)
                adj_d = [np.where(A_d[i] > 0)[0].tolist() for i in range(N)]
                demand_d = 0; D_d = 0
                for v in range(N):
                    p1 = bfs_path(adj_d, v, sigma[v])
                    p2 = bfs_path(adj_d, sigma[v], pi[v])
                    demand_d += (len(p1) - 1) + (len(p2) - 1)
                    D_d = max(D_d, len(p1) - 1, len(p2) - 1)
                T_d = ceil(demand_d / max(k, 1)) + D_d
                T_dense_trials.append(T_d)

            T_s_med = int(np.median(T_sparse_trials))
            T_d_med = int(np.median(T_dense_trials))
            winner = "sparse" if T_s_med <= T_d_med else "dense"
            ratio = T_s_med / max(T_d_med, 1)

            print(f"  {k_frac_name:>8} | {T_s_med:>9} {T_d_med:>9} {winner:>8} {ratio:>7.2f}")
        print()

    t_end = time.time()
    print(f"  Total computation time: {t_end - t_start:.1f}s")


if __name__ == "__main__":
    main()
