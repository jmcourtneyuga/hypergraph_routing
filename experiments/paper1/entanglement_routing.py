#!/usr/bin/env python3

import numpy as np
from numpy.linalg import eigvalsh
from math import ceil, log2, sqrt
import time

np.random.seed(42)

from routing_lib.graphs import random_regular_matching_union as random_regular_graph
from routing_lib.spectral import spectral_params_minimal as spectral_params
from routing_lib.routing import bfs_path, valiant_routing_depth
from routing_lib.greedy import grid_distance_torus as grid_distance

def main():
    t_start = time.time()

    print("  DIRECTION 4: ENTANGLEMENT-ASSISTED ROUTING")

    print("  PART 1: TELEPORTATION ROUTING DEPTH (Theorem 4.1)")
    print("  Once Bell pairs pre-shared, routing = matching on G_ent")

    print(f"\n  Routing depth on random Ramanujan overlay (teleportation model):")
    print(f"  No physical transport during routing — just LOCC.\n")

    print(f"  {'N':>5} {'d_ent':>5} | {'beta':>7} {'T_route':>8} {'T/logN':>7} "
      f"| {'Bell pairs':>11} {'per round':>10}")
    print(f"  {'-' * 65}")

    for n in [8, 10, 12, 16]:
        N = n * n
        logN = log2(N)
        for d_ent in [8, 16, 32]:
            if d_ent >= N:
                continue
            A_ent = random_regular_graph(N, d_ent)
            sp = spectral_params(A_ent)
            T_route = valiant_routing_depth(A_ent, n_trials=15)

            # Bell pair budget
            total_bell = d_ent * N // 2
            # Each routing round uses T matchings, each consuming N/2 Bell pairs
            bell_per_round = T_route * N // 2
            R_rounds = total_bell / max(bell_per_round, 1)

            print(f"  {N:>5} {d_ent:>5} | {sp['beta']:>7.4f} {T_route:>8} "
              f"{T_route / logN:>7.2f} | {total_bell:>11} {bell_per_round:>10}")
        print()


    print("  PART 2: ENTANGLEMENT DISTRIBUTION COST (Theorem 4.2)")
    print("  Physical transport to create Bell pairs along overlay edges")

    print(f"\n  For each edge (u,v) of G_ent embedded on n x n grid:")
    print(f"  Physical cost = grid distance d_grid(u,v)")
    print(f"  Total distribution = sum over all edges / (parallelism k)\n")

    print(f"  {'N':>5} {'d_ent':>5} | {'|E|':>8} {'avg dist':>9} {'max dist':>9} "
      f"{'total dist':>11} | {'T_dist(k=N)':>12} {'T_dist(k=N/logN)':>17}")
    print(f"  {'-' * 90}")

    dist_data = []
    for n in [8, 10, 12, 16, 20, 24, 32]:
        N = n * n
        logN = log2(N)
        for d_ent in [8, 16, 32]:
            if d_ent >= N:
                continue
            A_ent = random_regular_graph(N, d_ent)

            # Compute grid distances for all overlay edges
            edge_dists = []
            for u in range(N):
                for v in range(u + 1, N):
                    if A_ent[u, v] > 0:
                        d_grid = grid_distance(u, v, n)
                        edge_dists.extend([d_grid] * int(A_ent[u, v]))

            n_edges = len(edge_dists)
            avg_dist = np.mean(edge_dists)
            max_dist = np.max(edge_dists)
            total_dist = np.sum(edge_dists)

            # Distribution cost with parallelism k
            k_full = N
            k_sparse = max(1, int(N / logN))

            total_work = 2 * total_dist  # factor 2 for round-trip
            T_dist_full = ceil(total_work / k_full)
            T_dist_sparse = ceil(total_work / k_sparse)

            print(f"  {N:>5} {d_ent:>5} | {n_edges:>8} {avg_dist:>9.1f} "
              f"{max_dist:>9} {total_dist:>11} | {T_dist_full:>12} "
              f"{T_dist_sparse:>17}")

            dist_data.append({
                'N': N, 'n': n, 'd_ent': d_ent, 'n_edges': n_edges,
                'avg_dist': avg_dist, 'total_dist': total_dist,
                'T_dist_full': T_dist_full, 'T_dist_sparse': T_dist_sparse
            })
        print()

    # Verify scaling: avg_dist should be O(n) = O(sqrt(N)) for random overlay
    print(f"\n  Scaling verification: avg grid distance for d_ent=8 overlay edges")
    print(f"  {'N':>5} {'n':>4} {'avg_dist':>9} {'n/3':>6} {'ratio':>7}")
    print(f"  {'-' * 40}")
    for r in dist_data:
        if r['d_ent'] == 8:
            n = r['n']
            print(f"  {r['N']:>5} {n:>4} {r['avg_dist']:>9.2f} {n/3:>6.2f} "
              f"{r['avg_dist'] / (n/3):>7.3f}")



    print("  PART 3: AMORTIZED COST PER ROUTING ROUND (Corollary 4.3)")
    print("  T_amort = T_route + T_dist / R")

    print(f"\n  T_route = O(log N / (1-beta)), T_dist = O(d_ent * sqrt(N))")
    print(f"  R = d_ent / T_route (rounds per distribution cycle)\n")

    print(f"  {'N':>5} {'d_ent':>5} | {'T_route':>8} {'T_dist':>8} "
      f"{'R':>6} {'T_amort':>8} | {'T_phys':>7} {'speedup':>8} "
      f"{'R_break':>8}")
    print(f"  {'-' * 80}")

    for n in [8, 10, 12, 16, 20, 32]:
        N = n * n
        logN = log2(N)
        sqrtN = sqrt(N)

        for d_ent in [8, 16, 32, 64]:
            if d_ent >= N:
                continue

            beta = 2 * sqrt(d_ent - 1) / d_ent  # Friedman prediction
            T_route = max(3, int(round(0.35 * logN / (1 - beta))))

            avg_grid_dist = n / 3  # expected distance on torus
            T_dist = int(ceil(2 * d_ent * N / 2 * avg_grid_dist / N))
            R = d_ent / max(T_route, 1)

            T_amort = T_route + T_dist / max(R, 0.01)
            T_phys = int(round(1.5 * sqrtN))

            speedup = T_phys / T_amort if T_amort > 0 else float('inf')
            if T_phys > T_route:
                R_break = T_dist / (T_phys - T_route)
            else:
                R_break = 0  # teleportation always wins

            print(f"  {N:>5} {d_ent:>5} | {T_route:>8} {T_dist:>8} "
              f"{R:>6.1f} {T_amort:>8.1f} | {T_phys:>7} "
              f"{speedup:>8.2f}x {R_break:>8.1f}")
        print()

    print("\n" + "=" * 90)
    print("  PART 4: OPTIMAL OVERLAY DEGREE d_ent")
    print("  Balancing routing depth (wants large d) vs distribution cost (wants small d)")
    print("=" * 90)

    print(f"\n  For N = 1024 (32x32 grid):")
    N = 1024
    n = 32
    logN = log2(N)
    sqrtN = sqrt(N)

    print(f"  sqrt(N) = {sqrtN:.0f}, log2(N) = {logN:.0f}")
    print(f"\n  {'d_ent':>5} | {'beta':>7} {'T_route':>8} {'T_dist':>8} "
      f"{'R_max':>6} {'T_amort(R=1)':>13} {'T_amort(R=Rmax)':>16} "
      f"{'R_break':>8}")
    print(f"  {'-' * 85}")

    for d_ent in [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
        if d_ent >= N:
            continue
        beta = 2 * sqrt(d_ent - 1) / d_ent
        T_route = max(3, int(round(0.35 * logN / (1 - beta))))

        avg_dist = n / 3
        T_dist = int(ceil(d_ent * avg_dist))  # simplified: d_ent * sqrt(N)/3

        R_max = d_ent / max(T_route, 1)

        T_amort_R1 = T_route + T_dist  # single round
        T_amort_Rmax = T_route + T_dist / max(R_max, 0.01)

        T_phys = int(round(1.5 * sqrtN))
        R_break = T_dist / max(T_phys - T_route, 0.01) if T_phys > T_route else 0

        print(f"  {d_ent:>5} | {beta:>7.4f} {T_route:>8} {T_dist:>8} "
          f"{R_max:>6.1f} {T_amort_R1:>13} {T_amort_Rmax:>16.1f} "
          f"{R_break:>8.1f}")

    # Larger N analysis
    print(f"\n\n  Crossover analysis: R_break = T_dist / (T_phys - T_route)")
    print(f"  At what R does teleportation beat physical transport?\n")

    print(f"  {'N':>6} {'d_ent':>5} | {'T_route':>8} {'T_phys':>7} "
      f"{'T_dist':>8} {'R_break':>8} {'sqrtN/logN':>10}")
    print(f"  {'-' * 65}")

    for N in [64, 256, 1024, 4096, 10000, 40000]:
        n = int(round(sqrt(N)))
        N = n * n
        logN = log2(N)
        sqrtN = sqrt(N)

        d_ent = 16  # fixed moderate degree
        beta = 2 * sqrt(d_ent - 1) / d_ent
        T_route = max(3, int(round(0.35 * logN / (1 - beta))))
        T_dist = int(ceil(d_ent * n / 3))
        T_phys = int(round(1.5 * sqrtN))
        R_break = T_dist / max(T_phys - T_route, 0.01) if T_phys > T_route else 0

        print(f"  {N:>6} {d_ent:>5} | {T_route:>8} {T_phys:>7} "
          f"{T_dist:>8} {R_break:>8.1f} {sqrtN / logN:>10.1f}")

    print("  PART 5: HYBRID PROTOCOL")
    print("  Teleport long-range, physically route short-range")

    for n in [10, 16, 20, 32]:
        N = n * n
        dists = []
        n_samples = 50
        for _ in range(n_samples):
            pi = np.random.permutation(N)
            for v in range(N):
                dists.append(grid_distance(v, pi[v], n))

        dists = np.array(dists)
        mean_d = np.mean(dists)
        median_d = np.median(dists)

        print(f"  N={N:>5} (n={n:>2}): mean={mean_d:.1f}, median={median_d:.0f}, "
          f"max={np.max(dists)}")

        thresholds = [n // 4, n // 2, n, 2 * n]
        fracs = [np.mean(dists > t) for t in thresholds]
        print(f"    Fraction with dist > n/4={thresholds[0]}: {fracs[0]:.2%}")
        print(f"    Fraction with dist > n/2={thresholds[1]}: {fracs[1]:.2%}")
        print(f"    Fraction with dist > n={thresholds[2]}:   {fracs[2]:.2%}")
        print()

    print(f"\n  Hybrid cost model (N=1024, d_ent=16):")
    N = 1024; n = 32; logN = log2(N)
    d_ent = 16
    beta = 2 * sqrt(d_ent - 1) / d_ent
    T_route_tele = max(3, int(round(0.35 * logN / (1 - beta))))
    T_phys_full = int(round(1.5 * sqrt(N)))

    print(f"  T_teleport = {T_route_tele}, T_phys_full = {T_phys_full}")
    print(f"\n  {'D_thresh':>8} | {'frac_tele':>10} {'frac_phys':>10} | "
      f"{'T_tele':>7} {'T_phys_cleanup':>15} {'T_total':>8}")
    print(f"  {'-' * 70}")

    for D_thresh in [4, 8, 12, 16, 24, 32]:
        dists = []
        for _ in range(20):
            pi = np.random.permutation(N)
            for v in range(N):
                dists.append(grid_distance(v, pi[v], n))
        dists = np.array(dists)
        frac_tele = np.mean(dists > D_thresh)
        frac_phys = 1 - frac_tele

        T_tele = T_route_tele if frac_tele > 0 else 0
        T_phys_cleanup = int(ceil(frac_phys * D_thresh * 2))  # simplified

        # Total: max of parallel teleportation + sequential cleanup
        T_total = T_tele + T_phys_cleanup

        print(f"  {D_thresh:>8} | {frac_tele:>10.2%} {frac_phys:>10.2%} | "
          f"{T_tele:>7} {T_phys_cleanup:>15} {T_total:>8}")


    print("\n\n" + "=" * 90)
    print("  SUMMARY OF RESULTS")
    print("=" * 90)

    t_end = time.time()
    print(f"  Total computation time: {t_end - t_start:.1f}s")


if __name__ == "__main__":
    main()
