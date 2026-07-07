#!/usr/bin/env python3

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from collections import defaultdict, deque
import time

np.random.seed(42)

from routing_lib.graphs import (
    random_regular_matching_union as random_regular_graph,
    build_2d_grid_clique as build_2d_grid,
)
from routing_lib.spectral import (
    spectral_params_lam2_over_d as spectral_params,
    theoretical_routing_bound,
    ramanujan_bound,
)
from routing_lib.routing import (
    bfs_all,
    path_edges,
    valiant_routing_trial,
    routing_simulation,
    edge_layer_assignment,
    effective_capacity_with_crosstalk,
)


def build_layer_union(N, d0, L, independent=True):
    layers = []
    A_union = np.zeros((N, N))
    for ell in range(L):
        if independent:
            A_ell = random_regular_graph(N, d0)
        else:
            # Correlated layers: each layer is a perturbation of the first
            if ell == 0:
                A_base = random_regular_graph(N, d0)
                A_ell = A_base.copy()
            else:
                A_ell = random_regular_graph(N, d0)
                # Mix: 50% base + 50% random (for correlated test)
                # Actually keep independent for now
                pass
        layers.append(A_ell)
        A_union += A_ell
    return layers, A_union

def part1_spectral_scaling():
    print("  PART 1: SPECTRAL GAIN FROM LAYER UNION (Lemma 1.3)")
    print("  Union of L independent random d0-regular graphs")

    d0 = 8
    N_vals = [64, 100, 144, 256]
    L_vals = [1, 2, 4, 8, 16]

    print(f"\n  d0 = {d0} per layer, 5 independent trials per configuration")
    print(f"\n  {'N':>5} {'L':>3} | {'d_union':>8} {'lam2':>8} {'beta':>8} "
          f"{'1-beta':>8} | {'predicted':>9} {'Ram bound':>10} {'Ramanujan?':>10}")
    print(f"  {'-' * 85}")

    all_results = []

    for N in N_vals:
        for L in L_vals:
            betas = []
            lam2s = []
            d_primes = []

            for trial in range(5):
                np.random.seed(42 + trial * 100 + L * 10 + N)
                layers, A_union = build_layer_union(N, d0, L)
                sp = spectral_params(A_union)
                betas.append(sp['beta'])
                lam2s.append(sp['lambda_2'])
                d_primes.append(sp['d_prime'])

            beta_mean = np.mean(betas)
            lam2_mean = np.mean(lam2s)
            d_prime_mean = np.mean(d_primes)

            # Predicted beta from Friedman: lambda_2 ~ 2*sqrt(L*d0 - 1)
            predicted_lam2 = 2 * np.sqrt(L * d0 - 1)
            predicted_beta = predicted_lam2 / (L * d0)

            # Ramanujan check
            ram_bound = ramanujan_bound(d_prime_mean)
            is_ram = lam2_mean <= ram_bound + 0.1

            print(f"  {N:>5} {L:>3} | {d_prime_mean:>8.1f} {lam2_mean:>8.3f} "
                  f"{beta_mean:>8.4f} {1-beta_mean:>8.4f} | "
                  f"{predicted_beta:>9.4f} {ram_bound:>10.3f} "
                  f"{'YES' if is_ram else 'no':>10}")

            all_results.append({
                'N': N, 'L': L, 'd_prime': d_prime_mean,
                'lam2': lam2_mean, 'beta': beta_mean,
                'predicted_beta': predicted_beta,
                'is_ramanujan': is_ram,
            })

    # Verify scaling: beta ~ C / sqrt(L)
    print(f"\n  Scaling verification: beta_union vs 1/sqrt(L)")
    for N in N_vals:
        results_N = [r for r in all_results if r['N'] == N]
        Ls = [r['L'] for r in results_N]
        betas = [r['beta'] for r in results_N]
        beta_1 = betas[0]  # L=1 baseline
        print(f"    N={N:>3}:  " + "  ".join(
            f"L={L}: β={b:.4f} (pred {beta_1/np.sqrt(L):.4f})"
            for L, b in zip(Ls, betas)))

    return all_results


def part2_routing_depth():
    print(f"\n\n{'=' * 90}")
    print("  PART 2: MULTI-LAYER ROUTING DEPTH (Theorem 1.1)")
    print("  Simulated Valiant routing on union graphs")
    print("=" * 90)

    d0 = 8
    N_configs = [(8, 64), (10, 100), (12, 144)]
    L_vals = [1, 2, 4, 8]
    num_trials = 150

    for n, N in N_configs:
        print(f"\n  --- N = {N} ({n}x{n} grid), d0 = {d0} ---")

        # Grid baseline
        A_grid = build_2d_grid(n, r=3)
        sp_grid = spectral_params(A_grid)
        grid_bound = theoretical_routing_bound(sp_grid['d_prime'], sp_grid['beta'], N)

        # Grid routing simulation
        grid_results = routing_simulation(A_grid, N, num_trials=num_trials)
        grid_T = np.median([r[2] for r in grid_results])

        print(f"  {'Model':<25} {'d_prime':>7} {'beta':>7} {'1-beta':>7} "
              f"{'Thm bound':>9} {'Sim T_med':>9} {'Sim T_90':>8}")
        print(f"  {'-' * 80}")
        print(f"  {'2D grid (baseline)':<25} {sp_grid['d_prime']:>7.1f} "
              f"{sp_grid['beta']:>7.4f} {sp_grid['gap']:>7.4f} "
              f"{grid_bound:>9.0f} {grid_T:>9.0f} "
              f"{np.percentile([r[2] for r in grid_results], 90):>8.0f}")

        for L in L_vals:
            np.random.seed(42 + N * 100 + L)
            layers, A_union = build_layer_union(N, d0, L)
            sp_union = spectral_params(A_union)
            union_bound = theoretical_routing_bound(
                sp_union['d_prime'], sp_union['beta'], N)

            # Routing simulation on union graph
            union_results = routing_simulation(A_union, N, num_trials=num_trials)
            union_T = np.median([r[2] for r in union_results])
            union_T90 = np.percentile([r[2] for r in union_results], 90)

            min_k0 = N // (2 * L)

            print(f"  {'L=' + str(L) + ' layers':<25} {sp_union['d_prime']:>7.1f} "
                  f"{sp_union['beta']:>7.4f} {sp_union['gap']:>7.4f} "
                  f"{union_bound:>9.0f} {union_T:>9.0f} {union_T90:>8.0f}"
                  f"  [min k0={min_k0}]")


        print(f"\n  Speedup over 2D grid (simulated median T):")
        for L in L_vals:
            np.random.seed(42 + N * 100 + L)
            _, A_union = build_layer_union(N, d0, L)
            res = routing_simulation(A_union, N, num_trials=num_trials)
            T_med = np.median([r[2] for r in res])
            print(f"    L={L:>2}: T_med={T_med:>4.0f}, speedup={grid_T/T_med:>5.2f}x")


def part3_capacity_analysis():

    print("  PART 3: CAPACITY-CONSTRAINED ROUTING DEPTH")
    print("  How many layers are needed for Theta(log N)?")


    d0 = 8

    print(f"\n  Theoretical analysis: T = T_overlay * ceil(N / (2*L*k0))")
    print(f"  where T_overlay = O(log N / (1 - beta_union))")

    print(f"\n  {'N':>6} {'L':>3} {'k0':>6} {'L*k0':>7} {'beta_u':>7} "
          f"{'T_overlay':>10} {'substeps':>9} {'T_total':>8} {'vs grid':>8}")
    print(f"  {'-' * 75}")

    for N in [64, 256, 1024, 4096]:
        log2N = np.log2(N)

        # Grid baseline (approximate beta for large N)
        beta_grid = 1 - 1.5 / np.sqrt(N)
        d_grid = 8.0
        T_grid = theoretical_routing_bound(d_grid, min(beta_grid, 0.999), N)

        for L in [1, 2, 4, 8, 16]:
            d_union = L * d0
            # Predicted beta from Friedman
            beta_union = 2 * np.sqrt(L * d0 - 1) / (L * d0)
            beta_union = min(beta_union, 0.999)
            T_overlay = theoretical_routing_bound(d_union, beta_union, N)

            for k0_frac in [0.5, 0.1, 0.01]:
                k0 = max(1, int(k0_frac * N))
                total_cap = L * k0
                substeps = max(1, int(np.ceil(N / (2 * total_cap))))
                T_total = T_overlay * substeps
                ratio = T_total / T_grid if T_grid < float('inf') else float('inf')

                label = f"N/{int(1/k0_frac)}" if k0_frac < 1 else "N/2"
                better = " ***" if T_total < T_grid else ""
                print(f"  {N:>6} {L:>3} {label:>6} {total_cap:>7} "
                      f"{beta_union:>7.4f} {T_overlay:>10.0f} "
                      f"{substeps:>9} {T_total:>8.0f} {ratio:>7.2f}x{better}")

            if L == 16:
                print(f"  {'-' * 75}")


def part4_crosstalk():
    print(f"\n\n{'=' * 90}")
    print("  PART 4: CROSSTALK SENSITIVITY (Proposition 1.4)")
    print("  How inter-layer optical coupling affects effective capacity")
    print("=" * 90)

    k0 = 32  # per-layer capacity
    L_vals = [2, 4, 8, 16]
    gamma_vals = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    print(f"\n  Per-layer capacity k0 = {k0}")
    print(f"\n  {'L':>3} | " + " ".join(f"{'γ='+format(g,'.1f'):>8}" for g in gamma_vals))
    print(f"  {'-' * (6 + 9 * len(gamma_vals))}")

    for L in L_vals:
        caps = []
        for gamma in gamma_vals:
            k_eff = effective_capacity_with_crosstalk(L, k0, gamma)
            caps.append(k_eff)
        ideal = L * k0
        print(f"  {L:>3} | " + " ".join(f"{c:>8.0f}" for c in caps)
              + f"  (ideal: {ideal})")

    # Detailed analysis: routing depth under crosstalk
    print(f"\n\n  Routing depth under crosstalk (N=256, d0=8, k0=N/16=16)")
    N = 256
    d0 = 8
    k0 = N // 16  # 16 transfers per layer
    log2N = np.log2(N)

    print(f"\n  {'L':>3} {'gamma':>6} | {'k_eff':>7} {'substeps':>9} "
          f"{'beta_u':>7} {'T_overlay':>10} {'T_total':>8}")
    print(f"  {'-' * 60}")

    for L in [2, 4, 8, 16]:
        d_union = L * d0
        beta_union = min(2 * np.sqrt(L * d0 - 1) / (L * d0), 0.999)
        T_overlay = theoretical_routing_bound(d_union, beta_union, N)

        for gamma in [0, 0.2, 0.5]:
            k_eff = effective_capacity_with_crosstalk(L, k0, gamma)
            substeps = max(1, int(np.ceil(N / (2 * k_eff))))
            T_total = T_overlay * substeps

            print(f"  {L:>3} {gamma:>6.1f} | {k_eff:>7.0f} {substeps:>9} "
                  f"{beta_union:>7.4f} {T_overlay:>10.0f} {T_total:>8.0f}")


def part5_combined_simulation():
    print(f"\n\n{'=' * 90}")
    print("  PART 5: FULL SIMULATION VALIDATION")
    print("  Multi-layer Valiant routing with per-layer capacity constraints")
    print("=" * 90)

    d0 = 8
    N = 100  # 10x10 grid
    n = 10
    num_trials = 200

    # Grid baseline
    A_grid = build_2d_grid(n, r=3)
    sp_grid = spectral_params(A_grid)
    grid_results = routing_simulation(A_grid, N, num_trials=num_trials)
    grid_T = np.median([r[2] for r in grid_results])
    grid_C = np.median([r[0] for r in grid_results])

    print(f"\n  Baseline: 2D grid ({n}x{n}, N={N})")
    print(f"  d'={sp_grid['d_prime']:.0f}, beta={sp_grid['beta']:.4f}, "
          f"median T={grid_T:.0f}, median C={grid_C:.0f}")

    print(f"\n  {'Model':<30} {'d_prime':>7} {'beta':>7} {'T_med':>6} "
          f"{'C_med':>6} {'D_med':>6} {'speedup':>8}")
    print(f"  {'-' * 75}")
    print(f"  {'2D grid':<30} {sp_grid['d_prime']:>7.1f} "
          f"{sp_grid['beta']:>7.4f} {grid_T:>6.0f} {grid_C:>6.0f} "
          f"{np.median([r[1] for r in grid_results]):>6.0f} {'1.00x':>8}")

    for L in [1, 2, 4, 8]:
        np.random.seed(42 + L * 1000)
        layers, A_union = build_layer_union(N, d0, L)
        sp_u = spectral_params(A_union)

        results = routing_simulation(A_union, N, num_trials=num_trials)
        T_med = np.median([r[2] for r in results])
        C_med = np.median([r[0] for r in results])
        D_med = np.median([r[1] for r in results])
        speedup = grid_T / T_med if T_med > 0 else float('inf')

        print(f"  {f'L={L} random {d0}-reg layers':<30} {sp_u['d_prime']:>7.1f} "
              f"{sp_u['beta']:>7.4f} {T_med:>6.0f} {C_med:>6.0f} "
              f"{D_med:>6.0f} {speedup:>7.2f}x")

    # Also test: grid + L overlay layers
    print(f"\n  Grid + overlay layers (grid provides free local connectivity):")
    print(f"  {'Model':<30} {'d_prime':>7} {'beta':>7} {'T_med':>6} "
          f"{'C_med':>6} {'D_med':>6} {'speedup':>8}")
    print(f"  {'-' * 75}")

    for L in [1, 2, 4]:
        np.random.seed(42 + L * 2000)
        _, A_overlay = build_layer_union(N, d0, L)
        A_combined = A_grid + A_overlay
        sp_c = spectral_params(A_combined)

        results = routing_simulation(A_combined, N, num_trials=num_trials)
        T_med = np.median([r[2] for r in results])
        C_med = np.median([r[0] for r in results])
        D_med = np.median([r[1] for r in results])
        speedup = grid_T / T_med if T_med > 0 else float('inf')

        print(f"  {f'grid + {L} overlay layers':<30} {sp_c['d_prime']:>7.1f} "
              f"{sp_c['beta']:>7.4f} {T_med:>6.0f} {C_med:>6.0f} "
              f"{D_med:>6.0f} {speedup:>7.2f}x")


def main():
    t0 = time.time()

    print("╔" + "═" * 88 + "╗")
    print("║  DIRECTION 1: MULTI-LAYER 3D AOL ARCHITECTURES                                       ║")
    print("║  Stacking independent AOL layers for Theta(log N) routing                             ║")
    print("╚" + "═" * 88 + "╝")

    spectral_results = part1_spectral_scaling()
    part2_routing_depth()
    part3_capacity_analysis()
    part4_crosstalk()
    part5_combined_simulation()

    elapsed = time.time() - t0

    print(f"  Total computation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
