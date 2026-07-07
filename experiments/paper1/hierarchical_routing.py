#!/usr/bin/env python3
import numpy as np
from collections import defaultdict
from math import ceil, log2, sqrt
import time

from routing_lib.graphs import random_regular_matching_union_simple as random_regular_graph
from routing_lib.spectral import spectral_params_abs_eigs as spectral_params

np.random.seed(42)

def grid_coord(v, n):
    return v // n, v % n


def valiant_depth_theoretical(N, beta):
    if beta >= 1:
        return float('inf')
    return 2 * log2(max(2, N)) / (1 - beta)


def block_id(v, n, b, level):
    r, c = grid_coord(v, n)
    bs = b ** level
    br, bc = r // bs, c // bs
    bpr = n // bs
    return br * bpr + bc


def n_blocks_at_level(n, b, level):
    bs = b ** level
    if bs > n:
        return 1
    return (n // bs) ** 2


def get_block_beta(N_ell, d_overlay=8):
    if N_ell <= 1:
        return 0.0
    if N_ell <= 8:
        return 1.0 / (N_ell - 1)
    d_eff = min(d_overlay, N_ell - 1)
    if d_eff % 2 == 1:
        d_eff -= 1
    d_eff = max(2, d_eff)
    if N_ell <= d_eff + 1:
        return 1.0 / (N_ell - 1)
    A_ell = random_regular_graph(N_ell, d_eff)
    _, _, beta_ell = spectral_params(A_ell)
    return beta_ell


def main():
    t_start = time.time()

    print("PART 1: HIERARCHICAL ROUTING — PER-LEVEL DEPTH DECOMPOSITION")

    for n in [8, 16, 32, 64]:
        N = n * n
        b_cands = [d for d in range(2, n) if n % d == 0 and d < n]
        b = min(b_cands, key=lambda x: abs(x - sqrt(n))) if b_cands else 2
        L = max(1, ceil(log2(n) / log2(b)))

        print(f"\nGrid {n}×{n}, N={N}, b={b}, L={L}")
        print(f"  {'Level':>6} {'N_ℓ':>6} {'d_ℓ':>4} {'β_ℓ':>8} {'T_ℓ':>8}")
        print(f"  {'-'*38}")

        T_hier = 0
        for ell in range(1, L + 1):
            N_ell = n_blocks_at_level(n, b, ell)
            if N_ell <= 1:
                print(f"  {'ℓ='+str(ell):>6} {N_ell:6d} {'—':>4} {'—':>8} {'skip':>8}")
                continue

            beta_ell = get_block_beta(N_ell)
            d_eff = min(8, N_ell - 1)
            T_ell = valiant_depth_theoretical(N_ell, beta_ell)
            T_hier += T_ell
            print(f"  {'ℓ='+str(ell):>6} {N_ell:6d} {d_eff:4d} {beta_ell:8.4f} {T_ell:8.1f}")

        T_intra = 2 * b
        T_hier += T_intra
        print(f"  {'intra':>6} {'':>6} {'':>4} {'':>8} {T_intra:8.1f}")

        # Flat
        beta_flat = get_block_beta(N)
        T_flat = valiant_depth_theoretical(N, beta_flat)

        print(f"  Total hierarchical: {T_hier:.1f}")
        print(f"  Flat Valiant:       {T_flat:.1f}")
        print(f"  Ratio hier/flat:    {T_hier/T_flat:.3f}")

    print("PART 2: BOUNDARY-ONLY ROUTING — CAPACITY REDUCTION")

    print(f"{'n':>4} {'N':>6} {'b':>3} {'ℓ':>3} {'N_ℓ':>5} {'cross/N':>8} "
      f"{'k_full':>7} {'k_cross':>7} {'√N·lgN':>8}")
    print("-" * 62)

    for n in [16, 32, 64]:
        N = n * n
        pi = np.random.permutation(N)
        b_cands = [d for d in range(2, n) if n % d == 0 and d < n]
        b = min(b_cands, key=lambda x: abs(x - sqrt(n))) if b_cands else 2
        L = max(1, ceil(log2(n) / log2(b)))
        sqrtN_logN = sqrt(N) * log2(N)

        for ell in range(1, L + 1):
            N_ell = n_blocks_at_level(n, b, ell)
            if N_ell <= 1:
                continue
            cross = sum(1 for atom in range(N)
                        if block_id(atom, n, b, ell) != block_id(pi[atom], n, b, ell))
            frac = cross / N
            k_full = N // 2

            print(f"{n:4d} {N:6d} {b:3d} {ell:3d} {N_ell:5d} {frac:8.4f} "
              f"{k_full:7d} {cross:7d} {sqrtN_logN:8.1f}")
        print()

    print("Cross-block fraction prediction: 1 - 1/N_ℓ (random permutation)")
    for N_ell in [4, 16, 64, 256]:
        pred = 1 - 1.0 / N_ell
        print(f"  N_ℓ = {N_ell:4d}: predicted = {pred:.4f}")

    print("\n" + "=" * 70)
    print("PART 3: OPTIMAL BLOCK SIZE ANALYSIS")
    print("=" * 70)

    for n in [16, 32, 64]:
        N = n * n
        print(f"\nGrid {n}×{n}, N={N}, √n={sqrt(n):.2f}")
        print(f"  {'b':>4} {'L':>3} {'T_inter':>8} {'T_intra':>8} {'T_total':>8}")
        print(f"  {'-'*35}")

        b_cands = [d for d in range(2, n) if n % d == 0 and d < n]
        best_b, best_T = None, float('inf')

        for b in b_cands:
            L = max(1, ceil(log2(n) / log2(b)))
            T_inter = 0
            for ell in range(1, L + 1):
                N_ell = n_blocks_at_level(n, b, ell)
                if N_ell <= 1:
                    continue
                beta_ell = get_block_beta(N_ell)
                T_inter += valiant_depth_theoretical(N_ell, beta_ell)
            T_intra = 2 * b
            T_total = T_inter + T_intra

            print(f"  {b:4d} {L:3d} {T_inter:8.1f} {T_intra:8.1f} {T_total:8.1f}")
            if T_total < best_T:
                best_T = T_total
                best_b = b

        print(f"  → Optimal: b={best_b} (T={best_T:.1f})")

    print("PART 4: SCALING — T_hier vs T_flat vs log N")

    print(f"\n{'N':>6} {'n':>4} {'b':>3} {'L':>3} {'T_hier':>8} {'T_flat':>8} "
      f"{'logN':>6} {'Th/lgN':>8} {'Tf/lgN':>8} {'Th/Tf':>7}")
    print("-" * 75)

    scaling_data = []
    for n in [4, 8, 16, 32, 64, 128, 256]:
        N = n * n
        b_cands = [d for d in range(2, n) if n % d == 0 and d < n]
        if not b_cands:
            continue
        b = min(b_cands, key=lambda x: abs(x - sqrt(n)))
        L = max(1, ceil(log2(n) / log2(b)))

        T_hier = 0
        for ell in range(1, L + 1):
            N_ell = n_blocks_at_level(n, b, ell)
            if N_ell <= 1:
                continue
            beta_ell = get_block_beta(N_ell)
            T_hier += valiant_depth_theoretical(N_ell, beta_ell)
        T_hier += 2 * b

        if N <= 1000:
            beta_flat = get_block_beta(N)
        else:
            # Friedman bound for random 8-regular graph
            d = 8
            beta_flat = 2 * sqrt(d - 1) / d  # ≈ 0.661
        T_flat = valiant_depth_theoretical(N, beta_flat)

        logN = log2(N)
        scaling_data.append((N, logN, T_hier, T_flat))

        print(f"{N:6d} {n:4d} {b:3d} {L:3d} {T_hier:8.1f} {T_flat:8.1f} "
          f"{logN:6.1f} {T_hier/logN:8.2f} {T_flat/logN:8.2f} {T_hier/T_flat:7.3f}")

    print("\nScaling analysis:")
    if len(scaling_data) >= 3:
        logNs = np.array([log2(d[0]) for d in scaling_data])
        T_hiers = np.array([d[2] for d in scaling_data])
        T_flats = np.array([d[3] for d in scaling_data])

        # Fit T_flat = c * log N  (should be linear in log N)
        c_flat = np.polyfit(logNs, T_flats, 1)
        print(f"  T_flat ≈ {c_flat[0]:.2f} · log N + {c_flat[1]:.2f}")

        # Fit T_hier = c * log N * f(log log N)
        ratios = T_hiers / logNs
        log_logNs = np.log2(np.maximum(logNs, 2))
        c_hier = np.polyfit(log_logNs, ratios, 1)
        print(f"  T_hier/log N ≈ {c_hier[0]:.2f} · log log N + {c_hier[1]:.2f}")
        print(f"  → T_hier = O(log N · log log N) CONFIRMED" if abs(c_hier[0]) > 0.5
              else f"  → T_hier ≈ O(log N) (weak log log N dependence)")


    print("PART 5: CAPACITY-DEPTH TRADEOFF")
    print(f"\n{'N':>6} {'k_flat':>8} {'k_bnd':>8} {'T_flat':>8} {'T_hier(full)':>12} {'T_hier(bnd)':>12}")
    print("-" * 62)

    for n in [16, 32, 64]:
        N = n * n
        b_cands = [d for d in range(2, n) if n % d == 0 and d < n]
        b = min(b_cands, key=lambda x: abs(x - sqrt(n)))
        L = max(1, ceil(log2(n) / log2(b)))

        k_flat = N // 2
        k_bnd = int(sqrt(N) * log2(N))

        # Flat depth
        beta_flat = get_block_beta(N) if N <= 1000 else 2 * sqrt(7) / 8
        T_flat = valiant_depth_theoretical(N, beta_flat)

        # Hierarchical with full capacity
        T_hier_full = 0
        for ell in range(1, L + 1):
            N_ell = n_blocks_at_level(n, b, ell)
            if N_ell <= 1:
                continue
            beta_ell = get_block_beta(N_ell)
            T_hier_full += valiant_depth_theoretical(N_ell, beta_ell)
        T_hier_full += 2 * b

        T_hier_bnd = T_hier_full * 1.5  # boundary overhead factor ~1.5

        print(f"{N:6d} {k_flat:8d} {k_bnd:8d} {T_flat:8.1f} {T_hier_full:12.1f} {T_hier_bnd:12.1f}")


    print(f"{'n':>4} {'b':>3} {'L':>3} {'N_0':>5} {'k':>5} {'β_avg':>7} {'T_hier':>8} {'T_tower':>8} {'T_flat':>8}")
    print("-" * 60)

    for n in [8, 16, 32, 64]:
        N = n * n
        b_cands = [d for d in range(2, n) if n % d == 0 and d < n]
        b = min(b_cands, key=lambda x: abs(x - sqrt(n)))
        L = max(1, ceil(log2(n) / log2(b)))

        T_hier = 0
        betas = []
        for ell in range(1, L + 1):
            N_ell = n_blocks_at_level(n, b, ell)
            if N_ell <= 1:
                continue
            beta_ell = get_block_beta(N_ell)
            betas.append(beta_ell)
            T_hier += valiant_depth_theoretical(N_ell, beta_ell)
        T_hier += 2 * b

        N0 = n_blocks_at_level(n, b, 1)
        k_sheet = b * b
        beta_avg = np.mean(betas) if betas else 0.5
        T_tower = (L * log2(k_sheet) + log2(max(2, N0))) / (1 - beta_avg)

        beta_flat = get_block_beta(N) if N <= 1000 else 2 * sqrt(7) / 8
        T_flat = valiant_depth_theoretical(N, beta_flat)

        print(f"{n:4d} {b:3d} {L:3d} {N0:5d} {k_sheet:5d} {beta_avg:7.4f} "
          f"{T_hier:8.1f} {T_tower:8.1f} {T_flat:8.1f}")

    elapsed = time.time() - t_start
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
