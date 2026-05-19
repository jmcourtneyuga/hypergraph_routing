#!/usr/bin/env python3
"""
Block routing simulator for Ramanujan hypergraphs (Paper II).

Validates rt_B = Theta(d_C * log N_L) via:
  - clique-expansion construction of (d,r)-regular Ramanujan hosts
  - BFS-based block placement with guard zones
  - quotient-graph spectral analysis (Theorem 4.8)
  - block Valiant routing + LMR scheduling
  - experiments F2.1 / F2.2 / F3.1 / F5.1
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from collections import defaultdict, deque
import time

from routing_lib.graphs import (
    random_regular_config_model as random_regular_simple_graph,
    build_host_graph_d_dprime as build_host_graph,
)
from routing_lib.spectral import compute_graph_spectral_params

np.random.seed(42)


# ============================================================
# Block data structures
# ============================================================


from routing_lib.blocks import (
    Block,
    BlockConfig,
    random_block_config,
    quotient_graph,
)
from routing_lib.routing import (
    block_valiant_route,
    block_lmr_schedule,
    simulate_block_routing,
    point_routing_baseline,
    shortest_paths_all_pairs,
)


def experiment_F2_1():
    """Experiment F2.1: N_L scaling at fixed d_C."""
    print("\n" + "="*70)
    print("EXPERIMENT F2.1: N_L Scaling (fixed d_C)")
    print("="*70)

    d_C_values = [3, 5, 7]
    N_L_values = [4, 8, 16]

    for d_C in d_C_values:
        print(f"\nCode distance d_C = {d_C}")
        print("-" * 70)
        print(f"{'N_L':<6} {'N_phys':<8} {'T_mean':<10} {'T_median':<10} {'C_max':<10} {'D_Q':<8} {'beta_Q':<10}")
        print("-" * 70)

        for N_L in N_L_values:
            N_phys = max(N_L * d_C * d_C * 6, 256)
            if (N_phys * 4 * 2) % 2 != 0:
                N_phys += 1
            stats = simulate_block_routing(N_phys, d=4, r=3, d_C=d_C,
                                         guard_dist=1, N_L=N_L, n_trials=1)

            print(f"{N_L:<6} {N_phys:<8} {stats['T_mean']:<10.1f} {stats['T_median']:<10.1f} "
                  f"{stats['C_max_mean']:<10.1f} {stats['D_Q_mean']:<8.1f} {stats['beta_Q_mean']:<10.4f}")


def experiment_F2_2():
    """Experiment F2.2: d_C scaling at fixed N_L."""
    print("\n" + "="*70)
    print("EXPERIMENT F2.2: Code Distance Scaling (fixed N_L = 16)")
    print("="*70)

    N_L = 8
    d_C_values = [3, 5, 7, 9]

    print(f"\nLogical qubits N_L = {N_L}")
    print("-" * 70)
    print(f"{'d_C':<6} {'N_phys':<8} {'T_mean':<10} {'T_median':<10} {'C_max':<10} {'D_Q':<8}")
    print("-" * 70)

    for d_C in d_C_values:
        N_phys = max(N_L * d_C * d_C * 6, 256)
        if (N_phys * 4 * 2) % 2 != 0:
            N_phys += 1
        stats = simulate_block_routing(N_phys, d=4, r=3, d_C=d_C,
                                     guard_dist=1, N_L=N_L, n_trials=1)

        print(f"{d_C:<6} {N_phys:<8} {stats['T_mean']:<10.1f} {stats['T_median']:<10.1f} "
              f"{stats['C_max_mean']:<10.1f} {stats['D_Q_mean']:<8.1f}")


def experiment_F3_1():
    """Experiment F3.1: beta_Q vs beta (spectral inheritance)."""
    print("\n" + "="*70)
    print("EXPERIMENT F3.1: Spectral Inheritance (beta_Q vs beta)")
    print("="*70)

    d_values = [3, 4, 5, 6]
    d_C = 3
    N_L = 8

    print(f"\nCode distance d_C = {d_C}, Logical qubits N_L = {N_L}")
    print("-" * 70)
    print(f"{'d':<6} {'N_phys':<8} {'beta':<10} {'beta_Q':<10} {'epsilon_eq':<10}")
    print("-" * 70)

    for d in d_values:
        r = 3
        N_phys = max(N_L * d_C * d_C * 6, 256)
        if (N_phys * d * (r - 1)) % 2 != 0:
            N_phys += 1

        try:
            # Build random d'-regular expander
            G_cl = build_host_graph(N_phys, d, r)

            # Compute spectral params
            _, _, beta, _ = compute_graph_spectral_params(G_cl)

            # Place blocks and compute quotient graph
            config = random_block_config(G_cl, d_C, 1, N_L)
            q_result = quotient_graph(G_cl, config)

            print(f"{d:<6} {N_phys:<8} {beta:<10.4f} {q_result['beta_Q']:<10.4f} "
                  f"{q_result['epsilon_eq']:<10.4f}")
        except Exception as e:
            print(f"{d:<6} {N_phys:<8} --- (failed: {str(e)[:30]})")


def experiment_F5_1():
    """Experiment F5.1: Overhead ratio rt_B / rt (point routing)."""
    print("\n" + "="*70)
    print("EXPERIMENT F5.1: Block vs. Point Routing Overhead")
    print("="*70)

    d_C_values = [3, 5]
    N_L_values = [4, 8]

    print(f"\nBlock routing overhead (ratio rt_B / rt_point)")
    print("-" * 70)
    print(f"{'d_C':<6} {'N_L':<6} {'N_phys':<8} {'T_block':<10} {'T_point':<10} {'Overhead':<10}")
    print("-" * 70)

    for d_C in d_C_values:
        for N_L in N_L_values:
            N_phys = max(N_L * d_C * d_C * 6, 256)
            if (N_phys * 4 * 2) % 2 != 0:
                N_phys += 1

            try:
                stats = simulate_block_routing(N_phys, d=4, r=3, d_C=d_C,
                                              guard_dist=1, N_L=N_L, n_trials=2)

                # Point routing on same graph size (analytical bound)
                d_prime = 4 * (3 - 1)
                T_point = d_prime * np.log2(N_phys)
                overhead = stats['T_mean'] / T_point if T_point > 0 else 0

                print(f"{d_C:<6} {N_L:<6} {N_phys:<8} {stats['T_mean']:<10.1f} "
                      f"{T_point:<10.1f} {overhead:<10.2f}")
            except Exception as e:
                print(f"{d_C:<6} {N_L:<6} {N_phys:<8} --- (failed)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Block Routing Simulator — Paper II")
    print("Ramanujan Hypergraph Routing Surface Code Patches")
    print("="*70)

    # Run key experiments
    experiment_F2_1()
    experiment_F2_2()
    experiment_F3_1()
    experiment_F5_1()

    print("\n" + "="*70)
    print("Simulations complete")
    print("="*70)
