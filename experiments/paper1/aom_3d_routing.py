"""
Grid hypergraph models A (2D AOD) and B (3D AOL) (Paper I, Sections 3.5 and 11).

Model A: n x n torus grid; r-uniform hyperedges = row/column neighborhoods
         within the Rydberg blockade radius.
Model B: Model A + diagonal/skip hyperedges to capture inter-layer shuttling.

The matching model used here is a relaxation of the physical AOD non-crossing
constraint; bounds derived are therefore lower bounds on physical depth.
"""

import numpy as np
from collections import defaultdict, deque
from itertools import combinations

from routing_lib.graphs import (
    torus_grid_coords,
    coord_to_idx,
    build_2d_grid_hypergraph,
    build_3d_enhanced_hypergraph,
    hypergraph_to_clique_expansion,
)
from routing_lib.spectral import compute_spectrum
from routing_lib.routing import (
    bfs_diameter_and_paths,
    compute_routing_congestion,
)

def compute_vertex_degree_stats(adj, N):
    degrees = [len(adj[v]) for v in range(N)]
    return min(degrees), np.mean(degrees), max(degrees)

def compute_regularity(N, hyperedges, r):
    """Check vertex degrees in hypergraph."""
    deg = defaultdict(int)
    for e in hyperedges:
        for v in e:
            deg[v] += 1
    degs = [deg.get(v, 0) for v in range(N)]
    return min(degs), np.mean(degs), max(degs)

def run_analysis():
    print("=" * 95)
    print("  3D AOL ARCHITECTURE: HYPERGRAPH ROUTING MODEL")
    print("  2D Grid (Standard AOD) vs 3D-Enhanced (AOL with Inter-Layer Bypass)")
    print("=" * 95)

    test_configs = [
        (8, 3, "8x8 grid (N=64)"),
        (10, 3, "10x10 grid (N=100)"),
        (12, 3, "12x12 grid (N=144)"),
        (16, 3, "16x16 grid (N=256)"),
        (8, 4, "8x8 grid (N=64)"),
        (10, 4, "10x10 grid (N=100)"),
    ]

    results = []

    for n, r, label in test_configs:
        N = n * n
        print(f"\n{'─' * 95}")
        print(f"  {label}, r={r}")
        print(f"{'─' * 95}")

        # Model A: 2D Grid
        N_2d, hyp_2d = build_2d_grid_hypergraph(n, r)
        adj_2d = hypergraph_to_clique_expansion(N_2d, hyp_2d)
        dmin_h2d, dmean_h2d, dmax_h2d = compute_regularity(N_2d, hyp_2d, r)
        dmin_2d, dmean_2d, dmax_2d = compute_vertex_degree_stats(adj_2d, N_2d)
        D_2d, avg_d_2d = bfs_diameter_and_paths(adj_2d, N_2d)

        eigs_2d = compute_spectrum(adj_2d, N_2d, k=4)
        d_prime_2d = eigs_2d[0]
        lam2_2d = eigs_2d[1] if len(eigs_2d) > 1 else 0
        beta_2d = lam2_2d / d_prime_2d if d_prime_2d > 0 else 1

        print(f"\n  MODEL A (2D AOD):")
        print(f"    |Hyperedges| = {len(hyp_2d)}, Hypergraph degree: min={dmin_h2d}, mean={dmean_h2d:.1f}, max={dmax_h2d}")
        print(f"    G_cl degree d': min={dmin_2d}, mean={dmean_2d:.1f}, max={dmax_2d}")
        print(f"    Spectrum: lambda_1={eigs_2d[0]:.2f}, lambda_2={lam2_2d:.2f}, beta={beta_2d:.4f}")
        print(f"    Diameter D={D_2d}, Avg distance={avg_d_2d:.2f}")

        # Routing bound
        if beta_2d < 1 and d_prime_2d > 0:
            mu_2d = 2 * D_2d / d_prime_2d
            bound_2d = (4 * (d_prime_2d + 4)) / (d_prime_2d * np.log2(1/beta_2d)) * np.log2(N) + 5 * np.log2(N)
            print(f"    mu = 2D/d' = {mu_2d:.3f}")
            print(f"    Theorem 15.1 bound: rt(H) <= {bound_2d:.1f}")
        else:
            bound_2d = float('inf')
            mu_2d = float('inf')
            print(f"    beta >= 1 or d'=0: bound not applicable (not Ramanujan)")

        # Model B: 3D Enhanced
        N_3d, hyp_3d = build_3d_enhanced_hypergraph(n, r)
        adj_3d = hypergraph_to_clique_expansion(N_3d, hyp_3d)
        dmin_h3d, dmean_h3d, dmax_h3d = compute_regularity(N_3d, hyp_3d, r)
        dmin_3d, dmean_3d, dmax_3d = compute_vertex_degree_stats(adj_3d, N_3d)
        D_3d, avg_d_3d = bfs_diameter_and_paths(adj_3d, N_3d)

        eigs_3d = compute_spectrum(adj_3d, N_3d, k=4)
        d_prime_3d = eigs_3d[0]
        lam2_3d = eigs_3d[1] if len(eigs_3d) > 1 else 0
        beta_3d = lam2_3d / d_prime_3d if d_prime_3d > 0 else 1

        print(f"\n  MODEL B (3D AOL):")
        print(f"    |Hyperedges| = {len(hyp_3d)}, Hypergraph degree: min={dmin_h3d}, mean={dmean_h3d:.1f}, max={dmax_h3d}")
        print(f"    G_cl degree d': min={dmin_3d}, mean={dmean_3d:.1f}, max={dmax_3d}")
        print(f"    Spectrum: lambda_1={eigs_3d[0]:.2f}, lambda_2={lam2_3d:.2f}, beta={beta_3d:.4f}")
        print(f"    Diameter D={D_3d}, Avg distance={avg_d_3d:.2f}")

        if beta_3d < 1 and d_prime_3d > 0:
            mu_3d = 2 * D_3d / d_prime_3d
            bound_3d = (4 * (d_prime_3d + 4)) / (d_prime_3d * np.log2(1/beta_3d)) * np.log2(N) + 5 * np.log2(N)
            print(f"    mu = 2D/d' = {mu_3d:.3f}")
            print(f"    Theorem 15.1 bound: rt(H) <= {bound_3d:.1f}")
        else:
            bound_3d = float('inf')
            mu_3d = float('inf')
            print(f"    beta >= 1 or d'=0: bound not applicable")

        # Improvement
        if bound_2d < float('inf') and bound_3d < float('inf'):
            improvement = bound_2d / bound_3d
            D_improvement = D_2d / D_3d if D_3d > 0 else float('inf')
            print(f"\n  IMPROVEMENT (3D vs 2D):")
            print(f"    Diameter: {D_2d} -> {D_3d} ({D_improvement:.2f}x)")
            print(f"    beta: {beta_2d:.4f} -> {beta_3d:.4f}")
            print(f"    Routing bound: {bound_2d:.1f} -> {bound_3d:.1f} ({improvement:.2f}x)")

        # Actual routing congestion (for smaller instances)
        if N <= 144:
            print(f"\n  SIMULATED ROUTING (500 random permutations):")
            C_mean_2d, C_med_2d, C_max_2d = compute_routing_congestion(adj_2d, N_2d, num_trials=500)
            C_mean_3d, C_med_3d, C_max_3d = compute_routing_congestion(adj_3d, N_3d, num_trials=500)
            T_2d = 2 * (C_med_2d + D_2d)
            T_3d = 2 * (C_med_3d + D_3d)
            print(f"    2D: C_mean={C_mean_2d:.1f}, C_med={C_med_2d:.0f}, C_max={C_max_2d:.0f}, T~{T_2d:.0f}")
            print(f"    3D: C_mean={C_mean_3d:.1f}, C_med={C_med_3d:.0f}, C_max={C_max_3d:.0f}, T~{T_3d:.0f}")
            print(f"    Routing time speedup: {T_2d/T_3d:.2f}x")

            results.append({
                'label': label, 'N': N, 'r': r, 'n': n,
                'D_2d': D_2d, 'D_3d': D_3d,
                'dprime_2d': d_prime_2d, 'dprime_3d': d_prime_3d,
                'beta_2d': beta_2d, 'beta_3d': beta_3d,
                'bound_2d': bound_2d, 'bound_3d': bound_3d,
                'C_med_2d': C_med_2d, 'C_med_3d': C_med_3d,
                'T_2d': T_2d, 'T_3d': T_3d,
            })
        else:
            results.append({
                'label': label, 'N': N, 'r': r, 'n': n,
                'D_2d': D_2d, 'D_3d': D_3d,
                'dprime_2d': d_prime_2d, 'dprime_3d': d_prime_3d,
                'beta_2d': beta_2d, 'beta_3d': beta_3d,
                'bound_2d': bound_2d, 'bound_3d': bound_3d,
                'C_med_2d': None, 'C_med_3d': None,
                'T_2d': None, 'T_3d': None,
            })

    # Summary table
    print(f"\n\n{'=' * 95}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 95}")
    print(f"  {'Config':<20} {'N':>4} {'r':>2} | {'D_2D':>4} {'D_3D':>4} | "
          f"{'d_2D':>5} {'d_3D':>5} | {'β_2D':>6} {'β_3D':>6} | "
          f"{'Bnd_2D':>7} {'Bnd_3D':>7} {'Impr':>5}")
    print(f"  {'-'*90}")
    for res in results:
        impr = res['bound_2d'] / res['bound_3d'] if res['bound_3d'] > 0 and res['bound_3d'] < float('inf') else 0
        print(f"  {res['label']:<20} {res['N']:>4} {res['r']:>2} | "
              f"{res['D_2d']:>4} {res['D_3d']:>4} | "
              f"{res['dprime_2d']:>5.0f} {res['dprime_3d']:>5.0f} | "
              f"{res['beta_2d']:>6.3f} {res['beta_3d']:>6.3f} | "
              f"{res['bound_2d']:>7.1f} {res['bound_3d']:>7.1f} {impr:>5.2f}x")

    print(f"\n  ROUTING SIMULATION COMPARISON (where available):")
    print(f"  {'Config':<20} {'N':>4} | {'T_2D':>5} {'T_3D':>5} {'Speedup':>8}")
    print(f"  {'-'*50}")
    for res in results:
        if res['T_2d'] is not None:
            speedup = res['T_2d'] / res['T_3d'] if res['T_3d'] > 0 else 0
            print(f"  {res['label']:<20} {res['N']:>4} | "
                  f"{res['T_2d']:>5.0f} {res['T_3d']:>5.0f} {speedup:>7.2f}x")

    print(f"\n  NOTE: These bounds assume the full matching model (any matching in G_cl).")
    print(f"  The physical AOD constraint (row/column moves only) is more restrictive.")
    print(f"  Our bounds are therefore LOWER BOUNDS on the actual physical routing depth.")
    print(f"  The 3D/2D improvement ratio, however, is expected to hold qualitatively")
    print(f"  since the 3D bypass reduces diameter and increases connectivity in both models.")

if __name__ == "__main__":
    run_analysis()
