#!/usr/bin/env python3
import numpy as np

from routing_lib.graphs import random_regular_networkx as random_regular_graph
from routing_lib.greedy import (
    grid_distance_open as grid_distance,
    displacement_energy,
    n_displaced,
    greedy_matching as greedy_displacement_matching,
    apply_matching,
)


def main():
    print("=" * 75)
    print("GREEDY STALL TEST EXTENDED TO N=256")
    print("=" * 75)
    print(f"\n{'n':>4} {'N':>5} {'d':>3} {'T_stall':>8} "
          f"{'Phi_stall/Phi_0':>17} {'#disp':>7} {'Phi_0':>10}")
    print("-" * 75)

    grid_sides = [4, 6, 8, 10, 12, 14, 16]  # N = n^2 = 16 .. 256
    n_trials = 15
    max_steps = 500

    for n in grid_sides:
        N = n * n
        d = max(2, min(8, N - 1))
        if d % 2 == 1:
            d -= 1

        adjacency = random_regular_graph(N, d, seed=42 + n)
        if adjacency is None:
            print(f"  n={n}: graph construction failed")
            continue

        stall_steps, stall_ratios, displaced_at_stall, phi0_values = [], [], [], []

        for trial in range(n_trials):
            np.random.seed(1000 + trial * 17 + n)
            target = np.random.permutation(N).tolist()
            positions = list(range(N))
            phi0 = displacement_energy(positions, target, n)
            phi0_values.append(phi0)

            for step in range(max_steps):
                matching, _ = greedy_displacement_matching(
                    adjacency, positions, target, n)
                if not matching:
                    stall_steps.append(step)
                    phi_stall = displacement_energy(positions, target, n)
                    stall_ratios.append(phi_stall / phi0 if phi0 else 0)
                    displaced_at_stall.append(n_displaced(positions, target))
                    break
                positions = apply_matching(positions, matching)
                if displacement_energy(positions, target, n) == 0:
                    stall_steps.append(step + 1)
                    stall_ratios.append(0)
                    displaced_at_stall.append(0)
                    break

        if stall_steps:
            print(f"{n:>4} {N:>5} {d:>3} {np.mean(stall_steps):>8.1f} "
                  f"{np.mean(stall_ratios):>17.4f} "
                  f"{np.mean(displaced_at_stall):>7.1f} "
                  f"{np.mean(phi0_values):>10.0f}")

    print()
    print("Paper claim: Phi_stall/Phi_0 -> ~0.17 for N >= 64.")


import pytest


@pytest.mark.parametrize("n,phi_ratio_bound", [
    (8, 0.5),    # N=64 (paper says ~0.17)
    (10, 0.5),   # N=100
    (12, 0.5),   # N=144
    (14, 0.5),   # N=196
    (16, 0.5),   # N=256 (the headline N for this test)
])
def test_greedy_stall_phi_ratio(n, phi_ratio_bound):
    N = n * n
    d = max(2, min(8, N - 1))
    if d % 2 == 1:
        d -= 1

    adjacency = random_regular_graph(N, d, seed=42 + n)
    if adjacency is None:
        pytest.skip("graph construction returned None")

    n_trials = 15
    max_steps = 500
    stall_ratios = []

    for trial in range(n_trials):
        np.random.seed(1000 + trial * 17 + n)
        target = np.random.permutation(N).tolist()
        positions = list(range(N))
        phi0 = displacement_energy(positions, target, n)

        for step in range(max_steps):
            matching, _ = greedy_displacement_matching(
                adjacency, positions, target, n)
            if not matching:
                phi_stall = displacement_energy(positions, target, n)
                stall_ratios.append(phi_stall / phi0 if phi0 else 0)
                break
            positions = apply_matching(positions, matching)
            if displacement_energy(positions, target, n) == 0:
                stall_ratios.append(0)
                break

    assert stall_ratios, f"n={n}: no stall recorded in any trial"
    mean_ratio = np.mean(stall_ratios)
    assert mean_ratio <= phi_ratio_bound, (
        f"n={n}, N={N}: mean Phi_stall/Phi_0 = {mean_ratio:.4f} > {phi_ratio_bound}")


if __name__ == "__main__":
    main()
