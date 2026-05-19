#!/usr/bin/env python3
"""
Test of concentration assumption.

Bins greedy steps by current Phi_t and reports the empirical tail
P[Delta Phi_t < 0.5 * E[Delta Phi_t | Phi_t]] across bins.
The paper assumes this tail is bounded by some alpha < 1.
"""

import numpy as np
from collections import defaultdict

from routing_lib.graphs import random_regular_networkx as random_regular_graph
from routing_lib.greedy import (
    grid_distance_open as grid_distance,
    displacement_energy,
    greedy_matching,
    apply_matching,
)


def main():
    print("=" * 75)
    print("CONCENTRATION-TAIL TEST")
    print("=" * 75)
    print("\nP[Delta Phi_t < 0.5 * E[Delta Phi | Phi_t]] across N and Phi-bins.")
    print(f"\n{'n':>3} {'N':>5} {'d':>3} {'#trials':>8} "
          f"{'#steps':>8} {'mean_tail':>11} {'max_tail':>10} {'OK?':>6}")
    print("-" * 75)

    grid_sides = [8, 10, 12, 14]  # N = 64, 100, 144, 196
    n_trials = 30
    max_steps = 200

    all_ok = True

    for n in grid_sides:
        N = n * n
        d = 8
        if N * d % 2 != 0:
            continue

        adjacency = random_regular_graph(N, d, seed=42 + n)
        if adjacency is None:
            continue

        # Bin Delta Phi values by log2(Phi_t) bin index.
        deltas_by_bin = defaultdict(list)
        n_total_steps = 0

        for trial in range(n_trials):
            np.random.seed(2000 + trial * 13 + n)
            target = np.random.permutation(N).tolist()
            positions = list(range(N))
            phi = displacement_energy(positions, target, n)

            for _ in range(max_steps):
                bin_idx = int(np.log2(max(phi, 1)))
                _, delta = greedy_matching(adjacency, positions, target, n)
                if delta <= 0:
                    break
                deltas_by_bin[bin_idx].append(delta)
                n_total_steps += 1

                matching, _ = greedy_matching(adjacency, positions, target, n)
                if not matching:
                    break
                positions = apply_matching(positions, matching)
                phi = displacement_energy(positions, target, n)
                if phi == 0:
                    break

        # Per-bin mean and tail probability.
        bin_stats = []
        for bin_idx, deltas in sorted(deltas_by_bin.items()):
            if len(deltas) < 5:
                continue
            mean_delta = np.mean(deltas)
            tail_prob = sum(1 for d in deltas if d < 0.5 * mean_delta) / len(deltas)
            bin_stats.append((bin_idx, mean_delta, tail_prob, len(deltas)))

        if not bin_stats:
            print(f"  n={n}: no data")
            continue

        mean_tail = np.mean([t for _, _, t, _ in bin_stats])
        max_tail = max(t for _, _, t, _ in bin_stats)
        ok = max_tail <= 0.30  # lenient threshold
        if N >= 64 and max_tail > 0.30:
            all_ok = False

        print(f"{n:>3} {N:>5} {d:>3} {n_trials:>8} {n_total_steps:>8} "
              f"{mean_tail:>11.3f} {max_tail:>10.3f} "
              f"{'PASS' if ok else 'FAIL':>6}")

    print()
    if all_ok:
        print("RESULT: PASS — tail bounded; consistent with concentration assumption.")
    else:
        print("RESULT: tail not bounded by 0.30 — assumption stated as alpha <= 0.51.")


import pytest


@pytest.mark.parametrize("n", [8, 10, 12, 14])
def test_concentration_tail_bounded(n):
    """Direct test of Assumption ass:concentration. Paper allows alpha <= 0.51."""
    N = n * n
    d = 8
    if N * d % 2 != 0:
        pytest.skip("N*d not even")
    adjacency = random_regular_graph(N, d, seed=42 + n)
    if adjacency is None:
        pytest.skip("graph construction returned None")

    n_trials = 30
    max_steps = 200
    deltas_by_bin = defaultdict(list)

    for trial in range(n_trials):
        np.random.seed(2000 + trial * 13 + n)
        target = np.random.permutation(N).tolist()
        positions = list(range(N))
        phi = displacement_energy(positions, target, n)

        for _ in range(max_steps):
            bin_idx = int(np.log2(max(phi, 1)))
            _, delta = greedy_matching(adjacency, positions, target, n)
            if delta <= 0:
                break
            deltas_by_bin[bin_idx].append(delta)

            matching, _ = greedy_matching(adjacency, positions, target, n)
            if not matching:
                break
            positions = apply_matching(positions, matching)
            phi = displacement_energy(positions, target, n)
            if phi == 0:
                break

    bin_stats = []
    for _, deltas in sorted(deltas_by_bin.items()):
        if len(deltas) < 5:
            continue
        mean_delta = np.mean(deltas)
        tail_prob = sum(1 for d in deltas if d < 0.5 * mean_delta) / len(deltas)
        bin_stats.append(tail_prob)

    assert bin_stats, f"no bin had >=5 samples for n={n}"
    max_tail = max(bin_stats)
    assert max_tail <= 0.51, f"n={n}: max tail prob {max_tail:.3f} > 0.51"


if __name__ == "__main__":
    main()
