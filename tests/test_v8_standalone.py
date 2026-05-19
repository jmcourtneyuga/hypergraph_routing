#!/usr/bin/env python3
"""
Standalone V8 test: Koenig edge-coloring of the block-translation
bipartite multigraph. Validates Lemma 5.5 / 5.5b.
"""

import numpy as np

from routing_lib.routing import (
    koenig_edge_color_simple as koenig_edge_color,
    chromatic_index_block_translation,
    chromatic_index_corridor,
)


def _legacy_test_v8():
    print("=" * 70)
    print("V8 STANDALONE TEST: Koenig edge-coloring of block translation")
    print("=" * 70)

    # Direct (parallel + random-bijection) cases
    print(f"\n{'d_C':<6} {'n_atoms':<10} {'parallel':<10} "
          f"{'random_mean':<14} {'random_max':<12} {'OK?':<6}")
    print("-" * 64)

    all_ok = True
    for d_C in [3, 5, 7, 9]:
        r = chromatic_index_block_translation(d_C)
        ok = r['n_colors_parallel'] == 1 and r['max_n_colors_random'] <= d_C
        all_ok &= ok
        print(f"{d_C:<6} {r['n_atoms']:<10} {r['n_colors_parallel']:<10} "
              f"{r['mean_n_colors_random']:<14.1f} "
              f"{r['max_n_colors_random']:<12} {'PASS' if ok else 'FAIL':<6}")

    # Pessimistic corridor case
    print("\nPessimistic case: random corridor paths (multigraph, shared edges)")
    print("-" * 64)
    print(f"{'d_C':<6} {'path_len':<10} {'max_deg':<10} {'chromatic':<12} {'<=2*d_C?':<8}")
    print("-" * 64)

    pessimistic_ok = True
    for d_C in [3, 5, 7, 9]:
        max_deg = chromatic_index_corridor(d_C)
        ok = max_deg <= 2 * d_C  # factor-2 slack for the random simulation
        pessimistic_ok &= ok
        print(f"{d_C:<6} {d_C:<10} {max_deg:<10} {max_deg:<12} "
              f"{'YES' if ok else 'NO':<8}")

    print()
    if all_ok and pessimistic_ok:
        print("V8 OVERALL: PASS")
        print("Lemma 5.5 (regular host): chromatic index = 1 -> O(1) per quotient step.")
        print("Lemma 5.5b (general host): chromatic index <= O(d_C).")
    else:
        print("V8 OVERALL: FAIL")

    return all_ok and pessimistic_ok


import pytest


@pytest.mark.parametrize("d_C", [3, 5, 7, 9])
def test_chromatic_index_parallel_translation(d_C):
    """Lemma 5.5: parallel translation needs exactly 1 physical step."""
    r = chromatic_index_block_translation(d_C)
    assert r['n_colors_parallel'] == 1, (
        f"parallel translation at d_C={d_C} requires "
        f"{r['n_colors_parallel']} colors, expected 1")


@pytest.mark.parametrize("d_C", [3, 5, 7, 9])
def test_chromatic_index_random_translation(d_C):
    """Lemma 5.5b: near-rigid (random) translation needs <= d_C physical steps."""
    r = chromatic_index_block_translation(d_C)
    assert r['max_n_colors_random'] <= d_C, (
        f"random translation at d_C={d_C} requires "
        f"{r['max_n_colors_random']} colors, expected <= {d_C}")


@pytest.mark.parametrize("d_C", [3, 5, 7, 9])
def test_chromatic_index_corridor_pessimistic(d_C):
    """Pessimistic corridor case: chromatic index <= 2*d_C."""
    max_deg = chromatic_index_corridor(d_C)
    assert max_deg <= 2 * d_C, (
        f"corridor at d_C={d_C}: max_deg={max_deg} > 2*{d_C}")


if __name__ == "__main__":
    _legacy_test_v8()
