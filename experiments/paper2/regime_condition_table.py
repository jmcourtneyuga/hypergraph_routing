#!/usr/bin/env python3
"""
Recompute Remark 4.6 (rem:regime-practicality) of paper2.

The regime condition is d'_Q > beta * d', where d'_Q >= d' - delta_intra
and delta_intra is the maximum per-vertex intra-block degree.

Old bound (paper2 V1):  delta_intra <= s(r-1) = d_C^2 (r-1)
New bound (Issue 2):    delta_intra <= O(d_C) for surface-code patches with
                        bounded aspect ratio (Remark 2.3). We test two flavors:
                        (a) coefficient 1: delta_intra <= d_C
                        (b) coefficient 4: delta_intra <= 4 d_C
                            (4 nearest-neighbors per atom on the embedded grid)

Ramanujan saturation:  beta = 2*sqrt(d'-1)/d'  (the worst case allowed).

For each d_C, we find the minimum integer base degree d (with r = 3, so
d' = 2d) such that d' (1 - beta) > delta_intra.
"""

from math import sqrt, ceil


def ramanujan_beta(d_prime):
    if d_prime <= 1:
        return 1.0
    return 2.0 * sqrt(d_prime - 1) / d_prime


def min_d_satisfying(d_C, delta_intra_fn, r=3):
    """Smallest integer d such that d'(1 - beta(d')) > delta_intra(d_C),
    where d' = d*(r-1) and beta is Ramanujan-saturating."""
    delta = delta_intra_fn(d_C)
    for d in range(2, 5000):
        d_prime = d * (r - 1)
        beta = ramanujan_beta(d_prime)
        if d_prime * (1.0 - beta) > delta:
            return d, d_prime, beta, delta
    return None


def n_phys_estimate(d_C, d, n_logical_min=8):
    """Order-of-magnitude N_phys for at least n_logical_min logical qubits."""
    n_phys = max(n_logical_min * d_C * d_C, d * 4)  # need enough vertices
    # round up to nearest power of 10 for the table column
    if n_phys < 200:
        return r"$\sim 10^2$"
    if n_phys < 2000:
        return r"$\sim 10^3$"
    if n_phys < 20000:
        return r"$\sim 10^4$"
    return r"$\sim 10^5$"


def make_table(name, delta_intra_fn):
    print(f"\n{name}")
    print("=" * len(name))
    print(f"{'d_C':>4}  {'min d':>6}  {'d_prime':>8}  {'beta':>7}  "
          f"{'delta':>7}  {'1-beta':>7}  N_phys")
    rows = []
    for d_C in [3, 5, 7, 9]:
        result = min_d_satisfying(d_C, delta_intra_fn)
        if result is None:
            print(f"{d_C:>4}  no solution")
            continue
        d, d_prime, beta, delta = result
        nphys = n_phys_estimate(d_C, d)
        print(f"{d_C:>4}  {d:>6}  {d_prime:>8}  {beta:>7.4f}  "
              f"{delta:>7.1f}  {1-beta:>7.4f}  {nphys}")
        rows.append((d_C, d, d_prime, beta, delta, nphys))
    return rows


def latex_row(d_C, d, d_prime, n_phys_str):
    return f"{d_C} & {d} & {d_prime} & {n_phys_str} \\\\"


def main():
    print("Recomputing Remark 4.6 (rem:regime-practicality) for paper2.")
    print("All tables use r=3, so d' = 2d, with Ramanujan-saturating beta.")
    print("Threshold: d'*(1-beta) > delta_intra(d_C).")

    # Old bound (paper2 V1, line 313-328 of paper2_05.02.2026.tex)
    old = make_table(
        "Old bound:  delta_intra = s*(r-1) = 2*d_C^2",
        lambda d_C: 2 * d_C * d_C,
    )

    # Issue 2 fix, conservative coefficient 1 -- using the paper's own
    # Remark 2.3 wording "intra-block degree O(d_C)".
    new1 = make_table(
        "New bound (Issue 2, c=1):  delta_intra = (r-1)*d_C = 2*d_C",
        lambda d_C: 2 * d_C,
    )

    # Issue 2 fix, surface-code-tight: 2D grid patch has 4 nearest neighbors
    # per atom, regardless of d_C.  Constant in d_C.
    new4 = make_table(
        "New bound (surface-code tight):  delta_intra = 4 (constant)",
        lambda d_C: 4,
    )

    print()
    print("=" * 70)
    print("LaTeX rows (paste into Remark 4.6 of paper2):")
    print("=" * 70)
    print()
    print("Old bound (current table in V1, included for comparison):")
    for d_C, d, d_prime, beta, delta, nphys in old:
        print("  " + latex_row(d_C, d, d_prime, nphys))
    print()
    print("New bound (Issue 2 fix, conservative c=1):")
    for d_C, d, d_prime, beta, delta, nphys in new1:
        print("  " + latex_row(d_C, d, d_prime, nphys))
    print()
    print("New bound (surface-code tight, delta_intra = 4):")
    for d_C, d, d_prime, beta, delta, nphys in new4:
        print("  " + latex_row(d_C, d, d_prime, nphys))


if __name__ == "__main__":
    main()
