#!/usr/bin/env python3
from math import sqrt, ceil


def ramanujan_beta(d_prime):
    if d_prime <= 1:
        return 1.0
    return 2.0 * sqrt(d_prime - 1) / d_prime

def min_d_satisfying(d_C, delta_intra_fn, r=3):
    delta = delta_intra_fn(d_C)
    for d in range(2, 5000):
        d_prime = d * (r - 1)
        beta = ramanujan_beta(d_prime)
        if d_prime * (1.0 - beta) > delta:
            return d, d_prime, beta, delta
    return None


def n_phys_estimate(d_C, d, n_logical_min=8):
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

    old = make_table(
        "Old bound:  delta_intra = s*(r-1) = 2*d_C^2",
        lambda d_C: 2 * d_C * d_C,
    )

    new1 = make_table(
        "New bound (Issue 2, c=1):  delta_intra = (r-1)*d_C = 2*d_C",
        lambda d_C: 2 * d_C,
    )

    new4 = make_table(
        "New bound (surface-code tight):  delta_intra = 4 (constant)",
        lambda d_C: 4,
    )


if __name__ == "__main__":
    main()
