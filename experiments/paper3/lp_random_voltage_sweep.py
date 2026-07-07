#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
from numpy.linalg import svd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.qldpc.codes import lp_check_matrix_circulant

def biregular_3x5_mask():
    return np.array([
        [1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
    ], dtype=bool)


def random_voltages(M, L, rng):
    rows, cols = M.shape
    B = -np.ones((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if M[i, j]:
                B[i, j] = int(rng.integers(0, L))
    return B


def lift_spectrum_check(M, L, rng):
    B = random_voltages(M, L, rng)
    H_lift = lp_check_matrix_circulant(B, L)
    sigmas = svd(H_lift.astype(float), compute_uv=False)
    s1 = sigmas[0]
    s2 = sigmas[1] if len(sigmas) > 1 else 0.0
    beta_lift = s2 / s1 if s1 > 1e-10 else 1.0
    beta_lp = 0.5 + 0.5 * beta_lift

    # Feng-Li bipartite Ramanujan bound on the lifted classical Tanner graph
    col_w = H_lift.sum(axis=0).astype(float)
    row_w = H_lift.sum(axis=1).astype(float)
    c_eff = float(col_w.mean())
    r_eff = float(row_w.mean())
    feng_li = np.sqrt(max(c_eff - 1, 0)) + np.sqrt(max(r_eff - 1, 0))
    is_ramanujan = s2 <= feng_li + 1e-9

    return {
        'sigma_1': s1, 'sigma_2': s2,
        'beta_lift': beta_lift, 'beta_lp': beta_lp,
        'c_eff': c_eff, 'r_eff': r_eff,
        'feng_li_bound': feng_li, 'is_ramanujan': is_ramanujan,
    }


def main():
    M = biregular_3x5_mask()
    print(f"\n  Base 3x5 support mask:")
    for row in M.astype(int):
        print(f"    {row}")
    print(f"  Row weights: {M.sum(axis=1).tolist()}, "
          f"col weights: {M.sum(axis=0).tolist()}")
    print(f"  Total support: {int(M.sum())} nonzero entries")

    rng = np.random.default_rng(42)
    n_samples = 500

    print(f"\n  {'L':>4} {'N_LP':>7} {'frac Ram':>10} "
          f"{'min beta_lift':>14} {'mean beta_lift':>15} {'max beta_lift':>14} "
          f"{'time(s)':>9}")
    print(f"  {'-' * 80}")

    summary = []
    for L in [2, 3, 4, 5, 6, 8, 12, 16]:
        t0 = time.time()
        beta_lifts = []
        is_rams = []
        feng_li = None
        N_LP = (5 * L) ** 2 + (3 * L) ** 2  # HGP-of-lift formula
        for _ in range(n_samples):
            res = lift_spectrum_check(M, L, rng)
            beta_lifts.append(res['beta_lift'])
            is_rams.append(res['is_ramanujan'])
            feng_li = res['feng_li_bound']
        frac_ram = sum(is_rams) / len(is_rams)
        elapsed = time.time() - t0
        summary.append({
            'L': L, 'N_LP': N_LP, 'frac_ram': frac_ram,
            'min_beta': min(beta_lifts), 'mean_beta': float(np.mean(beta_lifts)),
            'max_beta': max(beta_lifts), 'feng_li': feng_li,
        })
        print(f"  {L:>4} {N_LP:>7} {frac_ram:>9.1%}  "
              f"{min(beta_lifts):>14.4f} {np.mean(beta_lifts):>15.4f} "
              f"{max(beta_lifts):>14.4f} {elapsed:>9.1f}")

    # Compute predicted asymptotic beta_lift
    M_arr = M.astype(int)
    col_w = M_arr.sum(axis=0).astype(float)
    row_w = M_arr.sum(axis=1).astype(float)
    c_eff = col_w.mean()
    r_eff = row_w.mean()
    feng_li = np.sqrt(c_eff - 1) + np.sqrt(r_eff - 1)
    # sigma_1 for biregular: sqrt(c * r) at large N
    sigma_1_pred = np.sqrt(c_eff * r_eff)
    beta_lift_pred = feng_li / sigma_1_pred
    beta_lp_pred = 0.5 + 0.5 * beta_lift_pred

    print(f"  Effective biregularity from base support: c = {c_eff:.2f}, r = {r_eff:.2f}")
    print(f"  Predicted Friedman/Feng-Li asymptotic:")
    print(f"    sigma_1 -> sqrt(c*r) = {sigma_1_pred:.4f}")
    print(f"    sigma_2 -> sqrt(c-1) + sqrt(r-1) = {feng_li:.4f}")
    print(f"    beta_lift -> {beta_lift_pred:.4f}")
    print(f"    beta_LP   -> (1 + beta_lift) / 2 = {beta_lp_pred:.4f}")

    print(f"\n  Empirical mean beta_lift across L values: "
          f"{np.mean([s['mean_beta'] for s in summary]):.4f}")
    print(f"  Asymptotic prediction:                         {beta_lift_pred:.4f}")

    print("\n" + "=" * 80)
    print("  Comparison to Paper I §7 (Fano plane covering tower)")
    print("=" * 80)
    print(f"\n  Paper I §7 result (for reference):")
    print(f"    Fano plane (N0=7, k=2):  ~94% Ramanujan over 128 voltage assigns")
    print(f"    Fano plane (N0=7, k=3):  ~80-90%  (random sample of 200)")
    print(f"    Fano plane (N0=7, k=5):  ~70-85%")
    print(f"    Fano plane (N0=7, k=7):  ~60-80%")
    print(f"\n  Paper III R2 result (3x5 base, this script):")
    print(f"    {'L':>3} {'frac Ramanujan':>18}")
    for s in summary:
        print(f"    {s['L']:>3} {s['frac_ram']:>17.1%}")


if __name__ == "__main__":
    main()
