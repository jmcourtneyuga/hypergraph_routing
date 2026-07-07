#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
from numpy.linalg import eigvalsh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.qldpc.published_codes import (
    BRAVYI_BB_CODES,
    get_bravyi_bb_tanner,
    bb_check_matrices,
)
from experiments.paper3.konig_optimal_test import bipartite_konig_coloring

def spectral_ratio_bipartite(A):
    eigs = np.sort(eigvalsh(A))[::-1]
    lam1 = eigs[0]
    nontrivial = eigs[1:-1]
    if len(nontrivial) == 0:
        return lam1, 0.0, 0.0
    lam2_nt = max(abs(nontrivial[0]), abs(nontrivial[-1]))
    beta = lam2_nt / lam1 if lam1 > 0 else 1.0
    return float(lam1), float(lam2_nt), float(beta)


def base_polynomial_spectrum(A_or_B_matrix):
    from numpy.linalg import svd
    sigmas = svd(A_or_B_matrix.astype(float), compute_uv=False)
    s1 = float(sigmas[0])
    s2 = float(sigmas[1]) if len(sigmas) > 1 else 0.0
    beta_base = s2 / s1 if s1 > 1e-10 else 1.0
    return s1, s2, beta_base


def main():
    print("DIRECT MEASUREMENT TABLE")
    print("=" * 100)
    print(f"  {'Code':>14}  {'N_total':>8}  {'Delta':>5}  {'chi'' (Konig)':>14}  "
          f"{'lambda_1':>9}  {'lambda_2nt':>10}  {'beta_BB':>8}  {'time(s)':>8}")
    print(f"  {'-' * 96}")

    rows = []
    for spec in BRAVYI_BB_CODES:
        t0 = time.time()
        A, H_X, H_Z = get_bravyi_bb_tanner(spec)
        N_total = A.shape[0]
        degrees = A.sum(axis=1).astype(int)
        delta = int(degrees.max())
        _, n_colors, delta_check = bipartite_konig_coloring(A)
        assert n_colors == delta, (
            f"Konig non-tight for {spec['label']}: chi'={n_colors}, "
            f"Delta={delta}"
        )
        lam1, lam2nt, beta_bb = spectral_ratio_bipartite(A)
        elapsed = time.time() - t0
        rows.append({
            "label": spec["label"], "N": N_total, "delta": delta,
            "chi_prime": n_colors, "lam1": lam1, "lam2nt": lam2nt,
            "beta_BB": beta_bb,
        })
        print(f"  {spec['label']:>14}  {N_total:>8}  {delta:>5}  "
              f"{n_colors:>14}  {lam1:>9.4f}  {lam2nt:>10.4f}  "
              f"{beta_bb:>8.4f}  {elapsed:>8.2f}")

    print()
    print("PER-CYCLE STEP COUNT vs L_layers (= ceil(chi'/L_layers))")
    print(f"  {'Code':>14}  {'chi''':>5}  {'L=1':>4}  {'L=2':>4}  {'L=4':>4}  "
          f"{'L=8':>4}  {'L=16':>5}  {'cap (saturated)':>17}")
    print(f"  {'-' * 60}")
    for r in rows:
        chi = r["chi_prime"]
        pc = [int(np.ceil(chi / L)) for L in (1, 2, 4, 8, 16)]
        print(f"  {r['label']:>14}  {chi:>5}  {pc[0]:>4}  {pc[1]:>4}  "
              f"{pc[2]:>4}  {pc[3]:>4}  {pc[4]:>5}  "
              f"L >= {chi} -> 1")

    print()
    print("=" * 100)
    print("COMPARISON TO HGP / LP (Paper III Tables 5.4, 5.5)")
    print("=" * 100)
    print()
    print(f"  HGP[H,H] (3,4)-biregular base: chi' = 2*delta_c = 8")
    print(f"  LP[H_lift,H_lift] (3,4)-biregular base: chi' = 2*delta_c = 8")
    print(f"  LP[H_lift,H_lift] (3,5)-biregular base: chi' = 2*delta_c = 10")
    print(f"  Bravyi BB codes (this script):")
    for r in rows:
        print(f"     {r['label']:>14}: chi' = {r['chi_prime']}")

    print(f"  {'Code':>14}  {'beta_BB':>8}  {'beta_A':>8}  {'beta_B':>8}  "
          f"{'(1+max(beta_A,beta_B))/2':>26}  {'match?':>8}")
    print(f"  {'-' * 80}")
    for spec in BRAVYI_BB_CODES:
        H_X, H_Z = bb_check_matrices(spec["l"], spec["m"],
                                       spec["A"], spec["B"])
        n_block = spec["l"] * spec["m"]
        A_mat = H_X[:, :n_block]
        B_mat = H_X[:, n_block:]
        _, _, beta_A = base_polynomial_spectrum(A_mat)
        _, _, beta_B = base_polynomial_spectrum(B_mat)
        row = next(r for r in rows if r["label"] == spec["label"])
        beta_BB = row["beta_BB"]
        hgp_like = (1.0 + max(beta_A, beta_B)) / 2.0
        match = "yes" if abs(beta_BB - hgp_like) < 0.01 else "no"
        diff_pct = 100.0 * abs(beta_BB - hgp_like) / beta_BB if beta_BB > 0 else 0.0
        print(f"  {spec['label']:>14}  {beta_BB:>8.4f}  {beta_A:>8.4f}  "
              f"{beta_B:>8.4f}  {hgp_like:>26.4f}  "
              f"{match:>5} ({diff_pct:>4.1f}%)")
    print()
    print("  LaTeX:")
    for r in rows:
        chi = r["chi_prime"]
        pc_L8 = int(np.ceil(chi / 8))
        pc_L16 = int(np.ceil(chi / 16))
        print(f"  BB {r['label']:<14} & {r['N']:>4} & {chi:>2} & "
              f"{pc_L8:>2} & {pc_L16:>2} & "
              f"$\\beta = {r['beta_BB']:.3f}$ \\\\")


if __name__ == "__main__":
    main()
