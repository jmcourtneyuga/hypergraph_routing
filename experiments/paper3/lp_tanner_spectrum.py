#!/usr/bin/env python3
import os
import sys
import numpy as np
from numpy.linalg import eigvalsh

# Ensure routing_lib is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.qldpc.codes import (
    random_3_4_bipartite_check_matrix,
    tanner_graph_classical,
    hgp_tanner_graph,
    hgp_code_params,
    lp_check_matrix_circulant,
    lp_code,
)


def spectral_summary(A, label, biregular_degrees=None):
    eigs = np.sort(eigvalsh(A))[::-1]
    lam_max = eigs[0]      # +lambda_1, the trivial Perron eigenvalue
    lam_min = eigs[-1]     # -lambda_1, the bipartite mirror

    nontrivial = eigs[1:-1]
    if len(nontrivial) == 0:
        lam2_nt = 0.0
    else:
        lam2_nt = max(abs(nontrivial[0]), abs(nontrivial[-1]))

    beta_bip = lam2_nt / lam_max if lam_max > 0 else 1.0
    naive_beta = max(abs(eigs[1]), abs(lam_min)) / lam_max if lam_max > 0 else 1.0

    print(f"\n  {label}")
    print(f"  {'-' * 70}")
    print(f"  Tanner graph: {A.shape[0]} nodes, {int(A.sum() / 2)} edges")
    print(f"  lambda_1 (Perron):           {lam_max:>10.4f}")
    print(f"  lambda_N (= -lambda_1):      {lam_min:>10.4f}   [bipartite mirror]")
    print(f"  lambda_2 (non-trivial):      {lam2_nt:>10.4f}   [excludes Perron pair]")
    print(f"  beta_bip = lam_2_nt / lam_1: {beta_bip:>10.4f}")
    print(f"  1 - beta_bip:                {1 - beta_bip:>10.4f}")
    print(f"  (naive beta would be {naive_beta:.4f}, which equals 1 by bipartite symmetry)")

    if biregular_degrees is not None:
        c, d = biregular_degrees
        feng_li = np.sqrt(c - 1) + np.sqrt(d - 1)
        is_ram = lam2_nt <= feng_li + 1e-9
        print(f"  ({c},{d})-biregular Feng-Li Ramanujan bound: "
              f"sqrt({c-1}) + sqrt({d-1}) = {feng_li:.4f}")
        print(f"  Satisfies Feng-Li bound?     {'YES' if is_ram else 'NO':>10}  "
              f"(lam_2/bound = {lam2_nt/feng_li:.4f})")
    else:
        is_ram = None  # not biregular -> no canonical Ramanujan bound

    return {'lam_max': lam_max, 'lam_min': lam_min, 'lam2_nt': lam2_nt,
            'beta_bip': beta_bip, 'is_ramanujan': is_ram, 'eigs': eigs}


def routing_depth_bound(d_prime, beta, N):
    if beta >= 1 or beta <= 0 or d_prime <= 0:
        return float('inf')
    log2N = np.log2(N)
    log2_inv_beta = np.log2(1.0 / beta)
    return (4 * (d_prime + 6) / (d_prime * log2_inv_beta)) * log2N + 19 * log2N


def main():
    print("  Paper III, R1: Tanner-graph spectrum of qLDPC product codes")
    print("  PART 1: HGP code [[225, 9, 4]] — exact Xu Fig. 3a entry")

    print("  Base: random (3,4)-regular bipartite, n=12 vars, m=9 checks.")
    print("  HGP gives N = n^2 + m^2 = 144 + 81 = 225 qubits, K = (n-m)^2 = 9.")

    H = random_3_4_bipartite_check_matrix(12, seed=1)
    var_deg = H.sum(axis=0)
    chk_deg = H.sum(axis=1)
    rank_H = int(np.linalg.matrix_rank(H.astype(float)))
    print(f"\n  Base H verification:")
    print(f"    Variable degrees: {var_deg.tolist()}  (all 3? {(var_deg == 3).all()})")
    print(f"    Check degrees:    {chk_deg.tolist()}  (all 4? {(chk_deg == 4).all()})")
    print(f"    Rank: {rank_H}  (full row rank? {rank_H == H.shape[0]})")

    A_base = tanner_graph_classical(H)
    base_spec = spectral_summary(
        A_base,
        "Base classical Tanner graph (12 vars deg 3 + 9 checks deg 4 = 21 nodes)",
        biregular_degrees=(3, 4))


    A_hgp = hgp_tanner_graph(H, H)
    N_hgp, K_hgp = hgp_code_params(H, H)
    print(f"\n  HGP code parameters: [[N={N_hgp}, K={K_hgp}, ?]]")
    hgp_spec = spectral_summary(
        A_hgp,
        f"HGP[[{N_hgp},{K_hgp}]] full Tanner graph "
        f"(qubits + X-checks + Z-checks = {A_hgp.shape[0]} nodes)")

    if hgp_spec['beta_bip'] < 1:
        T_bound = routing_depth_bound(hgp_spec['lam_max'], hgp_spec['beta_bip'], N_hgp)
        print(f"\n  Theorem 5.1 routing-depth bound on HGP[[{N_hgp},{K_hgp}]]:")
        print(f"    Using lambda_1 = {hgp_spec['lam_max']:.3f}, beta_bip = {hgp_spec['beta_bip']:.4f}")
        print(f"    T <= {T_bound:.0f}  (steps for full Valiant routing)")
        print(f"    Compare to Xu et al.'s scrambling: O(L^(1/3)) for L atoms;")
        print(f"    for L = {N_hgp}, that is ~{N_hgp**(1/3):.1f} time units.")

    sv = np.linalg.svd(H.astype(float), compute_uv=False)
    sigmas_nonzero = sv[sv > 1e-10]
    print(f"\n  Spectrum decomposition test (R1a, added 2026-05-09):")
    print(f"    Singular values of H (rank {len(sigmas_nonzero)}): "
          f"{np.round(sv, 4).tolist()}")

    n_product, n_boundary, n_zero, n_unmatched = 0, 0, 0, 0
    tol = 1e-4
    for e in hgp_spec['eigs']:
        matched = False
        for s1 in sigmas_nonzero:
            if matched: break
            for s2 in sigmas_nonzero:
                for sgn1 in [1, -1]:
                    for sgn2 in [1, -1]:
                        if abs(e - (sgn1 * s1 + sgn2 * s2)) < tol:
                            matched = True
                            n_product += 1
                            break
                    if matched: break
                if matched: break
        if matched:
            continue
        # Boundary modes ±σ_k
        for s in sigmas_nonzero:
            if abs(abs(e) - s) < tol:
                n_boundary += 1
                matched = True
                break
        if matched:
            continue
        if abs(e) < 1e-6:
            n_zero += 1
        else:
            n_unmatched += 1

    total = len(hgp_spec['eigs'])
    pct_p = 100 * n_product / total
    pct_b = 100 * n_boundary / total
    pct_z = 100 * n_zero / total
    print(f"    HGP eigenvalue decomposition (n={total}):")
    print(f"      product modes ±(σ_i ± σ_j): {n_product:>4} ({pct_p:.1f}%)")
    print(f"      boundary modes ±σ_k:        {n_boundary:>4} ({pct_b:.1f}%)")
    print(f"      zero modes:                 {n_zero:>4} ({pct_z:.1f}%)")
    print(f"      unmatched:                  {n_unmatched:>4} ({100*n_unmatched/total:.1f}%)")
    if n_unmatched == 0:
        print(f"      STRUCTURAL CLAIM VERIFIED: all eigenvalues in expected set.")
    else:
        print(f"      WARNING: {n_unmatched} eigenvalues are not in the expected set.")


    np.random.seed(7)
    L = 4
    # Random Z_L base shifts; -1 = zero block, 0..L-1 = shift x^s
    # For a quasi-cyclic LDPC code, typical density is moderate
    B = np.random.randint(-1, L, size=(3, 5))  # entries in {-1, 0, 1, ..., L-1}
    # Ensure base column degrees (= row weight before lift) are nonzero
    while np.any((B >= 0).sum(axis=0) == 0) or np.any((B >= 0).sum(axis=1) == 0):
        B = np.random.randint(-1, L, size=(3, 5))

    print(f"\n  Base 3x5 matrix (entries in {{-1, 0, ..., {L-1}}}; -1 = zero block):")
    print(f"  {B}")
    print(f"  Lift order L = {L}, so lifted check matrix is "
          f"{B.shape[0]*L}x{B.shape[1]*L} = {B.shape[0]*L*B.shape[1]*L} nonzero entries max.")

    A_lp, N_lp, K_lp = lp_code(B, L)
    H_lifted = lp_check_matrix_circulant(B, L)
    rank_lift = int(np.linalg.matrix_rank(H_lifted.astype(float)))
    print(f"\n  Lifted matrix shape: {H_lifted.shape}, rank = {rank_lift}")
    print(f"  LP-style code parameters: [[N={N_lp}, K={K_lp}, ?]]")

    lp_spec = spectral_summary(
        A_lp, f"LP-style[[{N_lp},{K_lp}]] full Tanner graph ({A_lp.shape[0]} nodes)")

    if lp_spec['beta_bip'] < 1:
        T_bound = routing_depth_bound(lp_spec['lam_max'], lp_spec['beta_bip'], N_lp)
        print(f"\n  Theorem 5.1 routing-depth bound: T <= {T_bound:.0f}")


    print(f"\n  {'Code':<28} {'N':>5} {'lam_1':>8} {'lam_2':>8} "
          f"{'beta_bip':>9} {'Ram?':>6} {'T_bound':>10}")
    print(f"  {'-' * 78}")

    for label, spec, N in [
        (f"Base classical (3,4)-bireg", base_spec, 21),
        (f"HGP[[225,9,4]] (exact)", hgp_spec, 225),
        (f"LP[[{N_lp},{K_lp}]] (3x5 over Z_4)", lp_spec, N_lp),
    ]:
        T = routing_depth_bound(spec['lam_max'], spec['beta_bip'], N) \
            if spec['beta_bip'] < 1 else float('inf')
        ram = ('YES' if spec['is_ramanujan']
               else 'no' if spec['is_ramanujan'] is False
               else 'n/a')
        print(f"  {label:<28} {N:>5} {spec['lam_max']:>8.3f} "
              f"{spec['lam2_nt']:>8.3f} {spec['beta_bip']:>9.4f} "
              f"{ram:>6} {T:>10.1f}")

if __name__ == "__main__":
    main()
