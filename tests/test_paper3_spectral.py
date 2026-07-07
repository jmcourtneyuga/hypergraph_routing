#!/usr/bin/env python3
import numpy as np

from routing_lib.qldpc.codes import (
    random_3_4_bipartite_check_matrix,
    hgp_tanner_graph,
)
from routing_lib.qldpc.published_codes import (
    BRAVYI_BB_CODES,
    bb_check_matrices,
)

TOL = 1e-9

def _nontrivial_bipartite_ratio(A):
    ev = np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1]
    lam1 = ev[0]
    k = 1
    while k < len(ev) and abs(ev[k] - lam1) < 1e-6 * lam1:
        k += 1
    return ev[k] / lam1


def hgp_closed_form_cases(sizes=(12, 16, 20), seeds=(1, 2, 3)):
    out = []
    for n in sizes:
        for s in seeds:
            H = random_3_4_bipartite_check_matrix(n, seed=s)
            sv = np.linalg.svd(H, compute_uv=False)
            sv = sv[sv > TOL]
            if len(sv) < 2 or sv[0] - sv[1] < 1e-6:
                continue  # skip degenerate top singular value (hypothesis fails)
            beta_base = sv[1] / sv[0]
            A = hgp_tanner_graph(H, H)
            beta_meas = _nontrivial_bipartite_ratio(np.asarray(A, dtype=float))
            out.append((n, s, beta_base, (1 + beta_base) / 2, beta_meas))
    return out


def test_hgp_closed_form():
    cases = hgp_closed_form_cases()
    assert cases, "no non-degenerate HGP cases generated"
    for n, s, beta_base, pred, meas in cases:
        assert abs(pred - meas) < 1e-6, (
            f"n={n} seed={s}: predicted {pred:.6f} != measured {meas:.6f}")


def _bb_symbol_singular_values(l, m, A_poly, B_poly):
    svals = []
    for a in range(l):
        for b in range(m):
            Ahat = sum(np.exp(2j * np.pi * (i * a / l + j * b / m)) for (i, j) in A_poly)
            Bhat = sum(np.exp(2j * np.pi * (i * a / l + j * b / m)) for (i, j) in B_poly)
            M = np.array([[Ahat, Bhat], [np.conj(Bhat), np.conj(Ahat)]])
            svals.extend(np.linalg.svd(M, compute_uv=False).tolist())
    return np.sort(np.array(svals))[::-1]


def bb_reduction_cases():
    out = []
    for code in BRAVYI_BB_CODES:
        l, m = code["l"], code["m"]
        H_X, H_Z = bb_check_matrices(l, m, code["A"], code["B"])
        # full check matrix H = [[A, B],[B^T, A^T]] singular values
        H = np.vstack([H_X, H_Z]).astype(float)
        sv_full = np.sort(np.linalg.svd(H, compute_uv=False))[::-1]
        sv_reduced = _bb_symbol_singular_values(l, m, code["A"], code["B"])
        # compare the top 2*l*m values (both should have that many nonzeros max)
        k = min(len(sv_full), len(sv_reduced))
        out.append((code["label"], sv_full[:k], sv_reduced[:k]))
    return out


def test_bb_fourier_reduction():
    cases = bb_reduction_cases()
    assert cases, "no BB codes found"
    for label, sv_full, sv_reduced in cases:
        max_disc = float(np.max(np.abs(sv_full - sv_reduced)))
        assert max_disc < 1e-9, f"{label}: sv discrepancy {max_disc:.2e}"


if __name__ == "__main__":
    print("Theorem 3.2 -- HGP closed form beta_HGP = (1 + beta_base)/2")
    ok = True
    for n, s, bb, pred, meas in hgp_closed_form_cases():
        good = abs(pred - meas) < 1e-6
        ok &= good
        print(f"  n={n:2d} seed={s}  beta_base={bb:.4f}  pred={pred:.4f}  "
              f"measured={meas:.4f}  {'PASS' if good else 'FAIL'}")
    print("\nTheorem 3.5 -- BB Fourier reduction (lm 2x2 SVDs vs full check matrix)")
    for label, sv_full, sv_reduced in bb_reduction_cases():
        disc = float(np.max(np.abs(sv_full - sv_reduced)))
        good = disc < 1e-9
        ok &= good
        print(f"  {label:>14}  max sv discrepancy = {disc:.2e}  "
              f"{'PASS' if good else 'FAIL'}")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
