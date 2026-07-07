#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
from numpy.linalg import eigvalsh, svd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.qldpc.published_codes import (
    BRAVYI_BB_CODES,
    bb_check_matrices,
    bb_tanner_graph,
)

RNG = np.random.default_rng(20260617)

def tanner_spectrum_direct(H_X, H_Z):
    A = bb_tanner_graph(H_X, H_Z)
    return np.sort(eigvalsh(A))[::-1], A


def spectral_ratio(sorted_eigs, tol=1e-9):
    lam1 = sorted_eigs[0]
    if lam1 <= tol:
        return 0.0, lam1, 0.0
    mags = np.abs(sorted_eigs)
    below = mags[mags < lam1 - tol]
    lam2 = float(below.max()) if below.size else 0.0
    return lam2 / lam1, float(lam1), lam2

def poly_fourier(poly, l, m, a, b):
    val = 0.0 + 0.0j
    for (i_pow, j_pow) in poly:
        val += np.exp(2j * np.pi * (i_pow * a / l + j_pow * b / m))
    return val


def tanner_spectrum_fourier(l, m, A_poly, B_poly):
    singular = []
    for a in range(l):
        for b in range(m):
            ah = poly_fourier(A_poly, l, m, a, b)
            bh = poly_fourier(B_poly, l, m, a, b)
            M = np.array([[ah, bh],
                          [np.conj(bh), np.conj(ah)]], dtype=complex)
            s = svd(M, compute_uv=False)  # two non-negative singular values
            singular.extend(s.tolist())
    singular = np.array(sorted(singular, reverse=True))
    eigs = np.concatenate([singular, -singular[::-1]])
    return np.sort(eigs)[::-1]


def block_beta_fourier(poly, l, m, tol=1e-9):
    vals = np.array([abs(poly_fourier(poly, l, m, a, b))
                     for a in range(l) for b in range(m)])
    vals = np.sort(vals)[::-1]
    s1 = vals[0]
    if s1 <= tol:
        return 0.0
    below = vals[vals < s1 - tol]
    s2 = below[0] if below.size else 0.0
    return float(s2 / s1)

def gf2_rank(M):
    M = (M.copy() % 2).astype(np.uint8)
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def bb_logical_qubits(H_X, H_Z):
    n_q = H_X.shape[1]
    return n_q - gf2_rank(H_X) - gf2_rank(H_Z)

def random_poly(l, m, weight, rng):
    seen = set()
    while len(seen) < weight:
        i, j = int(rng.integers(0, l)), int(rng.integers(0, m))
        seen.add((i, j))
    return [list(t) for t in seen]


def build_sweep():
    cases = []
    for spec in BRAVYI_BB_CODES:
        cases.append({"label": spec["label"], "l": spec["l"], "m": spec["m"],
                      "A": spec["A"], "B": spec["B"], "family": "Bravyi"})
    for (l, m) in [(6, 6), (9, 6), (12, 6), (12, 12), (15, 3), (10, 10)]:
        for k in range(6):
            cases.append({"label": f"rand-w3 l{l}m{m} #{k}", "l": l, "m": m,
                          "A": random_poly(l, m, 3, RNG),
                          "B": random_poly(l, m, 3, RNG), "family": "rand-w3"})
    # Other weights: weight-2 (Delta=4) and weight-4 (Delta=8).
    for (l, m) in [(8, 8), (12, 6)]:
        for w in (2, 4):
            for k in range(3):
                cases.append({"label": f"rand-w{w} l{l}m{m} #{k}",
                              "l": l, "m": m,
                              "A": random_poly(l, m, w, RNG),
                              "B": random_poly(l, m, w, RNG),
                              "family": f"rand-w{w}"})
    return cases


def main():
    cases = build_sweep()

    print("beta_BB FEASIBILITY PROBE -- is the BB Tanner spectrum reducible / beta_BB closed-form?")

    print(f"Sweep size: {len(cases)} BB codes "
          f"({sum(c['family']=='Bravyi' for c in cases)} published, "
          f"{sum(c['family']!='Bravyi' for c in cases)} random)\n")


    print("H1  Fourier per-frequency 2x2 SVD predictor  vs  direct Tanner diagonalization")

    print(f"  {'code':>22}  {'l':>3} {'m':>3} {'wt':>3}  {'Delta':>5}  {'K':>4}  "
          f"{'N_Tanner':>9}  {'max|spec_dir - spec_fourier|':>28}  {'beta_dir':>9}  {'beta_fft':>9}")


    h1_max_dev = 0.0
    rows = []
    for c in cases:
        H_X, H_Z = bb_check_matrices(c["l"], c["m"], c["A"], c["B"])
        wt = len(c["A"])
        A_adj = bb_tanner_graph(H_X, H_Z)
        delta = int(A_adj.sum(axis=1).max())
        spec_dir = np.sort(eigvalsh(A_adj))[::-1]
        spec_fft = tanner_spectrum_fourier(c["l"], c["m"], c["A"], c["B"])
        n = min(len(spec_dir), len(spec_fft))
        dev = float(np.max(np.abs(spec_dir[:n] - spec_fft[:n])))
        h1_max_dev = max(h1_max_dev, dev)
        beta_dir, lam1, _ = spectral_ratio(spec_dir)
        beta_fft, _, _ = spectral_ratio(spec_fft)
        K = bb_logical_qubits(H_X, H_Z)
        rows.append({**c, "wt": wt, "delta": delta, "K": K,
                     "beta_dir": beta_dir, "beta_fft": beta_fft, "dev": dev})
        print(f"  {c['label']:>22}  {c['l']:>3} {c['m']:>3} {wt:>3}  {delta:>5}  {K:>4}  "
              f"{len(spec_dir):>9}  {dev:>28.2e}  {beta_dir:>9.5f}  {beta_fft:>9.5f}")

    print()
    print(f"  ==> H1 max spectral deviation over the whole sweep: {h1_max_dev:.3e}")
    print(f"      {'PASS' if h1_max_dev < 1e-8 else 'FAIL'}: the Fourier 2x2-SVD reduction "
          f"{'reproduces' if h1_max_dev < 1e-8 else 'does NOT reproduce'} the full Tanner spectrum.")
    print()

    print("H2  Does beta_BB = (1 + max(beta_A, beta_B)) / 2  (literal HGP formula transplant)?")

    print(f"  {'code':>22}  {'beta_BB':>9}  {'beta_A':>8}  {'beta_B':>8}  "
          f"{'(1+max)/2':>10}  {'abs err':>9}  {'match<1%?':>9}")
    print(f"  {'-'*92}")
    h2_matches = 0
    h2_total = 0
    for r in rows:
        if r["K"] == 0:
            continue  # skip trivial codes
        beta_A = block_beta_fourier(r["A"], r["l"], r["m"])
        beta_B = block_beta_fourier(r["B"], r["l"], r["m"])
        pred = (1.0 + max(beta_A, beta_B)) / 2.0
        err = abs(r["beta_dir"] - pred)
        ok = err < 0.01 * max(r["beta_dir"], 1e-9)
        h2_matches += int(ok)
        h2_total += 1
        if r["family"] == "Bravyi" or h2_total <= 28:
            print(f"  {r['label']:>22}  {r['beta_dir']:>9.5f}  {beta_A:>8.4f}  "
                  f"{beta_B:>8.4f}  {pred:>10.5f}  {err:>9.4f}  {'YES' if ok else 'no':>9}")
    print()
    print(f"  ==> H2 holds for {h2_matches}/{h2_total} non-trivial codes "
          f"(<1% relative error).")
    print(f"      {'SUPPORTED' if h2_matches == h2_total else 'REFUTED'}: the literal "
          f"HGP one-liner {'fits' if h2_matches == h2_total else 'does NOT fit'} BB codes.")
    print()

    print("ANCHOR  Bravyi published codes -- beta_BB vs the values reported in section 6.1")

    expected = {"[[72,12,6]]": 0.667, "[[90,8,10]]": 0.872,
                "[[144,12,12]]": 0.828, "[[288,12,18]]": 0.828}
    for r in rows:
        if r["family"] != "Bravyi":
            continue
        exp = expected.get(r["label"])
        flag = "ok" if exp is not None and abs(r["beta_dir"] - exp) < 5e-3 else "CHECK"
        print(f"  {r['label']:>14}  beta_dir = {r['beta_dir']:.4f}  "
              f"(draft 6.1: {exp})  [{flag}]")
    print()

    print("STRUCTURE HINT  beta_BB grouped by (l, m, weight) -- does size or weight drive it?")

    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        if r["K"] == 0:
            continue
        grp[(r["wt"],)].append(r["beta_dir"])
    for key in sorted(grp):
        vals = np.array(grp[key])
        print(f"  weight={key[0]}:  n={len(vals):>2}  beta_BB in "
              f"[{vals.min():.4f}, {vals.max():.4f}]  "
              f"mean={vals.mean():.4f}  std={vals.std():.4f}")
    print()
    print("  Interpretation: a tight band within a weight class would hint at a")
    print("  weight-determined formula; a wide spread means beta_BB genuinely")
    print("  depends on the monomial *positions*, i.e. needs the Fourier reduction (H1).")


if __name__ == "__main__":
    main()
