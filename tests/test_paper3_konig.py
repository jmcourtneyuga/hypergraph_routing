#!/usr/bin/env python3
"""Paper III Koenig-tightness tests.

Validates the chromatic-index claim underpinning the per-cycle cost
chi'(G_T) = Delta(G_T) = 2*delta_c (Theorem 4.1 and Tables tab:konig-tight,
tab:head-to-head, tab:bb-codes): for a bipartite Tanner graph, Koenig's theorem
gives chi' = Delta, and the coloring is constructible in polynomial time by
repeated bipartite maximum matching on the Delta-regular extension.

We verify BOTH:
  (a) Delta(G_T) has the expected value (8 for (3,4)-biregular HGP, 6 for the
      Bravyi BB codes), and
  (b) an explicit edge coloring achieves exactly Delta colors (chi' = Delta),
      i.e. the construction is Koenig-tight, on small HGP and BB Tanner graphs.

Dual-mode: `pytest tests/test_paper3_konig.py` runs assertions;
`python tests/test_paper3_konig.py` prints a PASS/FAIL report.
"""
import numpy as np

from routing_lib.qldpc.codes import (
    random_3_4_bipartite_check_matrix,
    hgp_tanner_graph,
)
from routing_lib.qldpc.published_codes import (
    BRAVYI_BB_CODES,
    get_bravyi_bb_tanner,
)
# Reuse the validated Delta-regular-extension coloring from the experiment
# script (Koenig-optimal via Hall's theorem + repeated perfect matching).
from experiments.paper3.konig_optimal_test import bipartite_konig_coloring


def konig_edge_coloring_num_colors(A):
    """Number of colors used by the explicit Koenig-optimal coloring of the
    Tanner adjacency A.  Returns n_colors (== Delta when Koenig-tight)."""
    _colors, n_colors, _delta = bipartite_konig_coloring(np.asarray(A, dtype=float))
    return n_colors


def hgp_delta_cases(sizes=(12, 16), seeds=(1, 2)):
    out = []
    for n in sizes:
        for s in seeds:
            H = random_3_4_bipartite_check_matrix(n, seed=s)
            A = np.asarray(hgp_tanner_graph(H, H), dtype=float)
            Delta = int(A.sum(1).max())
            out.append((f"HGP(3,4) n={n} seed={s}", A, Delta, 8))
    return out


def bb_delta_cases():
    out = []
    for code in BRAVYI_BB_CODES:
        A, _, _ = get_bravyi_bb_tanner(code)
        A = np.asarray(A, dtype=float)
        Delta = int(A.sum(1).max())
        out.append((f"BB {code['label']}", A, Delta, 6))
    return out


def test_hgp_max_degree_is_8():
    for label, A, Delta, expect in hgp_delta_cases():
        assert Delta == expect, f"{label}: Delta={Delta}, expected {expect}"


def test_bb_max_degree_is_6():
    for label, A, Delta, expect in bb_delta_cases():
        assert Delta == expect, f"{label}: Delta={Delta}, expected {expect}"


def test_konig_tight_small_instances():
    # Explicit coloring achieves exactly Delta colors on the smallest instances.
    checks = []
    H = random_3_4_bipartite_check_matrix(12, seed=1)
    checks.append((np.asarray(hgp_tanner_graph(H, H), dtype=float), 8))
    A_bb, _, _ = get_bravyi_bb_tanner(BRAVYI_BB_CODES[0])  # [[72,12,6]]
    checks.append((np.asarray(A_bb, dtype=float), 6))
    for A, Delta in checks:
        colors = konig_edge_coloring_num_colors(A)
        assert colors == Delta, (
            f"explicit coloring used {colors} colors, expected chi'=Delta={Delta}")


if __name__ == "__main__":
    ok = True
    print("Koenig: Delta(G_T) for HGP (expect 8) and BB (expect 6)")
    for label, A, Delta, expect in hgp_delta_cases() + bb_delta_cases():
        good = Delta == expect
        ok &= good
        print(f"  {label:>24}  Delta={Delta}  {'PASS' if good else 'FAIL'}")
    print("\nExplicit edge coloring is Koenig-tight (chi' = Delta):")
    H = random_3_4_bipartite_check_matrix(12, seed=1)
    for A, Delta, name in [
        (np.asarray(hgp_tanner_graph(H, H), dtype=float), 8, "HGP(3,4) n=12"),
        (np.asarray(get_bravyi_bb_tanner(BRAVYI_BB_CODES[0])[0], dtype=float), 6, "BB [[72,12,6]]"),
    ]:
        colors = konig_edge_coloring_num_colors(A)
        good = colors == Delta
        ok &= good
        print(f"  {name:>16}  colors={colors}  chi'=Delta={Delta}  "
              f"{'PASS' if good else 'FAIL'}")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
