"""
Specific published code constructors for direct comparison with the
Bluvstein 2024 (BB codes) and Xu 2024 (LP codes) experiments.

References:
  Bravyi et al., Nature 627, 778 (2024) — bivariate bicycle codes
  Bluvstein et al., Nature 626, 58 (2024) — uses BB codes
  Xu et al., Nature Physics 20, 1084 (2024) — uses LP codes from
    (3,5)-biregular bases over various lifts

NOTE: BB codes are NOT HGP-product codes; they are a separate family
constructed from two cyclic permutations on a torus. The chi' = Delta
analysis still applies via Konig (the BB Tanner graph is bipartite),
but the per-cycle algorithm comparison structure differs.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bb_check_matrices(l, m, A_polynomial, B_polynomial):
    """Bivariate bicycle (BB) code check matrices.

    A BB code is defined on n = 2*l*m qubits arranged on an l x m torus.
    Two polynomials A(x, y) and B(x, y) in F_2[x, y]/(x^l - 1, y^m - 1)
    define the X- and Z-stabilizer matrices:
      H_X = [A | B]
      H_Z = [B^T | A^T]
    where A, B are l*m x l*m matrices encoding multiplication by the
    polynomials.

    A_polynomial, B_polynomial: lists of (i, j) tuples specifying x^i y^j
    monomials.

    Returns (H_X, H_Z) as sparse 0/1 matrices.
    """
    n_block = l * m

    def poly_to_matrix(poly):
        """Build the n_block x n_block matrix for multiplication by sum of x^i y^j terms."""
        M = np.zeros((n_block, n_block), dtype=np.uint8)
        for (i_pow, j_pow) in poly:
            for r in range(l):
                for c in range(m):
                    src = r * m + c
                    dst = ((r + i_pow) % l) * m + ((c + j_pow) % m)
                    M[dst, src] = 1
        return M

    A = poly_to_matrix(A_polynomial)
    B = poly_to_matrix(B_polynomial)
    # H_X = [A | B], shape (n_block, 2*n_block)
    H_X = np.hstack([A, B])
    # H_Z = [B^T | A^T], shape (n_block, 2*n_block)
    H_Z = np.hstack([B.T, A.T])
    return H_X.astype(np.uint8), H_Z.astype(np.uint8)


def bb_tanner_graph(H_X, H_Z):
    """Combined bipartite Tanner graph for BB code: qubits on one side,
    X- and Z-checks on the other. Returns adjacency matrix."""
    n_X, n_q = H_X.shape
    n_Z, _ = H_Z.shape
    N_total = n_q + n_X + n_Z
    A = np.zeros((N_total, N_total), dtype=np.float64)
    # Qubit-X-check edges: A[q, X] = H_X[X, q]
    for x in range(n_X):
        for q in range(n_q):
            if H_X[x, q]:
                A[q, n_q + x] = 1
                A[n_q + x, q] = 1
    # Qubit-Z-check edges
    for z in range(n_Z):
        for q in range(n_q):
            if H_Z[z, q]:
                A[q, n_q + n_X + z] = 1
                A[n_q + n_X + z, q] = 1
    return A


# Bravyi et al. 2024 published BB codes
# Reference: Bravyi et al., Nature 627, 778 (2024), Table 1.
# Each entry: (l, m, A_poly, B_poly, label)
# A and B polynomials given in their notation as lists of (x_pow, y_pow) tuples.
BRAVYI_BB_CODES = [
    # [[72, 12, 6]]: l=6, m=6, A = x^3 + y + y^2, B = y^3 + x + x^2
    {"label": "[[72,12,6]]", "l": 6, "m": 6,
     "A": [(3, 0), (0, 1), (0, 2)],
     "B": [(0, 3), (1, 0), (2, 0)],
     "N": 72, "K": 12, "d": 6},
    # [[90, 8, 10]]: l=15, m=3
    {"label": "[[90,8,10]]", "l": 15, "m": 3,
     "A": [(9, 0), (1, 2), (2, 0)],
     "B": [(0, 1), (2, 1), (7, 0)],
     "N": 90, "K": 8, "d": 10},
    # [[144, 12, 12]]: l=12, m=6
    {"label": "[[144,12,12]]", "l": 12, "m": 6,
     "A": [(3, 0), (0, 1), (0, 2)],
     "B": [(0, 3), (1, 0), (2, 0)],
     "N": 144, "K": 12, "d": 12},
    # [[288, 12, 18]]: l=12, m=12
    {"label": "[[288,12,18]]", "l": 12, "m": 12,
     "A": [(3, 0), (0, 2), (0, 7)],
     "B": [(0, 3), (1, 0), (2, 0)],
     "N": 288, "K": 12, "d": 18},
]


# Xu et al. 2024 LP codes from supplementary
# Their primary family is built from the 3 x 5 base support mask with random
# voltage assignments in Z_L for various L. The exact circulant exponents are
# not all in the main text; we use representative voltages.
XU_LP_CODES = [
    # Smallest in their Fig. 3a: their lift gives N = 225 for L=5 (different
    # construction than our HGP-of-lift). We use HGP-of-lift L=4 (N=400) as
    # the closest equivalent, plus larger sizes.
    {"label": "Xu LP family L=4", "L_lift": 4, "base_shape": (3, 5)},
    {"label": "Xu LP family L=8", "L_lift": 8, "base_shape": (3, 5)},
    {"label": "Xu LP family L=12", "L_lift": 12, "base_shape": (3, 5)},
    {"label": "Xu LP family L=20", "L_lift": 20, "base_shape": (3, 5)},
    # L=32 (N=34816) requires sparse construction; not in default list.
]


# Cases requiring sparse construction (skipped from default sweep)
XU_LP_CODES_SPARSE_NEEDED = [
    {"label": "Xu LP family L=32", "L_lift": 32, "base_shape": (3, 5)},
]


def get_bravyi_bb_tanner(code_spec):
    """Build the BB Tanner graph for a given code spec dict."""
    H_X, H_Z = bb_check_matrices(
        code_spec["l"], code_spec["m"],
        code_spec["A"], code_spec["B"],
    )
    A = bb_tanner_graph(H_X, H_Z)
    return A, H_X, H_Z


def get_xu_lp_tanner(code_spec, seed=42):
    """Build the LP Tanner graph for a given Xu LP code spec.
    Returns adjacency of HGP[H_lift, H_lift] where H_lift is the random
    voltage lift of a 3x5 base mask."""
    from routing_lib.qldpc.codes import (
        lp_check_matrix_circulant, hgp_tanner_graph,
    )
    rng = np.random.default_rng(seed)
    L = code_spec["L_lift"]
    rows, cols = code_spec["base_shape"]
    B = rng.integers(0, L, size=(rows, cols))
    H = lp_check_matrix_circulant(B, L)
    A = hgp_tanner_graph(H, H)
    return A, H


if __name__ == "__main__":
    print("Published code constructors")
    print("=" * 60)
    print("\nBravyi et al. BB codes:")
    for spec in BRAVYI_BB_CODES:
        A, H_X, H_Z = get_bravyi_bb_tanner(spec)
        N = A.shape[0]
        degs = A.sum(axis=1).astype(int)
        delta = int(degs.max())
        print(f"  {spec['label']:>20}: Tanner N={N}, "
              f"H_X shape={H_X.shape}, Delta={delta}, chi'={delta} (Konig)")
    print("\nXu LP codes (random voltage on 3x5 mask):")
    for spec in XU_LP_CODES:
        A, H = get_xu_lp_tanner(spec)
        N = A.shape[0]
        degs = A.sum(axis=1).astype(int)
        delta = int(degs.max())
        print(f"  {spec['label']:>22}: Tanner N={N}, "
              f"H shape={H.shape}, Delta={delta}, chi'={delta} (Konig)")
