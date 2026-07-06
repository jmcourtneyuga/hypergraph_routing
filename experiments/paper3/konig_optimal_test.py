"""
Question: For non-QC HGP Tanner graphs, is Konig-optimal edge coloring
(chi' = Delta exactly) always achievable, or does non-QC structure
fundamentally require more colors?

Konig's theorem GUARANTEES chi' = Delta for any bipartite graph. The
question is whether we can CONSTRUCT the coloring in polynomial time at
the scales of interest.

Algorithm: extend G to Delta-regular bipartite multigraph by adding
fake edges (Hall's theorem guarantees feasibility), then iteratively
extract Delta perfect matchings. Implemented in bipartite_konig_coloring
below.

USAGE:
    python konig_optimal_test.py                  # default sweep
    python konig_optimal_test.py --n 100 1000 10000  # custom n_base values
    python konig_optimal_test.py --max-N 50000    # cap N for safety
    python konig_optimal_test.py --extend         # extend default to large N

Cost scales as O(Delta * E * sqrt(V)) ~ O(N^1.5) for sparse Tanner graphs.
At N=10^4 each test ~ a few seconds; at N=10^5 ~ minutes; at N=10^6 ~ hours.
"""
import os
import sys
import numpy as np
import networkx as nx
from collections import deque

# Ensure routing_lib is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.qldpc.codes import (
    random_3_4_bipartite_check_matrix,
    lp_check_matrix_circulant,
    hgp_tanner_graph,
)


def bipartite_konig_coloring(A):
    """Construct Konig-optimal proper edge coloring chi' = Delta for a
    bipartite adjacency matrix A.

    Accepts dense np.ndarray or scipy.sparse adjacency.

    Algorithm: extend G to a Delta-regular bipartite multigraph by adding
    fake edges between vertices of degree < Delta (Hall's theorem
    guarantees this is possible). Then iteratively extract perfect
    matchings (each is a color class). Strip fake edges from each color.

    Returns (colors_dict, n_colors_used, delta).
    """
    n = A.shape[0]
    # Build graph and bipartition (sparse-aware iteration)
    from scipy.sparse import issparse
    G = nx.Graph()
    if issparse(A):
        A_csr = A.tocsr() if not hasattr(A, 'indptr') else A
        for i in range(n):
            for j in A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i+1]]:
                if i < j:
                    G.add_edge(int(i), int(j))
    else:
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] > 0:
                    G.add_edge(i, j)
    if G.number_of_edges() == 0:
        return {}, 0, 0

    side = [-1] * n
    for s in range(n):
        if side[s] != -1 or not G.has_node(s):
            continue
        side[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in G.neighbors(u):
                if side[v] == -1:
                    side[v] = 1 - side[u]
                    q.append(v)
    left = sorted([i for i in range(n) if side[i] == 0 and G.has_node(i)])
    right = sorted([i for i in range(n) if side[i] == 1 and G.has_node(i)])
    delta = max(G.degree(v) for v in G.nodes())

    # Pad both sides to equal size (add isolated dummies on smaller side)
    if len(left) < len(right):
        for k in range(len(right) - len(left)):
            dummy = -100 - k
            left.append(dummy)
    elif len(right) < len(left):
        for k in range(len(left) - len(right)):
            dummy = -200 - k
            right.append(dummy)

    # Build a Delta-regular bipartite MULTIGRAPH by adding fake edges
    # between vertices with degree < Delta. We use scipy assignment as a
    # convenient way to find perfect matchings on Delta-regular bipartite.
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    # Use multigraph representation: degrees dict + edge multiplicities
    L_idx = {u: i for i, u in enumerate(left)}
    R_idx = {v: i for i, v in enumerate(right)}
    # multi[(i, j)] = number of edges between left[i] and right[j]
    multi = {}
    for u, v in G.edges():
        if side[u] == 0:
            li, ri = L_idx[u], R_idx[v]
        else:
            li, ri = L_idx[v], R_idx[u]
        multi[(li, ri)] = multi.get((li, ri), 0) + 1
    # Track real-edge identity so we can strip fakes later
    real_edges = set(multi.keys())

    # Pad with fake edges to make Delta-regular multigraph
    deg_L = [0] * len(left)
    deg_R = [0] * len(right)
    for (li, ri), m in multi.items():
        deg_L[li] += m
        deg_R[ri] += m
    # Greedy fake-edge addition
    while True:
        L_low = [i for i in range(len(left)) if deg_L[i] < delta]
        R_low = [i for i in range(len(right)) if deg_R[i] < delta]
        if not L_low or not R_low:
            break
        for li in L_low:
            for ri in R_low:
                if deg_L[li] >= delta or deg_R[ri] >= delta:
                    continue
                multi[(li, ri)] = multi.get((li, ri), 0) + 1
                deg_L[li] += 1
                deg_R[ri] += 1
                if deg_L[li] >= delta:
                    break

    # Now iteratively extract perfect matchings.
    # Each matching reduces all degrees by 1; after delta iterations,
    # multigraph is empty.
    colors = {}
    for color in range(delta):
        # Build bipartite graph from current multi
        rows, cols = [], []
        for (li, ri), m in multi.items():
            if m > 0:
                rows.append(li)
                cols.append(ri)
        if not rows:
            break
        from scipy.sparse import csr_matrix
        adj = csr_matrix(([1] * len(rows), (rows, cols)),
                          shape=(len(left), len(right)))
        match = maximum_bipartite_matching(adj, perm_type='column')
        # Extract perfect matching (assumes Delta-regular gives perfect)
        pm = []
        for li in range(len(left)):
            ri = match[li]
            if ri >= 0 and (li, ri) in multi and multi[(li, ri)] > 0:
                pm.append((li, ri))
                multi[(li, ri)] -= 1
        # Color real edges in this matching
        for (li, ri) in pm:
            if (li, ri) in real_edges:
                u, v = left[li], right[ri]
                if u >= 0 and v >= 0:
                    colors[(min(u, v), max(u, v))] = color
    n_colors_used = max(colors.values()) + 1 if colors else 0
    return colors, n_colors_used, delta


def default_test_cases():
    return [
        ("QC LP[3x4 base, L=4]", lambda: lp_check_matrix_circulant(
            np.random.default_rng(42).integers(0, 5, size=(3, 4)), L=4)),
        ("QC LP[3x4 base, L=8]", lambda: lp_check_matrix_circulant(
            np.random.default_rng(42).integers(0, 5, size=(3, 4)), L=8)),
        ("non-QC random (3,4) n=16, seed=1", lambda: random_3_4_bipartite_check_matrix(16, seed=1)),
        ("non-QC random (3,4) n=16, seed=2", lambda: random_3_4_bipartite_check_matrix(16, seed=2)),
        ("non-QC random (3,4) n=24, seed=3", lambda: random_3_4_bipartite_check_matrix(24, seed=3)),
        ("non-QC random (3,4) n=32, seed=1", lambda: random_3_4_bipartite_check_matrix(32, seed=1)),
        ("non-QC random (3,4) n=48, seed=2", lambda: random_3_4_bipartite_check_matrix(48, seed=2)),
    ]


def extended_test_cases(n_values, seeds=(1, 2, 3)):
    """Generate cases at user-specified n_base values across QC and non-QC."""
    cases = []
    rng = np.random.default_rng(42)
    B_qc = rng.integers(0, 7, size=(3, 4))
    for n_base in n_values:
        # The LP[3x4, L_lift=n_base/4] has lifted size that matches HGP[n_base]
        L_lift = max(1, n_base // 4)
        cases.append((f"QC LP[3x4, L={L_lift}]",
                       lambda lift=L_lift: lp_check_matrix_circulant(B_qc, lift)))
        for seed in seeds:
            cases.append((f"non-QC random (3,4) n={n_base}, seed={seed}",
                           lambda nb=n_base, s=seed: random_3_4_bipartite_check_matrix(nb, seed=s)))
    return cases


def estimate_tanner_N(H):
    """Estimate the size of HGP[H, H]'s Tanner graph WITHOUT constructing it.
    Returns total node count (qubits + X-checks + Z-checks)."""
    m, n = H.shape
    N_qubits = n * n + m * m
    N_X = m * n
    N_Z = n * m
    return N_qubits + N_X + N_Z


def run_test(test_cases=None, max_N=None, verbose=True,
              dense_threshold_gb=4.0):
    import time
    if test_cases is None:
        test_cases = default_test_cases()

    print("=" * 100)
    print("KONIG-OPTIMAL COLORING TEST: QC vs non-QC HGP Tanner graphs")
    if max_N:
        print(f"max N cap: {max_N}")
    print(f"dense memory threshold: {dense_threshold_gb} GB "
          f"(switches to sparse construction beyond)")
    print("=" * 100)
    print()

    print(f"{'case':>42} | {'N':>6} | {'mode':>6} | {'Delta':>5} | "
          f"{'Konig':>5} | {'tight?':>7} | {'time(s)':>8}", flush=True)
    print("-" * 100, flush=True)
    for label, make_H in test_cases:
        try:
            H = make_H()
        except MemoryError as e:
            print(f"{label:>42} | {'?':>6} | {'-':>6} | {'-':>5} | "
                  f"{'-':>5} | {'-':>7} | MemoryError building H: {e}",
                  flush=True)
            continue
        N_estimate = estimate_tanner_N(H)
        if max_N is not None and N_estimate > max_N:
            print(f"{label:>42} | {N_estimate:>6} | {'-':>6} | {'-':>5} | "
                  f"{'-':>5} | {'-':>7} | (skipped, est. N > {max_N})",
                  flush=True)
            continue
        bytes_needed = N_estimate * N_estimate * 8
        use_sparse = bytes_needed > dense_threshold_gb * 2**30
        mode = "sparse" if use_sparse else "dense"
        try:
            A = hgp_tanner_graph(H, H, sparse=use_sparse)
        except MemoryError as e:
            print(f"{label:>42} | {N_estimate:>6} | {mode:>6} | {'-':>5} | "
                  f"{'-':>5} | {'-':>7} | MemoryError: {e}", flush=True)
            continue
        N = A.shape[0]
        if hasattr(A, 'sum'):
            degs = np.asarray(A.sum(axis=1)).astype(int).flatten()
        else:
            degs = A.sum(axis=1).astype(int)
        delta = int(degs.max())
        t0 = time.time()
        try:
            colors, n_colors, delta_check = bipartite_konig_coloring(A)
            elapsed = time.time() - t0
            tight = "YES" if n_colors == delta else f"no ({n_colors})"
            print(f"{label:>42} | {N:>6} | {mode:>6} | {delta:>5} | "
                  f"{n_colors:>5} | {tight:>7} | {elapsed:>8.2f}", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{label:>42} | {N:>6} | {mode:>6} | {delta:>5} | "
                  f"{'-':>5} | {'-':>7} | ERROR after {elapsed:.1f}s: "
                  f"{type(e).__name__}: {e}", flush=True)

    print()
    print("=" * 90)
    print("CONCLUSION:")
    print("=" * 90)
    print("Konig's theorem guarantees chi' = Delta for any bipartite graph.")
    print("If coloring is Konig-tight for non-QC random codes, the 'non-QC")
    print("overhead' (chi' = 9-10 from GREEDY) is purely an algorithmic")
    print("artifact, not a fundamental property of non-QC structure.")


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=None,
                    help="n_base values for extended sweep (e.g., 60 80 100 200)")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="seeds for random non-QC bases (default: 1 2 3)")
    p.add_argument("--max-N", type=int, default=None,
                    help="skip cases with N > max_N (safety cap)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.n is not None:
        cases = extended_test_cases(args.n, seeds=args.seeds)
    else:
        cases = default_test_cases()
    run_test(cases, max_N=args.max_N)
