from typing import Optional, Tuple

import numpy as np
import networkx as nx


def random_3_4_bipartite_check_matrix(n: int, seed: Optional[int] = None,
                                       max_attempts: int = 200) -> np.ndarray:
    if n % 4 != 0:
        raise ValueError(f"n must be divisible by 4 for (3,4)-regular; got n={n}")
    m = 3 * n // 4

    rng = np.random.default_rng(seed)
    var_stubs = np.repeat(np.arange(n), 3)
    chk_stubs = np.repeat(np.arange(m), 4)

    # Cold-start with rejection
    for attempt in range(max_attempts):
        rng.shuffle(var_stubs)
        rng.shuffle(chk_stubs)
        H_try = np.zeros((m, n), dtype=np.uint8)
        ok = True
        for v_stub, c_stub in zip(var_stubs, chk_stubs):
            if H_try[c_stub, v_stub] == 1:
                ok = False
                break
            H_try[c_stub, v_stub] = 1
        if ok:
            return H_try

    # Repair stage: build matrix anyway with parallel edges, then swap
    H = np.zeros((m, n), dtype=np.uint8)
    rng.shuffle(var_stubs)
    rng.shuffle(chk_stubs)
    edges = list(zip(chk_stubs, var_stubs))  # multi-edges possible
    for repair_round in range(max_attempts):
        # Find a duplicate edge (c1, v1)
        seen = {}
        dup = None
        for idx, (c, v) in enumerate(edges):
            key = (c, v)
            if key in seen:
                dup = (seen[key], idx)
                break
            seen[key] = idx
        if dup is None:
            # Build H from edges
            for c, v in edges:
                H[c, v] = 1
            return H
        # Swap to break duplicate: pick a random other edge (c2, v2)
        # and swap to (c1, v2), (c2, v1) provided neither is a duplicate
        i, j = dup
        c1, v1 = edges[i]
        for swap_try in range(20):
            k = rng.integers(0, len(edges))
            if k == i or k == j:
                continue
            c2, v2 = edges[k]
            new_a = (c1, v2)
            new_b = (c2, v1)
            edge_set = set(edges)
            edge_set.discard(edges[i])
            edge_set.discard(edges[k])
            if new_a in edge_set or new_b in edge_set:
                continue
            edges[i] = new_a
            edges[k] = new_b
            break
        else:
            # No swap worked; resample
            rng.shuffle(var_stubs)
            rng.shuffle(chk_stubs)
            edges = list(zip(chk_stubs, var_stubs))

    # Final attempt: networkx fallback (not guaranteed regular)
    G = nx.algorithms.bipartite.random_graph(m, n, p=4 / n, seed=seed)
    H = np.zeros((m, n), dtype=np.uint8)
    for u, v in G.edges():
        if u < m and v >= m:
            H[u, v - m] = 1
        elif v < m and u >= m:
            H[v, u - m] = 1
    return H


def tanner_graph_classical(H: np.ndarray) -> np.ndarray:
    m, n = H.shape
    A = np.zeros((n + m, n + m), dtype=np.float64)
    A[:n, n:] = H.T  # variable -> check
    A[n:, :n] = H    # check -> variable
    return A


def hgp_tanner_graph(H1: np.ndarray, H2: np.ndarray, sparse: bool = False):
    m1, n1 = H1.shape
    m2, n2 = H2.shape

    N_L = n1 * n2          # left-block qubits
    N_R = m1 * m2          # right-block qubits
    N_q = N_L + N_R        # total qubits
    N_X = m1 * n2          # X-checks
    N_Z = n1 * m2          # Z-checks
    N_total = N_q + N_X + N_Z

    # Index helpers
    def L(v1, v2):              # qubit at L-block (v1, v2)
        return v1 * n2 + v2
    def R(c1, c2):              # qubit at R-block (c1, c2)
        return N_L + c1 * m2 + c2
    def X(c1, v2):              # X-check at (c1, v2)
        return N_q + c1 * n2 + v2
    def Z(v1, c2):              # Z-check at (v1, c2)
        return N_q + N_X + v1 * m2 + c2

    if sparse:
        from scipy.sparse import coo_matrix
        rows, cols = [], []
        # X-check connections
        for c1 in range(m1):
            for v2 in range(n2):
                x = X(c1, v2)
                for v1 in range(n1):
                    if H1[c1, v1]:
                        q = L(v1, v2)
                        rows.append(x); cols.append(q)
                        rows.append(q); cols.append(x)
                for c2 in range(m2):
                    if H2[c2, v2]:
                        q = R(c1, c2)
                        rows.append(x); cols.append(q)
                        rows.append(q); cols.append(x)
        # Z-check connections
        for v1 in range(n1):
            for c2 in range(m2):
                z = Z(v1, c2)
                for v2 in range(n2):
                    if H2[c2, v2]:
                        q = L(v1, v2)
                        rows.append(z); cols.append(q)
                        rows.append(q); cols.append(z)
                for c1 in range(m1):
                    if H1[c1, v1]:
                        q = R(c1, c2)
                        rows.append(z); cols.append(q)
                        rows.append(q); cols.append(z)
        data = np.ones(len(rows), dtype=np.float64)
        coo = coo_matrix((data, (rows, cols)), shape=(N_total, N_total))
        coo.sum_duplicates()
        return coo.tocsr()

    A = np.zeros((N_total, N_total), dtype=np.float64)

    # X-check connections
    for c1 in range(m1):
        for v2 in range(n2):
            x = X(c1, v2)
            for v1 in range(n1):
                if H1[c1, v1]:
                    q = L(v1, v2)
                    A[x, q] = 1; A[q, x] = 1
            for c2 in range(m2):
                if H2[c2, v2]:
                    q = R(c1, c2)
                    A[x, q] = 1; A[q, x] = 1

    # Z-check connections
    for v1 in range(n1):
        for c2 in range(m2):
            z = Z(v1, c2)
            for v2 in range(n2):
                if H2[c2, v2]:
                    q = L(v1, v2)
                    A[z, q] = 1; A[q, z] = 1
            for c1 in range(m1):
                if H1[c1, v1]:
                    q = R(c1, c2)
                    A[z, q] = 1; A[q, z] = 1

    return A


def hgp_code_params(H1: np.ndarray, H2: np.ndarray) -> Tuple[int, int]:
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    rank_H1 = np.linalg.matrix_rank(H1.astype(np.float64))
    rank_H2 = np.linalg.matrix_rank(H2.astype(np.float64))
    k1 = n1 - rank_H1
    k2 = n2 - rank_H2
    k1_T = m1 - rank_H1
    k2_T = m2 - rank_H2
    N = n1 * n2 + m1 * m2
    K = k1 * k2 + k1_T * k2_T
    return int(N), int(K)


def lp_check_matrix_circulant(B: np.ndarray, L: int) -> np.ndarray:
    m, n = B.shape
    H = np.zeros((m * L, n * L), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            s = int(B[i, j])
            if s < 0:
                continue  # zero block
            # Add the L×L cyclic shift x^s
            for k in range(L):
                H[i * L + k, j * L + ((k + s) % L)] = 1
    return H


def lp_code(B: np.ndarray, L: int) -> Tuple[np.ndarray, int, int]:
    H_lifted = lp_check_matrix_circulant(B, L)
    A = hgp_tanner_graph(H_lifted, H_lifted)
    N, K = hgp_code_params(H_lifted, H_lifted)
    return A, N, K
