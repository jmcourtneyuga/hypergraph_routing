"""Graph and hypergraph constructors used by the validation scripts.

Several functions appear in multiple variants under distinct names; the
docstring of each names which original script(s) used it. This is
deliberate: the original scripts diverged on configuration-model vs
matching-union vs networkx-based construction, on whether the 2D grid
wraps as a torus, and on how odd-degree inputs are handled. The refactor
preserves bit-identical behavior for each call site.
"""

from collections import defaultdict, deque
from itertools import combinations

import numpy as np
import networkx as nx

# Random regular graphs (4 variants)

# Source: block_routing.py:27, validate_fixes.py:28 (identical bodies).
# Configuration model with 10 attempts, picks lowest-deficit graph.
def random_regular_config_model(N, d):
    """Build a random d-regular simple graph on N vertices.

    Uses configuration model with edge deduplication and self-loop removal.
    For large N, the result is approximately d-regular.
    """
    if (N * d) % 2 != 0:
        raise ValueError(f"N*d must be even, got N={N}, d={d}")

    stubs = []
    for v in range(N):
        stubs.extend([v] * d)

    best_A = None
    best_deficit = N * d

    for _ in range(10):
        np.random.shuffle(stubs)
        A = np.zeros((N, N), dtype=np.float64)
        for i in range(0, len(stubs) - 1, 2):
            u, v = stubs[i], stubs[i + 1]
            if u != v:
                A[u, v] = 1
                A[v, u] = 1

        deficit = np.sum(np.abs(A.sum(axis=1) - d))
        if deficit < best_deficit:
            best_A = A.copy()
            best_deficit = deficit
        if deficit == 0:
            break

    return best_A


# Source: multilayer_aol.py:24 (also entanglement_routing.py:24,
# sparse_overlay_routing.py:25 — identical bodies; the multilayer variant
# has an extra unused `attempts` arg).
# Builds via union of d perfect matchings; raises on odd d.
def random_regular_matching_union(N, d, attempts=1, sparse=False):
    """Build a random d-regular graph via d perfect matchings.

    Each matching contributes degree 1 per vertex, so d matchings give degree d.
    Returns adjacency matrix (may have multi-edges for small N).

    sparse=False (default): returns dense np.ndarray (back-compat).
    sparse=True: returns scipy.sparse.csr_matrix. Use for N >= 30000 where
    dense N x N float64 exceeds 7 GB.
    """
    if d % 2 == 1:
        raise ValueError("d must be even for this construction")
    if sparse:
        from scipy.sparse import coo_matrix
        rows, cols, data = [], [], []
        for _ in range(d):
            perm = np.random.permutation(N)
            for k in range(0, N - 1, 2):
                i, j = int(perm[k]), int(perm[k + 1])
                if i != j:
                    rows.append(i); cols.append(j); data.append(1.0)
                    rows.append(j); cols.append(i); data.append(1.0)
        coo = coo_matrix((data, (rows, cols)), shape=(N, N))
        coo.sum_duplicates()  # collapse multi-edges into weights
        return coo.tocsr()
    else:
        A = np.zeros((N, N))
        for _ in range(d):
            perm = np.random.permutation(N)
            for k in range(0, N - 1, 2):
                i, j = int(perm[k]), int(perm[k + 1])
                if i != j:
                    A[i, j] += 1
                    A[j, i] += 1
        return A


# Source: dynamic_adaptive.py:21 (also hierarchical_routing.py:25 — identical).
# Silently decrements odd d, has additional N<2 / d>=N edge cases.
def random_regular_matching_union_simple(N, d):
    if N < 2:
        return np.zeros((N, N))
    if d >= N:
        return np.ones((N, N)) - np.eye(N)
    if d % 2 == 1:
        d -= 1
    d = max(2, d)
    A = np.zeros((N, N))
    for _ in range(d):
        perm = np.random.permutation(N)
        for k in range(0, N - 1, 2):
            i, j = int(perm[k]), int(perm[k + 1])
            if i != j:
                A[i, j] += 1
                A[j, i] += 1
    return A


# Source: test_concentration.py:15, test_greedy_stall_n256.py:14.
# Uses networkx.random_regular_graph; seedable.
def random_regular_networkx(N, d, seed=42):
    if N * d % 2 != 0:
        return None
    return nx.to_numpy_array(nx.random_regular_graph(d, N, seed=seed), dtype=int)


# Source: aol_constrained_routing.py:79 (build_random_regular_graph).
# Best-of `best_of` random graphs by lowest beta.
def random_regular_best_of(N, d=10, best_of=3):
    """Random d-regular graph via repeated matching, picking lowest beta.

    By Friedman's theorem, lambda_2 <= 2*sqrt(d-1) + eps a.a.s.
    """
    from numpy.linalg import eigvalsh
    if d % 2 == 1:
        d += 1
    best_A = None
    best_beta = 1.0
    for _ in range(best_of):
        A = np.zeros((N, N))
        for _ in range(d // 2):
            perm = np.random.permutation(N)
            for k in range(0, N - 1, 2):
                i, j = int(perm[k]), int(perm[k + 1])
                if i != j:
                    A[i, j] = 1
                    A[j, i] = 1
        eigs = eigvalsh(A)
        eigs_sorted = np.sort(eigs)[::-1]
        d_eff = eigs_sorted[0]
        beta = eigs_sorted[1] / d_eff if d_eff > 0 else 1
        if beta < best_beta:
            best_beta = beta
            best_A = A.copy()
    return best_A

# Host graphs (clique expansion of (d,r)-regular hypergraphs)

# Source: block_routing.py:63 — 3-arg version (N_phys, d, r).
def build_host_graph_d_dprime(N_phys, d, r):
    """Random d'-regular expander on N_phys vertices, where d' = d(r-1)."""
    d_prime = d * (r - 1)
    if (N_phys * d_prime) % 2 != 0:
        N_phys += 1
    return random_regular_config_model(N_phys, d_prime)


# Source: validate_fixes.py:76 — 2-arg version (N_phys, d_prime).
def build_host_graph_dprime_only(N_phys, d_prime):
    """Build a random d'-regular expander graph on N_phys vertices."""
    if (N_phys * d_prime) % 2 != 0:
        N_phys += 1
    return random_regular_config_model(N_phys, d_prime)

# 2D / 3D grid clique expansions

# Source: aol_constrained_routing.py:28, aom_3d_routing.py:22 (identical).
def coord_to_idx(x, y, n):
    return (x % n) * n + (y % n)


# Source: aom_3d_routing.py:18
def torus_grid_coords(n):
    """Map vertex index to (row, col) on n x n torus."""
    return [(i // n, i % n) for i in range(n * n)]


# Source: multilayer_aol.py:68 (also sparse_overlay_routing.py:42 — identical).
def build_2d_grid_clique(n, r=3):
    """2D torus grid clique expansion for baseline comparison."""
    N = n * n
    A = np.zeros((N, N))
    for x in range(n):
        for y in range(n):
            verts = [(x * n + (y + k) % n) for k in range(r)]
            for a, b in combinations(verts, 2):
                A[a, b] += 1
                A[b, a] += 1
            verts = [((x + k) % n * n + y) for k in range(r)]
            for a, b in combinations(verts, 2):
                A[a, b] += 1
                A[b, a] += 1
    return A


# Source: aol_constrained_routing.py:32
def build_2d_grid_clique_torus(n, r=3):
    """Clique expansion adjacency matrix for 2D torus grid with r-uniform
    row/col hyperedges."""
    N = n * n
    A = np.zeros((N, N))
    for x in range(n):
        for y in range(n):
            verts = [coord_to_idx(x, y + k, n) for k in range(r)]
            for a, b in combinations(verts, 2):
                A[a, b] += 1
                A[b, a] += 1
            verts = [coord_to_idx(x + k, y, n) for k in range(r)]
            for a, b in combinations(verts, 2):
                A[a, b] += 1
                A[b, a] += 1
    return A


# Source: aol_constrained_routing.py:52
def build_3d_aol_clique(n, r=3):
    """Clique expansion for 3D AOL model: 2D grid + diagonal + skip edges."""
    N = n * n
    A = build_2d_grid_clique_torus(n, r)
    s = max(1, n // (4 * (r - 1)))

    for x in range(n):
        for y in range(n):
            for direction in [(1, 1), (1, -1)]:
                verts = [coord_to_idx(x + k * direction[0], y + k * direction[1], n)
                         for k in range(r)]
                for a, b in combinations(verts, 2):
                    A[a, b] += 1
                    A[b, a] += 1
            verts_row = [coord_to_idx(x, y + k * s, n) for k in range(r)]
            for a, b in combinations(verts_row, 2):
                A[a, b] += 1
                A[b, a] += 1
            verts_col = [coord_to_idx(x + k * s, y, n) for k in range(r)]
            for a, b in combinations(verts_col, 2):
                A[a, b] += 1
                A[b, a] += 1
    return A


# Source: aol_constrained_routing.py:105
def build_margulis_expander(n):
    """Margulis-Gabber-Galil expander on Z_n x Z_n. 8-regular."""
    N = n * n
    A = np.zeros((N, N))

    for x in range(n):
        for y in range(n):
            i = x * n + y
            neighbors = [
                ((x + y) % n,     y),
                ((x - y) % n,     y),
                (x,               (y + x) % n),
                (x,               (y - x) % n),
                ((x + y + 1) % n, y),
                ((x - y + 1) % n, y),
                (x,               (y + x + 1) % n),
                (x,               (y - x + 1) % n),
            ]
            for nx_, ny in neighbors:
                j = nx_ * n + ny
                if j != i:
                    A[i, j] = 1
                    A[j, i] = 1
    return A


# Source: aol_constrained_routing.py:134
def build_combined_grid_overlay(n, r=3, overlay_type='margulis'):
    """Combined model: 2D grid + virtual expander overlay."""
    A_grid = build_2d_grid_clique_torus(n, r)
    N = n * n

    if overlay_type == 'margulis':
        A_overlay = build_margulis_expander(n)
    elif overlay_type == 'random':
        A_overlay = random_regular_best_of(N, d=10)
    else:
        raise ValueError(f"Unknown overlay type: {overlay_type}")

    return A_grid + A_overlay, A_overlay


# Cayley graphs (Z_n^2 / abelian)

# Source: algebraic_overlay.py:26
def cayley_graph_Zn2(n, generators):
    """Build adjacency matrix of Cay(Z_n^2, S).

    generators: list of (s1, s2) pairs. The full generator set S = {+g, -g}.
    """
    N = n * n
    A = np.zeros((N, N))

    for g1, g2 in generators:
        for x in range(n):
            for y in range(n):
                v = x * n + y
                w1 = ((x + g1) % n) * n + ((y + g2) % n)
                w2 = ((x - g1) % n) * n + ((y - g2) % n)
                A[v, w1] += 1
                A[v, w2] += 1

    return A


# Source: algebraic_overlay.py:50
def cayley_eigenvalues_exact(n, generators):
    """Compute eigenvalues of Cay(Z_n^2, S) using the character formula."""
    eigs = []
    for a in range(n):
        for b in range(n):
            lam = 0.0
            for g1, g2 in generators:
                lam += 2 * np.cos(2 * np.pi / n * (a * g1 + b * g2))
            eigs.append(lam)
    return np.array(sorted(eigs, reverse=True))


# Source: algebraic_overlay.py:87
def margulis_gabber_galil_generators(n):
    """Margulis-Gabber-Galil generators on Z_n x Z_n."""
    return [(1, 0), (0, 1), (1, 1), (1, n - 1)]


# Source: algebraic_overlay.py:99
def quadratic_residue_generators(n, d):
    """Quadratic residue generators: S = {(x, x^2 mod n) : x = 1..d} and negatives."""
    gens = []
    for x in range(1, d + 1):
        g1 = x % n
        g2 = (x * x) % n
        gens.append((g1, g2))
    return gens


# Source: algebraic_overlay.py:110
def cubic_generators(n, d):
    """Cubic generators: S = {(x, x^3 mod n) : x = 1..d}."""
    gens = []
    for x in range(1, d + 1):
        g1 = x % n
        g2 = (x * x * x) % n
        gens.append((g1, g2))
    return gens


# Source: algebraic_overlay.py:120
def random_generators(n, d):
    """Random generators: d random elements of Z_n^2."""
    gens = []
    for _ in range(d):
        g1 = np.random.randint(0, n)
        g2 = np.random.randint(0, n)
        while g1 == 0 and g2 == 0:
            g1 = np.random.randint(0, n)
            g2 = np.random.randint(0, n)
        gens.append((g1, g2))
    return gens

# Affine maps & GL(2, Z_n) (algebraic_overlay.py)

# Source: algebraic_overlay.py:214
def is_prime(n):
    if n < 2:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if n == p:
            return True
        if n % p == 0:
            return False
    i = 37
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# Source: algebraic_overlay.py:206
def gl2_count(n):
    """Count |GL(2, Z_n)| for prime n = p."""
    if is_prime(n):
        return n * (n - 1) * (n - 1) * (n + 1)
    return None


# Source: algebraic_overlay.py:231
def apply_affine(n, A_mat, c, v):
    """Apply affine map sigma(v) = A*v + c on Z_n^2."""
    x, y = v // n, v % n
    nx_ = (A_mat[0][0] * x + A_mat[0][1] * y + c[0]) % n
    ny = (A_mat[1][0] * x + A_mat[1][1] * y + c[1]) % n
    return nx_ * n + ny


# Source: algebraic_overlay.py:241
def affine_permutation(n, A_mat, c):
    """Compute full permutation from affine map on Z_n^2."""
    N = n * n
    perm = np.zeros(N, dtype=int)
    for v in range(N):
        perm[v] = apply_affine(n, A_mat, c, v)
    return perm


# Source: algebraic_overlay.py:250
def det_mod(A_mat, n):
    """Determinant of 2x2 matrix mod n."""
    return (A_mat[0][0] * A_mat[1][1] - A_mat[0][1] * A_mat[1][0]) % n


# Source: algebraic_overlay.py:255
def enumerate_gl2(n, max_count=None):
    """Enumerate elements of GL(2, Z_n)."""
    count = 0
    matrices = []
    for a in range(n):
        for b in range(n):
            for c_ in range(n):
                for d_ in range(n):
                    det = (a * d_ - b * c_) % n
                    if det != 0:
                        matrices.append([[a, b], [c_, d_]])
                        count += 1
                        if max_count and count >= max_count:
                            return matrices
    return matrices


# Source: algebraic_overlay.py:272
def sample_gl2(n, count):
    """Sample random elements from GL(2, Z_n)."""
    matrices = []
    while len(matrices) < count:
        a, b, c_, d_ = [np.random.randint(0, n) for _ in range(4)]
        det = (a * d_ - b * c_) % n
        if det != 0:
            matrices.append([[a, b], [c_, d_]])
    return matrices

# Voltage coverings of hypergraphs (covering_tower.py)

# Source: covering_tower.py:26
def fano_plane():
    """Fano plane: 7 vertices, 7 hyperedges of size 3. (3,3)-regular."""
    hyperedges = [
        (0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6),
        (4, 5, 0), (5, 6, 1), (6, 0, 2)
    ]
    return 7, 3, hyperedges


# Source: covering_tower.py:39 — preserves audit-flagged dead first construction
# (lines 46-67 of the original) for output equivalence; see AUDIT.md §2.3.
def pg23():
    """PG(2,3): projective plane of order 3. 13 points, 13 lines of size 4. (4,4)-regular."""
    pts = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if (a, b, c) != (0, 0, 0):
                    v = [a, b, c]
                    for i in range(3):
                        if v[i] != 0:
                            inv = pow(v[i], 1, 3)
                            v = [(x * pow(v[i], -1, 3)) % 3 if v[i] != 0
                                 else x for x_idx, x in enumerate(v)]
                            first_nz = v[i]
                            if first_nz == 2:
                                v = [(x * 2) % 3 for x in [a, b, c]]
                            else:
                                v = [a, b, c]
                            break
                    v_tuple = tuple(v)
                    if v_tuple not in pts:
                        pts.append(v_tuple)

    points = []
    seen = set()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if (a, b, c) == (0, 0, 0):
                    continue
                v = [a, b, c]
                for i in range(3):
                    if v[i] != 0:
                        if v[i] == 2:
                            v = [(x * 2) % 3 for x in v]
                        break
                key = tuple(v)
                if key not in seen:
                    seen.add(key)
                    points.append(key)

    assert len(points) == 13, f"Expected 13 points, got {len(points)}"

    lines_set = set()
    hyperedges = []
    for la, lb, lc in points:
        line = []
        for idx, (pa, pb, pc) in enumerate(points):
            if (la * pa + lb * pb + lc * pc) % 3 == 0:
                line.append(idx)
        line = tuple(sorted(line))
        if line not in lines_set and len(line) == 4:
            lines_set.add(line)
            hyperedges.append(line)

    assert len(hyperedges) == 13, f"Expected 13 lines, got {len(hyperedges)}"
    return 13, 4, hyperedges


# Source: covering_tower.py:108
def clique_expansion(N, r, hyperedges):
    """Build clique expansion adjacency matrix from hypergraph."""
    A = np.zeros((N, N))
    for he in hyperedges:
        for a, b in combinations(he, 2):
            A[a, b] += 1
            A[b, a] += 1
    return A


# Source: covering_tower.py:123
def voltage_covering(N0, r, hyperedges, k, voltage_assignment):
    """Build k-fold covering of base hypergraph via voltage assignment on Z_k."""
    N_lift = N0 * k
    A_lift = np.zeros((N_lift, N_lift))

    v_idx = 0
    for he_idx, he in enumerate(hyperedges):
        v = voltage_assignment[he_idx % len(voltage_assignment)]
        for i in range(len(he)):
            for j_idx in range(i + 1, len(he)):
                u, w = he[i], he[j_idx]
                if i == 0:
                    edge_voltage = v
                else:
                    edge_voltage = 0

                for sheet in range(k):
                    u_lift = u * k + sheet
                    w_lift = w * k + (sheet + edge_voltage) % k
                    A_lift[u_lift, w_lift] += 1
                    A_lift[w_lift, u_lift] += 1

    return A_lift


# Source: covering_tower.py:171
def random_voltage_covering(N0, r, hyperedges, k):
    """Build a k-fold covering with random voltage assignment on Z_k."""
    voltages = [np.random.randint(0, k) for _ in hyperedges]
    return voltage_covering(N0, r, hyperedges, k, voltages), voltages


# Source: covering_tower.py:177
def trivial_covering(N0, r, hyperedges, k):
    """Build the trivial k-fold covering (all voltages = 0)."""
    voltages = [0] * len(hyperedges)
    return voltage_covering(N0, r, hyperedges, k, voltages), voltages


# Hypergraph (Model A / Model B) constructors (aom_3d_routing.py)

# Source: aom_3d_routing.py:25
def build_2d_grid_hypergraph(n, r):
    """r-uniform hypergraph on n x n torus grid: row/col hyperedges."""
    N = n * n
    hyperedges = set()

    for i in range(n):
        for j in range(n):
            row_edge = tuple(sorted([coord_to_idx(i, (j + k) % n, n) for k in range(r)]))
            hyperedges.add(row_edge)
            col_edge = tuple(sorted([coord_to_idx((i + k) % n, j, n) for k in range(r)]))
            hyperedges.add(col_edge)

    hyperedges = list(hyperedges)
    return N, hyperedges


# Source: aom_3d_routing.py:45
def build_3d_enhanced_hypergraph(n, r, bypass_range=None):
    """r-uniform hypergraph with 3D AOL bypass: 2D + diagonals + long-range skips."""
    N = n * n
    if bypass_range is None:
        bypass_range = max(2, n // 4)

    _, hyperedges_2d = build_2d_grid_hypergraph(n, r)
    hyperedges = set(tuple(e) for e in hyperedges_2d)

    coords = torus_grid_coords(n)

    def torus_dist(c1, c2):
        di = min(abs(c1[0] - c2[0]), n - abs(c1[0] - c2[0]))
        dj = min(abs(c1[1] - c2[1]), n - abs(c1[1] - c2[1]))
        return di + dj

    for i in range(n):
        for j in range(n):
            diag_edge = tuple(sorted([coord_to_idx((i + k) % n, (j + k) % n, n)
                                      for k in range(r)]))
            hyperedges.add(diag_edge)

            adiag_edge = tuple(sorted([coord_to_idx((i + k) % n, (j - k) % n, n)
                                       for k in range(r)]))
            hyperedges.add(adiag_edge)

    stride = max(2, bypass_range // (r - 1))
    for i in range(n):
        for j in range(n):
            skip_row = tuple(sorted([coord_to_idx(i, (j + k * stride) % n, n)
                                     for k in range(r)]))
            hyperedges.add(skip_row)

            skip_col = tuple(sorted([coord_to_idx((i + k * stride) % n, j, n)
                                     for k in range(r)]))
            hyperedges.add(skip_col)

    hyperedges = list(hyperedges)
    return N, hyperedges


# Source: aom_3d_routing.py:107
def hypergraph_to_clique_expansion(N, hyperedges):
    """Build clique expansion graph (as adjacency dict) from hyperedges."""
    adj = defaultdict(set)
    for e in hyperedges:
        for u, w in combinations(e, 2):
            adj[u].add(w)
            adj[w].add(u)
    return adj


# Circulant clique-graph builder (derandomization.py)

# Source: derandomization.py:14
def build_circulant_clique(N, offsets, r):
    """Build clique expansion of circulant r-uniform hypergraph."""
    hyperedges = []
    for v in range(N):
        base = [v] + [(v + off) % N for off in offsets]
        if len(base) == r:
            edge = tuple(sorted(base))
            if edge not in hyperedges:
                hyperedges.append(edge)
    hyperedges = list(set(hyperedges))

    adj = defaultdict(set)
    for e in hyperedges:
        for u, w in combinations(e, 2):
            adj[u].add(w)
            adj[w].add(u)

    return hyperedges, adj
