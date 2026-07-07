from collections import defaultdict, deque
from itertools import combinations

import numpy as np
import networkx as nx

def random_regular_config_model(N, d):
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

def random_regular_matching_union(N, d, attempts=1, sparse=False):
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


def random_regular_networkx(N, d, seed=42):
    if N * d % 2 != 0:
        return None
    return nx.to_numpy_array(nx.random_regular_graph(d, N, seed=seed), dtype=int)


def random_regular_best_of(N, d=10, best_of=3):
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

def build_host_graph_d_dprime(N_phys, d, r):
    d_prime = d * (r - 1)
    if (N_phys * d_prime) % 2 != 0:
        N_phys += 1
    return random_regular_config_model(N_phys, d_prime)


def build_host_graph_dprime_only(N_phys, d_prime):
    if (N_phys * d_prime) % 2 != 0:
        N_phys += 1
    return random_regular_config_model(N_phys, d_prime)


def coord_to_idx(x, y, n):
    return (x % n) * n + (y % n)



def torus_grid_coords(n):
    return [(i // n, i % n) for i in range(n * n)]


def build_2d_grid_clique(n, r=3):
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


def build_2d_grid_clique_torus(n, r=3):
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


def build_3d_aol_clique(n, r=3):
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


def build_margulis_expander(n):
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

def build_combined_grid_overlay(n, r=3, overlay_type='margulis'):
    A_grid = build_2d_grid_clique_torus(n, r)
    N = n * n

    if overlay_type == 'margulis':
        A_overlay = build_margulis_expander(n)
    elif overlay_type == 'random':
        A_overlay = random_regular_best_of(N, d=10)
    else:
        raise ValueError(f"Unknown overlay type: {overlay_type}")

    return A_grid + A_overlay, A_overlay


def cayley_graph_Zn2(n, generators):
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


def cayley_eigenvalues_exact(n, generators):
    eigs = []
    for a in range(n):
        for b in range(n):
            lam = 0.0
            for g1, g2 in generators:
                lam += 2 * np.cos(2 * np.pi / n * (a * g1 + b * g2))
            eigs.append(lam)
    return np.array(sorted(eigs, reverse=True))


def margulis_gabber_galil_generators(n):
    return [(1, 0), (0, 1), (1, 1), (1, n - 1)]


def quadratic_residue_generators(n, d):
    gens = []
    for x in range(1, d + 1):
        g1 = x % n
        g2 = (x * x) % n
        gens.append((g1, g2))
    return gens


def cubic_generators(n, d):
    gens = []
    for x in range(1, d + 1):
        g1 = x % n
        g2 = (x * x * x) % n
        gens.append((g1, g2))
    return gens


def random_generators(n, d):
    gens = []
    for _ in range(d):
        g1 = np.random.randint(0, n)
        g2 = np.random.randint(0, n)
        while g1 == 0 and g2 == 0:
            g1 = np.random.randint(0, n)
            g2 = np.random.randint(0, n)
        gens.append((g1, g2))
    return gens

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


def gl2_count(n):

    if is_prime(n):
        return n * (n - 1) * (n - 1) * (n + 1)
    return None



def apply_affine(n, A_mat, c, v):

    x, y = v // n, v % n
    nx_ = (A_mat[0][0] * x + A_mat[0][1] * y + c[0]) % n
    ny = (A_mat[1][0] * x + A_mat[1][1] * y + c[1]) % n
    return nx_ * n + ny


def affine_permutation(n, A_mat, c):

    N = n * n
    perm = np.zeros(N, dtype=int)
    for v in range(N):
        perm[v] = apply_affine(n, A_mat, c, v)
    return perm



def det_mod(A_mat, n):

    return (A_mat[0][0] * A_mat[1][1] - A_mat[0][1] * A_mat[1][0]) % n



def enumerate_gl2(n, max_count=None):

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



def sample_gl2(n, count):
    matrices = []
    while len(matrices) < count:
        a, b, c_, d_ = [np.random.randint(0, n) for _ in range(4)]
        det = (a * d_ - b * c_) % n
        if det != 0:
            matrices.append([[a, b], [c_, d_]])
    return matrices


def fano_plane():
    hyperedges = [
        (0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6),
        (4, 5, 0), (5, 6, 1), (6, 0, 2)
    ]
    return 7, 3, hyperedges



def pg23():
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


def clique_expansion(N, r, hyperedges):
    A = np.zeros((N, N))
    for he in hyperedges:
        for a, b in combinations(he, 2):
            A[a, b] += 1
            A[b, a] += 1
    return A


def voltage_covering(N0, r, hyperedges, k, voltage_assignment):
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


def random_voltage_covering(N0, r, hyperedges, k):
    voltages = [np.random.randint(0, k) for _ in hyperedges]
    return voltage_covering(N0, r, hyperedges, k, voltages), voltages


def trivial_covering(N0, r, hyperedges, k):
    voltages = [0] * len(hyperedges)
    return voltage_covering(N0, r, hyperedges, k, voltages), voltages



def build_2d_grid_hypergraph(n, r):
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

def build_3d_enhanced_hypergraph(n, r, bypass_range=None):
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


def hypergraph_to_clique_expansion(N, hyperedges):
    adj = defaultdict(set)
    for e in hyperedges:
        for u, w in combinations(e, 2):
            adj[u].add(w)
            adj[w].add(u)
    return adj

def build_circulant_clique(N, offsets, r):
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
