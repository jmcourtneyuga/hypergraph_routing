from collections import defaultdict, deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from routing_lib.blocks import BlockConfig
from routing_lib._sparse_compat import to_adj_list, adj_shape, is_sparse

def bfs_path(adj, s, t):
    N = len(adj)
    prev = [-1] * N
    visited = [False] * N
    visited[s] = True
    queue = [s]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        if u == t:
            break
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                prev[v] = u
                queue.append(v)
    path = []
    u = t
    while u != -1:
        path.append(u)
        u = prev[u]
    return path[::-1]

def bfs_distance_matrix(A):
    N = A.shape[0]
    dist = np.full((N, N), N + 1, dtype=int)
    adj = [np.where(A[i] > 0)[0] for i in range(N)]
    for s in range(N):
        dist[s, s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[s, v] > dist[s, u] + 1:
                    dist[s, v] = dist[s, u] + 1
                    queue.append(v)
    return dist


def bfs_all_parents(A, N):
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                adj[i].append(j)

    parents = {}
    dists = {}
    for src in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
        parents[src] = parent
        dists[src] = dist
    return parents, dists


def bfs_all(A, N):
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                adj[i].append(j)

    parents = {}
    dist = np.full((N, N), -1, dtype=int)
    for src in range(N):
        d = [-1] * N
        p = [-1] * N
        d[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if d[w] == -1:
                    d[w] = d[u] + 1
                    p[w] = u
                    queue.append(w)
        parents[src] = p
        dist[src] = d
    return parents, dist


def bfs_shortest_paths(adj, N, source):
    dist = [-1] * N
    parent = [-1] * N
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for w in sorted(adj[u]):
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                parent[w] = u
                queue.append(w)
    return dist, parent


def canonical_path(parent, source, target):
    if source == target:
        return [source]
    path = []
    v = target
    while v != source:
        path.append(v)
        v = parent[v]
    path.append(source)
    path.reverse()
    return path


def bfs_diameter_and_paths(adj, N):
    diam = 0
    total_dist = 0
    count = 0

    for src in range(N):
        dist = [-1] * N
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        for t in range(N):
            if t != src and dist[t] >= 0:
                diam = max(diam, dist[t])
                total_dist += dist[t]
                count += 1

    avg_dist = total_dist / count if count > 0 else 0
    return diam, avg_dist


def path_edges(parents_or_path, src=None, dst=None):
    if src is None and dst is None:
        # Form 2: path is a list of vertices
        path = parents_or_path
        edges = []
        for i in range(len(path) - 1):
            edges.append(frozenset([path[i], path[i + 1]]))
        return edges
    # Form 1: parents dict
    parents = parents_or_path
    if src == dst:
        return []
    edges = []
    v = dst
    while v != src:
        u = parents[src][v]
        edges.append((min(u, v), max(u, v)))
        v = u
    return edges

def get_path_edges(parents, src, dst):
    if src == dst:
        return []
    edges = []
    v = dst
    while v != src:
        u = parents[src][v]
        edges.append((min(u, v), max(u, v)))
        v = u
    return edges


def compute_congestion_perm(adj, perm_map):
    N = len(adj)
    edge_load = {}
    max_D = 0
    for v in range(N):
        t = perm_map[v]
        path = bfs_path(adj, v, t)
        d = len(path) - 1
        max_D = max(max_D, d)
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            e = (min(a, b), max(a, b))
            edge_load[e] = edge_load.get(e, 0) + 1
    C = max(edge_load.values()) if edge_load else 0
    return C, max_D

def compute_congestion_paths(adj, N, perm):
    edge_load = defaultdict(int)
    for v in range(N):
        if perm[v] == v:
            continue
        dist, parent = bfs_shortest_paths(adj, N, v)
        p = canonical_path(parent, v, perm[v])
        for e in path_edges(p):
            edge_load[e] += 1
    return max(edge_load.values()) if edge_load else 0


def compute_edge_congestion(A, pi):
    N = adj_shape(A)
    adj = to_adj_list(A)
    edge_load = {}
    for v in range(N):
        path = bfs_path(adj, v, pi[v])
        for i in range(len(path) - 1):
            a, b = int(path[i]), int(path[i + 1])
            key = (min(a, b), max(a, b))
            edge_load[key] = edge_load.get(key, 0) + 1
    C = max(edge_load.values()) if edge_load else 0
    return C, edge_load


def compute_routing_congestion(adj, N, num_trials=500):
    all_parent = {}
    for v in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[v] = 0
        queue = deque([v])
        while queue:
            u = queue.popleft()
            for w in sorted(adj[u]):
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
        all_parent[v] = parent

    def get_path(src, dst):
        if src == dst:
            return []
        path = []
        v = dst
        while v != src:
            path.append(v)
            v = all_parent[src][v]
        path.append(src)
        path.reverse()
        edges = []
        for i in range(len(path) - 1):
            edges.append(frozenset([path[i], path[i + 1]]))
        return edges

    congestions = []
    for _ in range(num_trials):
        sigma = np.random.permutation(N)
        pi = np.random.permutation(N)

        edge_load = defaultdict(int)
        for v in range(N):
            for e in get_path(v, sigma[v]):
                edge_load[e] += 1
        for v in range(N):
            for e in get_path(sigma[v], pi[v]):
                edge_load[e] += 1

        C = max(edge_load.values()) if edge_load else 0
        congestions.append(C)

    return np.mean(congestions), np.median(congestions), np.max(congestions)


def valiant_trial(A, pi=None, sigma=None):
    N = adj_shape(A)
    adj = to_adj_list(A)
    if pi is None:
        pi = np.random.permutation(N)
    if sigma is None:
        sigma = np.random.permutation(N)

    max_D = 0
    edge_load1 = {}
    for v in range(N):
        path = bfs_path(adj, v, sigma[v])
        d = len(path) - 1
        max_D = max(max_D, d)
        for i in range(len(path) - 1):
            e = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            edge_load1[e] = edge_load1.get(e, 0) + 1

    edge_load2 = {}
    for v in range(N):
        path = bfs_path(adj, sigma[v], pi[v])
        d = len(path) - 1
        max_D = max(max_D, d)
        for i in range(len(path) - 1):
            e = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            edge_load2[e] = edge_load2.get(e, 0) + 1

    C = max(max(edge_load1.values()) if edge_load1 else 0,
            max(edge_load2.values()) if edge_load2 else 0)
    return C + max_D, C, max_D


def valiant_trial_cayley(A, pi, sigma):
    N = adj_shape(A)
    adj = to_adj_list(A)

    C1, D1 = compute_congestion_perm(adj, sigma)

    gather_map = np.zeros(N, dtype=int)
    for v in range(N):
        gather_map[sigma[v]] = pi[v]
    C2, D2 = compute_congestion_perm(adj, gather_map)

    C = max(C1, C2)
    D = max(D1, D2)
    return C + D, C, D


def valiant_routing_trial_grid(A, pi=None):
    N = adj_shape(A)
    if pi is None:
        pi = np.random.permutation(N)
    sigma = np.random.permutation(N)

    adj = to_adj_list(A)

    max_D = 0
    edge_load1 = {}

    for v in range(N):
        path = bfs_path(adj, v, sigma[v])
        d = len(path) - 1
        max_D = max(max_D, d)
        for i in range(len(path) - 1):
            a, b = int(path[i]), int(path[i + 1])
            edge_load1[(a, b)] = edge_load1.get((a, b), 0) + 1

    C1 = max(edge_load1.values()) if edge_load1 else 0

    edge_load2 = {}
    for v in range(N):
        path = bfs_path(adj, sigma[v], pi[v])
        d = len(path) - 1
        max_D = max(max_D, d)
        for i in range(len(path) - 1):
            a, b = int(path[i]), int(path[i + 1])
            edge_load2[(a, b)] = edge_load2.get((a, b), 0) + 1

    C2 = max(edge_load2.values()) if edge_load2 else 0
    C = max(C1, C2)
    return C + max_D, C, max_D


def valiant_routing_depth(A, n_trials=20):
    N = adj_shape(A)
    adj = to_adj_list(A)
    Ts = []
    for _ in range(n_trials):
        pi = np.random.permutation(N)
        sigma = np.random.permutation(N)
        max_C = 0
        max_D = 0
        edge_load1 = {}
        for v in range(N):
            path = bfs_path(adj, v, sigma[v])
            d = len(path) - 1
            max_D = max(max_D, d)
            for i in range(len(path) - 1):
                e = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                edge_load1[e] = edge_load1.get(e, 0) + 1
        edge_load2 = {}
        for v in range(N):
            path = bfs_path(adj, sigma[v], pi[v])
            d = len(path) - 1
            max_D = max(max_D, d)
            for i in range(len(path) - 1):
                e = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                edge_load2[e] = edge_load2.get(e, 0) + 1
        C = max(max(edge_load1.values()) if edge_load1 else 0,
                max(edge_load2.values()) if edge_load2 else 0)
        Ts.append(C + max_D)
    return int(np.median(Ts))


def valiant_routing_simulation(A, N, num_trials=100):
    parents, dists = bfs_all_parents(A, N)
    results = []

    for _ in range(num_trials):
        pi = np.random.permutation(N)
        sigma = np.random.permutation(N)

        edge_load = defaultdict(int)
        max_dilation = 0

        for v in range(N):
            path1 = get_path_edges(parents, v, sigma[v])
            max_dilation = max(max_dilation, len(path1))
            for e in path1:
                edge_load[e] += 1

            path2 = get_path_edges(parents, sigma[v], pi[v])
            max_dilation = max(max_dilation, len(path2))
            for e in path2:
                edge_load[e] += 1

        C = max(edge_load.values()) if edge_load else 0
        D = max_dilation
        results.append((C, D, C + D))

    return results


def valiant_routing_trial(parents, dist_matrix, N, pi, sigma):
    edge_load = defaultdict(int)
    max_dilation = 0

    for v in range(N):
        p1 = path_edges(parents, v, sigma[v])
        max_dilation = max(max_dilation, len(p1))
        for e in p1:
            edge_load[e] += 1
        p2 = path_edges(parents, sigma[v], pi[v])
        max_dilation = max(max_dilation, len(p2))
        for e in p2:
            edge_load[e] += 1

    C = max(edge_load.values()) if edge_load else 0
    return C, max_dilation

def routing_simulation(A, N, num_trials=100):
    parents, dist_matrix = bfs_all(A, N)
    results = []
    for _ in range(num_trials):
        pi = np.random.permutation(N)
        sigma = np.random.permutation(N)
        C, D = valiant_routing_trial(parents, dist_matrix, N, pi, sigma)
        results.append((C, D, C + D))
    return results


def edge_layer_assignment(parents, layers, N, pi, sigma):
    L = len(layers)
    layer_loads = [defaultdict(int) for _ in range(L)]

    for v in range(N):
        for phase_dst in [sigma[v], pi[v]]:
            src = v if phase_dst == sigma[v] else sigma[v]
            edges = path_edges(parents, src, phase_dst)
            for e in edges:
                i, j = e
                assigned = False
                for ell in range(L):
                    if layers[ell][i, j] > 0:
                        layer_loads[ell][e] += 1
                        assigned = True
                        break
                if not assigned:
                    layer_loads[0][e] += 1

    return layer_loads


def effective_capacity_with_crosstalk(L, k0, gamma):
    if gamma == 0:
        return L * k0

    active_layers = (L + 1) // 2
    per_layer_capacity = k0 / (1 + gamma)
    total = active_layers * per_layer_capacity

    all_active_per_layer = k0 / (1 + 2 * gamma)
    all_active_total = L * all_active_per_layer

    return max(total, all_active_total)

def shortest_paths_all_pairs(Q):
    Q_sparse = csr_matrix(Q)
    D = shortest_path(Q_sparse, directed=False, unweighted=True)
    return D

def _shortest_path_route(Q, src, dst, D):
    if src == dst:
        return [src]

    path = [src]
    current = src

    max_iterations = len(Q) * 2
    iterations = 0

    while current != dst and iterations < max_iterations:
        iterations += 1
        neighbors = np.where(Q[current, :] > 0)[0]

        if len(neighbors) == 0:
            break

        if np.isfinite(D[neighbors, dst]).any():
            next_hop = neighbors[np.argmin(D[neighbors, dst])]
        else:
            next_hop = neighbors[0]

        path.append(next_hop)
        current = next_hop

    if current != dst:
        if Q[src, dst] > 0:
            return [src, dst]
        else:
            return [src, dst]

    return path


def block_valiant_route(Q, source_perm):
    N_L = Q.shape[0]
    D = shortest_paths_all_pairs(Q)

    intermediate_perm = np.random.permutation(N_L)

    paths = []
    edge_load = defaultdict(int)

    max_path_len = 0

    for src in range(N_L):
        path1 = _shortest_path_route(Q, src, intermediate_perm[src], D)
        paths.append(path1)
        max_path_len = max(max_path_len, len(path1) - 1)

        for i in range(len(path1) - 1):
            u, v = path1[i], path1[i + 1]
            edge = (min(u, v), max(u, v))
            edge_load[edge] += 1

    paths_phase2 = []
    for src in range(N_L):
        intermediate_pos = intermediate_perm[src]
        target = source_perm[src]
        path2 = _shortest_path_route(Q, intermediate_pos, target, D)
        paths_phase2.append(path2)
        max_path_len = max(max_path_len, len(path2) - 1)

        for i in range(len(path2) - 1):
            u, v = path2[i], path2[i + 1]
            edge = (min(u, v), max(u, v))
            edge_load[edge] += 1

    combined_paths = [p1 + p2[1:] for p1, p2 in zip(paths, paths_phase2)]

    D_Q = max_path_len
    C_Q = max(edge_load.values()) if edge_load else 0

    return {
        'paths': combined_paths,
        'D_Q': D_Q,
        'C_Q': int(C_Q),
        'intermediate_perm': intermediate_perm,
        'edge_load': dict(edge_load)
    }


def block_lmr_schedule(Q, routing_result, config):
    paths = routing_result['paths']
    D_Q = routing_result['D_Q']
    C_Q = routing_result['C_Q']

    N_L = len(paths)
    frames = []
    max_step = max(len(p) for p in paths)

    for step in range(max_step - 1):
        active_blocks = set()
        for block_id, path in enumerate(paths):
            if step < len(path) - 1:
                active_blocks.add(block_id)

        matching = []
        used = set()

        for b in active_blocks:
            if b in used:
                continue

            for b2 in active_blocks:
                if b2 <= b or b2 in used:
                    continue

                if Q[b, b2] == 0:
                    matching.append((b, b2))
                    used.add(b)
                    used.add(b2)
                    break

            if b not in used:
                matching.append((b,))
                used.add(b)

        if matching:
            frames.append(matching)

    T_Q = len(frames)
    T_physical = config.d_C * (C_Q + D_Q)

    return {
        'frames': frames,
        'T_Q': T_Q,
        'T_physical': T_physical,
        'D_Q': D_Q,
        'C_Q': C_Q
    }


def simulate_block_routing(N_phys, d, r, d_C, guard_dist, N_L, n_trials=5):
    from routing_lib.graphs import build_host_graph_d_dprime
    from routing_lib.spectral import compute_graph_spectral_params
    from routing_lib.blocks import random_block_config, quotient_graph

    results = {
        'T': [], 'C_max': [], 'D_Q': [], 'beta_Q': [], 'beta': []
    }

    successful_trials = 0
    max_attempts_per_trial = 50

    for trial in range(n_trials):
        for attempt in range(max_attempts_per_trial):
            try:
                G_cl = build_host_graph_d_dprime(N_phys, d, r)
                _, _, beta, _ = compute_graph_spectral_params(G_cl)

                config = random_block_config(G_cl, d_C, guard_dist, N_L)

                if len(config.blocks) < N_L:
                    continue

                q_result = quotient_graph(G_cl, config)
                Q = q_result['Q']
                beta_Q = q_result['beta_Q']

                perm = np.random.permutation(config.N_L)

                routing_result = block_valiant_route(Q, perm)
                D_Q = routing_result['D_Q']
                C_Q = routing_result['C_Q']

                lmr_result = block_lmr_schedule(Q, routing_result, config)
                T_physical = lmr_result['T_physical']

                results['T'].append(T_physical)
                results['C_max'].append(C_Q)
                results['D_Q'].append(D_Q)
                results['beta_Q'].append(beta_Q)
                results['beta'].append(beta)

                successful_trials += 1
                break

            except (ValueError, IndexError):
                continue

    if successful_trials == 0:
        return {
            'T_mean': 0, 'T_median': 0, 'T_max': 0,
            'C_max_mean': 0, 'C_max_median': 0,
            'D_Q_mean': 0, 'beta_Q_mean': 0, 'beta_mean': 0,
            'all_trials': results, 'successful_trials': 0
        }

    stats = {
        'T_mean': np.mean(results['T']),
        'T_median': np.median(results['T']),
        'T_max': np.max(results['T']),
        'C_max_mean': np.mean(results['C_max']),
        'C_max_median': np.median(results['C_max']),
        'D_Q_mean': np.mean(results['D_Q']),
        'beta_Q_mean': np.mean(results['beta_Q']),
        'beta_mean': np.mean(results['beta']),
        'all_trials': results,
        'successful_trials': successful_trials
    }

    return stats


def point_routing_baseline(N_phys, d, r, n_trials=5):
    from routing_lib.graphs import build_host_graph_d_dprime
    results = []

    for trial in range(n_trials):
        G_cl = build_host_graph_d_dprime(N_phys, d, r)
        D = shortest_paths_all_pairs(G_cl)
        D_max = np.max(D[np.isfinite(D)])
        C_est = int(d * np.log2(N_phys))
        T_point = int(C_est + D_max)
        results.append(T_point)

    return {
        'T_mean': np.mean(results),
        'T_median': np.median(results),
        'T_max': np.max(results)
    }


def shortest_path_bfs(Q, src, dst):
    N = Q.shape[0]
    if src == dst:
        return [src]

    parent = {src: None}
    queue = deque([src])

    while queue:
        v = queue.popleft()
        if v == dst:
            break
        for u in range(N):
            if Q[v, u] > 0 and u not in parent:
                parent[u] = v
                queue.append(u)

    if dst not in parent:
        return None

    path = []
    v = dst
    while v is not None:
        path.append(v)
        v = parent[v]
    return list(reversed(path))

def valiant_route_simple(Q, perm):
    N_L = Q.shape[0]
    intermediate = np.random.permutation(N_L)

    edge_load = defaultdict(int)
    max_path_len = 0
    paths = []

    for i in range(N_L):
        p1 = shortest_path_bfs(Q, i, intermediate[i])
        p2 = shortest_path_bfs(Q, intermediate[i], perm[i])

        if p1 is None or p2 is None:
            return None, None, None

        full_path = p1 + p2[1:]
        paths.append(full_path)
        max_path_len = max(max_path_len, len(full_path) - 1)

        for k in range(len(full_path) - 1):
            e = (min(full_path[k], full_path[k + 1]), max(full_path[k], full_path[k + 1]))
            edge_load[e] += 1

    C_Q = max(edge_load.values()) if edge_load else 0
    D_Q = max_path_len

    return C_Q, D_Q, paths


def derandomize_valiant(adj, N, pi, lam=None):
    all_dist = {}
    all_parent = {}
    for v in range(N):
        d, p = bfs_shortest_paths(adj, N, v)
        all_dist[v] = d
        all_parent[v] = p

    all_edges = set()
    for v in range(N):
        for w in adj[v]:
            all_edges.add(frozenset([v, w]))
    all_edges = list(all_edges)
    edge_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)

    path_edge_sets = {}
    for v in range(N):
        path_edge_sets[v] = {}
        for w in range(N):
            if v == w:
                path_edge_sets[v][w] = []
            else:
                p = canonical_path(all_parent[v], v, w)
                path_edge_sets[v][w] = [edge_idx[frozenset([p[i], p[i + 1]])] for i in range(len(p) - 1)]

    D = max(all_dist[v][w] for v in range(N) for w in range(N))
    d_prime = min(len(adj[v]) for v in range(N))
    mu = 2 * D / d_prime

    if lam is None:
        threshold = max(4 * mu + 6 * np.log(N), 2 * np.log(N) + 10)
        lam = np.log(2 * num_edges) / threshold if threshold > 0 else 0.5

    scatter_load = np.zeros(num_edges, dtype=float)
    gather_load = np.zeros(num_edges, dtype=float)

    sigma = [-1] * N
    available = set(range(N))

    for v in range(N):
        best_target = None
        best_potential = float('inf')

        for c in available:
            scatter_delta = np.zeros(num_edges)
            for ei in path_edge_sets[v][c]:
                scatter_delta[ei] = 1.0

            gather_delta = np.zeros(num_edges)
            for ei in path_edge_sets[c][pi[v]]:
                gather_delta[ei] = 1.0

            new_scatter = scatter_load + scatter_delta
            new_gather = gather_load + gather_delta

            potential = np.sum(np.exp(lam * new_scatter)) + np.sum(np.exp(lam * new_gather))

            if potential < best_potential:
                best_potential = potential
                best_target = c

        sigma[v] = best_target
        available.remove(best_target)

        for ei in path_edge_sets[v][best_target]:
            scatter_load[ei] += 1.0
        for ei in path_edge_sets[best_target][pi[v]]:
            gather_load[ei] += 1.0

    C_scatter = int(max(scatter_load))
    C_gather = int(max(gather_load))

    return sigma, C_scatter, C_gather


def random_valiant(adj, N, pi, num_trials=500):
    best_C = float('inf')

    all_parent = {}
    for v in range(N):
        _, p = bfs_shortest_paths(adj, N, v)
        all_parent[v] = p

    all_edges = set()
    for v in range(N):
        for w in adj[v]:
            all_edges.add(frozenset([v, w]))

    results = []
    for _ in range(num_trials):
        sigma = np.random.permutation(N)

        edge_load_s = defaultdict(int)
        for v in range(N):
            if sigma[v] == v:
                continue
            p = canonical_path(all_parent[v], v, sigma[v])
            for e in path_edges(p):
                edge_load_s[e] += 1
        C_s = max(edge_load_s.values()) if edge_load_s else 0

        edge_load_g = defaultdict(int)
        for v in range(N):
            dest = pi[v]
            src = sigma[v]
            if src == dest:
                continue
            p = canonical_path(all_parent[src], src, dest)
            for e in path_edges(p):
                edge_load_g[e] += 1
        C_g = max(edge_load_g.values()) if edge_load_g else 0

        C_total = max(C_s, C_g)
        results.append((C_s, C_g, C_total))

    return results


def koenig_edge_color(adj_pairs, max_left, max_right):
    n_edges = len(adj_pairs)
    colors = [-1] * n_edges

    left_used = [set() for _ in range(max_left)]
    right_used = [set() for _ in range(max_right)]

    for idx, (u, v) in enumerate(adj_pairs):
        v_idx = v - max_left
        c = 0
        while c in left_used[u] or c in right_used[v_idx]:
            c += 1
        colors[idx] = c
        left_used[u].add(c)
        right_used[v_idx].add(c)

    n_colors = max(colors) + 1 if colors else 0
    return colors, n_colors

def koenig_edge_color_simple(edges, n_left, n_right):
    colors = [-1] * len(edges)
    used_left = [set() for _ in range(n_left)]
    used_right = [set() for _ in range(n_right)]

    for i, (u, v) in enumerate(edges):
        c = 0
        while c in used_left[u] or c in used_right[v]:
            c += 1
        colors[i] = c
        used_left[u].add(c)
        used_right[v].add(c)

    return colors, (max(colors) + 1 if colors else 0)


def simulate_block_translation(d_C, n_trials=10):
    n_atoms = d_C * d_C

    src_to_tgt_parallel = []
    for i in range(d_C):
        for j in range(d_C):
            u_idx = i * d_C + j
            v_idx = i * d_C + j
            src_to_tgt_parallel.append((u_idx, v_idx + n_atoms))

    _, n_colors_parallel = koenig_edge_color(
        src_to_tgt_parallel, n_atoms, n_atoms)

    np.random.seed(0)
    chromatic_indices = []
    for trial in range(n_trials):
        np.random.seed(trial)
        target_perm = np.random.permutation(n_atoms)
        edges = []
        for src in range(n_atoms):
            tgt = target_perm[src] + n_atoms
            edges.append((src, tgt - n_atoms))
        _, n_colors_random = koenig_edge_color(edges, n_atoms, n_atoms)
        chromatic_indices.append(n_colors_random)

    return {
        'n_atoms': n_atoms,
        'n_colors_parallel': n_colors_parallel,
        'mean_n_colors_random': np.mean(chromatic_indices),
        'max_n_colors_random': max(chromatic_indices),
        'd_C': d_C,
        'expected_O_dC': d_C,
        'expected_O_1_for_parallel': 1,
    }


def chromatic_index_block_translation(d_C, n_trials=10):
    n_atoms = d_C * d_C

    parallel_edges = [(i, i) for i in range(n_atoms)]
    _, n_colors_parallel = koenig_edge_color_simple(parallel_edges, n_atoms, n_atoms)

    chromatic_indices = []
    for trial in range(n_trials):
        np.random.seed(trial)
        target_perm = np.random.permutation(n_atoms)
        edges = list(enumerate(target_perm))
        _, n_colors = koenig_edge_color_simple(edges, n_atoms, n_atoms)
        chromatic_indices.append(n_colors)

    return {
        'd_C': d_C,
        'n_atoms': n_atoms,
        'n_colors_parallel': n_colors_parallel,
        'mean_n_colors_random': np.mean(chromatic_indices),
        'max_n_colors_random': max(chromatic_indices),
    }


def chromatic_index_corridor(d_C):
    n_atoms = d_C * d_C
    np.random.seed(42 + d_C)
    edge_count = {}

    for _ in range(n_atoms):
        for _ in range(d_C):
            u = np.random.randint(n_atoms)
            v = np.random.randint(n_atoms)
            if u == v:
                v = (v + 1) % n_atoms
            edge = (min(u, v), max(u, v))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    return max(edge_count.values()) if edge_count else 0
