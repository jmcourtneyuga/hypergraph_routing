"""Block placement and quotient-graph machinery (Paper II).

Two block-placement variants are preserved:
    random_block_config — block_routing.py's class-based variant
                          (returns BlockConfig with `Block` objects)
    place_blocks_bfs    — validate_fixes.py's set-based variant
                          (returns list of vertex sets)

Two quotient-graph variants:
    quotient_graph       — block_routing.py's variant; raises if disconnected
    build_quotient_graph — validate_fixes.py's variant; returns None if
                           disconnected (does not raise)
"""

from collections import defaultdict, deque

import numpy as np

from routing_lib.spectral import compute_graph_spectral_params


# Source: block_routing.py:116
class Block:
    """Represents a rectangular block (surface code patch).

    Attributes:
        vertices: set of vertex indices in the host graph
        d_C: template width (d_C x d_C grid)
        center: center position vertex
        position: (row, col) position on logical grid
    """

    def __init__(self, vertices, d_C, center, position):
        self.vertices = set(vertices)
        self.d_C = d_C
        self.center = center
        self.position = position

    def __repr__(self):
        return f"Block(center={self.center}, pos={self.position}, size={len(self.vertices)})"


# Source: block_routing.py:136
class BlockConfig:
    """Block configuration on host graph G.

    Stores blocks and validates non-overlap and guard distance constraints.
    """

    def __init__(self, blocks, d_C, guard_dist=1):
        self.blocks = blocks
        self.d_C = d_C
        self.guard_dist = guard_dist
        self.N_L = len(blocks)
        self._validate()

    def _validate(self):
        """Check for overlaps and guard distance violations."""
        vertex_to_block = {}
        for b_idx, block in enumerate(self.blocks):
            for v in block.vertices:
                if v in vertex_to_block:
                    raise ValueError(f"Vertex {v} in multiple blocks")
                vertex_to_block[v] = b_idx

    def get_quotient_blocks(self):
        """Return list of block centers (quotient graph vertices)."""
        return [b.center for b in self.blocks]


# Source: block_routing.py:173
def random_block_config(G_cl, d_C, guard_dist, N_L, max_attempts=2000):
    """Place N_L blocks on G_cl by greedy BFS expansion from seed vertices.

    Blocks claim d_C^2 vertices each; guard zone is enforced softly.
    Returns a BlockConfig with as many blocks as could be placed before
    vertex pool ran out.
    """
    N_phys = G_cl.shape[0]
    blocks = []
    occupied = set()
    guard_zone = set()

    block_size = d_C * d_C
    min_block_size = max(2, int(block_size * 0.75))

    for block_idx in range(N_L):
        available_clean = [v for v in range(N_phys) if v not in occupied and v not in guard_zone]
        available_guard = [v for v in range(N_phys) if v not in occupied and v in guard_zone]
        candidates = available_clean + available_guard

        if not candidates:
            break

        np.random.shuffle(candidates)
        placed = False

        for seed in candidates[:100]:
            block_vertices = set()
            queue = deque([seed])
            visited = {seed}

            while len(block_vertices) < block_size and queue:
                v = queue.popleft()
                if v not in occupied:
                    block_vertices.add(v)

                if len(block_vertices) < block_size:
                    neighbors = np.where(G_cl[v, :] > 0)[0]
                    for u in neighbors:
                        if u not in visited and u not in occupied:
                            visited.add(u)
                            queue.append(u)

            if len(block_vertices) >= min_block_size:
                center = max(block_vertices,
                             key=lambda v: np.sum(G_cl[v, list(block_vertices)]))

                occupied.update(block_vertices)

                for v in block_vertices:
                    neighbors = np.where(G_cl[v, :] > 0)[0]
                    for u in neighbors:
                        if u not in occupied:
                            guard_zone.add(u)

                blocks.append(Block(block_vertices, d_C, center, (len(blocks), 0)))
                placed = True
                break

        if not placed:
            break

    return BlockConfig(blocks, d_C, guard_dist)


# Source: validate_fixes.py:124
def place_blocks_bfs(G, d_C, guard_dist, N_L):
    """Place N_L blocks of size d_C^2 on graph G using BFS.

    Returns list of blocks (each block = set of vertex indices), or None if failed.
    """
    N = G.shape[0]
    block_size = d_C * d_C
    blocks = []
    occupied = set()
    guard_zone = set()

    for _ in range(N_L):
        available_clean = [v for v in range(N) if v not in occupied and v not in guard_zone]
        available_guard = [v for v in range(N) if v not in occupied and v in guard_zone]

        candidates = available_clean + available_guard
        if not candidates:
            return None

        np.random.shuffle(candidates)
        placed = False

        for seed in candidates[:100]:
            block_verts = set()
            queue = deque([seed])
            visited = {seed}

            while len(block_verts) < block_size and queue:
                v = queue.popleft()
                if v not in occupied:
                    block_verts.add(v)

                if len(block_verts) < block_size:
                    neighbors = np.where(G[v, :] > 0)[0]
                    for u in neighbors:
                        if u not in visited and u not in occupied:
                            visited.add(u)
                            queue.append(u)

            if len(block_verts) >= block_size:
                blocks.append(block_verts)
                occupied.update(block_verts)

                boundary = set()
                for v in block_verts:
                    neighbors = np.where(G[v, :] > 0)[0]
                    for u in neighbors:
                        if u not in occupied and u not in block_verts:
                            boundary.add(v)
                            break

                for v in boundary:
                    neighbors = np.where(G[v, :] > 0)[0]
                    for u in neighbors:
                        if u not in occupied:
                            guard_zone.add(u)

                placed = True
                break

        if not placed:
            min_size = max(2, int(block_size * 0.75))
            for seed in candidates[:50]:
                block_verts = set()
                queue = deque([seed])
                visited = {seed}

                while len(block_verts) < block_size and queue:
                    v = queue.popleft()
                    if v not in occupied:
                        block_verts.add(v)
                    if len(block_verts) < block_size:
                        neighbors = np.where(G[v, :] > 0)[0]
                        for u in neighbors:
                            if u not in visited and u not in occupied:
                                visited.add(u)
                                queue.append(u)

                if len(block_verts) >= min_size:
                    blocks.append(block_verts)
                    occupied.update(block_verts)
                    placed = True
                    break

            if not placed:
                return None

    return blocks


# Source: block_routing.py:311
def _count_components(adj_matrix):
    """Count connected components in graph."""
    N = adj_matrix.shape[0]
    visited = set()
    components = 0

    for start in range(N):
        if start in visited:
            continue

        components += 1
        queue = deque([start])
        visited.add(start)

        while queue:
            v = queue.popleft()
            neighbors = np.where(adj_matrix[v, :] > 0)[0]
            for u in neighbors:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)

    return components


# Source: block_routing.py:246
def quotient_graph(G_cl, config):
    """Build the quotient graph Q(G, B) and return its spectral parameters.

    Raises ValueError if Q is disconnected.
    """
    N_L = config.N_L
    block_centers = config.get_quotient_blocks()

    vertex_to_block = {}
    for i, block in enumerate(config.blocks):
        for v in block.vertices:
            vertex_to_block[v] = i

    Q = np.zeros((N_L, N_L))
    edge_counts = defaultdict(int)

    for i, block_i in enumerate(config.blocks):
        for v in block_i.vertices:
            neighbors = np.where(G_cl[v, :] > 0)[0]
            for u in neighbors:
                if u in vertex_to_block:
                    j = vertex_to_block[u]
                    if i != j:
                        edge_counts[(min(i, j), max(i, j))] += 1

    for (i, j), count in edge_counts.items():
        Q[i, j] = max(count, 1)
        Q[j, i] = max(count, 1)

    if N_L > 1:
        component_count = _count_components(Q)
        if component_count > 1:
            raise ValueError(f"Quotient graph is disconnected ({component_count} components)")

    eigvals_Q, d_Q, beta_Q, lambda2_Q = compute_graph_spectral_params(Q)

    row_sums = Q.sum(axis=1)
    row_variation = np.max(row_sums) - np.min(row_sums) if len(row_sums) > 0 else 0
    epsilon_eq = row_variation / (d_Q + 1) if d_Q > 0 else 0

    return {
        'Q': Q,
        'd_Q': int(d_Q),
        'beta_Q': beta_Q,
        'lambda2_Q': lambda2_Q,
        'epsilon_eq': epsilon_eq,
        'eigvals_Q': eigvals_Q
    }


# Source: validate_fixes.py:226
def build_quotient_graph(G, blocks):
    """Build quotient graph from block configuration (validate_fixes variant).

    Q[i,j] = 1 iff there exist u in blocks[i], v in blocks[j] with G[u,v] > 0.
    Returns Q adjacency matrix, or None if disconnected.
    """
    N_L = len(blocks)
    v2b = {}
    for i, blk in enumerate(blocks):
        for v in blk:
            v2b[v] = i

    Q = np.zeros((N_L, N_L))
    for i, blk in enumerate(blocks):
        for v in blk:
            neighbors = np.where(G[v, :] > 0)[0]
            for u in neighbors:
                if u in v2b:
                    j = v2b[u]
                    if i != j:
                        Q[i, j] = 1
                        Q[j, i] = 1

    visited = set()
    queue = deque([0])
    visited.add(0)
    while queue:
        v = queue.popleft()
        for u in range(N_L):
            if Q[v, u] > 0 and u not in visited:
                visited.add(u)
                queue.append(u)

    if len(visited) < N_L:
        return None

    return Q
