"""Greedy displacement-matching helpers (Paper I Section 9, Theorems 9.3-9.4).

`grid_distance_open` and `grid_distance_torus` differ in whether the n x n
grid wraps. The greedy/displacement experiments use the open grid;
entanglement_routing uses the torus. Both are kept under distinct names
to avoid silent semantic substitution.

`displacement_energy`, `n_displaced`, `greedy_matching`, `random_matching`,
and `apply_matching` are used by `dynamic_adaptive.py` and the two
greedy-stall test scripts; the bodies are functionally identical across
all three sources, so there is one canonical version of each here.
"""

import numpy as np


# Source: dynamic_adaptive.py:40 (also test_concentration.py:21,
# test_greedy_stall_n256.py:21 — identical bodies)
def grid_distance_open(u, v, n):
    """L1 distance on the open n x n grid (no toroidal wrap)."""
    return abs(u // n - v // n) + abs(u % n - v % n)


# Source: entanglement_routing.py:70
def grid_distance_torus(u, v, n):
    """L1 distance on the n x n torus (with wraparound)."""
    ux, uy = u // n, u % n
    vx, vy = v // n, v % n
    dx = min(abs(ux - vx), n - abs(ux - vx))
    dy = min(abs(uy - vy), n - abs(uy - vy))
    return dx + dy


# Source: dynamic_adaptive.py:43
def displacement_energy(pos, target, n):
    return sum(grid_distance_open(pos[v], target[v], n) ** 2 for v in range(len(pos)))


# Source: dynamic_adaptive.py:46
def n_displaced(pos, target):
    return sum(1 for v in range(len(pos)) if pos[v] != target[v])


# Source: dynamic_adaptive.py:49
def greedy_matching(A, pos, target, n):
    """Greedy matching maximizing displacement reduction."""
    N = A.shape[0]
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if A[u, v] > 0:
                rho_u = grid_distance_open(pos[u], target[u], n)
                rho_v = grid_distance_open(pos[v], target[v], n)
                rho_u_new = grid_distance_open(pos[v], target[u], n)
                rho_v_new = grid_distance_open(pos[u], target[v], n)
                delta = (rho_u**2 + rho_v**2) - (rho_u_new**2 + rho_v_new**2)
                if delta > 0:
                    edges.append((delta, u, v))
    edges.sort(reverse=True)
    matched = set()
    matching = []
    total_delta = 0
    for delta, u, v in edges:
        if u not in matched and v not in matched:
            matching.append((u, v))
            matched.add(u)
            matched.add(v)
            total_delta += delta
    return matching, total_delta


# Source: dynamic_adaptive.py:75
def random_matching(A, N):
    """Random maximal matching."""
    adj = [[] for _ in range(N)]
    for u in range(N):
        for v in range(u + 1, N):
            if A[u, v] > 0:
                adj[u].append(v)
                adj[v].append(u)
    matched = set()
    matching = []
    for u in np.random.permutation(N):
        if u in matched:
            continue
        nbs = [v for v in adj[int(u)] if v not in matched]
        if nbs:
            v = nbs[np.random.randint(len(nbs))]
            matching.append((int(u), v))
            matched.add(int(u))
            matched.add(v)
    return matching


# Source: dynamic_adaptive.py:95
def apply_matching(pos, matching):
    new_pos = pos.copy()
    for u, v in matching:
        new_pos[u], new_pos[v] = pos[v], pos[u]
    return new_pos
