import numpy as np
import pytest

from routing_lib.greedy import (
    apply_matching,
    displacement_energy,
    greedy_matching,
    grid_distance_open,
    grid_distance_torus,
    n_displaced,
    random_matching,
)

def test_grid_distance_open_zero_to_self():
    assert grid_distance_open(0, 0, 4) == 0
    assert grid_distance_open(7, 7, 4) == 0


def test_grid_distance_open_adjacent_cells():
    # Vertex i = row * n + col on an n x n grid; cells (0,0) and (0,1) are
    # neighbors with L1 distance 1.
    n = 4
    assert grid_distance_open(0, 1, n) == 1  # right neighbor
    assert grid_distance_open(0, n, n) == 1  # down neighbor


def test_grid_distance_open_diagonal():
    n = 4
    # (0,0) to (1,1)
    assert grid_distance_open(0, n + 1, n) == 2


def test_grid_distance_open_far_corners():
    n = 4
    # (0,0) to (n-1, n-1) on the open grid = 2(n-1).
    assert grid_distance_open(0, n * n - 1, n) == 2 * (n - 1)


def test_grid_distance_torus_wraps_around():
    n = 4
    # On the open grid (0,0) -> (0, n-1) costs n-1; on the torus it costs 1.
    assert grid_distance_open(0, n - 1, n) == n - 1
    assert grid_distance_torus(0, n - 1, n) == 1


def test_grid_distance_open_triangle_inequality():
    n = 5
    rng = np.random.RandomState(0)
    for _ in range(50):
        u, v, w = rng.randint(0, n * n, size=3)
        d_uv = grid_distance_open(u, v, n)
        d_vw = grid_distance_open(v, w, n)
        d_uw = grid_distance_open(u, w, n)
        assert d_uw <= d_uv + d_vw

def test_displacement_energy_zero_when_at_target():
    n = 3
    pos = list(range(n * n))
    assert displacement_energy(pos, pos, n) == 0
    assert n_displaced(pos, pos) == 0


def test_displacement_energy_positive_after_swap():
    n = 4
    pos = list(range(n * n))
    target = list(range(n * n))
    # Swap two adjacent cells in pos relative to target.
    pos[0], pos[1] = pos[1], pos[0]
    e = displacement_energy(pos, target, n)
    assert e > 0
    # Each swap displaces both endpoints by distance 1, so energy = 1^2 + 1^2 = 2.
    assert e == 2
    assert n_displaced(pos, target) == 2


def _path_graph_adjacency(n):
    """Adjacency matrix of an n-vertex path 0 - 1 - 2 - ... - (n-1)."""
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A


def test_apply_matching_swaps_positions():
    pos = [10, 20, 30, 40]
    new_pos = apply_matching(pos, [(0, 2), (1, 3)])
    assert new_pos == [30, 40, 10, 20]


def test_apply_matching_empty_is_identity():
    pos = [1, 2, 3, 4]
    assert apply_matching(pos, []) == pos


def test_random_matching_is_valid_matching():
    np.random.seed(0)
    N = 8
    A = _path_graph_adjacency(N)
    m = random_matching(A, N)
    # Every vertex appears at most once.
    seen = set()
    for u, v in m:
        assert u not in seen and v not in seen
        seen.add(u)
        seen.add(v)
        # Edge must exist in A.
        assert A[u, v] > 0


def test_greedy_matching_reduces_displacement():
    n = 4
    N = n * n
    A = _path_graph_adjacency(N)
    target = list(range(N))
    pos = list(range(N))
    pos[0], pos[1] = pos[1], pos[0]  # swap two grid-adjacent atoms
    e_before = displacement_energy(pos, target, n)
    matching, total_delta = greedy_matching(A, pos, target, n)
    assert total_delta > 0
    pos_after = apply_matching(pos, matching)
    e_after = displacement_energy(pos_after, target, n)
    assert e_after < e_before
    assert e_after == 0  # the single swap fully restores the target


def test_greedy_matching_no_op_when_already_at_target():
    n = 3
    N = n * n
    A = _path_graph_adjacency(N)
    target = list(range(N))
    pos = list(range(N))
    matching, total_delta = greedy_matching(A, pos, target, n)
    assert matching == []
    assert total_delta == 0
