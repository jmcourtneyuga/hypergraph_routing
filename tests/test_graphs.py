import numpy as np
import pytest

from routing_lib.graphs import (  # noqa: I001 -- preserved order
    build_2d_grid_clique,
    build_2d_grid_clique_torus,
    build_3d_aol_clique,
    build_host_graph_d_dprime,
    fano_plane,
    is_prime,
    pg23,
    random_regular_config_model,
    random_regular_matching_union,
    random_regular_networkx,
)

def test_config_model_shape_and_symmetry():
    np.random.seed(0)
    A = random_regular_config_model(N=20, d=4)
    assert A.shape == (20, 20)
    np.testing.assert_array_equal(A, A.T)  # symmetric
    assert np.all(np.diag(A) == 0)  # no self-loops


def test_config_model_rejects_odd_Nd():
    with pytest.raises(ValueError):
        random_regular_config_model(N=5, d=3)


def test_matching_union_is_symmetric_and_even_only():
    np.random.seed(1)
    A = random_regular_matching_union(N=10, d=4)
    assert A.shape == (10, 10)
    np.testing.assert_array_equal(A, A.T)
    with pytest.raises(ValueError):
        random_regular_matching_union(N=10, d=3)


def test_matching_union_sparse_path_returns_csr():
    from scipy.sparse import issparse
    np.random.seed(2)
    A = random_regular_matching_union(N=12, d=2, sparse=True)
    assert issparse(A)


def test_networkx_random_is_d_regular():
    A = random_regular_networkx(N=20, d=4, seed=42)
    # Returned as an integer adjacency matrix; every row sum equals d.
    assert A.shape == (20, 20)
    np.testing.assert_array_equal(A.sum(axis=1), 4 * np.ones(20, dtype=A.dtype))


def test_networkx_random_returns_none_on_odd_Nd():
    assert random_regular_networkx(N=5, d=3) is None

def test_2d_grid_clique_open_vertex_count():
    n = 5
    A = build_2d_grid_clique(n, r=3)
    assert A.shape[0] == n * n
    np.testing.assert_array_equal(A, A.T)


def test_2d_grid_clique_torus_vertex_count():
    n = 5
    A = build_2d_grid_clique_torus(n, r=3)
    assert A.shape[0] == n * n
    np.testing.assert_array_equal(A, A.T)


def test_3d_aol_clique_is_2d_grid_plus_extra_hyperedges():
    n = 4
    A = build_3d_aol_clique(n, r=3)
    assert A.shape[0] == n * n
    np.testing.assert_array_equal(A, A.T)
    A_2d = build_2d_grid_clique_torus(n, r=3)
    assert np.count_nonzero(A) > np.count_nonzero(A_2d)

def test_host_graph_d_dprime_shape():
    np.random.seed(3)
    A = build_host_graph_d_dprime(N_phys=32, d=4, r=3)
    assert A.shape == (32, 32)
    np.testing.assert_array_equal(A, A.T)

def test_fano_plane_structure():
    N, r, hyperedges = fano_plane()
    assert N == 7
    assert r == 3
    assert len(hyperedges) == 7  # 7 lines of the Fano plane
    for line in hyperedges:
        assert len(line) == 3
    # Every pair of points lies on exactly one line.
    from collections import Counter
    pair_counts = Counter()
    for line in hyperedges:
        for i in range(len(line)):
            for j in range(i + 1, len(line)):
                pair_counts[(min(line[i], line[j]), max(line[i], line[j]))] += 1
    assert all(c == 1 for c in pair_counts.values())
    assert len(pair_counts) == 7 * 6 // 2  # all C(7,2) pairs covered


def test_pg23_structure():
    N, r, hyperedges = pg23()
    assert N == 13
    assert r == 4
    assert len(hyperedges) == 13  # 13 lines of PG(2,3)
    for line in hyperedges:
        assert len(line) == 4

@pytest.mark.parametrize("p", [2, 3, 5, 7, 11, 13, 17, 19, 23])
def test_is_prime_true_for_primes(p):
    assert is_prime(p)


@pytest.mark.parametrize("n", [0, 1, 4, 6, 9, 15, 21, 25])
def test_is_prime_false_for_composites_and_units(n):
    assert not is_prime(n)
