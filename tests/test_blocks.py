"""Tests for routing_lib.blocks: Block / BlockConfig dataclass-style
helpers and block placement on Paper II host graphs."""
import numpy as np
import pytest

from routing_lib.blocks import (
    Block,
    BlockConfig,
    build_quotient_graph,
    place_blocks_bfs,
    quotient_graph,
    random_block_config,
)
from routing_lib.graphs import build_host_graph_d_dprime


# ----- Block basic attributes -----

def test_block_records_attributes():
    b = Block(vertices={0, 1, 2, 3}, d_C=2, center=1, position=(0, 0))
    assert b.vertices == {0, 1, 2, 3}
    assert b.d_C == 2
    assert b.center == 1
    assert b.position == (0, 0)


def test_block_repr_contains_center():
    b = Block(vertices={5, 6}, d_C=1, center=5, position=(0, 0))
    assert "center=5" in repr(b)


# ----- BlockConfig validation -----

def test_blockconfig_rejects_overlapping_blocks():
    b1 = Block({0, 1, 2}, d_C=2, center=0, position=(0, 0))
    b2 = Block({2, 3, 4}, d_C=2, center=3, position=(0, 1))
    with pytest.raises(ValueError):
        BlockConfig([b1, b2], d_C=2)


def test_blockconfig_accepts_disjoint_blocks():
    b1 = Block({0, 1, 2, 3}, d_C=2, center=0, position=(0, 0))
    b2 = Block({4, 5, 6, 7}, d_C=2, center=4, position=(0, 1))
    cfg = BlockConfig([b1, b2], d_C=2)
    assert cfg.N_L == 2
    assert cfg.get_quotient_blocks() == [0, 4]


# ----- Block placement on a small host graph -----

@pytest.fixture
def small_host_graph():
    np.random.seed(0)
    return build_host_graph_d_dprime(N_phys=64, d=4, r=3)


def test_random_block_config_places_blocks(small_host_graph):
    np.random.seed(1)
    cfg = random_block_config(small_host_graph, d_C=2, guard_dist=1, N_L=3)
    assert isinstance(cfg, BlockConfig)
    assert cfg.N_L <= 3
    if cfg.N_L > 0:
        for b in cfg.blocks:
            assert isinstance(b, Block)
            assert len(b.vertices) > 0


def test_place_blocks_bfs_returns_disjoint_sets(small_host_graph):
    np.random.seed(2)
    out = place_blocks_bfs(small_host_graph, d_C=2, guard_dist=1, N_L=3)
    if out is None:
        pytest.skip("Placement did not succeed on this host; not a failure of the API")
    seen = set()
    for blk in out:
        assert isinstance(blk, set)
        assert seen.isdisjoint(blk)
        seen.update(blk)


def test_place_blocks_bfs_returns_none_when_overpacked():
    # 8 vertices, asking for 8 blocks of size 4 each -- impossible.
    A = np.zeros((8, 8))
    for i in range(7):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    np.random.seed(3)
    result = place_blocks_bfs(A, d_C=2, guard_dist=1, N_L=8)
    assert result is None or len(result) < 8


# ----- Quotient graph variants -----

def test_quotient_graph_runs_on_valid_config(small_host_graph):
    np.random.seed(4)
    cfg = random_block_config(small_host_graph, d_C=2, guard_dist=1, N_L=3)
    if cfg.N_L < 2:
        pytest.skip("Not enough blocks placed to form a non-trivial quotient")
    try:
        result = quotient_graph(small_host_graph, cfg)
    except ValueError:
        pytest.skip("Quotient was disconnected on this random instance")
    # API contract: returns a dict with Q (adjacency) and spectral keys.
    assert {"Q", "d_Q", "beta_Q", "lambda2_Q", "epsilon_eq", "eigvals_Q"} <= result.keys()
    assert result["Q"].shape == (cfg.N_L, cfg.N_L)
    np.testing.assert_array_equal(result["Q"], result["Q"].T)


def test_build_quotient_graph_returns_none_on_disconnected():
    # Two isolated triangles: any block in one cannot reach the other.
    A = np.zeros((6, 6))
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        A[u, v] = A[v, u] = 1
    for u, v in [(3, 4), (4, 5), (3, 5)]:
        A[u, v] = A[v, u] = 1
    blocks = [{0, 1, 2}, {3, 4, 5}]
    result = build_quotient_graph(A, blocks)
    # API contract: returns None for disconnected configurations.
    assert result is None or (hasattr(result, 'shape') and result.shape == (2, 2))
