"""Tests for routing_lib.spectral: Ramanujan bounds, beta conventions,
and the theoretical routing bound from Paper I."""
import numpy as np
import pytest

from routing_lib.graphs import build_2d_grid_clique, random_regular_networkx
from routing_lib.spectral import (
    compute_graph_spectral_params,
    compute_spectral_params,
    diameter_from_matrix,
    ramanujan_bound,
    ramanujan_bound_hyper,
    spectral_params_covering_tower,
    spectral_params_lam2_over_d,
    spectral_params_minimal,
    theoretical_routing_bound,
)


# ----- Ramanujan / Alon-Boppana bounds -----

def test_ramanujan_bound_formula():
    # 2*sqrt(d-1) for d >= 1, zero for d in {0, 1}.
    assert ramanujan_bound(0) == 0.0
    assert ramanujan_bound(1) == 0.0
    assert ramanujan_bound(2) == pytest.approx(2.0)
    assert ramanujan_bound(5) == pytest.approx(2 * np.sqrt(4))
    assert ramanujan_bound(10) == pytest.approx(2 * np.sqrt(9))


def test_ramanujan_bound_hyper_matches_graph_case_for_r_eq_2():
    # SFM bound 2*sqrt((d-1)(r-1)) with r=2 reduces to the graph bound.
    for d in range(2, 12):
        assert ramanujan_bound_hyper(d, 2) == pytest.approx(ramanujan_bound(d))


# ----- Beta convention consistency on a known graph -----

@pytest.fixture
def small_regular_adjacency():
    # 6-regular random graph on 20 vertices (returned as adjacency matrix).
    return random_regular_networkx(20, 6, seed=7).astype(float)


def test_lam2_over_d_returns_required_keys(small_regular_adjacency):
    out = spectral_params_lam2_over_d(small_regular_adjacency)
    for k in ("d_prime", "lambda_2", "lambda_N", "beta", "gap", "lambda_star"):
        assert k in out
    assert out["gap"] == pytest.approx(1 - out["beta"])
    assert out["lambda_star"] >= abs(out["lambda_N"]) - 1e-12
    assert out["lambda_star"] >= out["lambda_2"] - 1e-12


def test_minimal_subset_of_lam2_over_d(small_regular_adjacency):
    a = spectral_params_lam2_over_d(small_regular_adjacency)
    b = spectral_params_minimal(small_regular_adjacency)
    assert b["d_prime"] == pytest.approx(a["d_prime"])
    assert b["lambda_2"] == pytest.approx(a["lambda_2"])
    assert b["beta"] == pytest.approx(a["beta"])


def test_covering_tower_includes_all_eigs(small_regular_adjacency):
    out = spectral_params_covering_tower(small_regular_adjacency)
    assert "all_eigs" in out
    assert len(out["all_eigs"]) == small_regular_adjacency.shape[0]
    # eigenvalues sorted descending.
    eigs = out["all_eigs"]
    assert np.all(np.diff(eigs) <= 1e-10)


def test_compute_spectral_params_returns_6_tuple(small_regular_adjacency):
    d_prime, beta, lam_star, lam2, lamN, eigvals = compute_spectral_params(
        small_regular_adjacency)
    assert d_prime > 0
    assert 0 <= beta <= 1.0
    assert lam_star == pytest.approx(max(abs(lam2), abs(lamN)))


def test_compute_graph_spectral_params_returns_4_tuple(small_regular_adjacency):
    eigvals, d_prime, beta, lam2 = compute_graph_spectral_params(
        small_regular_adjacency)
    lambda1 = eigvals[0]
    assert beta == pytest.approx(
        max(abs(lam2), abs(eigvals[-1])) / lambda1, rel=1e-9
    )


# ----- Theoretical routing bound monotonicity -----

def test_routing_bound_handles_invalid_inputs():
    assert theoretical_routing_bound(0, 0.5, 64) == float("inf")
    assert theoretical_routing_bound(8, 0.0, 64) == float("inf")
    assert theoretical_routing_bound(8, 1.0, 64) == float("inf")


def test_routing_bound_increasing_in_N():
    b_small = theoretical_routing_bound(d_prime=8, beta=0.5, N=64)
    b_large = theoretical_routing_bound(d_prime=8, beta=0.5, N=1024)
    assert b_large > b_small


def test_routing_bound_increasing_in_beta_toward_1():
    # As beta -> 1, log2(1/beta) -> 0 and the bound diverges.
    b_low = theoretical_routing_bound(d_prime=8, beta=0.3, N=256)
    b_high = theoretical_routing_bound(d_prime=8, beta=0.9, N=256)
    assert b_high > b_low


# ----- Diameter helper -----

def test_diameter_of_connected_small_grid():
    A = build_2d_grid_clique(4, r=3)
    diam = diameter_from_matrix(A)
    assert isinstance(diam, int)
    assert diam > 0
    assert diam < A.shape[0]
