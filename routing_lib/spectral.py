"""Spectral-analysis helpers for adjacency matrices.

Three beta conventions and two lambda_star formulas coexist in the
original scripts; each is preserved here under a distinct name. The
docstring on each function names the scripts that use that variant.

Beta conventions:
    lam2_over_d        beta = lambda_2 / d_prime
    lambda_star_lam2   beta = max(lambda_2, |lambda_N|) / d_prime
    lambda_star_abs    beta = max(|lambda_2|, |lambda_N|) / lambda_1

Eigvalsh source: numpy.linalg vs scipy.linalg differ in LAPACK driver and
can produce slightly different eigenvalues at machine precision; per-
function aliases preserve which library each original script used.
"""

import numpy as np
from numpy.linalg import eigvalsh as _eigvalsh_np
from scipy.linalg import eigvalsh as _eigvalsh_sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh


# Exists in multilayer_aol.py:90 (verbatim in sparse_overlay_routing.py:61,
# entanglement_routing.py:38, aol_constrained_routing.py:160).
# Beta = lambda_2 / d_prime; lambda_star = max(lam2, abs(lamN)).
def spectral_params_lam2_over_d(A):
    """Compute full spectral parameters.

    Returns dict with keys: d_prime, lambda_2, lambda_N, beta, gap,
    lambda_star.
    """
    eigs = np.sort(_eigvalsh_sp(A))[::-1]
    d_prime = eigs[0]
    lam2 = eigs[1]
    lamN = eigs[-1]
    beta = lam2 / d_prime if d_prime > 0 else 1.0
    return {
        'd_prime': d_prime,
        'lambda_2': lam2,
        'lambda_N': lamN,
        'beta': beta,
        'gap': 1 - beta,
        'spectral_gap': 1 - beta,
        'lambda_star': max(lam2, abs(lamN)),
    }


# Exists in entanglement_routing.py:38 (slimmer dict — only beta/gap/d_prime/lambda_2).
# Kept as a separate variant so callers that only want the smaller dict
# continue to receive exactly those keys.
def spectral_params_minimal(A):
    eigs = np.sort(_eigvalsh_sp(A))[::-1]
    d_prime = eigs[0]
    lam2 = eigs[1]
    beta = lam2 / d_prime if d_prime > 0 else 1.0
    return {'d_prime': d_prime, 'lambda_2': lam2, 'beta': beta, 'gap': 1 - beta}


# Source: dynamic_adaptive.py:35 (also hierarchical_routing.py:45 — identical body).
# Returns a 3-tuple, not a dict; uses np.abs(eigvalsh) so eigenvalues are
# already non-negative when sorted ascending. Different beta semantics from
# the dict variants — keep separate.
def spectral_params_abs_eigs(A):
    eigs = np.sort(np.abs(_eigvalsh_np(A)))
    d, lam2 = eigs[-1], eigs[-2] if len(eigs) > 1 else 0
    return d, lam2, lam2 / d if d > 0 else 1.0


# Source: algebraic_overlay.py:66
def spectral_params_from_eigs(eigs):
    """Compute spectral parameters from a precomputed eigenvalue array."""
    d_prime = eigs[0]
    lam2 = eigs[1]
    lamN = eigs[-1]
    beta = lam2 / d_prime if d_prime > 0 else 1.0
    lam_star = max(abs(lam2), abs(lamN))
    return {
        'd_prime': d_prime, 'lambda_2': lam2, 'lambda_N': lamN,
        'lambda_star': lam_star, 'beta': beta, 'gap': 1 - beta,
    }


# Source: covering_tower.py:188 — uses numpy.linalg.eigvalsh AND
# lambda_star = max(abs(lam2), abs(lamN)). Different from the multilayer_aol
# variant. Returns dict with `all_eigs` included.
def spectral_params_covering_tower(A):
    eigs = np.sort(_eigvalsh_np(A))[::-1]
    d_prime = eigs[0]
    lam2 = eigs[1]
    lamN = eigs[-1]
    beta = lam2 / d_prime if d_prime > 0 else 1.0
    return {
        'd_prime': d_prime, 'lambda_2': lam2, 'lambda_N': lamN,
        'beta': beta, 'gap': 1 - beta,
        'lambda_star': max(abs(lam2), abs(lamN)),
        'all_eigs': eigs,
    }


# Source: validate_fixes.py:97 — Paper II convention, returns 6-tuple.
# Beta = max(|lam_2|, |lam_N|) / d_prime.
def compute_spectral_params(A):
    """Spectral parameters for Paper II validation; returns 6-tuple
    (d_prime, beta, lambda_star, lambda2, lambda_N, eigvals)."""
    degrees = A.sum(axis=1)
    d_prime = np.mean(degrees)

    eigvals = _eigvalsh_sp(A)
    eigvals = np.sort(eigvals)[::-1]

    lambda1 = eigvals[0]
    lambda2 = eigvals[1] if len(eigvals) > 1 else 0
    lambda_N = eigvals[-1]

    lambda_star = max(abs(lambda2), abs(lambda_N))
    beta = lambda_star / d_prime if d_prime > 0 else 1.0

    return d_prime, beta, lambda_star, lambda2, lambda_N, eigvals


# Source: block_routing.py:79 — Paper II convention, returns 4-tuple.
# Beta = max(|lam_2|, |lam_N|) / lambda_1 (note: lambda_1 not d_prime).
def compute_graph_spectral_params(A):
    """Compute spectral parameters of adjacency matrix per Paper I definition.

    Beta is the spectral ratio: lambda* / lambda_1 where
    lambda* = max(|lambda_2|, |lambda_N|) per Paper I.

    Returns: (eigenvalues, degree d', spectral ratio beta, second eigenvalue lambda2)
    """
    d_prime = np.mean(A.sum(axis=1))
    eigvals = _eigvalsh_sp(A)
    eigvals = np.sort(eigvals)[::-1]

    lambda1 = eigvals[0]
    lambda2 = eigvals[1] if len(eigvals) > 1 else 0
    lambda_N = eigvals[-1] if len(eigvals) > 0 else 0

    lambda_star = max(abs(lambda2), abs(lambda_N))
    beta = lambda_star / lambda1 if lambda1 > 0 else 0

    return eigvals, d_prime, beta, lambda2


# Source: algebraic_overlay.py:79 (also multilayer_aol.py:116 — identical),
# sparse_overlay_routing.py:75 — same body.
def ramanujan_bound(d):
    """Alon-Boppana bound: 2*sqrt(max(d-1, 0))."""
    return 2.0 * np.sqrt(max(d - 1, 0))


# Source: covering_tower.py:202
def ramanujan_bound_hyper(d, r):
    """SFM Ramanujan bound: |lambda_i - (r-2)| <= 2*sqrt((d-1)*(r-1))."""
    return 2 * np.sqrt((d - 1) * (r - 1))


# Source: multilayer_aol.py:107 (also at aol_constrained_routing.py:268 in
# capacity_analysis — identical formula)
def theoretical_routing_bound(d_prime, beta, N):
    """Tightened upper bound from the routing theorem (Paper I)."""
    if beta >= 1 or beta <= 0 or d_prime <= 0:
        return float('inf')
    log2N = np.log2(N)
    log2_inv_beta = np.log2(1.0 / beta)
    return (4 * (d_prime + 6) / (d_prime * log2_inv_beta)) * log2N + 19 * log2N


# Source: aom_3d_routing.py:116
def compute_spectrum(adj, N, k=6):
    """Compute top-k eigenvalues of an adjacency dict using sparse eigsh."""
    rows, cols, data = [], [], []
    for u in range(N):
        for w in adj[u]:
            rows.append(u)
            cols.append(w)
            data.append(1.0)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    try:
        eigenvalues = eigsh(A, k=min(k, N - 2), which='LM', return_eigenvectors=False)
        eigenvalues = sorted(eigenvalues, reverse=True)
    except Exception:
        eigenvalues = [0] * k

    return eigenvalues


# Source: aol_constrained_routing.py:178
def diameter_from_matrix(A):
    """Compute diameter using scipy shortest_path."""
    A_binary = csr_matrix(A > 0, dtype=float)
    dist = shortest_path(A_binary, directed=False, unweighted=True)
    if np.any(np.isinf(dist)):
        return float('inf')
    return int(np.max(dist))
