import numpy as np
from numpy.linalg import eigvalsh as _eigvalsh_np
from scipy.linalg import eigvalsh as _eigvalsh_sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh

def spectral_params_lam2_over_d(A):
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


def spectral_params_minimal(A):
    eigs = np.sort(_eigvalsh_sp(A))[::-1]
    d_prime = eigs[0]
    lam2 = eigs[1]
    beta = lam2 / d_prime if d_prime > 0 else 1.0
    return {'d_prime': d_prime, 'lambda_2': lam2, 'beta': beta, 'gap': 1 - beta}


def spectral_params_abs_eigs(A):
    eigs = np.sort(np.abs(_eigvalsh_np(A)))
    d, lam2 = eigs[-1], eigs[-2] if len(eigs) > 1 else 0
    return d, lam2, lam2 / d if d > 0 else 1.0


def spectral_params_from_eigs(eigs):
    d_prime = eigs[0]
    lam2 = eigs[1]
    lamN = eigs[-1]
    beta = lam2 / d_prime if d_prime > 0 else 1.0
    lam_star = max(abs(lam2), abs(lamN))
    return {
        'd_prime': d_prime, 'lambda_2': lam2, 'lambda_N': lamN,
        'lambda_star': lam_star, 'beta': beta, 'gap': 1 - beta,
    }


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


def compute_spectral_params(A):
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

def compute_graph_spectral_params(A):
    d_prime = np.mean(A.sum(axis=1))
    eigvals = _eigvalsh_sp(A)
    eigvals = np.sort(eigvals)[::-1]

    lambda1 = eigvals[0]
    lambda2 = eigvals[1] if len(eigvals) > 1 else 0
    lambda_N = eigvals[-1] if len(eigvals) > 0 else 0

    lambda_star = max(abs(lambda2), abs(lambda_N))
    beta = lambda_star / lambda1 if lambda1 > 0 else 0

    return eigvals, d_prime, beta, lambda2


def ramanujan_bound(d):
    return 2.0 * np.sqrt(max(d - 1, 0))


def ramanujan_bound_hyper(d, r):
    return 2 * np.sqrt((d - 1) * (r - 1))


def theoretical_routing_bound(d_prime, beta, N):
    if beta >= 1 or beta <= 0 or d_prime <= 0:
        return float('inf')
    log2N = np.log2(N)
    log2_inv_beta = np.log2(1.0 / beta)
    return (4 * (d_prime + 6) / (d_prime * log2_inv_beta)) * log2N + 19 * log2N


def compute_spectrum(adj, N, k=6):
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

def diameter_from_matrix(A):
    A_binary = csr_matrix(A > 0, dtype=float)
    dist = shortest_path(A_binary, directed=False, unweighted=True)
    if np.any(np.isinf(dist)):
        return float('inf')
    return int(np.max(dist))
