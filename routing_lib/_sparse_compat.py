"""Compatibility helpers for handling both dense numpy and scipy.sparse
adjacency matrices in routing primitives.

Use `to_adj_list(A)` to extract the adjacency list regardless of A's
representation. Use `is_sparse(A)` to dispatch on representation type.

Memory note: at N >= 30000, dense float64 adjacency exceeds 7 GB. Use
scipy.sparse.csr_matrix for these scales (typically <100 MB at degree 16).
"""
import numpy as np


def is_sparse(A):
    """True if A is any scipy.sparse matrix."""
    try:
        from scipy.sparse import issparse
        return issparse(A)
    except ImportError:
        return False


def to_adj_list(A):
    """Return list[list[int]] of neighbor indices for each vertex.

    Works for:
      - np.ndarray (dense): uses np.where on each row
      - scipy.sparse.csr_matrix or compatible: uses CSR indptr/indices

    Returns adjacency where adj[i] is a Python list of int neighbors.
    Self-loops are stripped (we have no use for them in routing).

    NOTE: returning Python lists (not numpy arrays) is ~5-10x faster for
    BFS iteration because Python's `for v in list` is much faster than
    `for v in ndarray` (no numpy unboxing per element). For routing
    primitives that BFS heavily, this matters at N >= 10000.
    """
    if is_sparse(A):
        # Convert to CSR for efficient row iteration
        from scipy.sparse import csr_matrix
        if not isinstance(A, csr_matrix):
            A = A.tocsr()
        N = A.shape[0]
        indptr = A.indptr
        indices = A.indices
        adj = []
        for i in range(N):
            cols = indices[indptr[i]:indptr[i+1]]
            # Use .tolist() (C-level conversion) for speed; strip self-loops
            lst = cols.tolist()
            if i in lst:
                lst = [v for v in lst if v != i]
            adj.append(lst)
        return adj
    else:
        # Dense numpy
        N = A.shape[0]
        adj = []
        for i in range(N):
            row_nbrs = np.where(A[i] > 0)[0]
            row_nbrs = row_nbrs[row_nbrs != i]
            adj.append(row_nbrs.tolist())
        return adj


def adj_shape(A):
    """Get N for either dense or sparse adjacency."""
    return A.shape[0]


def num_edges(A):
    """Number of undirected edges in symmetric adjacency.
    Counts each (i, j) once for i < j."""
    if is_sparse(A):
        from scipy.sparse import triu
        return int(triu(A, k=1).nnz)
    else:
        return int((A > 0).sum() // 2)


def estimate_dense_memory_gb(N):
    """Memory required for an N x N float64 dense adjacency, in GB."""
    return (N * N * 8) / (2 ** 30)


def estimate_sparse_memory_mb(N, avg_degree):
    """Approximate memory required for an N-node CSR with given avg degree, in MB.
    Each edge: 8 bytes data + 4 bytes col index = 12 bytes. Plus N+1 indptr."""
    nnz = N * avg_degree
    return (nnz * 12 + (N + 1) * 4) / (2 ** 20)


def should_use_sparse(N, avg_degree=16, dense_threshold_gb=8.0):
    """Decision helper: True if dense N x N would exceed `dense_threshold_gb`."""
    return estimate_dense_memory_gb(N) > dense_threshold_gb
