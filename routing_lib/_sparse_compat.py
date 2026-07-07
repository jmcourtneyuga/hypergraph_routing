import numpy as np


def is_sparse(A):
    try:
        from scipy.sparse import issparse
        return issparse(A)
    except ImportError:
        return False


def to_adj_list(A):
    if is_sparse(A):
        from scipy.sparse import csr_matrix
        if not isinstance(A, csr_matrix):
            A = A.tocsr()
        N = A.shape[0]
        indptr = A.indptr
        indices = A.indices
        adj = []
        for i in range(N):
            cols = indices[indptr[i]:indptr[i+1]]
            lst = cols.tolist()
            if i in lst:
                lst = [v for v in lst if v != i]
            adj.append(lst)
        return adj
    else:
        N = A.shape[0]
        adj = []
        for i in range(N):
            row_nbrs = np.where(A[i] > 0)[0]
            row_nbrs = row_nbrs[row_nbrs != i]
            adj.append(row_nbrs.tolist())
        return adj


def adj_shape(A):
    return A.shape[0]


def num_edges(A):
    if is_sparse(A):
        from scipy.sparse import triu
        return int(triu(A, k=1).nnz)
    else:
        return int((A > 0).sum() // 2)


def estimate_dense_memory_gb(N):
    return (N * N * 8) / (2 ** 30)


def estimate_sparse_memory_mb(N, avg_degree):
    nnz = N * avg_degree
    return (nnz * 12 + (N + 1) * 4) / (2 ** 20)


def should_use_sparse(N, avg_degree=16, dense_threshold_gb=8.0):
    return estimate_dense_memory_gb(N) > dense_threshold_gb