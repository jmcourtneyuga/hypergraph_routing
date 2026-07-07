import math
from typing import List

import numpy as np


def xu_1d_scramble_depth(perm: List[int]) -> int:
    return _scramble_depth_recursive(list(perm))


def _scramble_depth_recursive(perm: List[int]) -> int:
    n = len(perm)
    if n <= 1:
        return 0
    half = n // 2
    left = [p for p in perm if p < half]
    right = [p - half for p in perm if p >= half]
    subdepth = max(
        _scramble_depth_recursive(left) if left else 0,
        _scramble_depth_recursive(right) if right else 0,
    )
    return 1 + subdepth


def xu_2d_scramble_depth(perm_2d: np.ndarray, n: int) -> int:
    row_depths = []
    for r in range(n):
        row_targets = [perm_2d[r * n + c] for c in range(n)]
        col_targets = [t % n for t in row_targets]
        if sorted(col_targets) == list(range(n)):
            row_depths.append(xu_1d_scramble_depth(col_targets))
    row_depth = max(row_depths) if row_depths else 0

    col_depths = []
    for c in range(n):
        col_targets = [perm_2d[r * n + c] // n for r in range(n)]
        if sorted(col_targets) == list(range(n)):
            col_depths.append(xu_1d_scramble_depth(col_targets))
    col_depth = max(col_depths) if col_depths else 0

    return row_depth + col_depth


def xu_syndrome_extraction_depth(chromatic_index: int, n_qubits: int,
                                 amortize: str = 'pipelined_algorithm3',
                                 delta_c: int = 4,
                                 base_check_matrix: np.ndarray = None) -> int:
    if amortize == 'pipelined_algorithm3':
        if base_check_matrix is not None:
            H = np.asarray(base_check_matrix)
            row_deg = int(H.sum(axis=1).max())
            col_deg = int(H.sum(axis=0).max())
            delta_c_eff = max(row_deg, col_deg)
        else:
            delta_c_eff = delta_c
        return 2 * delta_c_eff

    n_side = int(round(math.sqrt(n_qubits)))
    log_side = max(1, math.ceil(math.log2(n_side)))
    per_layer = 2 * log_side

    if amortize == 'fresh':
        return chromatic_index * per_layer
    elif amortize == 'amortized':
        return per_layer + chromatic_index
    else:
        raise ValueError(f"unknown amortize mode: {amortize!r}")


def xu_random_2d_perm_depth_empirical(n: int, n_trials: int = 50,
                                       seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    depths = []
    for _ in range(n_trials):
        perm_2d = rng.permutation(n * n)
        depths.append(2 * xu_1d_scramble_depth(rng.permutation(n).tolist()))
    return {
        'median': float(np.median(depths)),
        'mean': float(np.mean(depths)),
        'max': int(np.max(depths)),
        'all': depths,
    }
