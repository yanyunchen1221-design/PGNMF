from __future__ import annotations

import numpy as np
from scipy import sparse


def is_sparse(x) -> bool:
    return sparse.issparse(x)


def matrix_shape(x):
    return x.shape


def matrix_max(x) -> float:
    if sparse.issparse(x):
        if x.nnz == 0:
            return 0.0
        return float(x.data.max())
    return float(np.asarray(x).max())


def take_values(x, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Return x[rows[k], cols[k]] as a 1-D float array without densifying all of x."""
    if len(rows) != len(cols):
        raise ValueError("rows and cols must have the same length")
    if sparse.issparse(x):
        csr = x.tocsr()
        # scipy fancy indexing can return matrix-shaped objects; flatten explicitly.
        return np.asarray(csr[rows, cols]).reshape(-1).astype(float)
    arr = np.asarray(x)
    return arr[rows, cols].astype(float, copy=False)


def set_values_zero(x, rows: np.ndarray, cols: np.ndarray):
    """Return a copy of x with selected coordinates set to zero."""
    if sparse.issparse(x):
        out = x.tolil(copy=True)
        out[rows, cols] = 0.0
        out = out.tocsr()
        out.eliminate_zeros()
        return out
    out = np.array(x, copy=True)
    out[rows, cols] = 0.0
    return out


def to_dense_float(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray().astype(float, copy=False)
    return np.asarray(x, dtype=float)
