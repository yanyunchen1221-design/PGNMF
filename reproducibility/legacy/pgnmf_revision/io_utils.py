from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import sparse


def load_imputed_matrix(path: str, layer: str = "imputed"):
    """Load an imputed matrix from .npy, .npz, or .h5ad."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return np.load(p)
    if suffix == ".npz":
        z = np.load(p)
        if "X_imputed" in z:
            return z["X_imputed"]
        if len(z.files) == 1:
            return z[z.files[0]]
        raise KeyError(".npz must contain 'X_imputed' or exactly one array")
    if suffix == ".h5ad":
        import scanpy as sc
        adata = sc.read_h5ad(p)
        if layer in adata.layers:
            return adata.layers[layer]
        # We do not silently assume .X is the imputed matrix unless the user explicitly asks.
        raise KeyError(f"Layer '{layer}' not found in {path}. Specify the correct layer.")
    raise ValueError("Supported imputed formats: .npy, .npz, .h5ad")
