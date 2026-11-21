from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy import sparse


@dataclass
class TopKInfo:
    rho: float
    n_zero: int
    k: int
    achieved_rate: float
    threshold: float | None
    tie_count_at_threshold: int
    tie_selected_at_threshold: int

    def to_dict(self):
        return asdict(self)


def exact_topk_preserve_observed(
    X_masked,
    X_hat_model: np.ndarray,
    model_gene_indices: np.ndarray,
    rho: float,
):
    """
    Preserve every observed value in X_masked exactly. Among zero entries restricted
    to the modeled genes, fill exactly floor(rho * N_zero_model) positions with the
    largest raw reconstruction values from X_hat_model.

    No z-score rescaling, no 0.8 factor, no quantile clipping, and no threshold-based
    over-selection at ties. np.argpartition selects exactly k positions.

    Returns a sparse full-gene matrix, plus selection metadata and the selected local
    row/gene coordinates.
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must lie in [0, 1]")
    model_gene_indices = np.asarray(model_gene_indices, dtype=np.int64)
    if X_hat_model.shape != (X_masked.shape[0], model_gene_indices.size):
        raise ValueError("X_hat_model shape does not match cells x modeled genes")

    if sparse.issparse(X_masked):
        base = X_masked.tocsr(copy=True)
        X_model_masked = base[:, model_gene_indices].toarray()
    else:
        base = sparse.csr_matrix(np.asarray(X_masked, dtype=np.float32))
        X_model_masked = np.asarray(X_masked[:, model_gene_indices], dtype=np.float32)

    zero_mask = X_model_masked <= 0
    zero_flat = np.flatnonzero(zero_mask.ravel())
    n_zero = int(zero_flat.size)
    k = int(np.floor(rho * n_zero))

    if k <= 0:
        info = TopKInfo(rho=float(rho), n_zero=n_zero, k=0, achieved_rate=0.0,
                        threshold=None, tie_count_at_threshold=0, tie_selected_at_threshold=0)
        return base, info, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    scores = np.asarray(X_hat_model, dtype=np.float32).ravel()[zero_flat]
    if np.any(~np.isfinite(scores)):
        raise ValueError("X_hat_model contains non-finite values at candidate zero positions")

    if k >= n_zero:
        selected_candidate_idx = np.arange(n_zero, dtype=np.int64)
    else:
        # Exact top-k. No '>=' threshold selection, so ties cannot inflate the fill rate.
        selected_candidate_idx = np.argpartition(scores, n_zero - k)[n_zero - k:]

    selected_flat = zero_flat[selected_candidate_idx]
    selected_scores = scores[selected_candidate_idx]
    local_rows, local_cols = np.unravel_index(selected_flat, zero_mask.shape)
    global_cols = model_gene_indices[local_cols]

    # The solver is nonnegative; guard against tiny numerical negatives anyway.
    out_dtype = base.dtype if hasattr(base, "dtype") else np.float32
    selected_values = np.maximum(selected_scores, 0.0).astype(out_dtype, copy=False)
    additions = sparse.coo_matrix(
        (selected_values, (local_rows, global_cols)), shape=base.shape, dtype=out_dtype
    ).tocsr()
    out = (base + additions).tocsr()
    out.eliminate_zeros()

    threshold = float(selected_scores.min()) if selected_scores.size else None
    if threshold is None:
        tie_count = tie_selected = 0
    else:
        tie_count = int(np.sum(scores == threshold))
        tie_selected = int(np.sum(selected_scores == threshold))

    info = TopKInfo(
        rho=float(rho),
        n_zero=n_zero,
        k=int(k),
        achieved_rate=float(k / n_zero) if n_zero else 0.0,
        threshold=threshold,
        tie_count_at_threshold=tie_count,
        tie_selected_at_threshold=tie_selected,
    )
    return out, info, local_rows.astype(np.int64), local_cols.astype(np.int64)
