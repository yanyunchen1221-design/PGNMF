"""PGNMF exact gene-block multiplicative-update solver.

This module intentionally exposes the exact gene-block solver only.
The full-matrix reference solver used during development is not included.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse


@dataclass
class SolverResult:
    W: np.ndarray
    H: np.ndarray
    losses: list[float]
    n_iter: int
    elapsed_seconds: float
    converged: bool

    def summary(self) -> dict:
        return {
            "n_iter": int(self.n_iter),
            "elapsed_seconds": float(self.elapsed_seconds),
            "converged": bool(self.converged),
            "initial_loss": float(self.losses[0]) if self.losses else None,
            "final_loss": float(self.losses[-1]) if self.losses else None,
            "loss_nonincreasing": (
                bool(all(
                    self.losses[i] <= self.losses[i - 1] * (1.0 + 1e-6) + 1e-7
                    for i in range(1, len(self.losses))
                )) if len(self.losses) > 1 else True
            ),
        }


def _h_times_sparse(H: np.ndarray, S: sparse.spmatrix) -> np.ndarray:
    return np.asarray((S.T @ H.T).T, dtype=np.float32)


def _sparse_times_w(S: sparse.spmatrix, W: np.ndarray) -> np.ndarray:
    return np.asarray(S @ W, dtype=np.float32)


def _weighted_prediction(
    WH: np.ndarray,
    observed: np.ndarray,
    zero_weight: float,
    zero_penalty_exclude_rows: np.ndarray | None = None,
    zero_penalty_exclude_cols: np.ndarray | None = None,
) -> np.ndarray:
    """Return Q⊙WH, with optional zero positions excluded from the weak penalty.

    Positive observed entries have weight 1; ordinary input zeros have
    ``zero_weight``; explicitly excluded held-out zeros have weight 0.
    """
    if zero_weight == 0.0:
        return WH * observed
    out = np.asarray(WH * np.float32(zero_weight), dtype=np.float32)
    out[observed] = WH[observed]
    if zero_penalty_exclude_rows is not None:
        out[zero_penalty_exclude_rows, zero_penalty_exclude_cols] = 0.0
    return out


def _objective_blockwise(
    X: np.ndarray,
    observed: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    zero_weight: float,
    S_cell: sparse.spmatrix,
    d_cell: np.ndarray,
    alpha_cell: float,
    S_gene: sparse.spmatrix,
    d_gene: np.ndarray,
    alpha_gene: float,
    lambda_h_gene: np.ndarray,
    lambda_reg: float,
    lambda_w_l1: float,
    block_size: int,
    zero_penalty_exclude_rows: np.ndarray | None = None,
    zero_penalty_exclude_cols: np.ndarray | None = None,
) -> float:
    """Evaluate the unchanged full objective without materializing full W@H."""
    n_genes = X.shape[1]
    loss = 0.0

    for start in range(0, n_genes, block_size):
        stop = min(start + block_size, n_genes)
        sl = slice(start, stop)
        WH_b = W @ H[:, sl]
        observed_b = observed[:, sl]
        residual = (X[:, sl] - WH_b)[observed_b]
        loss += 0.5 * float(np.dot(residual, residual))

        if zero_weight > 0:
            zero_pred = WH_b[~observed_b]
            zero_sq = float(np.dot(zero_pred, zero_pred))
            if zero_penalty_exclude_rows is not None:
                in_block = (
                    (zero_penalty_exclude_cols >= start)
                    & (zero_penalty_exclude_cols < stop)
                )
                if np.any(in_block):
                    local_cols = zero_penalty_exclude_cols[in_block] - start
                    excluded_pred = WH_b[
                        zero_penalty_exclude_rows[in_block], local_cols
                    ]
                    zero_sq -= float(np.dot(excluded_pred, excluded_pred))
                    zero_sq = max(zero_sq, 0.0)
            loss += 0.5 * float(zero_weight) * zero_sq

    if alpha_cell > 0:
        SW = _sparse_times_w(S_cell, W)
        tr_cell = float(
            np.sum(W * (d_cell[:, None] * W - SW), dtype=np.float64)
        )
        if tr_cell < 0 and abs(tr_cell) < 1e-4:
            tr_cell = 0.0
        loss += 0.5 * float(alpha_cell) * tr_cell

    if alpha_gene > 0:
        HS = _h_times_sparse(H, S_gene)
        tr_gene = float(
            np.sum(H * (H * d_gene[None, :] - HS), dtype=np.float64)
        )
        if tr_gene < 0 and abs(tr_gene) < 1e-4:
            tr_gene = 0.0
        loss += 0.5 * float(alpha_gene) * tr_gene

    if lambda_reg > 0:
        loss += 0.5 * float(lambda_reg) * (
            float(np.sum(W * W, dtype=np.float64))
            + float(np.sum(H * H, dtype=np.float64))
        )

    if lambda_w_l1 > 0:
        loss += float(lambda_w_l1) * float(np.sum(W, dtype=np.float64))

    if np.any(lambda_h_gene > 0):
        loss += float(np.sum(H * lambda_h_gene[None, :], dtype=np.float64))

    return float(loss)


def zero_aware_nmf_mu_blocked(
    X: np.ndarray,
    S_cell: sparse.spmatrix,
    degree_cell: np.ndarray,
    alpha_cell: float,
    S_gene: sparse.spmatrix,
    degree_gene: np.ndarray,
    alpha_gene: float,
    lambda_h_gene: np.ndarray,
    zero_weight: float = 0.0,
    rank: int = 20,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    lambda_reg: float = 1e-3,
    lambda_w_l1: float = 0.0,
    verbose: bool = True,
    tag: str = "PGNMF-ExactBlock",
    block_size: int = 500,
    zero_penalty_exclude_rows: np.ndarray | None = None,
    zero_penalty_exclude_cols: np.ndarray | None = None,
    iteration_callback=None,
) -> SolverResult:
    """
    Exact gene-block implementation of the PGNMF multiplicative updates.

    Genes are partitioned into disjoint contiguous blocks. The complete gene
    graph is retained: H @ S_gene is computed from the full H and full S_gene
    before any H block is updated. Thus every H block receives contributions
    from within-block and cross-block gene edges.

    The reconstruction, zero-background, cell-graph, gene-graph, L2, and
    gene-specific sparsity terms are unchanged. Reconstruction products are
    processed blockwise in the H update, W update, and objective evaluation.

    block_size is a computational chunk size, not a model hyperparameter.
    Optional zero-penalty exclusions are retained for compatibility with the
    biological held-out fitting path. If supplied, ``iteration_callback`` is
    called after each completed iteration as ``callback(iteration, loss, W, H)``.
    The callback is intended for read-only diagnostics and must not modify W or H.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or np.any(~np.isfinite(X)) or np.any(X < 0):
        raise ValueError("X must be a finite nonnegative 2-D matrix")
    if rank <= 0 or rank > min(X.shape):
        raise ValueError("invalid rank")
    if min(alpha_cell, alpha_gene, lambda_reg, lambda_w_l1, zero_weight) < 0:
        raise ValueError("regularization strengths and zero_weight must be nonnegative")
    if zero_weight > 1:
        raise ValueError("zero_weight should be <=1 so zeros remain weakly weighted")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    Sc = sparse.csr_matrix(S_cell, dtype=np.float32)
    Sg = sparse.csr_matrix(S_gene, dtype=np.float32)
    if Sc.shape != (X.shape[0], X.shape[0]):
        raise ValueError("S_cell shape mismatch")
    if Sg.shape != (X.shape[1], X.shape[1]):
        raise ValueError("S_gene shape mismatch")

    dc = np.asarray(degree_cell, dtype=np.float32).reshape(-1)
    dg = np.asarray(degree_gene, dtype=np.float32).reshape(-1)
    lh = np.asarray(lambda_h_gene, dtype=np.float32).reshape(-1)

    if dc.size != X.shape[0] or dg.size != X.shape[1] or lh.size != X.shape[1]:
        raise ValueError("degree/penalty size mismatch")
    if np.any(dc < 0) or np.any(dg < 0) or np.any(lh < 0):
        raise ValueError("degree/penalty values must be nonnegative")
    if (
        np.any(~np.isfinite(dc))
        or np.any(~np.isfinite(dg))
        or np.any(~np.isfinite(lh))
    ):
        raise ValueError("degree/penalty values must be finite")

    observed = X > 0
    if not np.any(observed):
        raise ValueError("X has no positive observed entries")

    if (zero_penalty_exclude_rows is None) != (zero_penalty_exclude_cols is None):
        raise ValueError("zero-penalty exclusion rows and columns must be supplied together")
    if zero_penalty_exclude_rows is not None:
        zero_penalty_exclude_rows = np.asarray(
            zero_penalty_exclude_rows, dtype=np.int64
        ).reshape(-1)
        zero_penalty_exclude_cols = np.asarray(
            zero_penalty_exclude_cols, dtype=np.int64
        ).reshape(-1)
        if zero_penalty_exclude_rows.size != zero_penalty_exclude_cols.size:
            raise ValueError("zero-penalty exclusion row/column lengths differ")
        if (
            np.any(zero_penalty_exclude_rows < 0)
            or np.any(zero_penalty_exclude_rows >= X.shape[0])
        ):
            raise IndexError("zero-penalty exclusion row index out of bounds")
        if (
            np.any(zero_penalty_exclude_cols < 0)
            or np.any(zero_penalty_exclude_cols >= X.shape[1])
        ):
            raise IndexError("zero-penalty exclusion column index out of bounds")
        if zero_penalty_exclude_rows.size:
            flat = (
                zero_penalty_exclude_rows * np.int64(X.shape[1])
                + zero_penalty_exclude_cols
            )
            if np.unique(flat).size != flat.size:
                raise ValueError("duplicate positions in zero-penalty exclusion")
            if np.any(
                observed[zero_penalty_exclude_rows, zero_penalty_exclude_cols]
            ):
                raise ValueError("zero-penalty exclusions must point only to zeros in X")

    def weighted_prediction_block(WH_b, observed_b, start, stop):
        # Keep compatibility with both the older 3-argument and newer
        # 5-argument _weighted_prediction helper definitions.
        QWH_b = _weighted_prediction(WH_b, observed_b, zero_weight)
        if zero_weight > 0 and zero_penalty_exclude_rows is not None:
            in_block = (
                (zero_penalty_exclude_cols >= start)
                & (zero_penalty_exclude_cols < stop)
            )
            if np.any(in_block):
                local_cols = zero_penalty_exclude_cols[in_block] - start
                QWH_b[
                    zero_penalty_exclude_rows[in_block], local_cols
                ] = 0.0
        return QWH_b

    rng = np.random.default_rng(random_state)
    mean_pos = float(X[observed].mean())
    scale = max(np.sqrt(mean_pos / max(rank, 1)), 1e-3)
    W = (rng.random((X.shape[0], rank), dtype=np.float32) + 0.1) * scale
    H = (rng.random((rank, X.shape[1]), dtype=np.float32) + 0.1) * scale

    eps = np.float32(1e-8)
    losses: list[float] = []
    converged = False
    start_time = time.perf_counter()
    n_genes = X.shape[1]

    for it in range(1, max_iter + 1):
        # Compute from the complete old H before updating any block.
        # This preserves all cross-block gene-graph contributions.
        HS_full = _h_times_sparse(H, Sg) if alpha_gene > 0 else None

        # H update: the graph contribution for each block is sliced from
        # the complete H_old @ S_gene product, not from a block-diagonal graph.
        for start in range(0, n_genes, block_size):
            stop = min(start + block_size, n_genes)
            sl = slice(start, stop)

            H_b = H[:, sl]
            WH_b = W @ H_b
            QWH_b = weighted_prediction_block(
                WH_b, observed[:, sl], start, stop
            )

            num_h = W.T @ X[:, sl]
            den_h = W.T @ QWH_b

            if alpha_gene > 0:
                num_h += np.float32(alpha_gene) * HS_full[:, sl]
                den_h += np.float32(alpha_gene) * (H_b * dg[None, sl])

            if lambda_reg > 0:
                den_h += np.float32(lambda_reg) * H_b

            den_h += lh[None, sl]
            H_b *= num_h / np.maximum(den_h, eps)
            np.maximum(H_b, eps, out=H_b)

            del WH_b, QWH_b, num_h, den_h

        # W update: sum reconstruction contributions over every gene block.
        num_w = np.zeros_like(W)
        den_w = np.zeros_like(W)

        for start in range(0, n_genes, block_size):
            stop = min(start + block_size, n_genes)
            sl = slice(start, stop)

            H_b = H[:, sl]
            WH_b = W @ H_b
            QWH_b = weighted_prediction_block(
                WH_b, observed[:, sl], start, stop
            )

            num_w += X[:, sl] @ H_b.T
            den_w += QWH_b @ H_b.T

            del WH_b, QWH_b

        if alpha_cell > 0:
            num_w += np.float32(alpha_cell) * _sparse_times_w(Sc, W)
            den_w += np.float32(alpha_cell) * (dc[:, None] * W)

        if lambda_reg > 0:
            den_w += np.float32(lambda_reg) * W

        if lambda_w_l1 > 0:
            den_w += np.float32(lambda_w_l1)

        W *= num_w / np.maximum(den_w, eps)
        np.maximum(W, eps, out=W)
        del num_w, den_w, HS_full

        loss = _objective_blockwise(
            X, observed, W, H, zero_weight,
            Sc, dc, alpha_cell, Sg, dg, alpha_gene,
            lh, lambda_reg, lambda_w_l1, block_size,
            zero_penalty_exclude_rows, zero_penalty_exclude_cols,
        )
        losses.append(loss)

        if iteration_callback is not None:
            iteration_callback(it, loss, W, H)

        if verbose and (it == 1 or it == max_iter or it % 10 == 0):
            print(f"[{tag}] iter={it:03d} objective={loss:.6e}")

        if len(losses) >= 2:
            prev = losses[-2]
            rel = abs(prev - loss) / max(abs(prev), 1e-12)
            if rel < tol:
                converged = True
                break

    return SolverResult(
        W=W,
        H=H,
        losses=losses,
        n_iter=len(losses),
        elapsed_seconds=time.perf_counter() - start_time,
        converged=converged,
    )
