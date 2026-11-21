from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy import sparse

from .matrix_utils import take_values, set_values_zero


@dataclass(frozen=True)
class MaskPlan:
    """
    Fixed held-out coordinates for simulated-missing evaluation.

    The plan is generated ONCE and must be reused for PGNMF and every baseline.
    `reference_values` are the ground-truth values in the common evaluation space
    (for PGNMF, normally log-CP10K before masking).
    """

    rows: np.ndarray
    cols: np.ndarray
    reference_values: np.ndarray
    shape: tuple[int, int]
    mask_ratio: float
    random_seed: int
    strategy: str = "per_cell_nonzero"

    @property
    def n_masked(self) -> int:
        return int(self.rows.size)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rows=self.rows.astype(np.int64),
            cols=self.cols.astype(np.int64),
            reference_values=self.reference_values.astype(np.float64),
            shape=np.asarray(self.shape, dtype=np.int64),
            mask_ratio=np.asarray([self.mask_ratio], dtype=np.float64),
            random_seed=np.asarray([self.random_seed], dtype=np.int64),
            strategy=np.asarray([self.strategy]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MaskPlan":
        z = np.load(path, allow_pickle=False)
        return cls(
            rows=z["rows"].astype(np.int64),
            cols=z["cols"].astype(np.int64),
            reference_values=z["reference_values"].astype(float),
            shape=tuple(int(x) for x in z["shape"].tolist()),
            mask_ratio=float(z["mask_ratio"][0]),
            random_seed=int(z["random_seed"][0]),
            strategy=str(z["strategy"][0]),
        )


def _row_nonzero_indices(X, i: int) -> np.ndarray:
    if sparse.issparse(X):
        csr = X if sparse.isspmatrix_csr(X) else X.tocsr()
        start, end = csr.indptr[i], csr.indptr[i + 1]
        cols = csr.indices[start:end]
        vals = csr.data[start:end]
        return cols[vals > 0]
    return np.flatnonzero(np.asarray(X[i]).reshape(-1) > 0)


def generate_mask_plan(
    X_reference,
    mask_ratio: float = 0.10,
    random_seed: int = 42,
    min_nonzero_per_cell: int = 10,
) -> MaskPlan:
    """
    Mask a fixed fraction of positive entries independently within each cell.

    Important differences from the legacy implementation:
      1. the coordinates are generated exactly once and saved;
      2. `create_masked_data()` never regenerates a second random mask;
      3. the plan stores the actual ground-truth values used by RMSE;
      4. no unknown/log-transformed matrix is rounded to integer counts.

    Cells with fewer than `min_nonzero_per_cell` positive values are not masked.
    """
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be between 0 and 1")

    rng = np.random.default_rng(random_seed)
    n_cells, n_genes = X_reference.shape
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []

    # Convert once for efficient row access, but never densify a large sparse matrix.
    X_rows = X_reference.tocsr() if sparse.issparse(X_reference) else X_reference

    for i in range(n_cells):
        nz = _row_nonzero_indices(X_rows, i)
        if nz.size < min_nonzero_per_cell:
            continue
        n_mask = int(np.floor(nz.size * mask_ratio))
        if n_mask <= 0:
            continue
        chosen = rng.choice(nz, size=n_mask, replace=False)
        rows_out.append(np.full(n_mask, i, dtype=np.int64))
        cols_out.append(np.asarray(chosen, dtype=np.int64))

    if not rows_out:
        raise ValueError("No entries were selected for masking; check the input matrix and settings")

    rows = np.concatenate(rows_out)
    cols = np.concatenate(cols_out)
    truth = take_values(X_reference, rows, cols)

    if np.any(truth <= 0):
        raise AssertionError("Mask plan contains non-positive reference values; this should never happen")

    return MaskPlan(
        rows=rows,
        cols=cols,
        reference_values=truth,
        shape=(n_cells, n_genes),
        mask_ratio=float(mask_ratio),
        random_seed=int(random_seed),
    )


def apply_mask(X, plan: MaskPlan):
    if tuple(X.shape) != tuple(plan.shape):
        raise ValueError(f"Matrix shape {X.shape} does not match mask plan {plan.shape}")
    return set_values_zero(X, plan.rows, plan.cols)


def write_mask_summary(plan: MaskPlan, path: str | Path) -> None:
    payload = {
        "shape": list(plan.shape),
        "mask_ratio_requested": plan.mask_ratio,
        "random_seed": plan.random_seed,
        "strategy": plan.strategy,
        "n_masked": plan.n_masked,
        "reference_min": float(plan.reference_values.min()),
        "reference_mean": float(plan.reference_values.mean()),
        "reference_max": float(plan.reference_values.max()),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
