from __future__ import annotations

import numpy as np
from .masking import MaskPlan
from .matrix_utils import take_values


def _predictions_at_mask(X_imputed, plan: MaskPlan) -> np.ndarray:
    if tuple(X_imputed.shape) != tuple(plan.shape):
        raise ValueError(f"Imputed matrix shape {X_imputed.shape} does not match mask plan {plan.shape}")
    pred = take_values(X_imputed, plan.rows, plan.cols)
    if not np.all(np.isfinite(pred)):
        bad = int((~np.isfinite(pred)).sum())
        raise ValueError(f"Imputed matrix has {bad} non-finite predictions at held-out positions")
    return pred


def masked_rmse(X_imputed, plan: MaskPlan) -> float:
    """RMSE ONLY on the fixed held-out entries, exactly matching the paper's simulated-missing definition."""
    pred = _predictions_at_mask(X_imputed, plan)
    truth = plan.reference_values
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def masked_mae(X_imputed, plan: MaskPlan) -> float:
    pred = _predictions_at_mask(X_imputed, plan)
    truth = plan.reference_values
    return float(np.mean(np.abs(truth - pred)))


def masked_bias(X_imputed, plan: MaskPlan) -> float:
    pred = _predictions_at_mask(X_imputed, plan)
    return float(np.mean(pred - plan.reference_values))


def masked_pearson(X_imputed, plan: MaskPlan) -> float:
    pred = _predictions_at_mask(X_imputed, plan)
    truth = plan.reference_values
    if truth.size < 2 or np.std(truth) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(truth, pred)[0, 1])
