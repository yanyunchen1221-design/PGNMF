from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
from scipy import sparse


@dataclass
class PreparedData:
    adata_reference: object
    input_space: str
    label_key: str | None


def _detect_label_key(adata, explicit: str | None = None) -> str | None:
    if explicit is not None:
        if explicit not in adata.obs.columns:
            raise KeyError(f"Requested label column '{explicit}' is not present in adata.obs")
        return explicit
    for key in ("celltype", "cell_type", "CellType", "label", "Label", "annotation", "Annotation"):
        if key in adata.obs.columns and adata.obs[key].nunique() > 1:
            return key
    return None


def _looks_integer_like(X, atol: float = 1e-6, sample_size: int = 200000) -> bool:
    if sparse.issparse(X):
        vals = X.data
    else:
        vals = np.asarray(X).reshape(-1)
    if vals.size == 0:
        return True
    if vals.size > sample_size:
        rng = np.random.default_rng(0)
        vals = rng.choice(vals, size=sample_size, replace=False)
    return bool(np.allclose(vals, np.round(vals), atol=atol, rtol=0))


def load_and_prepare_h5ad(
    path: str,
    input_space: str = "counts",
    label_key: str | None = None,
    min_genes: int = 200,
    min_cells: int = 3,
    max_mt_pct: float | None = 5.0,
    target_sum: float = 1e4,
):
    """
    Load one h5ad and return a QC-filtered reference matrix.

    input_space='counts':
        X must be raw/nonnegative count-like data. It is preserved in layers['counts'].
        X is then normalized once to log-CP10K for PGNMF and evaluation.

    input_space='logcp10k':
        X is assumed already normalized/log-transformed. It is NOT rounded, renormalized,
        or log-transformed again. This mode is acceptable for the Stage-1 PGNMF audit,
        but count-model baselines (SAVER/bayNorm) later require a true count matrix.
    """
    import scanpy as sc

    if input_space not in {"counts", "logcp10k"}:
        raise ValueError("input_space must be 'counts' or 'logcp10k'")

    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    # Basic nonzero-based filtering is valid for either counts or log-normalized data.
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Mitochondrial-percent QC should be computed from true counts, not from
    # log-normalized expression values.  For logcp10k inputs, reuse a
    # precomputed pct_counts_mt column if present; otherwise skip this filter.
    if max_mt_pct is not None:
        if input_space == "counts":
            names_upper = np.asarray([str(x).upper() for x in adata.var_names])
            adata.var["mt"] = np.char.startswith(names_upper.astype(str), "MT-")
            if bool(np.any(adata.var["mt"].values)):
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
                adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct, :].copy()
        elif "pct_counts_mt" in adata.obs.columns:
            adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct, :].copy()
        else:
            warnings.warn(
                "input_space='logcp10k' and no precomputed obs['pct_counts_mt'] was found; "
                "mitochondrial-percentage filtering is skipped because it should not be "
                "recomputed from log-normalized values.",
                RuntimeWarning,
            )

    detected_label = _detect_label_key(adata, label_key)

    if input_space == "counts":
        if np.min(adata.X.data) < 0 if sparse.issparse(adata.X) and adata.X.nnz else (np.asarray(adata.X).min() < 0):
            raise ValueError("Count matrix contains negative values")
        if not _looks_integer_like(adata.X):
            raise ValueError(
                "input_space='counts' was requested, but X is not integer-like. "
                "Do NOT round it. Use input_space='logcp10k' if X is already normalized, "
                "or provide an h5ad containing true counts."
            )
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        adata.uns["pgnmf_common_space"] = f"log1p(normalize_total={target_sum:g})"
    else:
        # Never guess and silently transform normalized data.
        if sparse.issparse(adata.X):
            min_val = float(adata.X.data.min()) if adata.X.nnz else 0.0
        else:
            min_val = float(np.asarray(adata.X).min())
        if min_val < 0:
            raise ValueError("logcp10k input must be nonnegative")
        adata.uns["pgnmf_common_space"] = "user-declared logcp10k"
        if "counts" not in adata.layers:
            warnings.warn(
                "No layers['counts'] found. Stage-1 PGNMF evaluation can proceed, "
                "but later SAVER/bayNorm runs must use true count data.",
                RuntimeWarning,
            )

    return PreparedData(adata_reference=adata, input_space=input_space, label_key=detected_label)


def strip_derived_analysis(adata):
    """
    Remove data-derived embeddings/graphs that may have been computed BEFORE
    simulated masking.  Keeping a precomputed X_pca/neighbors graph in a masked
    h5ad can leak held-out expression values into a downstream solver.

    Biological/sample annotations in .obs and gene annotations in .var are kept.
    Expression matrices and a true counts layer (if present) are kept.
    """
    # Embeddings such as X_pca/X_umap are functions of the expression matrix and
    # must be recomputed from the masked input.
    for key in list(adata.obsm.keys()):
        del adata.obsm[key]

    # Pairwise connectivities/distances likewise must not be inherited.
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]

    # Remove common analysis metadata that refers to deleted embeddings/graphs.
    for key in ("neighbors", "pca", "umap", "diffmap_evals", "paga"):
        if key in adata.uns:
            del adata.uns[key]

    return adata
