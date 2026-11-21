from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


@dataclass
class ClusteringResult:
    resolution: float
    n_clusters: int
    ari: float | None
    nmi: float | None
    silhouette: float | None
    label_key: str | None
    n_hvgs: int
    n_pcs: int
    n_neighbors: int
    metric: str

    def to_dict(self):
        return asdict(self)


def select_common_hvgs(adata_masked_reference, n_top_genes: int = 2000) -> np.ndarray:
    """
    Select HVGs ONCE from the common masked input, not separately from each method output.

    This prevents a second method-specific tuning layer. For log-normalized data we use
    Scanpy's 'seurat' flavor (the legacy code used 'seurat_v3', which expects counts).
    """
    import scanpy as sc

    tmp = adata_masked_reference.copy()
    sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_top_genes, tmp.n_vars), flavor="seurat", inplace=True)
    mask = np.asarray(tmp.var["highly_variable"].values, dtype=bool)
    if not np.any(mask):
        raise RuntimeError("No highly variable genes were selected")
    return mask


def evaluate_fixed_leiden(
    adata_template,
    X_imputed,
    hvg_mask: np.ndarray,
    label_key: str | None,
    resolution: float = 0.4,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    metric: str = "cosine",
    random_state: int = 42,
    compute_umap: bool = True,
):
    """
    Evaluate clustering WITHOUT using ground-truth labels to choose resolution.

    Ground-truth labels are touched only AFTER Leiden has produced cluster assignments.
    X_imputed must already be in the common log-expression evaluation space; this function
    deliberately does NOT call normalize_total() or log1p() again.
    """
    import anndata as ad
    import scanpy as sc

    if tuple(X_imputed.shape) != (adata_template.n_obs, adata_template.n_vars):
        raise ValueError("X_imputed shape does not match adata_template")
    if hvg_mask.shape[0] != adata_template.n_vars:
        raise ValueError("hvg_mask length does not match number of genes")

    eval_adata = ad.AnnData(
        X=X_imputed.copy() if hasattr(X_imputed, "copy") else np.array(X_imputed, copy=True),
        obs=adata_template.obs.copy(),
        var=adata_template.var.copy(),
    )
    eval_hvg = eval_adata[:, hvg_mask].copy()

    max_pcs = min(n_pcs, eval_hvg.n_vars - 1, eval_hvg.n_obs - 1)
    if max_pcs < 2:
        raise ValueError("Too few cells/genes for PCA after HVG selection")

    sc.tl.pca(eval_hvg, n_comps=max_pcs, random_state=random_state)
    sc.pp.neighbors(
        eval_hvg,
        n_neighbors=min(n_neighbors, max(2, eval_hvg.n_obs - 1)),
        n_pcs=max_pcs,
        use_rep="X_pca",
        metric=metric,
        random_state=random_state,
    )
    sc.tl.leiden(eval_hvg, resolution=resolution, key_added="leiden", random_state=random_state)
    if compute_umap:
        sc.tl.umap(eval_hvg, random_state=random_state)

    pred = eval_hvg.obs["leiden"].astype(str).to_numpy()
    n_clusters = int(pd.Series(pred).nunique())

    ari = nmi = None
    if label_key is not None and label_key in eval_hvg.obs.columns:
        truth = eval_hvg.obs[label_key].astype(str).to_numpy()
        if pd.Series(truth).nunique() > 1:
            # Critical rule: labels are used ONLY here, after clustering is fixed.
            ari = float(adjusted_rand_score(truth, pred))
            nmi = float(normalized_mutual_info_score(truth, pred))

    sil = None
    try:
        if n_clusters > 1 and n_clusters < eval_hvg.n_obs:
            sil = float(silhouette_score(eval_hvg.obsm["X_pca"], pred, metric="euclidean"))
    except Exception:
        sil = None

    result = ClusteringResult(
        resolution=float(resolution),
        n_clusters=n_clusters,
        ari=ari,
        nmi=nmi,
        silhouette=sil,
        label_key=label_key,
        n_hvgs=int(hvg_mask.sum()),
        n_pcs=int(max_pcs),
        n_neighbors=int(min(n_neighbors, max(2, eval_hvg.n_obs - 1))),
        metric=metric,
    )
    return result, eval_hvg
