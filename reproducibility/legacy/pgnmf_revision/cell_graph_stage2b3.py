from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class CellGraphInfo:
    n_cells: int
    n_model_genes: int
    n_pcs: int
    n_neighbors: int
    metric: str
    sigma: float
    n_edges_undirected: int
    mean_degree: float
    median_degree: float
    max_degree: float
    graph_source: str = "masked_input_only"
    graph_type: str = "sparse_knn_heat_kernel"

    def to_dict(self) -> dict:
        return asdict(self)


def _to_dense_float32(X):
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    return X


def pca_embedding(X, n_pcs: int = 30, random_state: int = 42) -> np.ndarray:
    X = _to_dense_float32(X)
    max_pcs = min(int(n_pcs), X.shape[0] - 1, X.shape[1] - 1)
    if max_pcs < 2:
        raise ValueError("Too few cells/genes for PCA")
    pca = PCA(n_components=max_pcs, svd_solver="randomized", random_state=random_state)
    return np.asarray(pca.fit_transform(X), dtype=np.float32)


def build_sparse_knn_cell_graph(
    X_model,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    random_state: int = 42,
):
    """
    Build a sparse, symmetric cell graph from the MASKED expression matrix only.

    Pipeline:
      masked model-gene matrix -> PCA -> kNN -> heat-kernel weights

    For each directed kNN edge i->j with distance d_ij,
        w_ij = exp(-d_ij^2 / (2*sigma^2)),
    where sigma is the median positive kNN distance. The graph is symmetrized by
    elementwise maximum, so no dense n_cells x n_cells distance matrix is created.
    """
    X = _to_dense_float32(X_model)
    n_cells, n_genes = X.shape
    if n_cells < 3:
        raise ValueError("Need at least 3 cells")
    k = min(int(n_neighbors), n_cells - 1)
    if k < 1:
        raise ValueError("n_neighbors must be positive")

    Z = pca_embedding(X, n_pcs=n_pcs, random_state=random_state)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
    nn.fit(Z)
    dist, ind = nn.kneighbors(Z, return_distance=True)

    # Usually the first neighbor is self. Remove self robustly row-by-row.
    rows = []
    cols = []
    dvals = []
    for i in range(n_cells):
        kept = 0
        for d, j in zip(dist[i], ind[i]):
            if int(j) == i:
                continue
            rows.append(i)
            cols.append(int(j))
            dvals.append(float(d))
            kept += 1
            if kept == k:
                break
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    dvals = np.asarray(dvals, dtype=np.float32)
    positive = dvals[dvals > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-8)
    weights = np.exp(-(dvals * dvals) / (2.0 * sigma * sigma)).astype(np.float32)

    S = sparse.coo_matrix((weights, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32).tocsr()
    S = S.maximum(S.T).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    degree = np.asarray(S.sum(axis=1)).reshape(-1).astype(np.float32)

    upper = sparse.triu(S, k=1)
    info = CellGraphInfo(
        n_cells=int(n_cells),
        n_model_genes=int(n_genes),
        n_pcs=int(Z.shape[1]),
        n_neighbors=int(k),
        metric=str(metric),
        sigma=float(sigma),
        n_edges_undirected=int(upper.nnz),
        mean_degree=float(np.mean(degree)),
        median_degree=float(np.median(degree)),
        max_degree=float(np.max(degree)),
    )
    diagnostics = {
        "distance_min_positive": float(np.min(positive)) if positive.size else 0.0,
        "distance_median_positive": float(np.median(positive)) if positive.size else 0.0,
        "distance_max": float(np.max(dvals)) if dvals.size else 0.0,
        "weight_min": float(S.data.min()) if S.nnz else 0.0,
        "weight_median": float(np.median(S.data)) if S.nnz else 0.0,
        "weight_max": float(S.data.max()) if S.nnz else 0.0,
    }
    return S, degree, Z, info, diagnostics


def knn_indices_from_expression(
    X_model,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Return fixed-size neighbor indices after PCA, excluding self."""
    X = _to_dense_float32(X_model)
    Z = pca_embedding(X, n_pcs=n_pcs, random_state=random_state)
    n_cells = Z.shape[0]
    k = min(int(n_neighbors), n_cells - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
    nn.fit(Z)
    _, ind = nn.kneighbors(Z, return_distance=True)
    out = np.empty((n_cells, k), dtype=np.int32)
    for i in range(n_cells):
        vals = [int(j) for j in ind[i] if int(j) != i][:k]
        if len(vals) != k:
            raise RuntimeError("Could not obtain fixed-size neighbor set")
        out[i] = vals
    return out


def neighborhood_preservation(reference_neighbors: np.ndarray, candidate_neighbors: np.ndarray) -> dict:
    """
    Mean kNN recall/overlap: |N_ref(i) ∩ N_candidate(i)| / k.
    This is label-free and ranges from 0 to 1.
    """
    ref = np.asarray(reference_neighbors)
    cand = np.asarray(candidate_neighbors)
    if ref.shape != cand.shape or ref.ndim != 2:
        raise ValueError("neighbor arrays must have identical 2-D shape")
    k = ref.shape[1]
    overlaps = np.empty(ref.shape[0], dtype=np.float32)
    for i in range(ref.shape[0]):
        overlaps[i] = len(set(ref[i].tolist()).intersection(cand[i].tolist())) / float(k)
    return {
        "mean_knn_overlap": float(np.mean(overlaps)),
        "median_knn_overlap": float(np.median(overlaps)),
        "std_knn_overlap": float(np.std(overlaps)),
        "k": int(k),
    }
