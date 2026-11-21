from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import rankdata


@dataclass
class GeneGraphInfo:
    n_model_genes: int
    n_retained_pathways: int
    n_pathway_model_genes: int
    n_edges_undirected: int
    beta_gene: float
    rank_tie_method: str
    double_zero_policy: str
    mean_valid_cells_per_edge: float
    median_valid_cells_per_edge: float
    mean_degree: float
    median_positive_degree: float

    def to_dict(self):
        return asdict(self)


def _upper_symbols(symbols: np.ndarray) -> np.ndarray:
    return np.asarray([str(x).strip().upper() for x in symbols], dtype=object)


def average_descending_ranks(X: np.ndarray) -> np.ndarray:
    """
    Per-cell descending ranks with average ranks for ties.

    Equal expression values, including the many zeros in scRNA-seq, receive the
    same average rank. This removes the arbitrary ordering introduced by
    argsort().argsort(). Higher expression has smaller rank (rank 1 is highest).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    # scipy.rankdata supports axis and average ties. Cast to float32 afterwards
    # to keep memory manageable for the 14k x 1k development matrices.
    return rankdata(-X, axis=1, method="average").astype(np.float32, copy=False)


def _retained_pathway_model_pairs(
    gmt_path: str | Path,
    all_gene_symbols: np.ndarray,
    model_gene_symbols: np.ndarray,
    min_pathway_genes: int = 10,
    max_pathway_genes: int = 500,
):
    """
    Return shared-pathway counts for pairs of modeled genes.

    Pathway retention uses overlap with the full dataset gene universe, matching
    the manuscript's retained-pathway logic. Edges are then formed only between
    modeled genes that co-occur in at least one retained pathway.
    """
    gmt_path = Path(gmt_path)
    if not gmt_path.exists():
        raise FileNotFoundError(gmt_path)
    if min_pathway_genes < 1 or max_pathway_genes < min_pathway_genes:
        raise ValueError("invalid pathway-size bounds")

    all_upper = _upper_symbols(all_gene_symbols)
    model_upper = _upper_symbols(model_gene_symbols)
    all_universe = {g for g in all_upper if g}
    model_symbol_to_indices: dict[str, list[int]] = {}
    for i, g in enumerate(model_upper):
        if g:
            model_symbol_to_indices.setdefault(g, []).append(i)
    model_universe = set(model_symbol_to_indices)

    pair_counts: dict[tuple[int, int], int] = {}
    pathway_model_gene_idx: set[int] = set()
    retained = 0

    with gmt_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            overlap_full = genes & all_universe
            if not (min_pathway_genes <= len(overlap_full) <= max_pathway_genes):
                continue
            retained += 1
            overlap_model_symbols = genes & model_universe
            if len(overlap_model_symbols) < 2:
                # Still a retained pathway, but it contributes no modeled edge.
                for sym in overlap_model_symbols:
                    pathway_model_gene_idx.update(model_symbol_to_indices[sym])
                continue
            idxs: list[int] = []
            for sym in overlap_model_symbols:
                idxs.extend(model_symbol_to_indices[sym])
            idxs = sorted(set(idxs))
            pathway_model_gene_idx.update(idxs)
            for u, v in combinations(idxs, 2):
                pair_counts[(u, v)] = pair_counts.get((u, v), 0) + 1

    return pair_counts, retained, pathway_model_gene_idx


def build_pathway_rank_gene_graph(
    X_model: np.ndarray,
    all_gene_symbols: np.ndarray,
    model_gene_symbols: np.ndarray,
    gmt_path: str | Path,
    min_pathway_genes: int = 10,
    max_pathway_genes: int = 500,
    beta_gene: float | None = None,
    edge_batch_size: int = 256,
):
    """
    Build the revised pathway-guided gene graph for Stage 2B-2.

    Revision made in direct response to the reviewer's zero-tie concern:
      1) ranks use average ties, so equal zeros never receive arbitrary ranks;
      2) for a gene pair (u,v), cells in which BOTH genes are zero contribute no
         rank-difference evidence. The mean rank difference is computed only on
         cells where at least one of the pair is expressed.

    For a pathway-linked pair, with c_uv shared retained pathways and Delta_uv the
    mean valid-cell rank difference,
        S_uv = c_uv * exp(-Delta_uv / beta_gene).
    If beta_gene is omitted, it is set to the median positive Delta over retained
    edges (a label-free, data-scale-adaptive choice).

    Returns sparse symmetric S, degree vector D, and metadata.
    """
    X = np.asarray(X_model, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(model_gene_symbols):
        raise ValueError("X_model/model_gene_symbols shape mismatch")
    if np.any(X < 0) or np.any(~np.isfinite(X)):
        raise ValueError("X_model must be finite and nonnegative")

    pair_counts, retained, pathway_model_idx = _retained_pathway_model_pairs(
        gmt_path, all_gene_symbols, model_gene_symbols,
        min_pathway_genes=min_pathway_genes,
        max_pathway_genes=max_pathway_genes,
    )
    if not pair_counts:
        raise RuntimeError(
            "No pathway-linked modeled gene pairs were found. Check gene symbols/GMT matching."
        )

    pairs = np.asarray(list(pair_counts.keys()), dtype=np.int64)
    shared = np.asarray([pair_counts[tuple(p)] for p in pairs], dtype=np.float32)

    ranks = average_descending_ranks(X)
    expressed = X > 0
    n_edges = pairs.shape[0]
    deltas = np.empty(n_edges, dtype=np.float32)
    valid_counts = np.empty(n_edges, dtype=np.int32)

    bs = max(1, int(edge_batch_size))
    for start in range(0, n_edges, bs):
        end = min(n_edges, start + bs)
        u = pairs[start:end, 0]
        v = pairs[start:end, 1]
        ru = ranks[:, u]
        rv = ranks[:, v]
        valid = expressed[:, u] | expressed[:, v]
        cnt = valid.sum(axis=0).astype(np.int32)
        diff = np.abs(ru - rv)
        # Exclude double-zero cells from the average; they provide no ordering evidence.
        diff *= valid
        sums = diff.sum(axis=0, dtype=np.float64)
        delta = np.divide(
            sums,
            np.maximum(cnt, 1),
            out=np.full(end - start, np.nan, dtype=np.float64),
            where=cnt > 0,
        )
        deltas[start:end] = delta.astype(np.float32)
        valid_counts[start:end] = cnt

    keep = np.isfinite(deltas) & (valid_counts > 0)
    if not np.any(keep):
        raise RuntimeError("All pathway-linked edges lacked valid rank evidence")
    pairs = pairs[keep]
    shared = shared[keep]
    deltas = deltas[keep]
    valid_counts = valid_counts[keep]

    if beta_gene is None:
        positive = deltas[deltas > 0]
        beta = float(np.median(positive)) if positive.size else 1.0
    else:
        beta = float(beta_gene)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta_gene must be positive and finite")

    weights = shared * np.exp(-deltas / np.float32(beta))
    positive_w = np.isfinite(weights) & (weights > 0)
    pairs = pairs[positive_w]
    weights = weights[positive_w].astype(np.float32, copy=False)
    valid_counts = valid_counts[positive_w]
    deltas = deltas[positive_w]
    if pairs.size == 0:
        raise RuntimeError("No positive-weight gene-graph edges remained")

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    vals = np.concatenate([weights, weights])
    n = X.shape[1]
    S = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    S.sum_duplicates()
    S.setdiag(0)
    S.eliminate_zeros()
    degree = np.asarray(S.sum(axis=1)).ravel().astype(np.float32)
    pos_degree = degree[degree > 0]

    info = GeneGraphInfo(
        n_model_genes=int(n),
        n_retained_pathways=int(retained),
        n_pathway_model_genes=int(len(pathway_model_idx)),
        n_edges_undirected=int(S.nnz // 2),
        beta_gene=float(beta),
        rank_tie_method="average",
        double_zero_policy="exclude cells where both genes are zero from Delta_uv",
        mean_valid_cells_per_edge=float(np.mean(valid_counts)),
        median_valid_cells_per_edge=float(np.median(valid_counts)),
        mean_degree=float(np.mean(degree)),
        median_positive_degree=float(np.median(pos_degree)) if pos_degree.size else 0.0,
    )
    diagnostics = {
        "delta_min": float(np.min(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_max": float(np.max(deltas)),
        "weight_min": float(np.min(weights)),
        "weight_median": float(np.median(weights)),
        "weight_max": float(np.max(weights)),
        "degree_max": float(np.max(degree)),
    }
    return S, degree, info, diagnostics
