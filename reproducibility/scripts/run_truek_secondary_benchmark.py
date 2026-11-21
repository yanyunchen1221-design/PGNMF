#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))  # maintained public PGNMF package
sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))

import scanpy as sc

from pgnmf_revision.masking import MaskPlan
from pgnmf_revision.pathway_sparsity import (
    get_gene_symbols,
    pathway_membership_from_gmt,
    make_two_level_penalty,
)
from pgnmf_revision.gene_graph_stage2b2 import build_pathway_rank_gene_graph
from pgnmf_revision.cell_graph_stage2b3 import build_sparse_knn_cell_graph
from pgnmf.solver import zero_aware_nmf_mu_blocked


def parse_csv_ints(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def dense_float32(X):
    if sparse.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def load_gene_mask(npz_path: str, key: str, n_vars: int):
    with np.load(npz_path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"gene-set key {key!r} not found; available={z.files}")
        m = np.asarray(z[key], dtype=bool)
    if m.shape != (n_vars,):
        raise ValueError(f"gene mask shape {m.shape} != ({n_vars},)")
    return m


def truek_eval(X, labels, *, true_k, pca_seed=42, kmeans_seed=42):
    X = np.asarray(X, dtype=np.float32)
    max_pcs = min(10, X.shape[1] - 1, X.shape[0] - 1)
    pcs = PCA(
        n_components=max_pcs,
        svd_solver="randomized",
        random_state=pca_seed,
    ).fit_transform(X)
    pred = KMeans(
        n_clusters=true_k,
        random_state=kmeans_seed,
        n_init=20,
    ).fit_predict(pcs)
    return {
        "ari": float(adjusted_rand_score(labels, pred)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "n_clusters": int(np.unique(pred).size),
    }


def heldout_rmse(X_eval, plan, model_mask):
    model_idx = np.flatnonzero(model_mask)
    global_to_local = np.full(model_mask.size, -1, dtype=np.int64)
    global_to_local[model_idx] = np.arange(model_idx.size, dtype=np.int64)
    keep = model_mask[plan.cols]
    rr = np.asarray(plan.rows[keep], dtype=np.int64)
    cc = global_to_local[np.asarray(plan.cols[keep], dtype=np.int64)]
    truth = np.asarray(plan.reference_values[keep], dtype=np.float64)
    pred = np.asarray(X_eval[rr, cc], dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--masked", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--mask-plan", required=True)
    p.add_argument("--gmt", required=True)
    p.add_argument("--gene-set-masks", required=True)
    p.add_argument("--gene-set-key", default="HVG1000+PATHWAY")
    p.add_argument("--label-key", default="majority_voting")
    p.add_argument("--gene-symbol-column", default=None)
    p.add_argument("--min-pathway-genes", type=int, default=10)
    p.add_argument("--max-pathway-genes", type=int, default=500)

    # Frozen PGNMF settings
    p.add_argument("--rank", type=int, default=80)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--lambda-reg", type=float, default=1e-3)
    p.add_argument("--sparsity-strength", type=float, default=10.0)
    p.add_argument("--pathway-weight", type=float, default=0.5)
    p.add_argument("--nonpathway-weight", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--alpha-cell", type=float, default=10.0)
    p.add_argument("--alpha-gene", type=float, default=2.0)
    p.add_argument("--graph-pcs", type=int, default=30)
    p.add_argument("--graph-neighbors", type=int, default=15)
    p.add_argument("--graph-metric", default="euclidean")
    p.add_argument("--gene-edge-batch-size", type=int, default=256)

    # Secondary true-K evaluation
    p.add_argument("--model-seeds", default="1,7,21,42,2026")
    p.add_argument("--pca-seed", type=int, default=42)
    p.add_argument("--kmeans-seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ad_masked = sc.read_h5ad(a.masked)
    ad_reference = sc.read_h5ad(a.reference)
    plan = MaskPlan.load(a.mask_plan)

    if ad_masked.shape != ad_reference.shape or tuple(plan.shape) != tuple(ad_masked.shape):
        raise ValueError("shape mismatch among masked/reference/mask plan")
    if not np.array_equal(ad_masked.var_names.astype(str), ad_reference.var_names.astype(str)):
        raise ValueError("masked/reference var_names are not aligned")
    if a.label_key not in ad_reference.obs.columns:
        raise KeyError(f"label key {a.label_key!r} not found")

    model_mask = load_gene_mask(a.gene_set_masks, a.gene_set_key, ad_masked.n_vars)
    X_masked = dense_float32(ad_masked[:, model_mask].X)
    X_reference = dense_float32(ad_reference[:, model_mask].X)

    labels = ad_reference.obs[a.label_key].astype(str).to_numpy()
    true_k = int(pd.Series(labels).nunique())
    seeds = parse_csv_ints(a.model_seeds)

    symbols_all, symbol_source = get_gene_symbols(ad_masked, a.gene_symbol_column)
    pathway_all, _ = pathway_membership_from_gmt(
        a.gmt,
        symbols_all,
        min_pathway_genes=a.min_pathway_genes,
        max_pathway_genes=a.max_pathway_genes,
        gene_symbol_source=symbol_source,
    )
    pathway_model = np.asarray(pathway_all[model_mask], dtype=bool)
    penalty = make_two_level_penalty(
        pathway_model,
        strength=a.sparsity_strength,
        pathway_weight=a.pathway_weight,
        nonpathway_weight=a.nonpathway_weight,
    ).astype(np.float32, copy=False)

    observed_mask = X_masked > 0
    zero_mask = ~observed_mask
    n_pos = int(np.count_nonzero(observed_mask))
    n_zero = int(np.count_nonzero(zero_mask))
    q0 = float(a.beta * (n_pos / n_zero))

    print("=" * 78)
    print("PGNMF TRUE-K SECONDARY BENCHMARK")
    print(f"cells={ad_masked.n_obs}; model_genes={int(model_mask.sum())}; true_K={true_k}")
    print(f"model seeds={seeds}; PCA10 seed={a.pca_seed}; KMeans seed={a.kmeans_seed}")
    print(f"frozen PGNMF: rank={a.rank}, beta={a.beta} -> q0={q0:.8f}, alpha_cell={a.alpha_cell}, alpha_gene={a.alpha_gene}")
    print("=" * 78)

    print("\nBuilding frozen pathway-rank gene graph ONCE ...")
    Sg_raw, dg_raw, gene_info, gene_diag = build_pathway_rank_gene_graph(
        X_masked,
        symbols_all,
        symbols_all[model_mask],
        a.gmt,
        min_pathway_genes=a.min_pathway_genes,
        max_pathway_genes=a.max_pathway_genes,
        edge_batch_size=a.gene_edge_batch_size,
    )
    pos_d = np.asarray(dg_raw)[np.asarray(dg_raw) > 0]
    if pos_d.size == 0:
        raise RuntimeError("gene graph has no positive degree")
    scale = float(np.median(pos_d))
    Sg = (Sg_raw / np.float32(scale)).tocsr()
    dg = np.asarray(dg_raw, dtype=np.float32) / np.float32(scale)
    print(f"gene graph scale={scale:.6f}; edges={int(Sg.nnz // 2)}")

    # Baselines: deterministic under fixed PCA/KMeans seeds
    masked_eval = truek_eval(
        X_masked, labels, true_k=true_k,
        pca_seed=a.pca_seed, kmeans_seed=a.kmeans_seed,
    )
    ref_eval = truek_eval(
        X_reference, labels, true_k=true_k,
        pca_seed=a.pca_seed, kmeans_seed=a.kmeans_seed,
    )

    print("\nFixed inputs:")
    print(f"MASKED    ARI={masked_eval['ari']:.6f} NMI={masked_eval['nmi']:.6f}")
    print(f"REFERENCE ARI={ref_eval['ari']:.6f} NMI={ref_eval['nmi']:.6f}")

    rows = []
    for seed in seeds:
        print(f"\n--- PGNMF model seed={seed} ---")
        Sc, dc, _, _, _ = build_sparse_knn_cell_graph(
            X_masked,
            n_pcs=a.graph_pcs,
            n_neighbors=a.graph_neighbors,
            metric=a.graph_metric,
            random_state=seed,
        )

        result = zero_aware_nmf_mu_blocked(
            X_masked,
            S_cell=Sc,
            degree_cell=dc,
            alpha_cell=a.alpha_cell,
            S_gene=Sg,
            degree_gene=dg,
            alpha_gene=a.alpha_gene,
            lambda_h_gene=penalty,
            zero_weight=q0,
            rank=a.rank,
            max_iter=a.max_iter,
            tol=a.tol,
            random_state=seed,
            lambda_reg=a.lambda_reg,
            lambda_w_l1=0.0,
            verbose=False,
            tag=f"trueK secondary seed={seed}",
        block_size=500,
        )
        Xhat = (result.W @ result.H).astype(np.float32, copy=False)
        X_dense = X_masked.copy()
        X_dense[zero_mask] = Xhat[zero_mask]

        ev = truek_eval(
            X_dense, labels, true_k=true_k,
            pca_seed=a.pca_seed, kmeans_seed=a.kmeans_seed,
        )
        rmse = heldout_rmse(X_dense, plan, model_mask)

        rows.append({
            "method": "PGNMF-DENSE_PRESERVE_OBS",
            "model_seed": seed,
            "pca_seed": a.pca_seed,
            "kmeans_seed": a.kmeans_seed,
            "true_k": true_k,
            "rmse_heldout": rmse,
            "ari": ev["ari"],
            "nmi": ev["nmi"],
            "n_clusters": ev["n_clusters"],
            "n_iter": int(result.n_iter),
            "elapsed_seconds": float(result.elapsed_seconds),
        })
        print(f"RMSE={rmse:.6f} ARI={ev['ari']:.6f} NMI={ev['nmi']:.6f} fit={result.elapsed_seconds:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "pgnmf_trueK_modelseed_results.csv", index=False)

    summary = pd.DataFrame([{
        "method": "PGNMF-DENSE_PRESERVE_OBS",
        "rmse_mean": float(df["rmse_heldout"].mean()),
        "rmse_sd": float(df["rmse_heldout"].std(ddof=1)),
        "ari_mean": float(df["ari"].mean()),
        "ari_sd": float(df["ari"].std(ddof=1)),
        "nmi_mean": float(df["nmi"].mean()),
        "nmi_sd": float(df["nmi"].std(ddof=1)),
        "n": int(len(df)),
    }])
    summary.to_csv(out / "pgnmf_trueK_modelseed_summary.csv", index=False)

    fixed = pd.DataFrame([
        {"view": "MASKED", **masked_eval},
        {"view": "REFERENCE", **ref_eval},
    ])
    fixed.to_csv(out / "trueK_fixed_input_results.csv", index=False)

    manifest = {
        "protocol": "PCA10 + KMeans(true K)",
        "label_key": a.label_key,
        "true_k": true_k,
        "model_seeds": seeds,
        "pca_seed": a.pca_seed,
        "kmeans_seed": a.kmeans_seed,
        "gene_set_key": a.gene_set_key,
        "frozen_model": {
            "rank": a.rank,
            "beta": a.beta,
            "effective_zero_weight": q0,
            "alpha_cell": a.alpha_cell,
            "alpha_gene": a.alpha_gene,
            "sparsity_strength": a.sparsity_strength,
            "pathway_weight": a.pathway_weight,
            "nonpathway_weight": a.nonpathway_weight,
            "lambda_reg": a.lambda_reg,
            "max_iter": a.max_iter,
            "tol": a.tol,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("FINAL TRUE-K SECONDARY SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("\nFixed inputs:")
    print(f"MASKED    ARI={masked_eval['ari']:.6f} NMI={masked_eval['nmi']:.6f}")
    print(f"REFERENCE ARI={ref_eval['ari']:.6f} NMI={ref_eval['nmi']:.6f}")
    print("\nSaved to:", out)


if __name__ == "__main__":
    main()
