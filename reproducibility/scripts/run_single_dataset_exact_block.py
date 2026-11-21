#!/usr/bin/env python3
"""Run the final PGNMF configuration on one prepared scRNA-seq dataset.

This is the public single-dataset entry point. It uses the maintained
pgnmf.zero_aware_nmf_mu_blocked exact gene-block solver, not the historical
full-matrix development solver.

Inputs are prepared by the user:
  * masked and reference AnnData (.h5ad), with identical cell/gene ordering;
  * a MaskPlan (.npz) generated from the corresponding reference matrix;
  * a gene-selection mask (.npz), e.g. key "HVG1000+PATHWAY";
  * a GMT pathway collection.

Run from the repository root after installing the experiment dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from pgnmf.cell_graph import (
    build_sparse_knn_cell_graph,
    knn_indices_from_expression,
    neighborhood_preservation,
)
from pgnmf.gene_graph import build_pathway_rank_gene_graph
from pgnmf.masking import MaskPlan
from pgnmf.pathway_sparsity import (
    get_gene_symbols,
    make_two_level_penalty,
    pathway_membership_from_gmt,
)
from pgnmf.solver import zero_aware_nmf_mu_blocked


def parse_int_list(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Run PGNMF once on a single prepared scRNA-seq dataset."
    )
    p.add_argument("--dataset", required=True, help="Dataset identifier used in output files.")
    p.add_argument("--masked", required=True, help="Masked AnnData input (.h5ad).")
    p.add_argument("--reference", required=True, help="Unmasked reference AnnData (.h5ad).")
    p.add_argument("--mask-plan", required=True, help="Fixed held-out MaskPlan (.npz).")
    p.add_argument("--gene-set-masks", required=True, help="Gene-selection masks (.npz).")
    p.add_argument("--gene-set-key", default="HVG1000+PATHWAY")
    p.add_argument("--gmt", required=True, help="Pathway gene sets in GMT format.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label-key", default="majority_voting")
    p.add_argument("--gene-symbol-column", default=None)

    # Frozen C11 configuration; override when the dataset's validated settings differ.
    p.add_argument("--rank", type=int, default=80)
    p.add_argument("--alpha-cell", type=float, default=10.0)
    p.add_argument("--alpha-gene", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=1.0,
                   help="Coefficient used to calculate the weak zero-background weight.")
    p.add_argument("--zero-weight-mode", choices=["balanced", "per_entry"], default="balanced")
    p.add_argument("--sparsity-strength", type=float, default=10.0)
    p.add_argument("--pathway-weight", type=float, default=0.5)
    p.add_argument("--nonpathway-weight", type=float, default=1.0)
    p.add_argument("--lambda-reg", type=float, default=1e-3)
    p.add_argument("--lambda-w-l1", type=float, default=0.0)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--seeds", type=parse_int_list, default=parse_int_list("1,7,21,42,2026"))

    # Label-based clustering evaluation protocol.
    p.add_argument("--resolution", type=float, required=True,
                   help="Dataset-specific Leiden resolution selected by the label-free calibration protocol.")
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--metric", default="cosine")

    # Cell-graph construction protocol.
    p.add_argument("--graph-pcs", type=int, default=30)
    p.add_argument("--graph-neighbors", type=int, default=15)
    p.add_argument("--graph-metric", default="euclidean")
    p.add_argument("--gene-edge-batch-size", type=int, default=256)
    p.add_argument("--min-pathway-genes", type=int, default=10)
    p.add_argument("--max-pathway-genes", type=int, default=500)

    # Fixed held-out validation/test split used for reporting.
    p.add_argument("--validation-seed", type=int, default=2026)
    p.add_argument("--validation-fraction", type=float, default=0.5)

    p.add_argument("--save-factors", action="store_true",
                   help="Save W/H and objective trace for each seed.")
    p.add_argument("--save-reconstruction", action="store_true",
                   help="Save model-gene reconstruction and observed-preserving dense matrix for each seed.")
    p.add_argument("--quiet-solver", action="store_true")
    return p.parse_args()


def dense_float32(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if sparse.issparse(obj):
        return {"shape": list(obj.shape), "nnz": int(obj.nnz)}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def load_gene_mask(path: str, key: str, n_vars: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"Gene mask key {key!r} not found; available keys: {z.files}")
        mask = np.asarray(z[key], dtype=bool)
    if mask.shape != (n_vars,):
        raise ValueError(f"Gene mask shape {mask.shape} does not match n_vars={n_vars}")
    if not np.any(mask):
        raise ValueError("Gene mask selects no model genes")
    return mask


def split_indices(n: int, validation_fraction: float, seed: int):
    if n < 2:
        raise ValueError("At least two held-out coordinates are required for validation/test reporting.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = int(round(n * validation_fraction))
    n_val = min(max(1, n_val), n - 1)
    return np.sort(order[:n_val]), np.sort(order[n_val:])


def vector_metrics(truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - truth
    pearson = float("nan")
    if truth.size > 1 and np.std(truth) > 0 and np.std(pred) > 0:
        pearson = float(np.corrcoef(truth, pred)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": pearson,
    }


def expression_leiden(X, obs, label_key, *, resolution, n_pcs, n_neighbors, metric, seed):
    # Imported lazily so --help and non-clustering utility use do not require Scanpy.
    import anndata as ad
    import scanpy as sc

    X = np.asarray(X, dtype=np.float32)
    tmp = ad.AnnData(X=X, obs=obs.copy())
    max_pcs = min(int(n_pcs), tmp.n_vars - 1, tmp.n_obs - 1)
    if max_pcs < 2:
        raise ValueError("Too few cells/genes for PCA-based Leiden clustering")
    sc.tl.pca(tmp, n_comps=max_pcs, random_state=seed)
    sc.pp.neighbors(
        tmp,
        n_neighbors=min(int(n_neighbors), max(2, tmp.n_obs - 1)),
        n_pcs=max_pcs,
        use_rep="X_pca",
        metric=metric,
        random_state=seed,
    )
    sc.tl.leiden(tmp, resolution=resolution, key_added="leiden", random_state=seed)
    pred = tmp.obs["leiden"].astype(str).to_numpy()
    truth = tmp.obs[label_key].astype(str).to_numpy()
    ari = float(adjusted_rand_score(truth, pred))
    nmi = float(normalized_mutual_info_score(truth, pred))
    n_clusters = int(pd.Series(pred).nunique())
    sil = float("nan")
    try:
        if 1 < n_clusters < tmp.n_obs:
            sil = float(silhouette_score(tmp.obsm["X_pca"], pred, metric="euclidean"))
    except Exception:
        pass
    return {"ari": ari, "nmi": nmi, "n_clusters": n_clusters, "silhouette": sil}


def evaluate_view(
    X_eval, *, seed, alpha_cell, alpha_gene, beta, q0, view,
    truth, held_rows, held_cols, val_idx, test_idx,
    reference_neighbors, obs, label_key, resolution, n_pcs, n_neighbors, metric,
    zero_mask, xhat,
):
    pred = np.asarray(X_eval[held_rows, held_cols], dtype=np.float64)
    val = vector_metrics(truth[val_idx], pred[val_idx])
    test = vector_metrics(truth[test_idx], pred[test_idx])
    allm = vector_metrics(truth, pred)

    neighbors = knn_indices_from_expression(
        X_eval, n_pcs=n_pcs, n_neighbors=n_neighbors, metric=metric, random_state=seed
    )
    overlap = neighborhood_preservation(reference_neighbors, neighbors)["mean_knn_overlap"]
    clustering = expression_leiden(
        X_eval, obs, label_key,
        resolution=resolution, n_pcs=n_pcs, n_neighbors=n_neighbors,
        metric=metric, seed=seed,
    )
    zero_predictions = np.asarray(xhat, dtype=np.float32)[zero_mask]
    return {
        "seed": int(seed),
        "alpha_cell": float(alpha_cell),
        "alpha_gene": float(alpha_gene),
        "beta": float(beta),
        "effective_zero_weight": float(q0),
        "view": view,
        "validation_rmse": val["rmse"],
        "validation_pearson": val["pearson"],
        "test_rmse": test["rmse"],
        "test_pearson": test["pearson"],
        "heldout_rmse": allm["rmse"],
        "heldout_pearson": allm["pearson"],
        "knn_overlap_vs_reference": float(overlap),
        "ari": float(clustering["ari"]),
        "nmi": float(clustering["nmi"]),
        "n_clusters": int(clustering["n_clusters"]),
        "silhouette": float(clustering["silhouette"]),
        "xhat_zero_rms": float(np.sqrt(np.mean(np.square(zero_predictions, dtype=np.float64)))),
        "xhat_zero_mean": float(np.mean(zero_predictions, dtype=np.float64)),
    }


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["alpha_cell", "alpha_gene", "beta", "effective_zero_weight", "view"]
    metrics = [
        "validation_rmse", "test_rmse", "validation_pearson", "test_pearson",
        "heldout_rmse", "heldout_pearson", "knn_overlap_vs_reference",
        "ari", "nmi", "n_clusters", "xhat_zero_rms", "xhat_zero_mean",
    ]
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_sd"] = float(np.nanstd(vals, ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.rank <= 0 or args.max_iter <= 0 or args.block_size <= 0:
        raise ValueError("rank, max_iter and block_size must be positive")
    if args.beta < 0 or args.sparsity_strength < 0:
        raise ValueError("beta and sparsity-strength must be nonnegative")
    if args.pathway_weight < 0 or args.nonpathway_weight < 0:
        raise ValueError("pathway weights must be nonnegative")
    if args.pathway_weight >= args.nonpathway_weight and args.sparsity_strength > 0:
        raise ValueError("pathway-weight must be smaller than nonpathway-weight")

    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as exc:
        raise SystemExit(
            "The single-dataset runner requires anndata and scanpy. "
            "Install reproducibility/requirements-experiments.txt first."
        ) from exc

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset} ...", flush=True)
    ad_masked = sc.read_h5ad(args.masked)
    ad_reference = sc.read_h5ad(args.reference)
    plan = MaskPlan.load(args.mask_plan)

    if ad_masked.shape != ad_reference.shape or tuple(plan.shape) != tuple(ad_masked.shape):
        raise ValueError("Shape mismatch among masked AnnData, reference AnnData and MaskPlan")
    if not np.array_equal(ad_masked.obs_names.astype(str), ad_reference.obs_names.astype(str)):
        raise ValueError("Masked/reference cell order is not aligned")
    if not np.array_equal(ad_masked.var_names.astype(str), ad_reference.var_names.astype(str)):
        raise ValueError("Masked/reference gene order is not aligned")
    if args.label_key not in ad_masked.obs.columns:
        raise KeyError(f"Label key {args.label_key!r} not found; available={list(ad_masked.obs.columns)}")
    if args.label_key not in ad_reference.obs.columns:
        raise KeyError(f"Label key {args.label_key!r} not found in reference AnnData")

    model_mask = load_gene_mask(args.gene_set_masks, args.gene_set_key, ad_masked.n_vars)
    model_idx = np.flatnonzero(model_mask)
    if args.rank > min(ad_masked.n_obs, model_idx.size):
        raise ValueError(f"rank={args.rank} exceeds min(n_cells, n_model_genes)={min(ad_masked.n_obs, model_idx.size)}")

    X_model = dense_float32(ad_masked[:, model_mask].X)
    X_reference_model = dense_float32(ad_reference[:, model_mask].X)
    symbols_all, symbol_source = get_gene_symbols(ad_masked, args.gene_symbol_column)
    model_symbols = np.asarray(symbols_all[model_mask], dtype=str)

    pathway_all, pathway_info = pathway_membership_from_gmt(
        args.gmt, symbols_all,
        min_pathway_genes=args.min_pathway_genes,
        max_pathway_genes=args.max_pathway_genes,
        gene_symbol_source=symbol_source,
    )
    pathway_model = np.asarray(pathway_all[model_mask], dtype=bool)
    lambda_h_gene = make_two_level_penalty(
        pathway_model,
        strength=args.sparsity_strength,
        pathway_weight=args.pathway_weight,
        nonpathway_weight=args.nonpathway_weight,
    )

    # Map held-out coordinates from the full gene space to selected model-gene space.
    plan_cols = np.asarray(plan.cols, dtype=np.int64)
    keep = model_mask[plan_cols]
    held_rows = np.asarray(plan.rows[keep], dtype=np.int64)
    held_cols_global = plan_cols[keep]
    global_to_local = np.full(ad_masked.n_vars, -1, dtype=np.int64)
    global_to_local[model_idx] = np.arange(model_idx.size, dtype=np.int64)
    held_cols = global_to_local[held_cols_global]
    truth = np.asarray(plan.reference_values[keep], dtype=np.float64)
    if truth.size < 2:
        raise ValueError("Fewer than two held-out coordinates fall within the selected model genes")
    val_idx, test_idx = split_indices(truth.size, args.validation_fraction, args.validation_seed)

    observed_mask = X_model > 0
    zero_mask = ~observed_mask
    n_positive = int(np.count_nonzero(observed_mask))
    n_zero = int(np.count_nonzero(zero_mask))
    if n_zero == 0:
        raise ValueError("Selected model matrix contains no zero entries")
    q0 = float(args.beta * n_positive / n_zero) if args.zero_weight_mode == "balanced" else float(args.beta)
    if not 0.0 <= q0 <= 1.0:
        raise ValueError(f"Effective zero-background weight q0={q0:.8g} is outside [0,1]")

    print(
        f"Input={ad_masked.shape}; model matrix={X_model.shape}; rank={args.rank}; "
        f"alpha_cell={args.alpha_cell}; alpha_gene={args.alpha_gene}; q0={q0:.8g}",
        flush=True,
    )
    print(
        f"Seeds={args.seeds}; held-out model coordinates={truth.size}; "
        f"validation/test={len(val_idx)}/{len(test_idx)}; block_size={args.block_size}",
        flush=True,
    )

    print("Building pathway-rank gene graph from masked model matrix ...", flush=True)
    graph_start = time.perf_counter()
    S_gene_raw, degree_gene_raw, gene_graph_info, gene_graph_diagnostics = (
        build_pathway_rank_gene_graph(
            X_model,
            symbols_all,
            model_symbols,
            args.gmt,
            min_pathway_genes=args.min_pathway_genes,
            max_pathway_genes=args.max_pathway_genes,
            edge_batch_size=args.gene_edge_batch_size,
        )
    )
    positive_degree = np.asarray(degree_gene_raw)[np.asarray(degree_gene_raw) > 0]
    if positive_degree.size == 0:
        raise RuntimeError("The pathway-guided gene graph has no positive-degree genes")
    gene_graph_scale = float(np.median(positive_degree))
    S_gene = (S_gene_raw / np.float32(gene_graph_scale)).tocsr()
    degree_gene = np.asarray(degree_gene_raw, dtype=np.float32) / np.float32(gene_graph_scale)
    graph_build_seconds = time.perf_counter() - graph_start

    result_rows = []
    fit_rows = []
    baseline_rows = []

    for seed in args.seeds:
        print(f"\n===== {args.dataset}: seed={seed} =====", flush=True)
        reference_neighbors = knn_indices_from_expression(
            X_reference_model, n_pcs=args.n_pcs, n_neighbors=args.n_neighbors,
            metric=args.metric, random_state=seed,
        )
        reference_clustering = expression_leiden(
            X_reference_model, ad_reference.obs, args.label_key,
            resolution=args.resolution, n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors, metric=args.metric, seed=seed,
        )
        masked_neighbors = knn_indices_from_expression(
            X_model, n_pcs=args.n_pcs, n_neighbors=args.n_neighbors,
            metric=args.metric, random_state=seed,
        )
        masked_clustering = expression_leiden(
            X_model, ad_masked.obs, args.label_key,
            resolution=args.resolution, n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors, metric=args.metric, seed=seed,
        )
        masked_overlap = neighborhood_preservation(reference_neighbors, masked_neighbors)["mean_knn_overlap"]
        baseline_rows.append({
            "dataset": args.dataset,
            "seed": int(seed),
            "resolution": float(args.resolution),
            "reference_ari": reference_clustering["ari"],
            "reference_nmi": reference_clustering["nmi"],
            "reference_n_clusters": reference_clustering["n_clusters"],
            "masked_ari": masked_clustering["ari"],
            "masked_nmi": masked_clustering["nmi"],
            "masked_n_clusters": masked_clustering["n_clusters"],
            "masked_knn_overlap_vs_reference": float(masked_overlap),
        })

        S_cell, degree_cell, _, cell_graph_info, cell_graph_diagnostics = (
            build_sparse_knn_cell_graph(
                X_model,
                n_pcs=args.graph_pcs,
                n_neighbors=args.graph_neighbors,
                metric=args.graph_metric,
                random_state=seed,
            )
        )

        fit_start = time.perf_counter()
        fit = zero_aware_nmf_mu_blocked(
            X_model,
            S_cell=S_cell,
            degree_cell=degree_cell,
            alpha_cell=args.alpha_cell,
            S_gene=S_gene,
            degree_gene=degree_gene,
            alpha_gene=args.alpha_gene,
            lambda_h_gene=lambda_h_gene,
            zero_weight=q0,
            rank=args.rank,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=seed,
            lambda_reg=args.lambda_reg,
            lambda_w_l1=args.lambda_w_l1,
            verbose=not args.quiet_solver,
            tag=f"PGNMF-{args.dataset}-seed{seed}",
            block_size=args.block_size,
        )
        fit_seconds = time.perf_counter() - fit_start
        X_hat = (fit.W @ fit.H).astype(np.float32, copy=False)
        zero_values = X_hat[zero_mask]
        fit_rows.append({
            "dataset": args.dataset,
            "seed": int(seed),
            "rank": int(args.rank),
            "alpha_cell": float(args.alpha_cell),
            "alpha_gene": float(args.alpha_gene),
            "beta": float(args.beta),
            "effective_zero_weight": float(q0),
            "sparsity_strength": float(args.sparsity_strength),
            "pathway_weight": float(args.pathway_weight),
            "nonpathway_weight": float(args.nonpathway_weight),
            "lambda_reg": float(args.lambda_reg),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "n_iter": int(fit.n_iter),
            "converged": bool(fit.converged),
            "initial_objective": float(fit.losses[0]) if fit.losses else float("nan"),
            "final_objective": float(fit.losses[-1]) if fit.losses else float("nan"),
            "fit_elapsed_seconds": float(fit.elapsed_seconds),
            "wall_elapsed_seconds": float(fit_seconds),
            "block_size": int(args.block_size),
            "n_blocks": int(math.ceil(X_model.shape[1] / args.block_size)),
            "xhat_zero_rms": float(np.sqrt(np.mean(np.square(zero_values, dtype=np.float64)))),
            "xhat_zero_mean": float(np.mean(zero_values, dtype=np.float64)),
            "cell_graph_edges": int(cell_graph_info.n_edges_undirected),
            "cell_graph_sigma": float(cell_graph_info.sigma),
        })

        dense_preserve = X_model.copy()
        dense_preserve[zero_mask] = X_hat[zero_mask]

        raw_record = evaluate_view(
            X_hat, seed=seed, alpha_cell=args.alpha_cell, alpha_gene=args.alpha_gene,
            beta=args.beta, q0=q0, view="RAW_WH",
            truth=truth, held_rows=held_rows, held_cols=held_cols,
            val_idx=val_idx, test_idx=test_idx,
            reference_neighbors=reference_neighbors,
            obs=ad_masked.obs, label_key=args.label_key,
            resolution=args.resolution, n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors, metric=args.metric,
            zero_mask=zero_mask, xhat=X_hat,
        )
        raw_record["dataset"] = args.dataset
        raw_record["resolution"] = float(args.resolution)
        result_rows.append(raw_record)

        dense_record = evaluate_view(
            dense_preserve, seed=seed, alpha_cell=args.alpha_cell, alpha_gene=args.alpha_gene,
            beta=args.beta, q0=q0, view="DENSE_PRESERVE_OBS",
            truth=truth, held_rows=held_rows, held_cols=held_cols,
            val_idx=val_idx, test_idx=test_idx,
            reference_neighbors=reference_neighbors,
            obs=ad_masked.obs, label_key=args.label_key,
            resolution=args.resolution, n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors, metric=args.metric,
            zero_mask=zero_mask, xhat=X_hat,
        )
        dense_record["dataset"] = args.dataset
        dense_record["resolution"] = float(args.resolution)
        result_rows.append(dense_record)

        if args.save_factors:
            np.savez_compressed(
                out / f"{args.dataset}_seed{seed}_factors.npz",
                W=fit.W, H=fit.H, losses=np.asarray(fit.losses, dtype=np.float64),
                model_gene_symbols=model_symbols,
            )
        if args.save_reconstruction:
            np.save(out / f"{args.dataset}_seed{seed}_X_hat_model.npy", X_hat)
            np.save(out / f"{args.dataset}_seed{seed}_X_dense_preserve_obs.npy", dense_preserve)

        print(
            f"seed={seed} iter={fit.n_iter} converged={fit.converged} "
            f"objective={fit.losses[-1]:.6g}; "
            f"DENSE test_RMSE={dense_record['test_rmse']:.5f}, "
            f"ARI={dense_record['ari']:.5f}, NMI={dense_record['nmi']:.5f}",
            flush=True,
        )

        del fit, X_hat, dense_preserve, S_cell, degree_cell

    results_df = pd.DataFrame(result_rows)
    fits_df = pd.DataFrame(fit_rows)
    baselines_df = pd.DataFrame(baseline_rows)
    summary_df = summarize_results(results_df)

    results_df.to_csv(out / "single_dataset_seed_results.csv", index=False)
    fits_df.to_csv(out / "single_dataset_fit_diagnostics.csv", index=False)
    baselines_df.to_csv(out / "single_dataset_reference_baselines.csv", index=False)
    summary_df.to_csv(out / "single_dataset_summary_mean_sd.csv", index=False)

    manifest = {
        "dataset": args.dataset,
        "masked_input": str(Path(args.masked)),
        "reference_input": str(Path(args.reference)),
        "mask_plan": str(Path(args.mask_plan)),
        "gene_set_masks": str(Path(args.gene_set_masks)),
        "gene_set_key": args.gene_set_key,
        "gmt": str(Path(args.gmt)),
        "label_key": args.label_key,
        "resolution": float(args.resolution),
        "model_matrix_shape": list(X_model.shape),
        "n_model_genes": int(model_idx.size),
        "n_pathway_model_genes": int(pathway_model.sum()),
        "n_nonpathway_model_genes": int((~pathway_model).sum()),
        "pathway_membership": pathway_info.to_dict(),
        "gene_graph": {
            "info": gene_graph_info.to_dict(),
            "diagnostics": gene_graph_diagnostics,
            "median_positive_degree_scale": gene_graph_scale,
            "build_seconds": graph_build_seconds,
        },
        "configuration": {
            "rank": args.rank,
            "alpha_cell": args.alpha_cell,
            "alpha_gene": args.alpha_gene,
            "beta": args.beta,
            "zero_weight_mode": args.zero_weight_mode,
            "effective_zero_weight": q0,
            "sparsity_strength": args.sparsity_strength,
            "pathway_weight": args.pathway_weight,
            "nonpathway_weight": args.nonpathway_weight,
            "lambda_reg": args.lambda_reg,
            "lambda_w_l1": args.lambda_w_l1,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "block_size": args.block_size,
            "seeds": args.seeds,
            "n_pcs": args.n_pcs,
            "n_neighbors": args.n_neighbors,
            "metric": args.metric,
            "graph_pcs": args.graph_pcs,
            "graph_neighbors": args.graph_neighbors,
            "graph_metric": args.graph_metric,
            "gene_edge_batch_size": args.gene_edge_batch_size,
            "validation_seed": args.validation_seed,
            "validation_fraction": args.validation_fraction,
        },
        "solver": "pgnmf.solver.zero_aware_nmf_mu_blocked",
        "solver_implementation": "exact gene-block multiplicative updates retaining the complete gene graph",
        "outputs": [
            "single_dataset_seed_results.csv",
            "single_dataset_fit_diagnostics.csv",
            "single_dataset_reference_baselines.csv",
            "single_dataset_summary_mean_sd.csv",
        ],
        "note": (
            "This is a single-dataset runner. Repeat it with the appropriate inputs, "
            "label key and frozen Leiden resolution for each dataset. The defaults "
            "reflect the confirmed C11 Stage2C8 configuration and should be changed "
            "only when the dataset-specific experiment protocol requires it."
        ),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print("\n=== Mean +/- SD across seeds ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
