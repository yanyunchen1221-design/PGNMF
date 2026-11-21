#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PGNMF sensitivity follow-up:
  1) convergence: rerun the two rank-100 / weak-gene-graph configurations
     with max_iter=600, keeping seed/split/graphs/evaluation fixed;
  2) seed_stability: rerun baseline and two low-ARI configurations with
     model initialization seeds 42, 1, and 2026.

Important: only the NMF initialization seed varies in seed_stability.
The cell graph, held-out split, and Leiden evaluation seed remain fixed at 42
(or validation_seed=2026 for the held-out split), so the comparison isolates
initialization sensitivity rather than changing the evaluation protocol.
Run from the PGNMF repository root.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz

ROOT = Path(__file__).resolve().parents[2]
S54_PATH = ROOT / "reproducibility" / "scripts" / "run_parameter_sensitivity.py"


def load_s54_module():
    if not S54_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot find the original sensitivity script: {S54_PATH}"
        )
    spec = importlib.util.spec_from_file_location("pgnmf_sensitivity_stage2c37", S54_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {S54_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--model-seeds must contain at least one integer.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate model seeds: {seeds}")
    return seeds


def parse_args():
    p = argparse.ArgumentParser(
        description="PGNMF convergence and random-initialization follow-up."
    )
    p.add_argument("--mode", required=True, choices=["convergence", "seed_stability"])
    p.add_argument("--dataset", default="C11")
    p.add_argument("--masked", default="work/C11/masked_logcp10k.h5ad")
    p.add_argument("--mask-plan", default="work/C11/mask_plan.npz")
    p.add_argument(
        "--gene-set-masks",
        default="audit/C11/stage2cpre_gene_set_diagnostic/gene_set_masks.npz",
    )
    p.add_argument(
        "--gmt",
        default="data/c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt",
    )
    p.add_argument("--gene-set-key", default="HVG1000+PATHWAY")
    p.add_argument("--gene-symbol-column", default=None)
    p.add_argument("--label-key", default="majority_voting")

    # Keep all model/evaluation settings from the completed stage2c37 run.
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lambda-reg", type=float, default=0.001)
    p.add_argument("--pathway-weight", type=float, default=0.5)
    p.add_argument("--nonpathway-weight", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--tol", type=float, default=1e-4)

    p.add_argument("--validation-seed", type=int, default=2026)
    p.add_argument("--validation-fraction", type=float, default=0.5)
    p.add_argument("--resolution", type=float, default=0.30)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--metric", default="cosine")

    p.add_argument("--graph-pcs", type=int, default=30)
    p.add_argument("--graph-neighbors", type=int, default=15)
    p.add_argument("--graph-metric", default="euclidean")
    p.add_argument("--gene-edge-batch-size", type=int, default=256)
    p.add_argument("--min-pathway-genes", type=int, default=10)
    p.add_argument("--max-pathway-genes", type=int, default=500)

    # These seeds are deliberately fixed for graph construction and Leiden.
    # Only the NMF initialization seed changes in seed_stability mode.
    p.add_argument("--fixed-graph-seed", type=int, default=42)
    p.add_argument("--fixed-metric-seed", type=int, default=42)
    p.add_argument("--model-seeds", default="42,1,2026")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def config_key(cfg: dict, seed: int) -> str:
    return (
        f"ac={cfg['alpha_cell']:g}|ag={cfg['alpha_gene']:g}|"
        f"ls={cfg['lambda_sparse']:g}|rank={cfg['rank']}|model_seed={seed}"
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def get_configs(mode: str) -> list[dict]:
    if mode == "convergence":
        # The only two configurations that reached max_iter=300 without
        # satisfying the original tol=1e-4 stopping criterion.
        return [
            {"alpha_cell": 10.0, "alpha_gene": 0.5, "lambda_sparse": 10.0, "rank": 100},
            {"alpha_cell": 10.0, "alpha_gene": 1.0, "lambda_sparse": 10.0, "rank": 100},
        ]
    # Original seed-42 baseline plus the two configurations with lower ARI.
    return [
        {"alpha_cell": 10.0, "alpha_gene": 2.0, "lambda_sparse": 10.0, "rank": 80},
        {"alpha_cell": 10.0, "alpha_gene": 2.0, "lambda_sparse": 20.0, "rank": 60},
        {"alpha_cell": 10.0, "alpha_gene": 2.0, "lambda_sparse": 20.0, "rank": 100},
    ]



def evaluate_with_cluster_details(
    s54,
    X_dense: np.ndarray,
    obs: pd.DataFrame,
    label_key: str,
    *,
    resolution: float,
    seed: int,
    n_pcs: int,
    n_neighbors: int,
    metric: str,
):
    """Replicate frozen PCA -> neighbors -> Leiden evaluation and return graph."""
    import anndata as ad
    import scanpy as sc

    tmp = ad.AnnData(X=np.asarray(X_dense, dtype=np.float32), obs=obs.copy())
    n_pc = min(int(n_pcs), tmp.n_vars - 1, tmp.n_obs - 1)
    if n_pc < 2:
        raise ValueError("Too few cells/features for PCA.")
    sc.tl.pca(tmp, n_comps=n_pc, random_state=seed)
    sc.pp.neighbors(
        tmp,
        n_neighbors=min(int(n_neighbors), max(2, tmp.n_obs - 1)),
        n_pcs=n_pc,
        use_rep="X_pca",
        metric=metric,
        random_state=seed,
    )
    sc.tl.leiden(
        tmp,
        resolution=float(resolution),
        key_added="leiden",
        random_state=seed,
    )
    pred = tmp.obs["leiden"].astype(str).to_numpy()
    truth = tmp.obs[label_key].astype(str).to_numpy()
    ari = float(s54.adjusted_rand_score(truth, pred))
    nmi = float(s54.normalized_mutual_info_score(truth, pred))
    n_clusters = int(pd.Series(pred).nunique())
    composition = pd.crosstab(
        pd.Series(pred, name="predicted_cluster"),
        pd.Series(truth, name="reference_cell_type"),
    )
    return ari, nmi, n_clusters, composition, tmp.obsp["connectivities"].tocsr()


def compare_knn_graphs(reference, current):
    """Mean per-cell neighbor retention and Jaccard similarity."""
    reference = reference.tocsr()
    current = current.tocsr()
    if reference.shape != current.shape:
        raise ValueError("Reference/current kNN graph shapes differ.")
    retention, jaccard = [], []
    for i in range(reference.shape[0]):
        a = set(reference.indices[reference.indptr[i]:reference.indptr[i + 1]])
        b = set(current.indices[current.indptr[i]:current.indptr[i + 1]])
        a.discard(i)
        b.discard(i)
        if not a:
            continue
        inter = len(a & b)
        retention.append(inter / len(a))
        union = len(a | b)
        jaccard.append(inter / union if union else 1.0)
    return (
        float(np.mean(retention)) if retention else np.nan,
        float(np.mean(jaccard)) if jaccard else np.nan,
    )

def main():
    args = parse_args()
    s54 = load_s54_module()

    if args.max_iter <= 0 or args.tol <= 0:
        raise ValueError("--max-iter and --tol must be positive.")
    if args.beta < 0 or args.beta > 1:
        raise ValueError("--beta must be in [0, 1].")

    configs = get_configs(args.mode)
    model_seeds = [42] if args.mode == "convergence" else parse_seed_list(args.model_seeds)

    default_out = (
        ROOT / "audit" / args.dataset / "stage2c38_convergence600"
        if args.mode == "convergence"
        else ROOT / "audit" / args.dataset / "stage2c39_seed_stability"
    )
    output_dir = s54.resolve_path(args.output_dir, default_out)
    output_dir.mkdir(parents=True, exist_ok=True)

    masked_path = s54.resolve_path(
        args.masked, ROOT / "work" / args.dataset / "masked_logcp10k.h5ad"
    )
    mask_plan_path = s54.resolve_path(
        args.mask_plan, ROOT / "work" / args.dataset / "mask_plan.npz"
    )
    gene_masks_path = s54.resolve_path(
        args.gene_set_masks,
        ROOT / "audit" / args.dataset / "stage2cpre_gene_set_diagnostic" / "gene_set_masks.npz",
    )
    gmt_path = s54.choose_gmt(args.gmt)
    for path in (masked_path, mask_plan_path, gene_masks_path, gmt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file does not exist: {path}")

    run_config = {
        "mode": args.mode,
        "dataset": args.dataset,
        "masked_path": str(masked_path),
        "mask_plan_path": str(mask_plan_path),
        "gene_masks_path": str(gene_masks_path),
        "gmt_path": str(gmt_path),
        "gene_set_key": args.gene_set_key,
        "label_key": args.label_key,
        "configs": configs,
        "model_seeds": model_seeds,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "beta": args.beta,
        "lambda_reg": args.lambda_reg,
        "pathway_weight": args.pathway_weight,
        "nonpathway_weight": args.nonpathway_weight,
        "validation_seed": args.validation_seed,
        "validation_fraction": args.validation_fraction,
        "resolution": args.resolution,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "metric": args.metric,
        "graph_pcs": args.graph_pcs,
        "graph_neighbors": args.graph_neighbors,
        "graph_metric": args.graph_metric,
        "gene_edge_batch_size": args.gene_edge_batch_size,
        "min_pathway_genes": args.min_pathway_genes,
        "max_pathway_genes": args.max_pathway_genes,
        "fixed_graph_seed": args.fixed_graph_seed,
        "fixed_metric_seed": args.fixed_metric_seed,
        "protocol": (
            "Only NMF initialization seed varies in seed_stability; "
            "cell graph and Leiden seed are fixed; held-out test coordinates are excluded "
            "from weak-zero penalty."
        ),
    }
    signature = hashlib.sha256(
        json.dumps(run_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest_path = output_dir / "followup_manifest.json"
    results_path = output_dir / "followup_results.csv"

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("--resume requested but followup_manifest.json is missing.")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("run_signature") != signature:
            raise RuntimeError(
                "The existing output directory has different settings. "
                "Use a new --output-dir; do not mix runs."
            )
        if results_path.is_file():
            results_df = pd.read_csv(results_path)
        else:
            results_df = pd.DataFrame()
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use a new --output-dir or add --resume if this is the same run."
            )
        results_df = pd.DataFrame()

    manifest = {"run_signature": signature, "configuration": run_config}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 88)
    print("PGNMF SENSITIVITY FOLLOW-UP")
    print("=" * 88)
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print(f"Configs: {len(configs)}; model seeds: {model_seeds}")
    print(f"max_iter={args.max_iter}; tol={args.tol}")
    print(f"Fixed graph seed={args.fixed_graph_seed}; fixed Leiden seed={args.fixed_metric_seed}")
    print(f"Validation seed={args.validation_seed}; fraction={args.validation_fraction}")

    # Load the same masked matrix, mask plan, gene set, and held-out split.
    import scanpy as sc
    ad_masked = sc.read_h5ad(masked_path)
    if args.label_key not in ad_masked.obs.columns:
        raise KeyError(
            f"Label key {args.label_key!r} not found. "
            f"Available columns: {ad_masked.obs.columns.tolist()}"
        )

    model_gene_mask = s54.load_gene_mask(
        gene_masks_path, args.gene_set_key, ad_masked.n_vars
    )
    model_gene_indices = np.flatnonzero(model_gene_mask)
    X = s54.dense32(ad_masked[:, model_gene_mask].X)
    if X.ndim != 2 or not np.all(np.isfinite(X)) or np.any(X < 0):
        raise ValueError("Masked model matrix must be finite and nonnegative.")
    obs = ad_masked.obs.copy()

    plan = s54.MaskPlan.load(str(mask_plan_path))
    if tuple(plan.shape) != tuple(ad_masked.shape):
        raise ValueError(
            f"MaskPlan shape {tuple(plan.shape)} != masked AnnData shape {ad_masked.shape}"
        )

    test_rows, test_cols_global, test_truth, n_masked_positive = s54.heldout_test_positions(
        plan, model_gene_mask, args.validation_fraction, args.validation_seed
    )
    global_to_local = np.full(ad_masked.n_vars, -1, dtype=np.int64)
    global_to_local[model_gene_indices] = np.arange(model_gene_indices.size, dtype=np.int64)
    test_cols_local = global_to_local[test_cols_global]
    if np.any(test_cols_local < 0):
        raise RuntimeError("A held-out test gene is missing from the model feature set.")
    if np.any(X[test_rows, test_cols_local] != 0):
        raise RuntimeError("Held-out test positions must be zero in the masked input.")

    heldout_hash = s54.coordinate_hash(test_rows, test_cols_global)
    observed_mask = X > 0
    zero_mask = ~observed_mask
    n_positive = int(np.count_nonzero(observed_mask))
    n_zero = int(np.count_nonzero(zero_mask))
    n_penalized_zero = n_zero - int(test_rows.size)
    if n_penalized_zero <= 0:
        raise RuntimeError("No zero positions remain for the weak zero penalty.")
    q0 = float(args.beta) * n_positive / n_penalized_zero
    if q0 > 1:
        raise ValueError(f"Effective zero weight q0={q0:.6g} exceeds 1.")

    print("\n--- Frozen evaluation split ---")
    print(f"X shape={X.shape}; selected genes={model_gene_indices.size}")
    print(f"Masked-positive positions={n_masked_positive}; test positions={test_truth.size}")
    print(f"Test-coordinate SHA256={heldout_hash}")
    print(f"Observed positives={n_positive}; zeros={n_zero}; penalized zeros={n_penalized_zero}")
    print(f"Balanced weak-zero weight q0={q0:.10f}")

    # Build both graphs once and keep them fixed for all follow-up fits.
    print("\n--- Building fixed KEGG pathway-rank gene graph ---")
    t0 = time.perf_counter()
    all_symbols, symbol_source = s54.get_gene_symbols(ad_masked, args.gene_symbol_column)
    all_symbols = np.asarray(all_symbols, dtype=str)
    pathway_all, _ = s54.pathway_membership_from_gmt(
        str(gmt_path),
        all_symbols,
        min_pathway_genes=args.min_pathway_genes,
        max_pathway_genes=args.max_pathway_genes,
        gene_symbol_source=symbol_source,
    )
    pathway_model = np.asarray(pathway_all[model_gene_mask], dtype=bool)

    Sg_raw, dg_raw, _, _ = s54.build_pathway_rank_gene_graph(
        X,
        all_symbols,
        all_symbols[model_gene_mask],
        str(gmt_path),
        min_pathway_genes=args.min_pathway_genes,
        max_pathway_genes=args.max_pathway_genes,
        edge_batch_size=args.gene_edge_batch_size,
    )
    positive_gene_degree = np.asarray(dg_raw, dtype=np.float64)
    positive_gene_degree = positive_gene_degree[positive_gene_degree > 0]
    if positive_gene_degree.size == 0:
        raise RuntimeError("The pathway-rank gene graph has no positive degree.")
    gene_graph_scale = float(np.median(positive_gene_degree))
    S_gene = (Sg_raw / np.float32(gene_graph_scale)).tocsr()
    degree_gene = np.asarray(dg_raw, dtype=np.float32) / np.float32(gene_graph_scale)
    print(f"Gene graph ready in {time.perf_counter() - t0:.1f}s")

    print("\n--- Building fixed union-kNN cell graph ---")
    t0 = time.perf_counter()
    S_cell, degree_cell, _, _, _ = s54.build_sparse_knn_cell_graph(
        X,
        n_pcs=args.graph_pcs,
        n_neighbors=args.graph_neighbors,
        metric=args.graph_metric,
        random_state=args.fixed_graph_seed,
    )
    print(f"Cell graph ready in {time.perf_counter() - t0:.1f}s")

    # Resume only successful fits; failed fits are retried.
    completed = set()
    if not results_df.empty and "status" in results_df.columns:
        successful = results_df[results_df["status"] == "success"]
        completed = set(successful["fit_key"].astype(str))

    fit_jobs = [(cfg, seed) for cfg in configs for seed in model_seeds]
    print(f"\n--- Fit jobs: {len(fit_jobs)} ---")

    for job_i, (cfg, model_seed) in enumerate(fit_jobs, start=1):
        fit_key = config_key(cfg, model_seed)
        if fit_key in completed:
            print(f"\n[{job_i}/{len(fit_jobs)}] SKIP cached {fit_key}")
            continue

        print(f"\n[{job_i}/{len(fit_jobs)}] FIT {fit_key}", flush=True)
        t0 = time.perf_counter()
        result = None
        penalty = None
        X_hat = None
        X_dense = None
        H = None
        losses = None
        composition = None
        knn_graph = None
        reference_knn = None
        try:
            penalty = s54.make_two_level_penalty(
                pathway_model,
                strength=float(cfg["lambda_sparse"]),
                pathway_weight=float(args.pathway_weight),
                nonpathway_weight=float(args.nonpathway_weight),
            ).astype(np.float32, copy=False)

            result = s54.zero_aware_nmf_mu_blocked(
                X,
                S_cell=S_cell,
                degree_cell=degree_cell,
                alpha_cell=float(cfg["alpha_cell"]),
                S_gene=S_gene,
                degree_gene=degree_gene,
                alpha_gene=float(cfg["alpha_gene"]),
                lambda_h_gene=penalty,
                zero_weight=float(q0),
                rank=int(cfg["rank"]),
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                random_state=int(model_seed),
                lambda_reg=float(args.lambda_reg),
                lambda_w_l1=0.0,
                verbose=False,
                tag=f"FOLLOWUP {fit_key}",
                zero_penalty_exclude_rows=test_rows,
                zero_penalty_exclude_cols=test_cols_local,
            block_size=500,
            )
            X_hat = np.asarray(result.W @ result.H, dtype=np.float32)
            test_pred = np.asarray(X_hat[test_rows, test_cols_local], dtype=np.float64)
            rmse = float(np.sqrt(np.mean((test_pred - test_truth) ** 2)))

            X_dense = X.copy()
            X_dense[zero_mask] = X_hat[zero_mask]
            ari, nmi, n_clusters, composition, knn_graph = evaluate_with_cluster_details(
                s54,
                X_dense,
                obs,
                args.label_key,
                resolution=float(args.resolution),
                seed=int(args.fixed_metric_seed),
                n_pcs=int(args.n_pcs),
                n_neighbors=int(args.n_neighbors),
                metric=str(args.metric),
            )
            composition_name = "cluster_composition_" + safe_name(fit_key) + ".csv"
            composition.to_csv(output_dir / composition_name, encoding="utf-8-sig")
            knn_name = "knn_connectivities_" + safe_name(fit_key) + ".npz"
            save_npz(output_dir / knn_name, knn_graph)

            reference_knn_path = output_dir / "knn_reference_baseline_seed42.npz"
            baseline_key = config_key(configs[0], 42)
            if args.mode == "seed_stability" and fit_key == baseline_key:
                save_npz(reference_knn_path, knn_graph)
                knn_retention, knn_jaccard = 1.0, 1.0
            elif args.mode == "seed_stability" and reference_knn_path.is_file():
                reference_knn = load_npz(reference_knn_path)
                knn_retention, knn_jaccard = compare_knn_graphs(reference_knn, knn_graph)
            else:
                knn_retention, knn_jaccard = np.nan, np.nan

            losses = np.asarray(result.losses, dtype=np.float64)
            if losses.size >= 2:
                final_rel_change = float(
                    abs(losses[-1] - losses[-2]) / max(abs(losses[-2]), 1e-12)
                )
            else:
                final_rel_change = np.nan

            H = np.asarray(result.H)
            h_floor_fraction = float(np.mean(H <= 1.01e-8))
            h_small_fraction = float(np.mean(H <= 1e-6))
            active_genes_per_factor = float(np.mean(np.sum(H > 1e-6, axis=1)))

            trace_name = "objective_" + safe_name(fit_key) + ".csv"
            pd.DataFrame({
                "iteration": np.arange(1, losses.size + 1),
                "objective": losses,
            }).to_csv(output_dir / trace_name, index=False, encoding="utf-8-sig")

            record = {
                "fit_key": fit_key,
                "mode": args.mode,
                "alpha_cell": float(cfg["alpha_cell"]),
                "alpha_gene": float(cfg["alpha_gene"]),
                "lambda_sparse": float(cfg["lambda_sparse"]),
                "rank": int(cfg["rank"]),
                "model_seed": int(model_seed),
                "fixed_graph_seed": int(args.fixed_graph_seed),
                "fixed_metric_seed": int(args.fixed_metric_seed),
                "validation_seed": int(args.validation_seed),
                "heldout_test_coordinate_sha256": heldout_hash,
                "rmse": rmse,
                "ari": float(ari),
                "nmi": float(nmi),
                "n_clusters": int(n_clusters),
                "knn_neighbor_retention_vs_baseline_seed42": knn_retention,
                "knn_mean_jaccard_vs_baseline_seed42": knn_jaccard,
                "cluster_composition_file": composition_name,
                "knn_graph_file": knn_name,
                "n_iter": int(result.n_iter),
                "converged": bool(result.converged),
                "final_objective": float(losses[-1]) if losses.size else np.nan,
                "final_relative_objective_change": final_rel_change,
                "objective_nonincreasing": bool(
                    np.all(np.diff(losses) <= np.maximum(np.abs(losses[:-1]) * 1e-6, 1e-7))
                ) if losses.size > 1 else True,
                "H_fraction_le_1e-8": h_floor_fraction,
                "H_fraction_le_1e-6": h_small_fraction,
                "H_mean_active_genes_per_factor_gt_1e-6": active_genes_per_factor,
                "objective_trace_file": trace_name,
                "fit_seconds": float(result.elapsed_seconds),
                "status": "success",
                "error": "",
            }

            if results_df.empty:
                results_df = pd.DataFrame([record])
            else:
                results_df = pd.concat([results_df, pd.DataFrame([record])], ignore_index=True)
            results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
            completed.add(fit_key)

            print(
                f"  RMSE={rmse:.6f} | ARI={ari:.6f} | NMI={nmi:.6f} "
                f"| clusters={n_clusters} | iter={result.n_iter} "
                f"| converged={result.converged} | final_rel={final_rel_change:.3e} "
                f"| elapsed={time.perf_counter()-t0:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "fit_key": fit_key,
                "mode": args.mode,
                "alpha_cell": float(cfg["alpha_cell"]),
                "alpha_gene": float(cfg["alpha_gene"]),
                "lambda_sparse": float(cfg["lambda_sparse"]),
                "rank": int(cfg["rank"]),
                "model_seed": int(model_seed),
                "status": "failed",
                "error": repr(exc),
            }
            if results_df.empty:
                results_df = pd.DataFrame([failure])
            else:
                results_df = pd.concat([results_df, pd.DataFrame([failure])], ignore_index=True)
            results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
            print(f"  FAILED: {exc!r}", flush=True)
        finally:
            # Release the large dense arrays before starting the next fit.
            result = None
            penalty = None
            X_hat = None
            X_dense = None
            H = None
            losses = None
            composition = None
            knn_graph = None
            reference_knn = None
            gc.collect()

    # Reload and write an aggregate table for the multi-seed mode.
    results_df = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    if args.mode == "seed_stability" and not results_df.empty:
        successful = results_df[results_df["status"] == "success"].copy()
        metric_cols = [
            "rmse", "ari", "nmi", "n_clusters",
            "H_fraction_le_1e-8", "H_fraction_le_1e-6",
            "H_mean_active_genes_per_factor_gt_1e-6",
            "knn_neighbor_retention_vs_baseline_seed42",
            "knn_mean_jaccard_vs_baseline_seed42",
        ]
        group_cols = ["alpha_cell", "alpha_gene", "lambda_sparse", "rank"]
        agg = successful.groupby(group_cols)[metric_cols].agg(["mean", "std"])
        agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
        agg = agg.reset_index()
        conv = successful.groupby(group_cols)["converged"].agg(["sum", "count"]).reset_index()
        conv = conv.rename(columns={"sum": "n_converged", "count": "n_successful"})
        agg = agg.merge(conv, on=group_cols, how="left")
        agg.to_csv(output_dir / "seed_stability_mean_sd.csv", index=False, encoding="utf-8-sig")

    print("\n=== FOLLOW-UP COMPLETE ===")
    print(f"Results: {results_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Objective traces: {output_dir}")
    if args.mode == "seed_stability":
        print(f"Mean/SD summary: {output_dir / 'seed_stability_mean_sd.csv'}")
    print("Check status and converged columns before using any result.")


if __name__ == "__main__":
    main()
