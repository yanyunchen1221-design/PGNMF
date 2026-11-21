#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run PGNMF using a prepared masked matrix and gene-symbol files.

Required inputs:
  X_model.npy             dense, nonnegative cells x selected-model-genes matrix
  all_gene_symbols.npy    gene symbols for the complete input gene universe
  model_gene_symbols.npy  gene symbols in exactly the same order as X_model columns
  pathways.gmt            GMT pathway collection

The script builds the pathway-rank gene graph, cell kNN graph, pathway-aware
H sparsity penalties, and then runs the exact gene-block solver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pgnmf.cell_graph import build_sparse_knn_cell_graph
from pgnmf.gene_graph import build_pathway_rank_gene_graph
from pgnmf.pathway_sparsity import (
    pathway_membership_from_gmt,
    make_two_level_penalty,
)
from pgnmf.solver import zero_aware_nmf_mu_blocked


def parse_args():
    p = argparse.ArgumentParser(description="Run exact gene-block PGNMF.")
    p.add_argument("--x-model", required=True, help="Masked model-gene matrix (.npy)")
    p.add_argument("--all-gene-symbols", required=True, help="Full gene symbols (.npy)")
    p.add_argument("--model-gene-symbols", required=True, help="Model gene symbols (.npy)")
    p.add_argument("--gmt", required=True, help="Pathway GMT file")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha-cell", type=float, required=True)
    p.add_argument("--alpha-gene", type=float, required=True)
    p.add_argument("--zero-weight", type=float, required=True,
                   help="Effective zero-background weight q0 in [0, 1]")
    p.add_argument("--sparsity-strength", type=float, required=True)
    p.add_argument("--pathway-weight", type=float, default=0.5)
    p.add_argument("--nonpathway-weight", type=float, default=1.0)

    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-reg", type=float, default=1e-3)
    p.add_argument("--lambda-w-l1", type=float, default=0.0)
    p.add_argument("--block-size", type=int, default=500)

    p.add_argument("--min-pathway-genes", type=int, default=10)
    p.add_argument("--max-pathway-genes", type=int, default=500)
    p.add_argument("--gene-edge-batch-size", type=int, default=256)
    p.add_argument("--graph-pcs", type=int, default=30)
    p.add_argument("--graph-neighbors", type=int, default=15)
    p.add_argument("--graph-metric", default="euclidean")
    p.add_argument("--zero-penalty-exclusions", default=None,
                   help="Optional .npz with local model-gene coordinates arrays named rows and cols")
    p.add_argument("--save-reconstruction", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = np.asarray(np.load(a.x_model, allow_pickle=False), dtype=np.float32)
    all_symbols = np.asarray(np.load(a.all_gene_symbols, allow_pickle=False), dtype=str)
    model_symbols = np.asarray(np.load(a.model_gene_symbols, allow_pickle=False), dtype=str)

    if X.ndim != 2 or X.shape[1] != model_symbols.size:
        raise ValueError("X_model columns must match model_gene_symbols in length and order")
    if X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError("X_model must contain at least 3 cells and 2 model genes")
    if np.any(~np.isfinite(X)) or np.any(X < 0):
        raise ValueError("X_model must be finite and nonnegative")
    if not 0.0 <= a.zero_weight <= 1.0:
        raise ValueError("--zero-weight must lie in [0, 1]")

    print("Building pathway membership and gene-specific sparsity penalties...", flush=True)
    membership_all, membership_info = pathway_membership_from_gmt(
        a.gmt,
        all_symbols,
        min_pathway_genes=a.min_pathway_genes,
        max_pathway_genes=a.max_pathway_genes,
    )
    # Model symbols are expected to be an ordered subset of the full gene universe.
    # Match by symbol rather than assuming the model genes are a contiguous slice.
    symbol_to_membership = {}
    for symbol, member in zip(all_symbols, membership_all):
        symbol_to_membership.setdefault(str(symbol).strip().upper(), bool(member))
    model_membership = np.asarray(
        [symbol_to_membership.get(str(s).strip().upper(), False) for s in model_symbols],
        dtype=bool,
    )
    penalty = make_two_level_penalty(
        model_membership,
        strength=a.sparsity_strength,
        pathway_weight=a.pathway_weight,
        nonpathway_weight=a.nonpathway_weight,
    )

    print("Building pathway-rank gene graph...", flush=True)
    Sg_raw, dg_raw, gene_info, gene_diag = build_pathway_rank_gene_graph(
        X,
        all_symbols,
        model_symbols,
        a.gmt,
        min_pathway_genes=a.min_pathway_genes,
        max_pathway_genes=a.max_pathway_genes,
        edge_batch_size=a.gene_edge_batch_size,
    )
    positive_degree = dg_raw[dg_raw > 0]
    if positive_degree.size == 0:
        raise RuntimeError("The pathway-rank gene graph has no positive-degree genes")
    graph_scale = float(np.median(positive_degree))
    Sg = (Sg_raw / np.float32(graph_scale)).tocsr()
    dg = np.asarray(dg_raw, dtype=np.float32) / np.float32(graph_scale)

    print("Building masked-input cell kNN graph...", flush=True)
    Sc, dc, _, cell_info, cell_diag = build_sparse_knn_cell_graph(
        X,
        n_pcs=a.graph_pcs,
        n_neighbors=a.graph_neighbors,
        metric=a.graph_metric,
        random_state=a.seed,
    )

    exclusion_rows = exclusion_cols = None
    if a.zero_penalty_exclusions:
        with np.load(a.zero_penalty_exclusions, allow_pickle=False) as z:
            if "rows" not in z or "cols" not in z:
                raise KeyError("Exclusion .npz must contain arrays named rows and cols")
            exclusion_rows = np.asarray(z["rows"], dtype=np.int64)
            exclusion_cols = np.asarray(z["cols"], dtype=np.int64)
        if exclusion_rows.shape != exclusion_cols.shape:
            raise ValueError("Exclusion rows and cols must have identical shapes")
        if np.any(exclusion_rows < 0) or np.any(exclusion_rows >= X.shape[0]):
            raise IndexError("Exclusion row index is outside X_model")
        if np.any(exclusion_cols < 0) or np.any(exclusion_cols >= X.shape[1]):
            raise IndexError("Exclusion column index is outside X_model")

    print("Fitting exact gene-block PGNMF...", flush=True)
    fit = zero_aware_nmf_mu_blocked(
        X,
        S_cell=Sc,
        degree_cell=dc,
        alpha_cell=a.alpha_cell,
        S_gene=Sg,
        degree_gene=dg,
        alpha_gene=a.alpha_gene,
        lambda_h_gene=penalty,
        zero_weight=a.zero_weight,
        rank=a.rank,
        max_iter=a.max_iter,
        tol=a.tol,
        random_state=a.seed,
        lambda_reg=a.lambda_reg,
        lambda_w_l1=a.lambda_w_l1,
        block_size=a.block_size,
        zero_penalty_exclude_rows=exclusion_rows,
        zero_penalty_exclude_cols=exclusion_cols,
        verbose=True,
        tag="PGNMF-exact-gene-block",
    )

    factors_path = out / "pgnmf_factors.npz"
    np.savez_compressed(
        factors_path,
        W=fit.W.astype(np.float32, copy=False),
        H=fit.H.astype(np.float32, copy=False),
        losses=np.asarray(fit.losses, dtype=np.float64),
        model_gene_symbols=model_symbols,
    )
    if a.save_reconstruction:
        np.save(out / "X_hat_model.npy", (fit.W @ fit.H).astype(np.float32))

    summary = {
        "solver": "zero_aware_nmf_mu_blocked",
        "matrix_shape": list(X.shape),
        "rank": a.rank,
        "block_size": a.block_size,
        "n_gene_blocks": int(np.ceil(X.shape[1] / a.block_size)),
        "zero_weight": a.zero_weight,
        "n_zero_penalty_exclusions": int(0 if exclusion_rows is None else exclusion_rows.size),
        "parameters": {
            "alpha_cell": a.alpha_cell,
            "alpha_gene": a.alpha_gene,
            "sparsity_strength": a.sparsity_strength,
            "pathway_weight": a.pathway_weight,
            "nonpathway_weight": a.nonpathway_weight,
            "max_iter": a.max_iter,
            "tol": a.tol,
            "seed": a.seed,
            "lambda_reg": a.lambda_reg,
            "lambda_w_l1": a.lambda_w_l1,
        },
        "solver_summary": fit.summary(),
        "pathway_membership": membership_info.to_dict(),
        "gene_graph": gene_info.to_dict(),
        "gene_graph_diagnostics": gene_diag,
        "cell_graph": cell_info.to_dict(),
        "cell_graph_diagnostics": cell_diag,
        "gene_graph_degree_scale": graph_scale,
        "factors_file": factors_path.name,
    }
    (out / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["solver_summary"], indent=2), flush=True)
    print(f"Saved factors: {factors_path}", flush=True)
    print(f"Saved summary: {out / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
