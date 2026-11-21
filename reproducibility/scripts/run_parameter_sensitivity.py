#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PGNMF two-parameter sensitivity analysis for the current zero-aware model.

Reproduces the three parameter pairs used in the manuscript's Figure 5:
  1) alpha_cell x lambda_sparse
  2) alpha_gene x rank
  3) lambda_sparse x rank

For every parameter combination:
  - fit the maintained exact gene-block PGNMF solver;
  - exclude the fixed held-out TEST coordinates from the weak-zero penalty;
  - calculate held-out TEST RMSE on the same frozen test split;
  - calculate ARI/NMI on DENSE_PRESERVE_OBS using the frozen Leiden protocol.

The model is fitted with one fixed seed (42) and one fixed mask/split. ARI/NMI
are post-hoc descriptive metrics only; they must not be used to select parameters.

Run from the PGNMF repository root, with this file in scripts/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))  # maintained public PGNMF package
sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))

import anndata as ad
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


# Default grids are centered around the latest C11 configuration:
# alpha_cell=10, alpha_gene=2, lambda_sparse/sparsity_strength=10, rank=80.
DEFAULT_ALPHA_CELL_GRID = "2,5,10,20"
DEFAULT_LAMBDA_SPARSE_GRID = "2.5,5,10,20"
DEFAULT_ALPHA_GENE_GRID = "0.5,1,2,4"
DEFAULT_RANK_GRID = "40,60,80,100"


def parse_float_grid(text: str, name: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two values.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values: {values}")
    if any((not np.isfinite(x)) or x < 0 for x in values):
        raise ValueError(f"{name} must contain finite nonnegative values.")
    return values


def parse_rank_grid(text: str) -> List[int]:
    raw = [x.strip() for x in text.split(",") if x.strip()]
    values = [int(x) for x in raw]
    if len(values) < 2:
        raise ValueError("--rank-grid must contain at least two integer values.")
    if len(set(values)) != len(values):
        raise ValueError(f"--rank-grid contains duplicates: {values}")
    if any(x <= 0 for x in values):
        raise ValueError("--rank-grid values must be positive.")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Three two-parameter PGNMF sensitivity sweeps (RMSE, ARI, NMI)."
    )
    p.add_argument("--dataset", default="C11")
    p.add_argument("--masked", default=None)
    p.add_argument("--mask-plan", default=None)
    p.add_argument("--gene-set-masks", default=None)
    p.add_argument("--gmt", default=None)
    p.add_argument("--gene-set-key", default="HVG1000+PATHWAY")
    p.add_argument("--gene-symbol-column", default=None)
    p.add_argument("--label-key", default="majority_voting")
    p.add_argument("--output-dir", default=None)

    # Frozen current-model baseline.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank-baseline", type=int, default=80)
    p.add_argument("--alpha-cell-baseline", type=float, default=10.0)
    p.add_argument("--alpha-gene-baseline", type=float, default=2.0)
    p.add_argument("--lambda-sparse-baseline", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lambda-reg", type=float, default=1e-3)
    p.add_argument("--pathway-weight", type=float, default=0.5)
    p.add_argument("--nonpathway-weight", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--tol", type=float, default=1e-4)

    # Sensitivity grids.
    p.add_argument("--alpha-cell-grid", default=DEFAULT_ALPHA_CELL_GRID)
    p.add_argument("--lambda-sparse-grid", default=DEFAULT_LAMBDA_SPARSE_GRID)
    p.add_argument("--alpha-gene-grid", default=DEFAULT_ALPHA_GENE_GRID)
    p.add_argument("--rank-grid", default=DEFAULT_RANK_GRID)

    # Frozen held-out evaluation protocol.
    p.add_argument("--validation-seed", type=int, default=2026)
    p.add_argument("--validation-fraction", type=float, default=0.5)
    p.add_argument("--resolution", type=float, default=0.30)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--metric", default="cosine")

    # Frozen graph construction protocol.
    p.add_argument("--graph-pcs", type=int, default=30)
    p.add_argument("--graph-neighbors", type=int, default=15)
    p.add_argument("--graph-metric", default="euclidean")
    p.add_argument("--gene-edge-batch-size", type=int, default=256)
    p.add_argument("--min-pathway-genes", type=int, default=10)
    p.add_argument("--max-pathway-genes", type=int, default=500)

    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from sensitivity_fit_results_unique.csv after checking the run signature.",
    )
    return p.parse_args()


def resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def choose_gmt(value: str | None) -> Path:
    if value:
        path = resolve_path(value, ROOT / "data" / "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt")
        if not path.is_file():
            raise FileNotFoundError(f"GMT file does not exist: {path}")
        return path
    preferred = ROOT / "data" / "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted(x for x in ROOT.rglob("*.gmt") if x.is_file())
    kegg = [x for x in candidates if "kegg" in x.name.lower()]
    candidates = kegg or candidates
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            "No GMT file found. Pass --gmt with the original KEGG GMT path."
        )
    raise RuntimeError(
        "Multiple GMT files found; pass --gmt explicitly:\n"
        + "\n".join(str(x) for x in candidates[:20])
    )


def dense32(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def split_indices(n: int, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("At least two masked-positive positions are required.")
    if not (0.0 < fraction < 1.0):
        raise ValueError("--validation-fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_validation = min(max(1, int(round(n * fraction))), n - 1)
    return np.sort(order[:n_validation]), np.sort(order[n_validation:])


def heldout_test_positions(
    mask_plan: MaskPlan,
    model_gene_mask: np.ndarray,
    validation_fraction: float,
    validation_seed: int,
):
    """
    Match the current Stage2C27/44 held-out protocol.

    The reference expression values are returned only for post-fit scoring.
    Only row/column coordinates are passed to the solver.
    """
    relevant = np.asarray(model_gene_mask, dtype=bool)[
        np.asarray(mask_plan.cols, dtype=np.int64)
    ]
    rows = np.asarray(mask_plan.rows[relevant], dtype=np.int64)
    cols_global = np.asarray(mask_plan.cols[relevant], dtype=np.int64)
    truth = np.asarray(mask_plan.reference_values[relevant], dtype=np.float64)

    if truth.size < 2:
        raise RuntimeError("Too few masked-positive entries in the model gene set.")

    _, test_idx = split_indices(truth.size, validation_fraction, validation_seed)
    return rows[test_idx], cols_global[test_idx], truth[test_idx], int(truth.size)


def coordinate_hash(rows: np.ndarray, cols: np.ndarray) -> str:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    if rows.size != cols.size:
        raise ValueError("Coordinate arrays have different lengths.")
    order = np.lexsort((cols, rows))
    xy = np.column_stack((rows[order], cols[order])).astype(np.int64, copy=False)
    return hashlib.sha256(xy.tobytes()).hexdigest()


def load_gene_mask(path: Path, key: str, n_vars: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"Gene-set key {key!r} not found in {path}; keys={z.files}")
        mask = np.asarray(z[key], dtype=bool)
    if mask.shape != (n_vars,):
        raise ValueError(f"Gene mask shape {mask.shape} != ({n_vars},)")
    if not np.any(mask):
        raise ValueError("Gene mask contains no selected genes.")
    return mask


def leiden_metrics(
    X_dense: np.ndarray,
    obs: pd.DataFrame,
    label_key: str,
    *,
    resolution: float,
    seed: int,
    n_pcs: int,
    n_neighbors: int,
    metric: str,
) -> Tuple[float, float, int]:
    """
    Same downstream metric protocol as the current C11 convergence/evaluation code:
    PCA -> neighbors -> Leiden at a frozen label-free resolution -> ARI/NMI.
    """
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
    ari = float(adjusted_rand_score(truth, pred))
    nmi = float(normalized_mutual_info_score(truth, pred))
    n_clusters = int(pd.Series(pred).nunique())
    return ari, nmi, n_clusters


def config_key(alpha_cell: float, alpha_gene: float, lambda_sparse: float, rank: int) -> str:
    return (
        f"ac={alpha_cell:.12g}|ag={alpha_gene:.12g}|"
        f"ls={lambda_sparse:.12g}|rank={int(rank)}"
    )


def build_sweeps(args, alpha_cell_grid, lambda_sparse_grid, alpha_gene_grid, rank_grid):
    """
    Matrix orientation follows the original plotting script:
    rows = first/x parameter; columns = second/y parameter.
    """
    baseline = {
        "alpha_cell": float(args.alpha_cell_baseline),
        "alpha_gene": float(args.alpha_gene_baseline),
        "lambda_sparse": float(args.lambda_sparse_baseline),
        "rank": int(args.rank_baseline),
    }
    sweeps = [
        {
            "sweep": "alpha_cell_x_lambda_sparse",
            "x_param": "alpha_cell",
            "y_param": "lambda_sparse",
            "x_values": alpha_cell_grid,
            "y_values": lambda_sparse_grid,
            "configs": [
                (float(ac), baseline["alpha_gene"], float(ls), baseline["rank"])
                for ac in alpha_cell_grid
                for ls in lambda_sparse_grid
            ],
        },
        {
            "sweep": "alpha_gene_x_rank",
            "x_param": "alpha_gene",
            "y_param": "rank",
            "x_values": alpha_gene_grid,
            "y_values": rank_grid,
            "configs": [
                (baseline["alpha_cell"], float(ag), baseline["lambda_sparse"], int(r))
                for ag in alpha_gene_grid
                for r in rank_grid
            ],
        },
        {
            "sweep": "lambda_sparse_x_rank",
            "x_param": "lambda_sparse",
            "y_param": "rank",
            "x_values": lambda_sparse_grid,
            "y_values": rank_grid,
            "configs": [
                (baseline["alpha_cell"], baseline["alpha_gene"], float(ls), int(r))
                for ls in lambda_sparse_grid
                for r in rank_grid
            ],
        },
    ]
    return sweeps


def run_one_fit(
    *,
    X: np.ndarray,
    zero_mask: np.ndarray,
    obs: pd.DataFrame,
    label_key: str,
    S_cell,
    degree_cell,
    S_gene,
    degree_gene,
    pathway_model: np.ndarray,
    test_rows: np.ndarray,
    test_cols_local: np.ndarray,
    test_truth: np.ndarray,
    q0: float,
    alpha_cell: float,
    alpha_gene: float,
    lambda_sparse: float,
    rank: int,
    args,
) -> Dict:
    penalty = make_two_level_penalty(
        pathway_model,
        strength=float(lambda_sparse),
        pathway_weight=float(args.pathway_weight),
        nonpathway_weight=float(args.nonpathway_weight),
    ).astype(np.float32, copy=False)

    result = zero_aware_nmf_mu_blocked(
        X,
        S_cell=S_cell,
        degree_cell=degree_cell,
        alpha_cell=float(alpha_cell),
        S_gene=S_gene,
        degree_gene=degree_gene,
        alpha_gene=float(alpha_gene),
        lambda_h_gene=penalty,
        zero_weight=float(q0),
        rank=int(rank),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        random_state=int(args.seed),
        lambda_reg=float(args.lambda_reg),
        lambda_w_l1=0.0,
        verbose=False,
        tag=f"SENS ac={alpha_cell:g} ag={alpha_gene:g} ls={lambda_sparse:g} r={rank}",
        zero_penalty_exclude_rows=test_rows,
        zero_penalty_exclude_cols=test_cols_local,
    block_size=500,
    )
    X_hat = np.asarray(result.W @ result.H, dtype=np.float32)

    # Test RMSE is calculated only on the frozen masked-positive TEST coordinates.
    test_pred = np.asarray(X_hat[test_rows, test_cols_local], dtype=np.float64)
    test_rmse = float(np.sqrt(np.mean((test_pred - test_truth) ** 2)))

    # Same DENSE_PRESERVE_OBS view used in the latest PGNMF evaluation:
    # keep observed entries unchanged and fill all input-zero entries with W@H.
    X_dense = X.copy()
    X_dense[zero_mask] = X_hat[zero_mask]

    ari, nmi, n_clusters = leiden_metrics(
        X_dense,
        obs,
        label_key,
        resolution=float(args.resolution),
        seed=int(args.seed),
        n_pcs=int(args.n_pcs),
        n_neighbors=int(args.n_neighbors),
        metric=str(args.metric),
    )

    record = {
        "config_key": config_key(alpha_cell, alpha_gene, lambda_sparse, rank),
        "alpha_cell": float(alpha_cell),
        "alpha_gene": float(alpha_gene),
        "lambda_sparse": float(lambda_sparse),
        "sparsity_strength": float(lambda_sparse),
        "rank": int(rank),
        "rmse": test_rmse,
        "test_rmse": test_rmse,
        "ari": float(ari),
        "nmi": float(nmi),
        "n_clusters": int(n_clusters),
        "n_test": int(test_truth.size),
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "final_objective": float(result.losses[-1]) if result.losses else np.nan,
        "fit_seconds": float(result.elapsed_seconds),
    }

    del X_hat, X_dense, penalty, result
    gc.collect()
    return record


def write_matrix_csv(
    df_sweep: pd.DataFrame,
    x_values: List[float],
    y_values: List[float],
    metric: str,
    output_path: Path,
):
    pivot = df_sweep.pivot(index="x_value", columns="y_value", values=metric)
    pivot = pivot.reindex(index=x_values, columns=y_values)
    pivot.index.name = "x_value"
    pivot.columns.name = "y_value"
    pivot.to_csv(output_path, float_format="%.10g")
    return pivot


def python_literal_matrix(pivot: pd.DataFrame) -> str:
    rows = []
    for _, row in pivot.iterrows():
        values = []
        for value in row.to_numpy(dtype=float):
            values.append("np.nan" if not np.isfinite(value) else repr(float(value)))
        rows.append("[" + ", ".join(values) + "]")
    return "[\n    " + ",\n    ".join(rows) + "\n]"


def write_plot_arrays(sweep_tables: Dict[str, pd.DataFrame], sweeps: List[Dict], out: Path):
    """Write ready-to-paste NumPy arrays matching the old Figure 5 plot script."""
    lookup = {s["sweep"]: s for s in sweeps}
    matrices = {}
    for sweep_name, sweep in lookup.items():
        sub = sweep_tables[sweep_name]
        for metric in ("rmse", "ari", "nmi"):
            pivot = sub.pivot(index="x_value", columns="y_value", values=metric)
            pivot = pivot.reindex(index=sweep["x_values"], columns=sweep["y_values"])
            matrices[(sweep_name, metric)] = pivot

    lines = [
        "# Auto-generated from the current PGNMF two-parameter sensitivity run.",
        "# Replace the old hard-coded arrays in 灵敏度双参数绘图.py with this block.",
        "import numpy as np",
        "",
        "# (a)-(c): alpha_cell x lambda_sparse",
        f"alpha_vals = {lookup['alpha_cell_x_lambda_sparse']['x_values']!r}",
        f"lambda_vals = {lookup['alpha_cell_x_lambda_sparse']['y_values']!r}",
        "lambda_vals_rev = list(reversed(lambda_vals))",
    ]
    for metric in ("rmse", "ari", "nmi"):
        name = f"{metric}_alpha_lambda"
        lines.append(f"{name} = np.array({python_literal_matrix(matrices[('alpha_cell_x_lambda_sparse', metric)])}, dtype=float)")
        lines.append(f"{name}_rev = np.array([list(reversed(row)) for row in {name}])")
    lines += [
        "",
        "# (d)-(f): alpha_gene x rank",
        f"alpha_gene_vals = {lookup['alpha_gene_x_rank']['x_values']!r}",
        f"rank_vals = {lookup['alpha_gene_x_rank']['y_values']!r}",
        "rank_vals_rev = list(reversed(rank_vals))",
    ]
    for metric in ("rmse", "ari", "nmi"):
        name = f"{metric}_alpha_gene_rank"
        lines.append(f"{name} = np.array({python_literal_matrix(matrices[('alpha_gene_x_rank', metric)])}, dtype=float)")
        lines.append(f"{name}_rev = np.array([list(reversed(row)) for row in {name}])")
    lines += [
        "",
        "# (g)-(i): lambda_sparse x rank",
        f"lambda_full_vals = {lookup['lambda_sparse_x_rank']['x_values']!r}",
        "lambda_full_vals_rev = list(reversed(lambda_full_vals))",
        f"rank_full_vals = {lookup['lambda_sparse_x_rank']['y_values']!r}",
        "rank_full_vals_rev = list(reversed(rank_full_vals))",
    ]
    for metric in ("rmse", "ari", "nmi"):
        name = f"{metric}_lambda_rank"
        lines.append(f"{name} = np.array({python_literal_matrix(matrices[('lambda_sparse_x_rank', metric)])}, dtype=float)")
        lines.append(
            f"{name}_rev = np.array([list(reversed(row)) for row in reversed({name})])"
        )

    lines += [
        "",
        "# The current metrics are on a narrower scale than the old Figure 5.",
        "# In plot_3d_bars, prefer vmin/vmax=None (automatic per-panel scaling),",
        "# or set bounds from each matrix's finite min/max.",
        "",
    ]
    (out / "plot_arrays_for_Figure5.py").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    if not args.label_key:
        raise ValueError("--label-key must identify the annotation column.")
    required_solver_args = {"zero_penalty_exclude_rows", "zero_penalty_exclude_cols"}
    solver_args = set(inspect.signature(zero_aware_nmf_mu_blocked).parameters)
    if not required_solver_args.issubset(solver_args):
        raise RuntimeError(
            "The maintained exact-block solver must support held-out zero-penalty exclusions."
        )
    if args.beta < 0 or args.beta > 1:
        raise ValueError("--beta must be in [0, 1].")
    if args.rank_baseline <= 0 or args.max_iter <= 0 or args.tol <= 0:
        raise ValueError("Invalid rank/max-iter/tolerance.")
    if args.pathway_weight < 0 or args.nonpathway_weight < 0:
        raise ValueError("Pathway sparsity weights must be nonnegative.")

    alpha_cell_grid = parse_float_grid(args.alpha_cell_grid, "--alpha-cell-grid")
    lambda_sparse_grid = parse_float_grid(args.lambda_sparse_grid, "--lambda-sparse-grid")
    alpha_gene_grid = parse_float_grid(args.alpha_gene_grid, "--alpha-gene-grid")
    rank_grid = parse_rank_grid(args.rank_grid)

    masked_path = resolve_path(
        args.masked, ROOT / "work" / args.dataset / "masked_logcp10k.h5ad"
    )
    mask_plan_path = resolve_path(
        args.mask_plan, ROOT / "work" / args.dataset / "mask_plan.npz"
    )
    gene_masks_path = resolve_path(
        args.gene_set_masks,
        ROOT / "audit" / args.dataset / "stage2cpre_gene_set_diagnostic" / "gene_set_masks.npz",
    )
    gmt_path = choose_gmt(args.gmt)
    output_dir = resolve_path(
        args.output_dir,
        ROOT / "audit" / args.dataset / "stage2c37_two_parameter_sensitivity",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (masked_path, mask_plan_path, gene_masks_path, gmt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file does not exist: {path}")

    print("=" * 88)
    print("PGNMF CURRENT-MODEL TWO-PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 88)
    print(f"Dataset: {args.dataset}")
    print(f"Masked input: {masked_path}")
    print(f"Mask plan: {mask_plan_path}")
    print(f"Gene masks: {gene_masks_path}")
    print(f"KEGG GMT: {gmt_path}")
    print(f"Label key: {args.label_key}")
    print(f"Baseline: alpha_cell={args.alpha_cell_baseline}, alpha_gene={args.alpha_gene_baseline}, "
          f"lambda_sparse={args.lambda_sparse_baseline}, rank={args.rank_baseline}")
    print(f"Grids: alpha_cell={alpha_cell_grid}; lambda_sparse={lambda_sparse_grid}; "
          f"alpha_gene={alpha_gene_grid}; rank={rank_grid}")
    print(f"Seed={args.seed}; max_iter={args.max_iter}; tol={args.tol}; "
          f"Leiden resolution={args.resolution}")
    print("RMSE = held-out TEST RMSE; ARI/NMI = post-hoc DENSE_PRESERVE_OBS Leiden metrics.")
    print("ARI/NMI are not used for parameter selection.")

    ad_masked = sc.read_h5ad(masked_path)
    if args.label_key not in ad_masked.obs.columns:
        raise KeyError(
            f"Label key {args.label_key!r} not found. Available columns: "
            f"{ad_masked.obs.columns.tolist()}"
        )

    model_gene_mask = load_gene_mask(gene_masks_path, args.gene_set_key, ad_masked.n_vars)
    model_gene_indices = np.flatnonzero(model_gene_mask)
    X = dense32(ad_masked[:, model_gene_mask].X)
    if X.ndim != 2 or not np.all(np.isfinite(X)) or np.any(X < 0):
        raise ValueError("Masked model matrix must be finite and nonnegative.")
    obs = ad_masked.obs.copy()

    plan = MaskPlan.load(str(mask_plan_path))
    if tuple(plan.shape) != tuple(ad_masked.shape):
        raise ValueError(
            f"MaskPlan shape {tuple(plan.shape)} != masked AnnData shape {ad_masked.shape}"
        )

    test_rows, test_cols_global, test_truth, n_masked_positive = heldout_test_positions(
        plan,
        model_gene_mask,
        args.validation_fraction,
        args.validation_seed,
    )
    global_to_local = np.full(ad_masked.n_vars, -1, dtype=np.int64)
    global_to_local[model_gene_indices] = np.arange(model_gene_indices.size, dtype=np.int64)
    test_cols_local = global_to_local[test_cols_global]
    if np.any(test_cols_local < 0):
        raise RuntimeError("A held-out test gene is missing from the model feature set.")
    if np.any(X[test_rows, test_cols_local] != 0):
        raise RuntimeError("Held-out test positions must be zero in the masked model input.")

    heldout_hash = coordinate_hash(test_rows, test_cols_global)
    observed_mask = X > 0
    zero_mask = ~observed_mask
    n_positive = int(np.count_nonzero(observed_mask))
    n_zero = int(np.count_nonzero(zero_mask))
    n_penalized_zero = n_zero - int(test_rows.size)
    if n_penalized_zero <= 0:
        raise RuntimeError("No zero positions remain for the weak zero penalty.")
    # Current model uses the balanced zero-weight rule:
    # q0 = beta * n_observed_positive / n_remaining_penalized_zero.
    q0 = float(args.beta) * n_positive / n_penalized_zero
    if q0 > 1:
        raise ValueError(f"Effective zero weight q0={q0:.6g} exceeds 1.")

    all_symbols, symbol_source = get_gene_symbols(ad_masked, args.gene_symbol_column)
    all_symbols = np.asarray(all_symbols, dtype=str)
    if all_symbols.size != ad_masked.n_vars:
        raise ValueError("Gene-symbol vector length differs from AnnData n_vars.")

    pathway_all, _ = pathway_membership_from_gmt(
        str(gmt_path),
        all_symbols,
        min_pathway_genes=args.min_pathway_genes,
        max_pathway_genes=args.max_pathway_genes,
        gene_symbol_source=symbol_source,
    )
    pathway_model = np.asarray(pathway_all[model_gene_mask], dtype=bool)

    print("\n--- Frozen evaluation split ---")
    print(f"Model matrix: {X.shape}; selected genes={model_gene_indices.size}")
    print(f"Pathway genes={int(pathway_model.sum())}; non-pathway genes={int((~pathway_model).sum())}")
    print(f"Masked-positive positions in model genes={n_masked_positive}")
    print(f"Test positions={test_truth.size}; validation fraction={args.validation_fraction}")
    print(f"Test coordinate SHA256={heldout_hash}")
    print(f"Observed positives={n_positive}; zero entries={n_zero}; penalized zeros={n_penalized_zero}")
    print(f"Balanced weak-zero weight q0={q0:.10f}")

    print("\n--- Building fixed KEGG pathway-rank gene graph ONCE ---")
    t0 = time.perf_counter()
    Sg_raw, dg_raw, gene_info, gene_diag = build_pathway_rank_gene_graph(
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
    print(f"Gene graph built in {time.perf_counter() - t0:.1f}s; "
          f"median positive degree={gene_graph_scale:.6g}")

    print("\n--- Building fixed union-kNN Euclidean cell graph ONCE ---")
    t0 = time.perf_counter()
    S_cell, degree_cell, cell_info, _, _ = build_sparse_knn_cell_graph(
        X,
        n_pcs=args.graph_pcs,
        n_neighbors=args.graph_neighbors,
        metric=args.graph_metric,
        random_state=args.seed,
    )
    print(f"Cell graph built in {time.perf_counter() - t0:.1f}s; "
          f"edges={getattr(cell_info, 'n_edges_undirected', 'NA')}")

    sweeps = build_sweeps(
        args, alpha_cell_grid, lambda_sparse_grid, alpha_gene_grid, rank_grid
    )
    unique_configs = list(dict.fromkeys(
        cfg for sweep in sweeps for cfg in sweep["configs"]
    ))
    print("\n--- Sweep size ---")
    print(f"Parameter combinations across the three panels: "
          f"{sum(len(s['configs']) for s in sweeps)}")
    print(f"Unique model fits after reusing overlapping combinations: {len(unique_configs)}")

    fixed_config = {
        "dataset": args.dataset,
        "masked_path": str(masked_path),
        "masked_size": masked_path.stat().st_size,
        "masked_mtime_ns": masked_path.stat().st_mtime_ns,
        "mask_plan_path": str(mask_plan_path),
        "mask_plan_size": mask_plan_path.stat().st_size,
        "mask_plan_mtime_ns": mask_plan_path.stat().st_mtime_ns,
        "gene_masks_path": str(gene_masks_path),
        "gene_masks_size": gene_masks_path.stat().st_size,
        "gene_masks_mtime_ns": gene_masks_path.stat().st_mtime_ns,
        "gmt_path": str(gmt_path),
        "gmt_mtime_ns": gmt_path.stat().st_mtime_ns,
        "gene_set_key": args.gene_set_key,
        "gene_mask_sha256": hashlib.sha256(
            np.asarray(model_gene_mask, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "heldout_test_coordinate_sha256": heldout_hash,
        "label_key": args.label_key,
        "seed": args.seed,
        "rank_baseline": args.rank_baseline,
        "alpha_cell_baseline": args.alpha_cell_baseline,
        "alpha_gene_baseline": args.alpha_gene_baseline,
        "lambda_sparse_baseline": args.lambda_sparse_baseline,
        "beta": args.beta,
        "effective_zero_weight_q0": q0,
        "lambda_reg": args.lambda_reg,
        "pathway_weight": args.pathway_weight,
        "nonpathway_weight": args.nonpathway_weight,
        "max_iter": args.max_iter,
        "tol": args.tol,
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
        "alpha_cell_grid": alpha_cell_grid,
        "lambda_sparse_grid": lambda_sparse_grid,
        "alpha_gene_grid": alpha_gene_grid,
        "rank_grid": rank_grid,
        "sensitivity_protocol": "current zero-aware PGNMF; held-out test positions excluded from zero penalty",
    }
    signature = hashlib.sha256(
        json.dumps(fixed_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest_path = output_dir / "sensitivity_manifest.json"
    unique_csv = output_dir / "sensitivity_fit_results_unique.csv"

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "--resume requested, but sensitivity_manifest.json is missing."
            )
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("run_signature") != signature:
            raise RuntimeError(
                "Existing sensitivity output has a different run signature. "
                "Use a new --output-dir; do not mix results from different inputs/settings."
            )
        if unique_csv.is_file():
            unique_df = pd.read_csv(unique_csv)
            completed = set(
                unique_df.loc[unique_df["status"] == "success", "config_key"].astype(str)
            )
        else:
            unique_df = pd.DataFrame()
            completed = set()
        print(f"Resuming: {len(completed)} successful unique fits already exist.")
    else:
        if unique_csv.exists():
            raise FileExistsError(
                f"{unique_csv} already exists. Use --resume to continue, "
                "or choose a new --output-dir."
            )
        unique_df = pd.DataFrame()
        completed = set()

    manifest = {
        "run_signature": signature,
        "fixed_configuration": fixed_config,
        "baseline": {
            "alpha_cell": args.alpha_cell_baseline,
            "alpha_gene": args.alpha_gene_baseline,
            "lambda_sparse": args.lambda_sparse_baseline,
            "rank": args.rank_baseline,
        },
        "sweeps": [
            {
                "sweep": s["sweep"],
                "x_param": s["x_param"],
                "y_param": s["y_param"],
                "x_values": s["x_values"],
                "y_values": s["y_values"],
                "n_combinations": len(s["configs"]),
            }
            for s in sweeps
        ],
        "metric_definitions": {
            "rmse": "RMSE on masked-positive held-out TEST coordinates only, in log-CP10K space.",
            "ari_nmi": "Leiden clustering metrics on DENSE_PRESERVE_OBS (observed entries retained; zero entries replaced by W@H), using the frozen label-free resolution.",
            "labels": "Used only after fitting for post-hoc ARI/NMI; never used by the PGNMF solver.",
        },
        "interpretation_note": (
            "C11 was used for model development. Sensitivity results on C11 describe "
            "local robustness around the frozen C11 setting and are not an independent external validation."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for fit_i, (alpha_cell, alpha_gene, lambda_sparse, rank) in enumerate(unique_configs, start=1):
        key = config_key(alpha_cell, alpha_gene, lambda_sparse, rank)
        if key in completed:
            print(f"\n[{fit_i}/{len(unique_configs)}] SKIP cached {key}")
            continue

        print(f"\n[{fit_i}/{len(unique_configs)}] FIT {key}", flush=True)
        t0 = time.perf_counter()
        try:
            record = run_one_fit(
                X=X,
                zero_mask=zero_mask,
                obs=obs,
                label_key=args.label_key,
                S_cell=S_cell,
                degree_cell=degree_cell,
                S_gene=S_gene,
                degree_gene=degree_gene,
                pathway_model=pathway_model,
                test_rows=test_rows,
                test_cols_local=test_cols_local,
                test_truth=test_truth,
                q0=q0,
                alpha_cell=alpha_cell,
                alpha_gene=alpha_gene,
                lambda_sparse=lambda_sparse,
                rank=rank,
                args=args,
            )
            record["status"] = "success"
            record["error"] = ""
            if unique_df.empty:
                unique_df = pd.DataFrame([record])
            else:
                unique_df = pd.concat([unique_df, pd.DataFrame([record])], ignore_index=True)
            unique_df.to_csv(unique_csv, index=False, encoding="utf-8-sig")
            completed.add(key)
            print(
                f"  RMSE={record['rmse']:.6f} | ARI={record['ari']:.6f} | "
                f"NMI={record['nmi']:.6f} | iter={record['n_iter']} "
                f"| converged={record['converged']} | fit={time.perf_counter()-t0:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "config_key": key,
                "alpha_cell": alpha_cell,
                "alpha_gene": alpha_gene,
                "lambda_sparse": lambda_sparse,
                "rank": rank,
                "error": repr(exc),
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            failures_path = output_dir / "sensitivity_fit_failures.csv"
            if failures_path.exists():
                old_failures = pd.read_csv(failures_path)
                old_failures = pd.concat([old_failures, pd.DataFrame([failure])], ignore_index=True)
            else:
                old_failures = pd.DataFrame([failure])
            old_failures.to_csv(failures_path, index=False, encoding="utf-8-sig")
            print(f"  FAILED: {exc!r}", flush=True)
            print("  This configuration will be retried when you rerun with --resume.", flush=True)

    if unique_df.empty:
        raise RuntimeError("No sensitivity fits completed successfully.")

    # Re-read the latest on-disk fit table to include any completed resume work.
    unique_df = pd.read_csv(unique_csv)
    successful = unique_df[unique_df["status"] == "success"].copy()
    result_lookup = {
        str(row["config_key"]): row.to_dict()
        for _, row in successful.iterrows()
    }

    all_sweep_rows = []
    sweep_tables: Dict[str, pd.DataFrame] = {}
    missing_cells = []
    for sweep in sweeps:
        rows = []
        for cfg in sweep["configs"]:
            ac, ag, ls, rank = cfg
            key = config_key(ac, ag, ls, rank)
            rec = result_lookup.get(key)
            if rec is None:
                missing_cells.append((sweep["sweep"], key))
                continue
            row = dict(rec)
            row.update({
                "sweep": sweep["sweep"],
                "x_param": sweep["x_param"],
                "y_param": sweep["y_param"],
                "x_value": ac if sweep["x_param"] == "alpha_cell"
                    else ag if sweep["x_param"] == "alpha_gene"
                    else ls,
                "y_value": ls if sweep["y_param"] == "lambda_sparse" else rank,
            })
            rows.append(row)
            all_sweep_rows.append(row)
        sweep_tables[sweep["sweep"]] = pd.DataFrame(rows)

    all_df = pd.DataFrame(all_sweep_rows)
    all_csv = output_dir / "sensitivity_two_parameter_all.csv"
    all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    metric_files = []
    for sweep in sweeps:
        sub = sweep_tables[sweep["sweep"]]
        for metric in ("rmse", "ari", "nmi"):
            matrix_path = output_dir / f"{sweep['sweep']}_{metric}_matrix.csv"
            write_matrix_csv(
                sub,
                sweep["x_values"],
                sweep["y_values"],
                metric,
                matrix_path,
            )
            metric_files.append(matrix_path.name)

    if not missing_cells:
        write_plot_arrays(sweep_tables, sweeps, output_dir)
        print("\nAll parameter cells completed.")
        print("Ready-to-paste plotting arrays: " + str(output_dir / "plot_arrays_for_Figure5.py"))
    else:
        print("\nWARNING: Some parameter combinations are still missing:")
        for sweep_name, key in missing_cells:
            print(f"  {sweep_name}: {key}")
        print("Rerun the same command with --resume to complete failed combinations.")
        print("Do NOT use the partially filled matrices to make the final Figure 5.")

    print("\n=== SENSITIVITY ANALYSIS OUTPUTS ===")
    print("Unique fits :", unique_csv)
    print("Long table  :", all_csv)
    print("Matrices    :", output_dir)
    for name in metric_files:
        print("  -", name)
    if not missing_cells:
        print("Plot arrays :", output_dir / "plot_arrays_for_Figure5.py")
    print("\nInterpretation reminder: ARI/NMI are post-hoc diagnostics, not tuning objectives.")


if __name__ == "__main__":
    main()
