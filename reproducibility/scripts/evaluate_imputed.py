#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))

import scanpy as sc

from pgnmf_revision.masking import MaskPlan
from pgnmf_revision.metrics import masked_rmse, masked_mae, masked_bias, masked_pearson
from pgnmf_revision.io_utils import load_imputed_matrix
from pgnmf_revision.clustering import select_common_hvgs, evaluate_fixed_leiden


def parse_args():
    p = argparse.ArgumentParser(description="Correct RMSE + fixed-Leiden evaluation for one imputation output.")
    p.add_argument("--reference", required=True, help="reference_logcp10k.h5ad from step 01")
    p.add_argument("--masked", required=True, help="masked_logcp10k.h5ad from step 01")
    p.add_argument("--mask-plan", required=True)
    p.add_argument("--imputed", required=True, help="Imputed .h5ad/.npy/.npz in the SAME common expression space")
    p.add_argument("--imputed-layer", default="imputed")
    p.add_argument("--label-key", default="celltype")
    p.add_argument("--resolution", type=float, default=0.4, help="Pre-specified Leiden resolution; NEVER selected by ARI/NMI")
    p.add_argument("--n-hvgs", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    reference = sc.read_h5ad(a.reference)
    masked = sc.read_h5ad(a.masked)
    plan = MaskPlan.load(a.mask_plan)
    X_imp = load_imputed_matrix(a.imputed, layer=a.imputed_layer)

    if tuple(reference.shape) != tuple(masked.shape) or tuple(reference.shape) != tuple(X_imp.shape):
        raise ValueError(f"Shape mismatch: reference={reference.shape}, masked={masked.shape}, imputed={X_imp.shape}")

    # 1) Correct simulated-missing metrics: ONLY exact held-out positions.
    metrics = {
        "RMSE_masked": masked_rmse(X_imp, plan),
        "MAE_masked": masked_mae(X_imp, plan),
        "Bias_masked": masked_bias(X_imp, plan),
        "Pearson_masked": masked_pearson(X_imp, plan),
        "n_masked": plan.n_masked,
        "mask_ratio_requested": plan.mask_ratio,
        "mask_seed": plan.random_seed,
    }

    # 2) One common HVG set selected from the common MASKED INPUT (not from each method output).
    hvg_mask = select_common_hvgs(masked, n_top_genes=a.n_hvgs)

    # 3) Fixed Leiden: no scan over resolutions; labels are read only after clustering.
    label_key = a.label_key if a.label_key in reference.obs.columns else None
    cluster_result, eval_adata = evaluate_fixed_leiden(
        adata_template=reference,
        X_imputed=X_imp,
        hvg_mask=hvg_mask,
        label_key=label_key,
        resolution=a.resolution,
        n_pcs=a.n_pcs,
        n_neighbors=a.n_neighbors,
        metric=a.metric,
        random_state=a.seed,
        compute_umap=True,
    )
    metrics.update({f"clustering_{k}": v for k, v in cluster_result.to_dict().items()})

    (out / "evaluation.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    eval_adata.obs[["leiden"] + ([label_key] if label_key else [])].to_csv(out / "leiden_assignments.csv")
    eval_adata.write_h5ad(out / "clustering_eval.h5ad")

    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Saved: {out / 'evaluation.json'}")


if __name__ == "__main__":
    main()
