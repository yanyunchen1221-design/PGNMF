#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))

from pgnmf_revision.data import load_and_prepare_h5ad, strip_derived_analysis
from pgnmf_revision.masking import generate_mask_plan, apply_mask, write_mask_summary


def parse_args():
    p = argparse.ArgumentParser(description="Prepare a fixed simulated-missing dataset for PGNMF revision experiments.")
    p.add_argument("--input", required=True, help="Original h5ad")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--input-space", choices=["counts", "logcp10k"], default="counts")
    p.add_argument("--label-key", default=None)
    p.add_argument("--mask-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--max-mt-pct", type=float, default=5.0)
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prepared = load_and_prepare_h5ad(
        a.input,
        input_space=a.input_space,
        label_key=a.label_key,
        min_genes=a.min_genes,
        min_cells=a.min_cells,
        max_mt_pct=a.max_mt_pct,
    )
    # IMPORTANT: remove PCA/UMAP/neighbors inherited from the source h5ad.
    # They may have been computed from the unmasked matrix and would leak held-out
    # expression values into any solver that reuses them.
    ref = strip_derived_analysis(prepared.adata_reference.copy())

    plan = generate_mask_plan(ref.X, mask_ratio=a.mask_ratio, random_seed=a.seed)
    masked = ref.copy()
    masked.X = apply_mask(ref.X, plan)
    masked.uns["pgnmf_mask_plan"] = {
        "mask_ratio": a.mask_ratio,
        "seed": a.seed,
        "n_masked": plan.n_masked,
        "common_space": ref.uns.get("pgnmf_common_space", "unknown"),
    }

    # Save the exact pair used later for every algorithm.
    ref.write_h5ad(out / "reference_logcp10k.h5ad")
    masked.write_h5ad(out / "masked_logcp10k.h5ad")
    plan.save(out / "mask_plan.npz")
    write_mask_summary(plan, out / "mask_summary.json")

    print("Prepared Stage-1 evaluation inputs")
    print(f"  reference: {out / 'reference_logcp10k.h5ad'}")
    print(f"  masked:    {out / 'masked_logcp10k.h5ad'}")
    print(f"  mask plan: {out / 'mask_plan.npz'}")
    print(f"  label key: {prepared.label_key}")
    print(f"  masked entries: {plan.n_masked:,}")
    print(f"  inherited embeddings removed: obsm={list(masked.obsm.keys())}, obsp={list(masked.obsp.keys())}")


if __name__ == "__main__":
    main()
