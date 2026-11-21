#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-hoc biological interpretation of the frozen C11 PGNMF rank-80 model.

This script DOES NOT fit/refit PGNMF. It loads the verified Stage2C27 W/H
cache, aligns H columns through the existing HVG1000+PATHWAY feature mask,
and produces:
  A. factor-by-cell-type activity heatmap (top cell-type-specific factors)
  B. factor-associated KEGG over-representation analysis (hypergeometric + BH FDR)
  C. UMAP of the frozen PGNMF W representation, coloured by representative factors

Run from the PGNMF repository root:
python -u reproducibility/scripts/analyze_c11_factor_biology.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
# Add project source directory to Python import path
SRC_DIR = ROOT / "reproducibility" / "legacy"

if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
elif (ROOT / "reproducibility" / "legacy" / "pgnmf_revision").is_dir():
    sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))
else:
    raise FileNotFoundError(
        f"Cannot locate PGNMF source package under {ROOT}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Biological interpretation of frozen C11 PGNMF W/H factors; no refitting."
    )
    p.add_argument("--dataset", default="C11")
    p.add_argument("--label-key", default="majority_voting")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=80)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--gene-set-key", default="HVG1000+PATHWAY")
    p.add_argument("--gene-symbol-column", default=None)
    p.add_argument(
        "--cache-dir",
        default=r"audit/C11/stage2c27_marker_figures_converged215",
    )
    p.add_argument(
        "--output-dir",
        default=r"audit/C11/stage2c34_factor_biological_interpretation_rank80",
    )
    p.add_argument(
        "--gmt",
        default=r"data/c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt",
    )
    p.add_argument("--top-genes", type=int, default=50)
    p.add_argument("--min-overlap", type=int, default=3)
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--heatmap-factors", type=int, default=20)
    p.add_argument("--umap-factors", type=int, default=4)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def sha256_bool_mask(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjusted p-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    if n == 0:
        return pvalues
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted


def load_gmt(path, background, min_size=10, max_size=500):
    pathways = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            name = fields[0]
            genes = {x.strip().upper() for x in fields[2:] if x.strip()}
            genes &= background
            if min_size <= len(genes) <= max_size:
                pathways[name] = genes
    if not pathways:
        raise RuntimeError(
            "No GMT pathways overlap the model-gene universe after size filtering."
        )
    return pathways


def unique_top_genes(scores, symbols, top_n):
    """Rank H loadings, retaining the first occurrence of each gene symbol."""
    order = np.argsort(np.asarray(scores))[::-1]
    selected = []
    seen = set()
    for j in order:
        gene = str(symbols[j]).strip().upper()
        if not gene or gene in {"NAN", "NONE", "NA"} or gene in seen:
            continue
        seen.add(gene)
        selected.append((int(j), gene, float(scores[j])))
        if len(selected) >= top_n:
            break
    return selected


def run_factor_enrichment(H, model_symbols, pathways, top_n, min_overlap):
    background = {
        str(g).strip().upper()
        for g in model_symbols
        if str(g).strip() and str(g).strip().upper() not in {"NAN", "NONE", "NA"}
    }
    universe_n = len(background)
    if universe_n < 100:
        raise RuntimeError(f"Unexpectedly small model-gene universe: {universe_n}")

    gene_rows = []
    enrichment_rows = []
    factor_best = {}

    for factor in range(H.shape[0]):
        selected = unique_top_genes(H[factor], model_symbols, top_n)
        top_set = {g for _, g, _ in selected}
        for rank, (local_idx, gene, loading) in enumerate(selected, start=1):
            gene_rows.append({
                "factor": factor,
                "factor_label": f"F{factor + 1:02d}",
                "rank_in_factor": rank,
                "gene_symbol": gene,
                "H_loading": loading,
            })

        tests_all = []
        for pathway_name, pathway_genes in pathways.items():
            overlap = top_set & pathway_genes
            k = len(overlap)
            K = len(pathway_genes)
            n = len(top_set)
            pvalue = float(hypergeom.sf(k - 1, universe_n, K, n)) if k else 1.0
            tests_all.append({
                "factor": factor,
                "factor_label": f"F{factor + 1:02d}",
                "pathway": pathway_name,
                "universe_genes": universe_n,
                "pathway_genes_in_universe": K,
                "top_genes_tested": n,
                "overlap_n": k,
                "overlap_ratio": k / max(K, 1),
                "pvalue": pvalue,
                "overlap_genes": ",".join(sorted(overlap)),
            })

        if tests_all:
            # Correct across the complete set of tested GMT pathways for this factor.
            qvalues = bh_adjust([x["pvalue"] for x in tests_all])
            for row, qvalue in zip(tests_all, qvalues):
                row["qvalue_BH_within_factor"] = float(qvalue)
                if row["overlap_n"] >= min_overlap:
                    enrichment_rows.append(row)
            best = min(tests_all, key=lambda x: x["qvalue_BH_within_factor"])
            factor_best[factor] = {
                "best_pathway": best["pathway"],
                "best_pvalue": best["pvalue"],
                "best_qvalue": float(best["qvalue_BH_within_factor"]),
            }
        else:
            factor_best[factor] = {
                "best_pathway": "",
                "best_pvalue": np.nan,
                "best_qvalue": 1.0,
            }

    return (
        pd.DataFrame(gene_rows),
        pd.DataFrame(enrichment_rows),
        factor_best,
        background,
    )


def compute_celltype_activity(W, labels, min_cells):
    labels = np.asarray(labels, dtype=str)
    counts = pd.Series(labels).value_counts()
    cell_types = counts[counts >= min_cells].index.tolist()
    if len(cell_types) < 2:
        raise RuntimeError(
            f"Only {len(cell_types)} cell types have >= {min_cells} cells; "
            "lower --min-cells if justified."
        )

    means = np.vstack([
        W[labels == cell_type].mean(axis=0)
        for cell_type in cell_types
    ])
    sd = means.std(axis=0, ddof=0)
    sd[sd < 1e-12] = 1.0
    zscores = (means - means.mean(axis=0, keepdims=True)) / sd[None, :]

    mean_df = pd.DataFrame(
        means,
        index=cell_types,
        columns=[f"F{i + 1:02d}" for i in range(W.shape[1])],
    )
    z_df = pd.DataFrame(
        zscores,
        index=cell_types,
        columns=[f"F{i + 1:02d}" for i in range(W.shape[1])],
    )
    return counts, cell_types, mean_df, z_df


def make_factor_summary(z_df, mean_df, factor_best):
    rows = []
    for factor in range(z_df.shape[1]):
        col = f"F{factor + 1:02d}"
        dominant_cell = str(z_df[col].idxmax())
        rows.append({
            "factor": factor,
            "factor_label": col,
            "dominant_cell_type_by_mean_Z": dominant_cell,
            "max_celltype_Z": float(z_df[col].max()),
            "dominant_celltype_mean_W": float(mean_df.loc[dominant_cell, col]),
            "best_KEGG_pathway": factor_best[factor]["best_pathway"],
            "best_KEGG_pvalue": factor_best[factor]["best_pvalue"],
            "best_KEGG_qvalue": factor_best[factor]["best_qvalue"],
            "has_KEGG_FDR_lt_0.05": bool(factor_best[factor]["best_qvalue"] < 0.05),
        })
    return pd.DataFrame(rows).sort_values(
        ["max_celltype_Z", "best_KEGG_qvalue"],
        ascending=[False, True],
    )


def set_publication_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_figure(fig, out_dir, stem):
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(
            path,
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        print("Saved:", path, flush=True)
    plt.close(fig)


def plot_celltype_heatmap(z_df, factor_summary, n_factors, out_dir):
    chosen = factor_summary.head(min(n_factors, len(factor_summary)))
    factor_cols = chosen["factor_label"].tolist()
    plot_data = z_df[factor_cols].T
    plot_data.index = factor_cols

    width = max(14, 0.34 * plot_data.shape[1])
    height = max(6.5, 0.34 * plot_data.shape[0])
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    sns.heatmap(
        plot_data,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Cell-type mean activity (Z-score)"},
    )
    ax.set_title(
        "(A) Cell-type-associated activity of representative PGNMF factors",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Cell type")
    ax.set_ylabel("PGNMF latent factor")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure51A_factor_celltype_activity_heatmap")


def plot_pathway_bubble(enrichment_df, out_dir):
    if enrichment_df.empty:
        print("No factor-pathway overlaps passed the minimum overlap; skipping pathway plot.")
        return

    plot_df = enrichment_df.sort_values(
        ["qvalue_BH_within_factor", "pvalue"], ascending=True
    ).head(25).copy()
    plot_df["pathway_short"] = (
        plot_df["pathway"]
        .str.replace("^KEGG_", "", regex=True)
        .str.replace("_", " ", regex=False)
    )
    plot_df["neglog10_q"] = -np.log10(
        np.clip(plot_df["qvalue_BH_within_factor"].astype(float), 1e-300, 1.0)
    )
    factors = list(dict.fromkeys(plot_df["factor_label"].tolist()))
    pathways_order = (
        plot_df.groupby("pathway_short")["neglog10_q"]
        .max()
        .sort_values()
        .index.tolist()
    )
    x_map = {name: i for i, name in enumerate(factors)}
    y_map = {name: i for i, name in enumerate(pathways_order)}

    fig_h = max(5.5, 0.34 * len(pathways_order) + 2.0)
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(factors) + 5), fig_h), dpi=180)
    sizes = 28 + 22 * plot_df["overlap_n"].to_numpy()
    sca = ax.scatter(
        [x_map[x] for x in plot_df["factor_label"]],
        [y_map[x] for x in plot_df["pathway_short"]],
        s=sizes,
        c=plot_df["neglog10_q"],
        cmap="viridis",
        edgecolors="white",
        linewidths=0.6,
    )
    ax.set_xticks(range(len(factors)), factors)
    ax.set_yticks(range(len(pathways_order)), pathways_order)
    ax.set_xlabel("PGNMF factor")
    ax.set_ylabel("KEGG pathway")
    ax.set_title(
        "(B) KEGG over-representation among factor-associated genes",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    cb = fig.colorbar(sca, ax=ax, pad=0.02)
    cb.set_label(r"$-\log_{10}$(BH-adjusted P)")
    fig.text(
        0.99, 0.01,
        "FDR is adjusted within each factor; q < 0.05 indicates significance.",
        ha="right", fontsize=7.5, color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure51B_factor_KEGG_enrichment")


def compute_and_plot_umap(W, factor_summary, n_factors, n_neighbors, random_state, out_dir):
    try:
        import umap
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "UMAP dependencies are unavailable. Install umap-learn and scikit-learn "
            "in the active environment, then rerun this plotting script."
        ) from exc

    n_cells = W.shape[0]
    nn = min(max(2, n_neighbors), n_cells - 1)
    W_scaled = StandardScaler().fit_transform(W)
    embedding = umap.UMAP(
        n_neighbors=nn,
        min_dist=0.3,
        n_components=2,
        metric="euclidean",
        random_state=random_state,
        transform_seed=random_state,
    ).fit_transform(W_scaled)

    chosen = factor_summary.head(min(n_factors, len(factor_summary)))
    factors = chosen["factor"].astype(int).tolist()
    ncols = min(4, len(factors))
    nrows = int(np.ceil(len(factors) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        dpi=180,
        squeeze=False,
    )

    for ax, factor in zip(axes.flat, factors):
        activity = np.asarray(W[:, factor], dtype=float)
        lo, hi = np.quantile(activity, [0.01, 0.99])
        if hi <= lo:
            lo, hi = float(activity.min()), float(activity.max() + 1e-8)
        sca = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=activity,
            s=2.2,
            alpha=0.78,
            cmap="viridis",
            vmin=lo,
            vmax=hi,
            linewidths=0,
            rasterized=True,
        )
        row = factor_summary.loc[factor_summary["factor"] == factor].iloc[0]
        title = f"{row['factor_label']} | {row['dominant_cell_type_by_mean_Z']}"
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.025)
        cb.set_label("W activity", fontsize=7)

    for ax in list(axes.flat)[len(factors):]:
        ax.axis("off")

    fig.suptitle(
        "(C) Representative factor activity on the PGNMF latent-space UMAP",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure51C_representative_factor_activity_UMAP")

    return embedding, factors


def main():
    args = parse_args()

    work_dir = ROOT / "work" / args.dataset
    masked_path = work_dir / "masked_logcp10k.h5ad"
    mask_path = (
        ROOT / "audit" / args.dataset
        / "stage2cpre_gene_set_diagnostic" / "gene_set_masks.npz"
    )
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    gmt_path = Path(args.gmt)
    if not gmt_path.is_absolute():
        gmt_path = ROOT / gmt_path
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (masked_path, mask_path, gmt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    npz_path = cache_dir / f"pgnmf_factors_seed{args.seed}_iter{args.max_iter}.npz"
    meta_path = cache_dir / f"pgnmf_factors_seed{args.seed}_iter{args.max_iter}.json"
    for path in (npz_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen factor cache file not found: {path}")

    import scanpy as sc
    from pgnmf_revision.pathway_sparsity import get_gene_symbols

    print("\n=== C11 RANK-80 FACTOR BIOLOGICAL INTERPRETATION ===", flush=True)
    print("No PGNMF fitting/refitting will be performed.", flush=True)
    print("Masked input:", masked_path, flush=True)
    print("Factor cache:", cache_dir, flush=True)
    print("Output:", output_dir, flush=True)

    masked = sc.read_h5ad(masked_path, backed="r")
    try:
        if args.label_key not in masked.obs.columns:
            raise KeyError(
                f"Label key {args.label_key!r} not found. "
                f"Available obs columns: {masked.obs.columns.tolist()}"
            )

        with np.load(mask_path, allow_pickle=False) as z:
            if args.gene_set_key not in z.files:
                raise KeyError(
                    f"{args.gene_set_key!r} missing from {mask_path}; available: {z.files}"
                )
            feature_mask = np.asarray(z[args.gene_set_key], dtype=bool)

        if feature_mask.shape != (masked.n_vars,):
            raise ValueError(
                f"Feature-mask shape {feature_mask.shape} != n_vars {masked.n_vars}"
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = meta.get("configuration", {})
        expected = {
            "dataset": args.dataset,
            "seed": args.seed,
            "rank": args.rank,
            "max_iter": args.max_iter,
            "gene_set_key": args.gene_set_key,
        }
        mismatches = {
            key: (cfg.get(key), value)
            for key, value in expected.items()
            if cfg.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Factor-cache configuration mismatch: {mismatches}")

        current_mask_hash = sha256_bool_mask(feature_mask)
        if cfg.get("gene_mask_sha256") != current_mask_hash:
            raise ValueError("Current gene-mask SHA256 does not match the frozen W/H cache.")
        if cfg.get("masked_file_size") != masked_path.stat().st_size:
            raise ValueError("Masked h5ad file size differs from the frozen cache metadata.")

        with np.load(npz_path, allow_pickle=False) as z:
            if not {"W", "H"}.issubset(set(z.files)):
                raise KeyError(f"Cache must contain W and H; found {z.files}")
            W = np.asarray(z["W"], dtype=np.float32)
            H = np.asarray(z["H"], dtype=np.float32)

        expected_w = (masked.n_obs, args.rank)
        expected_h = (args.rank, int(feature_mask.sum()))
        if W.shape != expected_w or H.shape != expected_h:
            raise ValueError(
                f"Unexpected factor shapes: W={W.shape} (expected {expected_w}); "
                f"H={H.shape} (expected {expected_h})"
            )
        if not np.isfinite(W).all() or not np.isfinite(H).all():
            raise ValueError("W or H contains non-finite values.")

        actual_iter = meta.get("n_iter", "not recorded")
        converged = meta.get("converged", "not recorded")
        print(
            f"Verified cache: W={W.shape}, H={H.shape}; "
            f"actual_iterations={actual_iter}; converged={converged}",
            flush=True,
        )

        all_symbols = np.asarray(
            get_gene_symbols(masked, args.gene_symbol_column)[0],
            dtype=str,
        )
        model_symbols = all_symbols[feature_mask]
        if len(model_symbols) != H.shape[1]:
            raise ValueError("Model gene-symbol order does not align with H columns.")

        labels = masked.obs[args.label_key].astype(str).to_numpy()
        if len(labels) != W.shape[0]:
            raise ValueError("Cell labels are not aligned with W rows.")

        background = {
            str(g).strip().upper()
            for g in model_symbols
            if str(g).strip() and str(g).strip().upper() not in {"NAN", "NONE", "NA"}
        }
        pathways = load_gmt(gmt_path, background, min_size=10, max_size=500)

        top_genes_df, enrichment_df, factor_best, background = run_factor_enrichment(
            H,
            model_symbols,
            pathways,
            top_n=args.top_genes,
            min_overlap=args.min_overlap,
        )
        top_genes_df.to_csv(
            output_dir / "factor_top_genes.csv",
            index=False,
            encoding="utf-8-sig",
        )
        enrichment_df.to_csv(
            output_dir / "factor_KEGG_enrichment_hypergeometric_FDR.csv",
            index=False,
            encoding="utf-8-sig",
        )

        counts, cell_types, mean_df, z_df = compute_celltype_activity(
            W, labels, min_cells=args.min_cells
        )
        mean_df.to_csv(
            output_dir / "factor_celltype_mean_activity.csv",
            encoding="utf-8-sig",
        )
        z_df.to_csv(
            output_dir / "factor_celltype_activity_Zscore.csv",
            encoding="utf-8-sig",
        )

        factor_summary = make_factor_summary(z_df, mean_df, factor_best)
        factor_summary.to_csv(
            output_dir / "factor_biological_interpretation_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        set_publication_style()
        plot_celltype_heatmap(
            z_df,
            factor_summary,
            n_factors=args.heatmap_factors,
            out_dir=output_dir,
        )
        plot_pathway_bubble(enrichment_df, output_dir)

        embedding, selected_factors = compute_and_plot_umap(
            W,
            factor_summary,
            n_factors=args.umap_factors,
            n_neighbors=args.n_neighbors,
            random_state=args.random_state,
            out_dir=output_dir,
        )
        umap_df = pd.DataFrame({
            "cell_id": masked.obs_names.astype(str),
            "cell_type": labels,
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
        })
        umap_df.to_csv(
            output_dir / "PGNMF_W_UMAP_coordinates.csv",
            index=False,
            encoding="utf-8-sig",
        )

        manifest = {
            "dataset": args.dataset,
            "label_key": args.label_key,
            "seed": args.seed,
            "rank": args.rank,
            "max_iter_setting": args.max_iter,
            "actual_iterations": actual_iter,
            "converged": converged,
            "n_cells": int(W.shape[0]),
            "n_model_genes": int(H.shape[1]),
            "gene_set_key": args.gene_set_key,
            "gene_mask_sha256": current_mask_hash,
            "model_gene_universe_unique_symbols": int(len(background)),
            "n_GMT_pathways_after_background_and_size_filter": int(len(pathways)),
            "min_cells_per_cell_type": args.min_cells,
            "cell_types_in_heatmap": cell_types,
            "heatmap_factors_selected_by": "highest maximum cell-type mean-activity Z-score across cell types with sufficient cell counts",
            "umap_factors_selected_by": "cell-type specificity ranking; FDR-significant pathway evidence used as a secondary ordering criterion",
            "representative_umap_factors": [int(x) for x in selected_factors],
            "pathway_test": "hypergeometric over-representation; BH correction performed separately within each factor",
            "cell_labels_used_only_for_posthoc_interpretation": True,
            "no_model_refit": True,
            "umap_input": "column-standardized frozen PGNMF W cell-factor matrix",
            "caution": (
                "Cell-type labels are CellTypist-inferred annotations. "
                "Factor-pathway enrichment is post-hoc interpretation of a model "
                "that already uses KEGG pathway priors; it is not an independent validation."
            ),
        }
        (output_dir / "stage2c34_factor_interpretation_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n=== COMPLETE ===", flush=True)
        print("Cell types retained:", len(cell_types), flush=True)
        print("Unique model-gene symbols:", len(background), flush=True)
        print("GMT pathways tested:", len(pathways), flush=True)
        print("Representative factors:", [f"F{x + 1:02d}" for x in selected_factors], flush=True)
        print("Output directory:", output_dir, flush=True)
        print("No PGNMF model was fitted or refitted.", flush=True)
    finally:
        if getattr(masked, "file", None) is not None:
            masked.file.close()


if __name__ == "__main__":
    main()
