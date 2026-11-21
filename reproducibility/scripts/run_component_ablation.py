#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PGNMF Stage 2C-22: reproducible component-level ablation.

Place in reproducibility/scripts/; run from project root, e.g.:
 python -u reproducibility/scripts/run_component_ablation.py --dataset C11 --seeds 42

Uses the maintained exact gene-block solver from src/pgnmf/solver.py, exactly the
model-gene universe / masked/reference/mask_plan / held-out split / kNN and
Leiden protocol of Stage 2C-8, and the frozen Stage 2C-19 resolution.

Variants each change only the explicitly listed coefficients:
 full                 = all frozen components
 no_zero              = beta=0 (all other terms retained)
 no_cell              = alpha_cell=0
 no_gene              = alpha_gene=0 (pathway-aware sparsity retained)
 uniform_sparsity     = omega_P=omega_N=1 (overall lambda_sparse fixed)
 plain_zeroaware      = alpha_cell=alpha_gene=0; lambda_sparse=0;
                        beta remains 1 and L2 stabilization remains enabled.

PGNMF outputs are DENSE_PRESERVE_OBS (positive masked-input entries unchanged).
No labels or held-out test errors are used for hyperparameter selection.
No source scripts or previous official benchmark files are changed.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))  # maintained public PGNMF package
sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))

from pgnmf_revision.masking import MaskPlan
from pgnmf_revision.pathway_sparsity import (
    get_gene_symbols, pathway_membership_from_gmt, make_two_level_penalty,
)
from pgnmf_revision.gene_graph_stage2b2 import build_pathway_rank_gene_graph
from pgnmf_revision.cell_graph_stage2b3 import (
    build_sparse_knn_cell_graph, knn_indices_from_expression,
    neighborhood_preservation,
)
from pgnmf.solver import zero_aware_nmf_mu_blocked

VARIANTS = (
    'full', 'no_zero', 'no_cell', 'no_gene',
    'uniform_sparsity', 'plain_zeroaware',
)
METRICS = ('test_rmse', 'ari', 'nmi', 'knn_overlap_vs_reference',
           'validation_rmse', 'n_clusters', 'xhat_zero_rms',
           'fit_seconds', 'n_iter')


def choose_gmt(explicit):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f'GMT does not exist: {p}')
        return p.resolve()
    preferred = ROOT / 'data' / 'c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt'
    if preferred.is_file():
        return preferred
    matches = sorted(p for p in (ROOT / 'data').rglob('*.gmt') if p.is_file())
    matches = [p for p in matches if 'kegg' in p.name.lower()] or matches
    if len(matches) != 1:
        raise ValueError('Expected exactly one KEGG GMT. Pass --gmt explicitly. Found: ' + str(matches))
    return matches[0]


def read_resolution(dataset, explicit):
    if explicit is not None:
        return float(explicit)
    f = ROOT / 'audit' / dataset / 'stage2c19_label_free_resolution' / 'stage2c19_selected_resolution.txt'
    if not f.is_file():
        raise FileNotFoundError(f'Frozen Leiden resolution not found: {f}; pass --resolution')
    return float(f.read_text(encoding='utf-8-sig').strip())


def dense32(x):
    return x.toarray().astype(np.float32, copy=False) if sparse.issparse(x) else np.asarray(x, dtype=np.float32)


def csv_ints(s):
    x = [int(t.strip()) for t in s.split(',') if t.strip()]
    if len(x) != len(set(x)) or not x:
        raise ValueError('Seeds must be unique and nonempty')
    return x


def split_indices(n, fraction, seed):
    if n < 2:
        raise ValueError('Need at least two held-out entries')
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    nv = min(max(1, int(round(n * fraction))), n - 1)
    return np.sort(order[:nv]), np.sort(order[nv:])


def load_gene_mask(f, key, n):
    with np.load(f, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f'{key!r} not in {f}; available={z.files}')
        m = np.asarray(z[key], dtype=bool)
    if m.shape != (n,):
        raise ValueError(f'Unexpected gene mask shape {m.shape}, expected {(n,)}')
    return m


def leiden_metrics(X, obs, key, resolution, seed, n_pcs, n_neighbors, metric):
    # Identical preprocessing and clustering calls to Stage 2C-8.
    tmp = ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs.copy())
    npc = min(int(n_pcs), tmp.n_vars-1, tmp.n_obs-1)
    if npc < 2:
        raise ValueError('Too few components')
    sc.tl.pca(tmp, n_comps=npc, random_state=seed)
    sc.pp.neighbors(tmp, n_neighbors=min(n_neighbors, max(2, tmp.n_obs-1)),
                    n_pcs=npc, use_rep='X_pca', metric=metric, random_state=seed)
    sc.tl.leiden(tmp, resolution=resolution, key_added='leiden', random_state=seed)
    truth = tmp.obs[key].astype(str).to_numpy()
    predicted = tmp.obs['leiden'].astype(str).to_numpy()
    values = {
        'ari': float(adjusted_rand_score(truth, predicted)),
        'nmi': float(normalized_mutual_info_score(truth, predicted)),
        'n_clusters': int(pd.Series(predicted).nunique()),
    }
    del tmp
    return values


def describe_variant(name, a, pathway_mask):
    beta = a.beta
    alpha_cell = a.alpha_cell
    alpha_gene = a.alpha_gene
    sparse_strength = a.sparsity_strength
    wp = a.pathway_weight
    wn = a.nonpathway_weight
    if name == 'no_zero':
        beta = 0.0
    elif name == 'no_cell':
        alpha_cell = 0.0
    elif name == 'no_gene':
        alpha_gene = 0.0
    elif name == 'uniform_sparsity':
        wp = wn = 1.0
    elif name == 'plain_zeroaware':
        alpha_cell = 0.0
        alpha_gene = 0.0
        sparse_strength = 0.0
    elif name != 'full':
        raise ValueError(f'Unknown variant {name!r}')
    # Ablations must allow equal gene penalties and a zero penalty vector.
    # The production helper deliberately enforces pathway_weight <
    # nonpathway_weight, which is correct for the full model but incompatible
    # with the uniform-sparsity ablation (wp == wn).
    if sparse_strength == 0.0:
        penalty = np.zeros(pathway_mask.shape, dtype=np.float32)
    elif wp == wn:
        # Each gene gets the same coefficient, while the overall L1
        # strength remains frozen.  Do not call make_two_level_penalty here.
        penalty = np.full(pathway_mask.shape,
                          np.float32(sparse_strength * wn), dtype=np.float32)
    else:
        penalty = make_two_level_penalty(
            pathway_mask, strength=sparse_strength,
            pathway_weight=wp, nonpathway_weight=wn,
        ).astype(np.float32, copy=False)
    return {
        'beta': float(beta), 'alpha_cell': float(alpha_cell),
        'alpha_gene': float(alpha_gene), 'sparsity_strength': float(sparse_strength),
        'pathway_weight': float(wp), 'nonpathway_weight': float(wn),
        'penalty': penalty,
    }


def parse_args():
    p = argparse.ArgumentParser(description='Stage 2C-22 PGNMF component-wise ablation')
    p.add_argument('--dataset', required=True, help='e.g. C11, SCIB, C02')
    p.add_argument('--seeds', default='42', help='e.g. 1,7,21,42,2026')
    p.add_argument('--variants', default=','.join(VARIANTS), help='comma-separated subset')
    p.add_argument('--label-key', default=None,
                   help='Default C01/C11=majority_voting, other datasets=celltype')
    p.add_argument('--resolution', type=float, default=None)
    p.add_argument('--gmt', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--gene-set-masks', default=None)
    p.add_argument('--gene-set-key', default='HVG1000+PATHWAY')
    p.add_argument('--gene-symbol-column', default=None)
    p.add_argument('--min-pathway-genes', type=int, default=10)
    p.add_argument('--max-pathway-genes', type=int, default=500)
    p.add_argument('--gene-edge-batch-size', type=int, default=256)
    p.add_argument('--rank', type=int, default=80)
    p.add_argument('--max-iter', type=int, default=100)
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--lambda-reg', type=float, default=1e-3)
    p.add_argument('--alpha-cell', type=float, default=10.0)
    p.add_argument('--alpha-gene', type=float, default=2.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--sparsity-strength', type=float, default=10.0)
    p.add_argument('--pathway-weight', type=float, default=0.5)
    p.add_argument('--nonpathway-weight', type=float, default=1.0)
    p.add_argument('--validation-seed', type=int, default=2026)
    p.add_argument('--validation-fraction', type=float, default=0.5)
    p.add_argument('--n-pcs', type=int, default=30)
    p.add_argument('--n-neighbors', type=int, default=15)
    p.add_argument('--metric', default='cosine')
    p.add_argument('--graph-pcs', type=int, default=30)
    p.add_argument('--graph-neighbors', type=int, default=15)
    p.add_argument('--graph-metric', default='euclidean')
    p.add_argument('--reuse-full', action='store_true',
                   help='Use Full PGNMF rows from existing Stage 2C-20 100-iter results (strict checks)')
    p.add_argument('--no-resume', action='store_true',
                   help='Do not resume saved seed-variant rows; output dir must be new')
    return p.parse_args()


def read_existing_full(a, dataset, seeds, resolution, label_key):
    # Only compatible with exactly the frozen final configuration.
    strict = (
        a.max_iter == 100 and abs(a.tol - 1e-4) < 1e-15 and
        a.rank == 80 and a.alpha_cell == 10 and a.alpha_gene == 2 and
        a.beta == 1 and a.lambda_reg == 0.001 and
        a.sparsity_strength == 10 and a.pathway_weight == 0.5 and
        a.nonpathway_weight == 1 and a.gene_set_key == 'HVG1000+PATHWAY' and
        a.validation_seed == 2026 and a.validation_fraction == 0.5 and
        a.n_pcs == 30 and a.n_neighbors == 15 and a.metric == 'cosine' and
        a.graph_pcs == 30 and a.graph_neighbors == 15 and a.graph_metric == 'euclidean'
    )
    if not strict:
        raise ValueError('--reuse-full requires the original frozen Stage2C20 config (100 iterations etc.)')
    expected_label = 'majority_voting' if dataset in ('C01','C11') else 'celltype'
    if label_key != expected_label:
        raise ValueError(f'--reuse-full expects original evaluation label {expected_label!r}, received {label_key!r}')
    f = ROOT / 'audit' / dataset / 'stage2c20_final_label_free_protocol' / 'stage2c8_graph_view_seed_results.csv'
    if not f.is_file():
        raise FileNotFoundError(f'Original Full PGNMF per-seed file missing: {f}')
    table = pd.read_csv(f)
    required = {'seed','view','alpha_cell','alpha_gene','beta','test_rmse','ari','nmi','knn_overlap_vs_reference','validation_rmse','n_clusters','xhat_zero_rms'}
    if missing := required.difference(table.columns):
        raise ValueError(f'Original full output missing columns: {sorted(missing)}')
    filt = ((table['view'].astype(str) == 'DENSE_PRESERVE_OBS') &
            np.isclose(pd.to_numeric(table['alpha_cell']), 10) &
            np.isclose(pd.to_numeric(table['alpha_gene']), 2) &
            np.isclose(pd.to_numeric(table['beta']), 1))
    table = table.loc[filt].copy()
    lookup = {}
    for seed in seeds:
        found = table.loc[pd.to_numeric(table.seed) == seed]
        if len(found) != 1:
            raise ValueError(f'Expected exactly one frozen Full row for {dataset} seed={seed}; got {len(found)}')
        data = found.iloc[0]
        lookup[seed] = {
            'dataset': dataset, 'seed': seed, 'variant': 'full',
            'view': 'DENSE_PRESERVE_OBS', 'resolution': resolution,
            'label_key': label_key,
            'test_rmse': float(data.test_rmse),
            'validation_rmse': float(data.validation_rmse),
            'ari': float(data.ari), 'nmi': float(data.nmi),
            'knn_overlap_vs_reference': float(data.knn_overlap_vs_reference),
            'n_clusters': int(data.n_clusters),
            'xhat_zero_rms': float(data.xhat_zero_rms),
            'source': str(f),
            'fit_seconds': float('nan'),
            'n_iter': 100, 'converged': False,
            'beta': 1.0, 'alpha_cell': 10.0, 'alpha_gene': 2.0,
            'sparsity_strength': 10.0,
            'pathway_weight': 0.5, 'nonpathway_weight': 1.0,
        }
    return lookup


def save_outputs(out, rows, seeds, variants):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(['variant','seed']).reset_index(drop=True)
    tmp = out / '_ablation_per_seed.tmp.csv'
    df.to_csv(tmp, index=False)
    tmp.replace(out / 'ablation_per_seed.csv')

    summary = []
    for v in variants:
        d = df[df.variant == v]
        if d.empty:
            continue
        r = {'variant': v, 'completed_seeds': len(d), 'planned_seeds': len(seeds)}
        for m in METRICS:
            if m not in d:
                continue
            x = pd.to_numeric(d[m], errors='coerce').dropna().to_numpy(dtype=float)
            r[m + '_mean'] = float(np.mean(x)) if len(x) else np.nan
            r[m + '_sd'] = float(np.std(x, ddof=0)) if len(x)>1 else np.nan
        summary.append(r)
    sd = pd.DataFrame(summary)
    sd.to_csv(out / 'ablation_summary_mean_sd.csv', index=False)
    print('\n=== ABLATION PROGRESS ===', flush=True)
    display = ['variant', 'completed_seeds', 'planned_seeds',
               'test_rmse_mean', 'ari_mean', 'nmi_mean', 'knn_overlap_vs_reference_mean']
    print(sd[[x for x in display if x in sd]].to_string(index=False), flush=True)


def main():
    a = parse_args()
    seeds = csv_ints(a.seeds)
    variants = [t.strip() for t in a.variants.split(',') if t.strip()]
    if not variants or len(variants) != len(set(variants)) or any(v not in VARIANTS for v in variants):
        raise ValueError('Invalid --variants; available: ' + ','.join(VARIANTS))
    dataset = a.dataset
    label_key = a.label_key or ('majority_voting' if dataset in ('C01','C11') else 'celltype')
    resolution = read_resolution(dataset, a.resolution)
    gmt = choose_gmt(a.gmt)
    w = ROOT / 'work' / dataset
    masked = w / 'masked_logcp10k.h5ad'
    reference = w / 'reference_logcp10k.h5ad'
    mask_plan_path = w / 'mask_plan.npz'
    mask_file = Path(a.gene_set_masks) if a.gene_set_masks else ROOT / 'audit' / dataset / 'stage2cpre_gene_set_diagnostic' / 'gene_set_masks.npz'
    for p in (masked, reference, mask_plan_path, mask_file):
        if not p.is_file():
            raise FileNotFoundError(f'Required input missing: {p}')
    out = Path(a.output_dir) if a.output_dir else ROOT / 'audit' / dataset / 'stage2c22_component_ablation'
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    official = (ROOT / 'audit' / dataset / 'stage2c20_final_label_free_protocol').resolve()
    if out == official or official in out.parents:
        raise ValueError('Refusing to write into original official Stage2C20 result directory')
    out.mkdir(parents=True, exist_ok=True)

    config = {
        'stage': '2C-22', 'dataset': dataset, 'label_key': label_key,
        'resolution': resolution, 'seeds': seeds, 'variants': variants,
        'gmt': str(gmt.resolve()), 'masked': str(masked.resolve()),
        'reference': str(reference.resolve()), 'mask_plan': str(mask_plan_path.resolve()),
        'gene_mask': str(mask_file.resolve()),
        'gene_set_key': a.gene_set_key,
        'min_pathway_genes': a.min_pathway_genes,
        'max_pathway_genes': a.max_pathway_genes,
        'gene_edge_batch_size': a.gene_edge_batch_size,
        'rank': a.rank, 'max_iter': a.max_iter, 'tol': a.tol,
        'lambda_reg': a.lambda_reg, 'alpha_cell': a.alpha_cell,
        'alpha_gene': a.alpha_gene, 'beta': a.beta,
        'sparsity_strength': a.sparsity_strength, 'pathway_weight': a.pathway_weight,
        'nonpathway_weight': a.nonpathway_weight,
        'validation_seed': a.validation_seed,
        'validation_fraction': a.validation_fraction,
        'n_pcs': a.n_pcs, 'n_neighbors': a.n_neighbors, 'metric': a.metric,
        'graph_pcs': a.graph_pcs, 'graph_neighbors': a.graph_neighbors,
        'graph_metric': a.graph_metric, 'reuse_full': bool(a.reuse_full),
        'definitions': {
            'full': 'Frozen full model',
            'no_zero': 'beta=0; retain all other terms',
            'no_cell': 'alpha_cell=0; retain other terms',
            'no_gene': 'alpha_gene=0; pathway sparsity retained',
            'uniform_sparsity': 'omega_P=omega_N=1; lambda_sparse fixed',
            'plain_zeroaware': 'alpha_cell=alpha_gene=lambda_sparse=0; beta=1 and L2 retained',
        },
    }
    manifest = out / 'ablation_manifest.json'
    if manifest.is_file():
        old = json.loads(manifest.read_text(encoding='utf-8'))
        if old != config:
            raise ValueError('Output directory has a different configuration; choose a new --output-dir')
        if a.no_resume:
            raise ValueError('--no-resume requires an empty output directory')
    else:
        if (out / 'ablation_per_seed.csv').exists():
            raise ValueError('Found result CSV without manifest; choose a new output directory')
        manifest.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')

    existing = {}
    if (out / 'ablation_per_seed.csv').is_file():
        for r in pd.read_csv(out / 'ablation_per_seed.csv').to_dict(orient='records'):
            existing[(int(r['seed']), str(r['variant']))] = r
    if a.reuse_full and 'full' in variants:
        reused = read_existing_full(a, dataset, seeds, resolution, label_key)
        for seed, row in reused.items():
            existing[(seed,'full')] = row
        save_outputs(out, list(existing.values()), seeds, variants)

    todo = [(seed, v) for seed in seeds for v in variants if (seed,v) not in existing]
    print('\n=== PGNMF COMPONENT ABLATION ===', flush=True)
    print(f'dataset={dataset} label={label_key} resolution={resolution:g} seeds={seeds}', flush=True)
    print(f'variants={variants}  remaining={len(todo)}  max_iter={a.max_iter}', flush=True)
    print(f'output={out}', flush=True)
    if not todo:
        print('Everything completed; no computation needed.', flush=True)
        return

    ad_masked = sc.read_h5ad(masked)
    ad_ref = sc.read_h5ad(reference)
    plan = MaskPlan.load(mask_plan_path)
    if ad_masked.shape != ad_ref.shape or tuple(plan.shape) != tuple(ad_masked.shape):
        raise ValueError('Shape mismatch between masked/reference/mask_plan')
    if not np.array_equal(ad_masked.var_names.astype(str), ad_ref.var_names.astype(str)):
        raise ValueError('Gene order mismatch')
    if label_key not in ad_masked.obs.columns:
        raise KeyError(f'Label key {label_key!r} absent; available={list(ad_masked.obs.columns)}')
    model_mask = load_gene_mask(mask_file, a.gene_set_key, ad_masked.n_vars)
    model_idx = np.flatnonzero(model_mask)
    X = dense32(ad_masked[:, model_mask].X)
    X_ref = dense32(ad_ref[:, model_mask].X)
    gene_symbols, source = get_gene_symbols(ad_masked, a.gene_symbol_column)
    pathway_membership, _ = pathway_membership_from_gmt(
        str(gmt), gene_symbols, min_pathway_genes=a.min_pathway_genes,
        max_pathway_genes=a.max_pathway_genes, gene_symbol_source=source)
    pathway_model = np.asarray(pathway_membership[model_mask], dtype=bool)
    to_local = np.full(ad_masked.n_vars, -1, dtype=np.int64)
    to_local[model_idx] = np.arange(len(model_idx), dtype=np.int64)
    keep = model_mask[plan.cols]
    rows = np.asarray(plan.rows[keep], dtype=np.int64)
    cols = to_local[np.asarray(plan.cols[keep], dtype=np.int64)]
    truth = np.asarray(plan.reference_values[keep], dtype=np.float64)
    val_idx, test_idx = split_indices(len(truth), a.validation_fraction, a.validation_seed)
    zero_mask = X <= 0
    n_pos = int(np.count_nonzero(~zero_mask))
    n_zero = int(np.count_nonzero(zero_mask))
    if n_zero <= 0:
        raise ValueError('No candidate zeros')
    balance_ratio = n_pos / n_zero
    print(f'model matrix={X.shape}; held-out={len(truth)} test={len(test_idx)} '
          f'positive={n_pos} zero={n_zero} balanced q0={a.beta*balance_ratio:.8g}',flush=True)

    need_graph = any(v in variants for v in ('full','no_zero','no_cell','no_gene','uniform_sparsity'))
    # Build frozen gene graph ONCE for all seeds/variants; no reselecting edges.
    if need_graph:
        print('Building frozen pathway-rank gene graph ONCE ...', flush=True)
        t0 = time.perf_counter()
        Sg_raw, dg_raw, _, _ = build_pathway_rank_gene_graph(
            X, gene_symbols, gene_symbols[model_mask], str(gmt),
            min_pathway_genes=a.min_pathway_genes,
            max_pathway_genes=a.max_pathway_genes,
            edge_batch_size=a.gene_edge_batch_size)
        pos_deg = np.asarray(dg_raw)[np.asarray(dg_raw)>0]
        if len(pos_deg) == 0:
            raise RuntimeError('Gene graph has no positive degree')
        scale = float(np.median(pos_deg))
        Sg = (Sg_raw / np.float32(scale)).tocsr()
        dg = np.asarray(dg_raw, dtype=np.float32) / np.float32(scale)
        print(f'Gene graph built in {time.perf_counter()-t0:.1f}s; scale={scale:.6g}', flush=True)
        del Sg_raw, dg_raw
    else:
        Sg = sparse.csr_matrix((X.shape[1], X.shape[1]), dtype=np.float32)
        dg = np.zeros(X.shape[1], dtype=np.float32)

    for seed in seeds:
        pending = [v for v in variants if (seed,v) not in existing]
        if not pending:
            print(f'Already completed all variants for seed={seed}; skipping.', flush=True)
            continue
        print(f'\n=== Seed {seed} ({len(pending)} remaining variants) ===', flush=True)
        reference_neighbors = knn_indices_from_expression(
            X_ref, n_pcs=a.n_pcs, n_neighbors=a.n_neighbors,
            metric=a.metric, random_state=seed)
        if any(v != 'plain_zeroaware' for v in pending):
            Sc, dc, _, _, _ = build_sparse_knn_cell_graph(
                X, n_pcs=a.graph_pcs, n_neighbors=a.graph_neighbors,
                metric=a.graph_metric, random_state=seed)
        else:
            Sc = sparse.csr_matrix((X.shape[0], X.shape[0]), dtype=np.float32)
            dc = np.zeros(X.shape[0], dtype=np.float32)
        for variant in pending:
            cfg = describe_variant(variant, a, pathway_model)
            q0 = cfg['beta'] * balance_ratio
            if not 0 <= q0 <= 1:
                raise ValueError(f'{variant}: invalid balanced q0={q0}')
            print(f'\n--- {variant} seed={seed} ac={cfg["alpha_cell"]} '
                  f'ag={cfg["alpha_gene"]} beta={cfg["beta"]} '
                  f'sparsity={cfg["sparsity_strength"]} ---', flush=True)
            fit = zero_aware_nmf_mu_blocked(
                X, S_cell=Sc, degree_cell=dc, alpha_cell=cfg['alpha_cell'],
                S_gene=Sg, degree_gene=dg, alpha_gene=cfg['alpha_gene'],
                lambda_h_gene=cfg['penalty'], zero_weight=q0,
                rank=a.rank, max_iter=a.max_iter, tol=a.tol, random_state=seed,
                lambda_reg=a.lambda_reg, lambda_w_l1=0.0,
                verbose=False, tag=f'Ablation-{dataset}-{variant}-{seed}',
            block_size=500,
            )
            Xhat = (fit.W @ fit.H).astype(np.float32, copy=False)
            Xdense = X.copy()
            Xdense[zero_mask] = Xhat[zero_mask]
            preds = np.asarray(Xdense[rows, cols], dtype=np.float64)
            vrmse = float(np.sqrt(np.mean((preds[val_idx]-truth[val_idx])**2)))
            trmse = float(np.sqrt(np.mean((preds[test_idx]-truth[test_idx])**2)))
            neighbor = knn_indices_from_expression(
                Xdense, n_pcs=a.n_pcs, n_neighbors=a.n_neighbors,
                metric=a.metric, random_state=seed)
            overlap = float(neighborhood_preservation(reference_neighbors, neighbor)['mean_knn_overlap'])
            cm = leiden_metrics(
                Xdense, ad_masked.obs, label_key, resolution, seed,
                a.n_pcs, a.n_neighbors, a.metric)
            z = Xhat[zero_mask]
            row = {
                'dataset': dataset, 'seed': int(seed), 'variant': variant,
                'view': 'DENSE_PRESERVE_OBS', 'label_key': label_key,
                'resolution': float(resolution), 'test_rmse': trmse,
                'validation_rmse': vrmse, 'ari': cm['ari'], 'nmi': cm['nmi'],
                'n_clusters': cm['n_clusters'], 'knn_overlap_vs_reference': overlap,
                'xhat_zero_rms': float(np.sqrt(np.mean(np.square(z, dtype=np.float64)))),
                'fit_seconds': float(fit.elapsed_seconds),
                'n_iter': int(fit.n_iter), 'converged': bool(fit.converged),
                'beta': float(cfg['beta']), 'alpha_cell': float(cfg['alpha_cell']),
                'alpha_gene': float(cfg['alpha_gene']),
                'sparsity_strength': float(cfg['sparsity_strength']),
                'pathway_weight': float(cfg['pathway_weight']),
                'nonpathway_weight': float(cfg['nonpathway_weight']),
                'source': 'new_stage2c22_fitting',
            }
            existing[(seed,variant)] = row
            print(f"RESULT {variant}: RMSE={trmse:.6f} ARI={cm['ari']:.5f} "
                  f"NMI={cm['nmi']:.5f} kNN={overlap:.5f} "
                  f"n_iter={fit.n_iter} converged={fit.converged} "
                  f"fit={fit.elapsed_seconds:.1f}s", flush=True)
            save_outputs(out, list(existing.values()), seeds, variants)
            del Xhat, Xdense, preds, neighbor, fit, z
            gc.collect()
        del reference_neighbors, Sc, dc
        gc.collect()

    print('\n=== COMPLETE ===', flush=True)
    print('Per-seed:', out / 'ablation_per_seed.csv', flush=True)
    print('Summary :', out / 'ablation_summary_mean_sd.csv', flush=True)
    print('Manifest:', out / 'ablation_manifest.json', flush=True)


if __name__ == '__main__':
    main()
