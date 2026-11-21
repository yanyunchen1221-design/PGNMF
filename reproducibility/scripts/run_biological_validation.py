#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PGNMF Stage2C32: biological validation of marker recovery/leakage and KEGG coexpression.

Place under PGNMF_revision_stage1/scripts, run from the project root.
Uses the Stage2C27 FULL W/H cache fitted with held-out positions excluded from the zero-aware penalty; validates the exact split and model configuration.
Other variants are optionally fitted from the same frozen solver and cached.
Never uses cell labels, held-out truths or coexpression scores to tune PGNMF.

Suggested:
  python -u reproducibility/scripts/run_biological_validation.py --dataset C11 --mode inspect
  python -u reproducibility/scripts/run_biological_validation.py --dataset C11 --mode plot
  python -u reproducibility/scripts/run_biological_validation.py --dataset C11 --mode plot --fit-ablations no_zero,no_gene

Interpretation: 'reference zeros' in scRNA-seq are *observed sequencing zeros*,
NOT verified biological zero expression. Non-target expression is not necessarily
false positive. Report results transparently; do not select favorable pathways.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import patheffects

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))  # maintained public PGNMF package
sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))

# Defined from cell biology, NOT selected from PGNMF performance.
MARKER_TARGETS = {
    'CD3D': ('Tcm/Naive helper T cells', 'Tcm/Naive cytotoxic T cells', 'Tem/Temra cytotoxic T cells'),
    'CD3E': ('Tcm/Naive helper T cells', 'Tcm/Naive cytotoxic T cells', 'Tem/Temra cytotoxic T cells'),
    'MS4A1': ('Naive B cells', 'Age-associated B cells'),
    'CD79A': ('Naive B cells', 'Age-associated B cells'),
    'CD79B': ('Naive B cells', 'Age-associated B cells'),
    'GNLY': ('CD16+ NK cells', 'Tem/Temra cytotoxic T cells'),
    'CD14': ('Classical monocytes',),
    'PPBP': ('Megakaryocytes/platelets',),
    'PF4': ('Megakaryocytes/platelets',),
}
DEFAULT_PATHWAYS = 'T_CELL_RECEPTOR_SIGNALING;B_CELL_RECEPTOR_SIGNALING'
VIEW_COLORS = {
    'Masked input': '#9aa4ad', 'PGNMF': '#1e8199', 'PGNMF-mask-only': '#c28a3a', 'PGNMF-heldout': '#4b956c',
    'w/o Zero-aware': '#d97a45', 'w/o Gene graph': '#995ea7',
}


def parse_args():
    p = argparse.ArgumentParser(description='C11 marker recovery/leakage + KEGG coexpression analysis')
    p.add_argument('--dataset', default='C11')
    p.add_argument('--label-key', default=None)
    p.add_argument('--mode', choices=['inspect', 'plot'], default='inspect')
    p.add_argument('--cache-dir', default=None, help='Stage2C27 held-out-excluded factor cache directory')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--gmt', default=None)
    p.add_argument('--gene-set-key', default='HVG1000+PATHWAY')
    p.add_argument('--gene-symbol-column', default=None)
    p.add_argument('--pathways', default=DEFAULT_PATHWAYS, help='Semicolon-delimited KEGG name fragments, fixed before results')
    p.add_argument('--max-pathway-genes', type=int, default=24)
    p.add_argument('--min-pathway-genes', type=int, default=8)
    p.add_argument('--min-marker-heldout', type=int, default=25)
    p.add_argument('--expression-threshold', type=float, default=1.0,
                   help='log-CP10K threshold to quantify expression at REFERENCE zeros')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rank', type=int, default=80)
    p.add_argument('--max-iter', type=int, default=100)
    p.add_argument('--validation-seed', type=int, default=2026)
    p.add_argument('--validation-fraction', type=float, default=0.5)
    p.add_argument('--fit-ablations', default='',
                   help='Optional: no_zero,no_gene. Fits once and caches W/H. Empty = no new training')
    p.add_argument('--force-refit', action='store_true')
    return p.parse_args()


def fullpath(p, default):
    v = Path(p) if p else default
    return (ROOT / v).resolve() if not v.is_absolute() else v.resolve()


def dense32(a):
    return a.toarray().astype(np.float32, copy=False) if sparse.issparse(a) else np.asarray(a, dtype=np.float32)


def file_hash(arr):
    return hashlib.sha256(np.asarray(arr, dtype=np.uint8).tobytes()).hexdigest()


def load_mask(p, key, n):
    with np.load(p, allow_pickle=False) as z:
        if key not in z:
            raise KeyError(f'{key!r} unavailable in {p}; choices: {z.files}')
        m = np.asarray(z[key], dtype=bool)
    if m.shape != (n,):
        raise ValueError(f'Gene mask incorrect shape: {m.shape} vs {(n,)}')
    return m


def choose_gmt(p):
    if p:
        x = fullpath(p, ROOT / 'data' / 'unused.gmt')
    else:
        x = ROOT / 'data' / 'c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt'
        if not x.is_file():
            found = [f for f in (ROOT/'data').rglob('*.gmt') if 'kegg' in f.name.lower()]
            if len(found) != 1:
                raise FileNotFoundError(f'KEGG GMT ambiguous; use --gmt. Candidates: {found}')
            x = found[0]
    if not x.is_file():
        raise FileNotFoundError(f'KEGG GMT missing: {x}')
    return x.resolve()


def read_gmt(path):
    entries = {}
    with path.open('r', encoding='utf-8-sig') as f:
        for line in f:
            fields = line.rstrip('\n\r').split('\t')
            if len(fields) < 3: continue
            name = fields[0].strip()
            genes = tuple(dict.fromkeys(g.strip().upper() for g in fields[2:] if g.strip()))
            entries[name] = genes
    return entries


def choose_pathways(gmt, specified, usable_symbols, min_genes):
    entries = read_gmt(gmt)
    selected = []
    for term in [x.strip() for x in specified.split(';') if x.strip()]:
        matched = [(name, genes) for name, genes in entries.items() if term.upper().replace(' ', '_') in name.upper().replace(' ', '_')]
        if len(matched) > 1:
            raise ValueError(f'Pathway name {term!r} is ambiguous: {[x[0] for x in matched]}; specify a more exact fragment')
        if not matched:
            print(f'WARNING: pathway not found in GMT: {term}; available T/B-related examples: '
                  f'{[x for x in entries if "RECEPTOR" in x.upper()][:16]}', flush=True)
            continue
        name, genes = matched[0]
        # Independent, predeclared pathway panels may share genes.
        overlap = [g for g in genes if g in usable_symbols]
        if len(overlap) < min_genes:
            print(f'WARNING: pathway {name} has only {len(overlap)} model genes (<{min_genes}); skipped', flush=True)
            continue
        selected.append((name, overlap))
    return selected


def valid_full_factor(
    cache_dir, dataset, gene_mask, seed, rank, iters, expected_n_cells, masked_file,
    validation_seed, validation_fraction, expected_heldout_hash, expected_n_heldout
):
    npz = cache_dir / f'pgnmf_factors_seed{seed}_iter{iters}.npz'
    meta = cache_dir / f'pgnmf_factors_seed{seed}_iter{iters}.json'
    if not npz.is_file() or not meta.is_file():
        raise FileNotFoundError(
            f'PGNMF factor cache is missing: {npz}.\nFirst run the held-out-exclusion version of run_marker_preservation_analysis.py '
            f'--dataset {dataset} --mode plot --seed {seed} --rank {rank} --max-iter {iters} '
            f'in its original output directory, or specify --cache-dir.')
    data = json.loads(meta.read_text(encoding='utf-8'))
    config = data.get('configuration', {})
    required = {'dataset': dataset, 'seed': seed, 'rank': rank, 'max_iter': iters,
                'gene_set_key': 'HVG1000+PATHWAY', 'alpha_cell': 10.0, 'alpha_gene': 2.0,
                'beta': 1.0, 'sparsity_strength': 10.0,
                'pathway_weight': 0.5, 'nonpathway_weight': 1.0,
                'zero_penalty_exclusion': True,
                'heldout_protocol_version': 'zero-penalty-exclusion-v1',
                'validation_seed': int(validation_seed),
                'validation_fraction': float(validation_fraction),
                'heldout_split_sha256': expected_heldout_hash,
                'n_heldout_excluded': int(expected_n_heldout)}
    wrong = {k: (config.get(k), val) for k, val in required.items() if config.get(k) != val}
    if wrong:
        raise ValueError(f'Full factor cache is not the frozen Full PGNMF: {wrong}')
    if config.get('gene_mask_sha256') != file_hash(gene_mask):
        raise ValueError('Full factor cache gene selection hash differs from current gene_set_masks.npz')
    if config.get('masked_file_size') != masked_file.stat().st_size:
        raise ValueError('Masked h5ad file size differs from when W/H were fitted; refusing unsafe cache reuse')
    if config.get('masked_file_mtime_ns') != masked_file.stat().st_mtime_ns:
        raise ValueError('Masked h5ad modification time differs from when W/H were fitted; refusing unsafe cache reuse')
    with np.load(npz, allow_pickle=False) as z:
        W = np.asarray(z['W'], dtype=np.float32)
        H = np.asarray(z['H'], dtype=np.float32)
    if W.shape != (expected_n_cells, rank) or H.shape != (rank, int(gene_mask.sum())):
        raise ValueError(f'Cached W/H dimensions mismatch: {W.shape}, {H.shape}')
    return (W, H), data


def ref_zero_prediction(mask_col, ref_col, WH_col, off_target, threshold):
    # IMPORTANT: NOT a biological false-positive rate. Ref zeros are sequencing zeros.
    positions = off_target & (ref_col == 0)
    vals = np.where(mask_col[positions] > 0, mask_col[positions], WH_col[positions])
    return {
        'n_reference_zero_non_target': int(positions.sum()),
        'reference_zero_above_threshold_fraction': float(np.mean(vals > threshold)) if len(vals) else np.nan,
        'reference_zero_mean_logcp10k': float(np.mean(vals)) if len(vals) else np.nan,
    }


def selected_mask_positions(mask_plan, gene_mask):
    keep = gene_mask[np.asarray(mask_plan.cols, dtype=np.int64)]
    rows = np.asarray(mask_plan.rows[keep], dtype=np.int64)
    cols = np.asarray(mask_plan.cols[keep], dtype=np.int64)
    truth = np.asarray(mask_plan.reference_values[keep], dtype=np.float64)
    if len(truth) < 2:
        raise RuntimeError('Too few masked positions in selected gene universe')
    return rows, cols, truth


def heldout_splits(mask_plan, gene_mask, fraction, seed):
    rows, cols, truth = selected_mask_positions(mask_plan, gene_mask)
    idx = np.random.default_rng(seed).permutation(len(truth))
    nv = min(max(1, int(round(len(truth)*fraction))), len(truth)-1)
    return rows[idx[nv:]], cols[idx[nv:]], truth[idx[nv:]]


def heldout_position_hash(rows, cols):
    """Order-independent SHA256 of held-out coordinates; expression values are not hashed."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    if rows.size != cols.size:
        raise ValueError('Held-out row/column lengths differ')
    order = np.lexsort((cols, rows))
    coordinates = np.column_stack((rows[order], cols[order])).astype(np.int64, copy=False)
    return hashlib.sha256(coordinates.tobytes()).hexdigest()


def compute_marker_metrics(marker_gene_index, targets, labels, ref, masked, view_preds, test_triplet, g_to_l, threshold):
    test_rows, test_cols, test_ref = test_triplet
    records=[]
    for gene, glob in marker_gene_index.items():
        if gene not in targets: continue
        local = int(g_to_l[glob])
        if local < 0: continue
        true_vec = dense32(ref[:, glob].X).reshape(-1)
        mask_vec = dense32(masked[:, glob].X).reshape(-1)
        is_target = np.isin(labels, list(targets[gene]))
        if not is_target.any() or np.all(is_target):
            print(f'WARNING: skipping {gene}: no matching target or non-target labels',flush=True)
            continue
        test_idx = (test_cols == glob)
        t = test_ref[test_idx]
        rr = test_rows[test_idx]
        if not len(t):
            continue
        for name, factors in [('Masked input', None), *view_preds.items()]:
            if factors is None:
                pred = np.zeros(len(t), dtype=np.float64)  # forcibly held-out positive positions
                whole = mask_vec
            else:
                W,H=factors
                pred = W[rr].astype(np.float64) @ H[:, local].astype(np.float64)
                full_pred = W @ H[:, local]
                whole = np.where(mask_vec > 0, mask_vec, full_pred)
            rzero = (true_vec == 0) & (~is_target)
            nonzero_ref = (~rzero) & (~is_target)
            records.append({
                'marker_gene':gene,'method':name,'n_heldout':int(len(t)),
                'heldout_rmse':float(np.sqrt(np.mean((pred-t)**2))),
                'heldout_mae':float(np.mean(np.abs(pred-t))),
                'n_target_cells':int(is_target.sum()), 'n_non_target_cells':int((~is_target).sum()),
                'n_reference_zero_non_target':int(rzero.sum()),
                'non_target_refzero_gt_threshold':float(np.mean(whole[rzero]>threshold)) if rzero.any() else np.nan,
                'non_target_refzero_mean_expr':float(np.mean(whole[rzero])) if rzero.any() else np.nan,
                'target_mean_expr':float(np.mean(whole[is_target])),
                'non_target_mean_expr':float(np.mean(whole[~is_target])),
                'non_target_reference_positive_count':int(nonzero_ref.sum()),
            })
    return pd.DataFrame(records)


def expr_for_genes(ad, globcols):
    return dense32(ad[:, globcols].X)



def predicted_expression(masked_cols, factors, local_cols, cell_indices=None):
    # Existing PGNMF-full view: preserve observed nonzero values and fill every
    # zero in the selected matrix. This includes both artificial mask positions
    # and original sequencing zeros, so interpret it as the full-zero-fill view.
    W, H = factors
    if cell_indices is not None:
        W = W[cell_indices]
    pred = W @ H[:, local_cols]
    return np.where(masked_cols > 0, masked_cols, pred).astype(np.float32, copy=False)


def predicted_expression_at_positions(masked_cols, factors, local_cols, global_cols,
                                      cell_indices, positions_triplet, position_name):
    """Fill ONLY the supplied artificial mask positions within this subset.

    All other entries, including original sequencing zeros and artificial
    masked positives outside positions_triplet, remain unchanged. Reference
    values in positions_triplet are never used to construct predictions.
    """
    W, H = factors
    pos_rows, pos_cols, pos_truth = positions_triplet

    pos_rows = np.asarray(pos_rows, dtype=np.int64)
    pos_cols = np.asarray(pos_cols, dtype=np.int64)
    pos_truth = np.asarray(pos_truth)

    if not (len(pos_rows) == len(pos_cols) == len(pos_truth)):
        raise ValueError(f"{position_name} rows, columns and truth arrays have different lengths")
    if W.shape[0] <= (int(pos_rows.max()) if len(pos_rows) else -1):
        raise ValueError(f"{position_name} row index exceeds PGNMF W dimensions")

    # cell_indices are sorted (arange for all cells, flatnonzero for subgroups).
    # Map global cell row indices to local rows in masked_cols without reordering.
    local_rows = np.searchsorted(cell_indices, pos_rows)
    in_scope = local_rows < len(cell_indices)
    valid = np.zeros(len(pos_rows), dtype=bool)
    valid[in_scope] = cell_indices[local_rows[in_scope]] == pos_rows[in_scope]

    out = np.asarray(masked_cols, dtype=np.float32).copy()
    n_filled = 0

    for j, (global_col, local_col) in enumerate(zip(global_cols, local_cols)):
        selected = valid & (pos_cols == int(global_col))
        if not np.any(selected):
            continue

        rows = pos_rows[selected]
        rows_local = local_rows[selected]

        # The selected artificial mask positions must be zero in masked input.
        if np.any(out[rows_local, j] != 0):
            raise ValueError(
                f"{position_name} positions are not zero in masked input for global gene column {global_col}"
            )

        prediction = (
            W[rows].astype(np.float64, copy=False)
            @ H[:, int(local_col)].astype(np.float64, copy=False)
        )
        out[rows_local, j] = prediction.astype(np.float32, copy=False)
        n_filled += int(len(rows))

    return out, n_filled


def corr_mat(arr):
    # Ref, mask, and imputed share the same cells and gene columns.
    c = np.corrcoef(np.asarray(arr, dtype=np.float64), rowvar=False)
    c = np.asarray(c, dtype=float)
    c[~np.isfinite(c)] = 0.0
    np.fill_diagonal(c, 1.0)
    return c


def corr_dist(a, b):
    ii=np.triu_indices_from(a,k=1)
    return float(np.mean(np.abs(a[ii]-b[ii]))),float(np.sqrt(np.mean((a[ii]-b[ii])**2)))



def compute_pathways(entries, symbol_to_global, g_to_l, masked, ref, views,
                     labels, mask_positions_triplet, test_triplet, max_genes, min_genes):
    labels = np.asarray(labels, dtype=str)
    n_obs = int(masked.n_obs)

    if labels.shape[0] != n_obs:
        raise ValueError(
            f'Label count ({labels.shape[0]}) does not match masked cells ({n_obs})'
        )
    if int(ref.n_obs) != n_obs:
        raise ValueError('Masked and Reference cell counts differ')

    # NK cells are not included in the T-cell group.
    # Plasma cells and plasmablasts are excluded from conventional B cells.
    normalized = [x.strip().casefold() for x in labels]

    t_mask = np.asarray([
        ('t cell' in x) or ('mait' in x) or (x in {'cd8a/a', 'cd8 a/a'})
        for x in normalized
    ], dtype=bool)

    b_mask = np.asarray([
        (('b cell' in x) and ('plasma' not in x) and ('plasmablast' not in x))
        for x in normalized
    ], dtype=bool)

    scopes = [('All cells', np.arange(n_obs, dtype=np.int64))]
    if int(t_mask.sum()) >= 30:
        scopes.append(('T cells', np.flatnonzero(t_mask)))
    else:
        print(f'WARNING: T-cell group has only {int(t_mask.sum())} cells; skipped.', flush=True)

    if int(b_mask.sum()) >= 30:
        scopes.append(('Conventional B cells', np.flatnonzero(b_mask)))
    else:
        print(f'WARNING: conventional B-cell group has only {int(b_mask.sum())} cells; skipped.', flush=True)

    print('\nPathway analysis cell groups:', flush=True)
    for scope_name, cell_indices in scopes:
        print(f'  {scope_name}: {len(cell_indices)} cells', flush=True)

    # All factor matrices must preserve the same full cell order as the input.
    for view_name, factors in views.items():
        W, H = factors
        if W.shape[0] != n_obs:
            raise ValueError(
                f'{view_name} W has {W.shape[0]} rows, but masked input has {n_obs} cells'
            )

    out = []
    for scope_name, cell_indices in scopes:
        for pathway_name, pool in entries:
            avail = [
                g for g in pool
                if g in symbol_to_global and g_to_l[symbol_to_global[g]] >= 0
            ]
            if not avail:
                print(f'WARNING {scope_name} / {pathway_name}: no model genes available; skipped.',
                      flush=True)
                continue

            globals_all = np.asarray(
                [symbol_to_global[g] for g in avail], dtype=np.int64
            )

            # Select variable pathway genes using masked input only,
            # separately within each cell group.
            masked_pool = dense32(masked[cell_indices, globals_all].X)
            variances = np.var(masked_pool, axis=0)
            valid = np.isfinite(variances) & (variances > 1e-10)

            order = sorted(
                np.flatnonzero(valid).tolist(),
                key=lambda i: (-float(variances[i]), avail[i])
            )[:max_genes]

            if len(order) < min_genes:
                print(
                    f'WARNING {scope_name} / {pathway_name}: only {len(order)} '
                    f'variable model genes (minimum {min_genes}); skipped.',
                    flush=True
                )
                continue

            genes = [avail[i] for i in order]
            global_cols = globals_all[order]
            local_cols = np.asarray(
                [g_to_l[k] for k in global_cols], dtype=np.int64
            )

            masked_arr = masked_pool[:, order]
            reference_arr = dense32(ref[cell_indices, global_cols].X)

            matrices = {
                'Reference': corr_mat(reference_arr),
                'Masked input': corr_mat(masked_arr)
            }

            maskonly_n_filled = 0
            heldout_n_filled = 0
            for view_name, factors in views.items():
                # Existing full-zero-fill result: fills both artificial mask
                # positions and original sequencing zeros.
                matrices[view_name] = corr_mat(
                    predicted_expression(
                        masked_arr, factors, local_cols,
                        cell_indices=cell_indices
                    )
                )

                if view_name == 'PGNMF':
                    # Primary diagnostic: fill ALL artificially masked positives
                    # from the mask plan, but leave original sequencing zeros at zero.
                    maskonly_arr, maskonly_n_filled = predicted_expression_at_positions(
                        masked_arr, factors, local_cols, global_cols,
                        cell_indices, mask_positions_triplet, 'All artificial-mask'
                    )
                    matrices['PGNMF-mask-only'] = corr_mat(maskonly_arr)

                    # Strict held-out diagnostic: fill only the predeclared test
                    # half of artificial mask positions.
                    heldout_arr, heldout_n_filled = predicted_expression_at_positions(
                        masked_arr, factors, local_cols, global_cols,
                        cell_indices, test_triplet, 'Held-out'
                    )
                    matrices['PGNMF-heldout'] = corr_mat(heldout_arr)

            metrics = []
            for view_name, corr_matrix in matrices.items():
                mae, rmse = corr_dist(matrices['Reference'], corr_matrix)
                metrics.append({
                    'scope': scope_name,
                    'n_cells': int(len(cell_indices)),
                    'pathway': pathway_name,
                    'method': view_name,
                    'n_genes': int(len(genes)),
                    'n_artificial_mask_positions_filled': (
                        int(maskonly_n_filled) if view_name == 'PGNMF-mask-only' else 0
                    ),
                    'n_heldout_positions_filled': (
                        int(heldout_n_filled) if view_name == 'PGNMF-heldout' else 0
                    ),
                    'corr_MAE_to_reference': mae,
                    'corr_RMSE_to_reference': rmse,
                    'genes': ';'.join(genes)
                })

            out.append({
                'scope': scope_name,
                'n_cells': int(len(cell_indices)),
                'name': pathway_name,
                'genes': genes,
                'matrices': matrices,
                'metrics': metrics
            })

    return out

def style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':9.0,
        'axes.titlesize':10,'axes.labelsize':9,
        'xtick.labelsize':7,'ytick.labelsize':8,
        'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
        'axes.spines.right':False, 'figure.facecolor':'white', 'savefig.facecolor':'white'
    })


def save(fig, out, stem):
    for ext in ('png','pdf','svg'):
        filepath=out/f'{stem}.{ext}'
        fig.savefig(filepath, dpi=600 if ext=='png' else None, bbox_inches='tight', facecolor='white')
        print('Saved:',filepath,flush=True)
    plt.close(fig)


def draw_marker_tradeoff(df,out,threshold,min_n):
    markers = [g for g in MARKER_TARGETS if g in df.marker_gene.unique()]
    good=[]
    for g in markers:
        grp=df[df.marker_gene==g]
        if (grp.n_heldout.min()>=min_n and grp.method.eq('PGNMF').any()
                and grp.n_reference_zero_non_target.min()>0):
            good.append(g)
    if len(good)<2:
        print('WARNING: insufficient markers for marker recovery/leakage panel',flush=True)
        return
    methods=[c for c in ['Masked input','PGNMF','w/o Zero-aware'] if c in df.method.unique()]
    vals=[]; ect=[]
    for g in good:
        r=[]; e=[]
        for method in methods:
            row=df[(df.marker_gene==g)&(df.method==method)]
            r.append(float(row.heldout_rmse.iloc[0]) if not row.empty else np.nan)
            e.append(100*float(row.non_target_refzero_gt_threshold.iloc[0]) if not row.empty else np.nan)
        vals.append(r); ect.append(e)
    vals=np.asarray(vals,float);ect=np.asarray(ect,float)
    fig,axs=plt.subplots(1,2,figsize=(max(8.5, 0.45*len(good)+7),5.4),gridspec_kw={'width_ratios':[1,1.05]},dpi=160)
    for ax,arr,title,fmt,scale in [(axs[0],vals,'(a) Recovery at masked marker entries','{:.2f}',1),
                                   (axs[1],ect,'(b) Expression at non-target reference zeros','{:.1f}',1)]:
        im=ax.imshow(arr,cmap='RdYlBu_r',aspect='auto',vmin=0,vmax=max(float(np.nanmax(arr)),1e-6))
        ax.set_xticks(range(len(methods)),methods,rotation=20,ha='right')
        ax.set_yticks(range(len(good)),good)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5,len(methods),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(good),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=2)
        ax.tick_params(which='minor',bottom=False,left=False)
        ax.set_title(title,pad=12,weight='bold')
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i,j]):
                    rgba=im.cmap(im.norm(arr[i,j]))
                    lum=0.2126*rgba[0]+0.7152*rgba[1]+0.0722*rgba[2]
                    ax.text(j,i,fmt.format(arr[i,j]),ha='center',va='center',fontsize=8,
                            color='black' if lum>.53 else 'white',weight='medium')
        cb=fig.colorbar(im,ax=ax,shrink=.62,pad=.02)
        cb.set_label('RMSE (log-CP10K)' if ax is axs[0] else f'% with expression > {threshold:g}',fontsize=8)
    fig.subplots_adjust(wspace=.48,bottom=.16,top=.88,left=.08,right=.99)
    save(fig,out,'Figure49A_marker_recovery_vs_reference_zero_expression')



def draw_pathways(pathways, out):
    if not pathways:
        return

    methods = list(pathways[0]['matrices'])
    rows = len(pathways)
    cols = len(methods)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3.1 * cols + 1.0, 3.0 * rows + 0.6),
        dpi=170, squeeze=False
    )

    for r, entry in enumerate(pathways):
        genes = entry['genes']
        score = {
            item['method']: item['corr_MAE_to_reference']
            for item in entry['metrics']
        }

        for c, method in enumerate(methods):
            ax = axes[r, c]
            im = ax.imshow(
                entry['matrices'][method],
                vmin=-1, vmax=1, cmap='RdBu_r',
                interpolation='nearest', aspect='equal'
            )

            fs = 5.5 if len(genes) > 19 else 6.6
            ax.set_xticks(range(len(genes)), genes, rotation=90, fontsize=fs)

            if c == 0:
                ax.set_yticks(range(len(genes)), genes, fontsize=fs)
                pathway_label = (
                    entry['name'].replace('KEGG_', '').replace('_', ' ').title()
                )
                ax.set_ylabel(
                    f"{entry['scope']}\n{pathway_label}",
                    rotation=0, ha='right', va='center',
                    labelpad=76, fontsize=8.0, weight='bold'
                )
            else:
                ax.set_yticks([])

            ax.tick_params(length=0)
            short_method = (
                method.replace('Masked input', 'Masked')
                      .replace('w/o Gene graph', 'w/o gene graph')
            )
            title = f'({chr(97 + c)}) {short_method}' if r == 0 else short_method
            if method != 'Reference':
                title += f"\nCorrelation MAE = {score[method]:.3f}"
            ax.set_title(title, fontsize=8.2, pad=8)

            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.subplots_adjust(
        wspace=0.26, hspace=0.68,
        left=0.27, right=0.90, top=0.92, bottom=0.14
    )
    cax = fig.add_axes([0.92, 0.29, 0.012, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Gene-gene Pearson correlation', fontsize=9)
    save(fig, out, 'Figure49B_KEGG_pathway_coexpression_heatmaps')

    # Each x-axis category is a distinct scope/pathway pair.
    kept_methods = [method for method in methods if method != 'Reference']

    if kept_methods:
        x = np.arange(len(pathways))
        width = 0.78 / len(kept_methods)
        xlabels = [
            f"{p['scope']}\n{p['name'].replace('KEGG_', '').replace('_', ' ').title()}"
            for p in pathways
        ]

        fig, ax = plt.subplots(
            figsize=(max(8.0, len(pathways) * 2.0 + 3.0), 4.6),
            dpi=170
        )

        for i, method in enumerate(kept_methods):
            values = []
            for pathway in pathways:
                record = next(
                    (item for item in pathway['metrics']
                     if item['method'] == method),
                    None
                )
                values.append(
                    np.nan if record is None
                    else float(record['corr_MAE_to_reference'])
                )

            ax.bar(
                x - 0.39 + width * (i + 0.5),
                values,
                width=width * 0.92,
                color=VIEW_COLORS.get(method, '#446'),
                label=method
            )

        ax.set_xticks(x, xlabels, fontsize=8, rotation=12, ha='right')
        ax.set_ylabel('Correlation matrix MAE vs. Reference (↓)')
        ax.set_ylim(bottom=0)
        ax.legend(
            frameon=False, ncol=min(3, len(kept_methods)),
            loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=8
        )
        ax.grid(axis='y', alpha=0.15)
        ax.set_axisbelow(True)
        fig.tight_layout()
        save(fig, out, 'Figure49C_KEGG_correlation_recovery_summary')

def fit_missing_ablations(names,out,masked,features_mask,all_symbols,gmts,seed,rank,iters,
                          full_config,heldout_rows,heldout_cols,heldout_hash,
                          validation_seed,validation_fraction,
                          gene_symbol_column=None,force_refit=False):
    from pgnmf_revision.pathway_sparsity import (pathway_membership_from_gmt,make_two_level_penalty,get_gene_symbols)
    from pgnmf_revision.gene_graph_stage2b2 import build_pathway_rank_gene_graph
    from pgnmf_revision.cell_graph_stage2b3 import build_sparse_knn_cell_graph
    from pgnmf.solver import zero_aware_nmf_mu_blocked
    if not names:return {}
    print('Loading model input matrix once for ablation fits...',flush=True)
    X=dense32(masked[:,features_mask].X)
    heldout_rows=np.asarray(heldout_rows,dtype=np.int64)
    heldout_cols=np.asarray(heldout_cols,dtype=np.int64)
    if len(heldout_rows)!=len(heldout_cols) or len(heldout_rows)==0:
        raise ValueError('Invalid held-out coordinates for ablation fitting')
    if np.any(X[heldout_rows,heldout_cols] != 0):
        raise ValueError('Held-out coordinates must be zero in the masked model input')
    npos=int(np.count_nonzero(X>0));nzero=int(np.count_nonzero(X==0))
    n_penalized_zero=nzero-len(heldout_rows)
    if n_penalized_zero<=0:
        raise ValueError('No zero positions remain for zero-aware ablation penalty')
    q0=float(npos/n_penalized_zero)
    res={}
    cached={}
    todo=[]
    for name in names:
        cache=out/f'{name}_factors_seed{seed}_iter{iters}.npz'
        meta=out/f'{name}_factors_seed{seed}_iter{iters}.json'
        expected={'variant':name,'dataset':full_config['dataset'],'seed':seed,'rank':rank,
                  'max_iter':iters,'gene_mask_sha256':file_hash(features_mask),
                  'gmt_mtime_ns':gmts.stat().st_mtime_ns,
                  'masked_file_size':(ROOT/'work'/full_config['dataset']/'masked_logcp10k.h5ad').stat().st_size,
                  'zero_penalty_exclusion':True,
                  'heldout_protocol_version':'zero-penalty-exclusion-v1',
                  'validation_seed':int(validation_seed),
                  'validation_fraction':float(validation_fraction),
                  'heldout_split_sha256':heldout_hash,
                  'n_heldout_excluded':int(len(heldout_rows))}
        if cache.exists() and meta.exists() and not force_refit:
            old=json.loads(meta.read_text(encoding='utf-8'))
            if all(old.get(k)==v for k,v in expected.items()):
                with np.load(cache,allow_pickle=False) as z:
                    W=z['W'];H=z['H']
                if W.shape==(X.shape[0],rank) and H.shape==(rank,X.shape[1]):
                    print('Reusing ablation factors:',cache,flush=True)
                    cached[name]=(W,H);continue
            raise ValueError(f'Existing ablation cache mismatches current configuration: {cache}; use --force-refit to rebuild')
        todo.append((name,cache,meta,expected))
    if not todo:return cached
    pathway_all,_=pathway_membership_from_gmt(str(gmts),all_symbols,
        min_pathway_genes=10,max_pathway_genes=500,
        gene_symbol_source=get_gene_symbols(masked,gene_symbol_column)[1])
    penalty=make_two_level_penalty(np.asarray(pathway_all[features_mask],dtype=bool),
         strength=10,pathway_weight=.5,nonpathway_weight=1.).astype(np.float32,copy=False)
    print('Building frozen union-kNN cell graph ONCE...',flush=True)
    Sc,dc,_,_,_=build_sparse_knn_cell_graph(X,n_pcs=30,n_neighbors=15,metric='euclidean',random_state=seed)
    if any(name=='no_zero' for name,_,_,_ in todo):
        print('Building frozen pathway-rank gene graph for w/o Zero-aware...',flush=True)
        raw,dg_raw,_,_=build_pathway_rank_gene_graph(X,all_symbols,all_symbols[features_mask],str(gmts),
           min_pathway_genes=10,max_pathway_genes=500,edge_batch_size=256)
        positive=np.asarray(dg_raw)[np.asarray(dg_raw)>0]
        if not len(positive): raise RuntimeError('Empty KEGG gene graph')
        scale=np.float32(np.median(positive))
        Sg=(raw/scale).tocsr();dg=np.asarray(dg_raw,dtype=np.float32)/scale
    else:
        Sg=sparse.csr_matrix((X.shape[1],X.shape[1]),dtype=np.float32)
        dg=np.zeros(X.shape[1],dtype=np.float32)
    for name,cache,meta,expected in todo:
        alpha_gene=0.0 if name=='no_gene' else 2.0
        zero_weight=0.0 if name=='no_zero' else q0
        print(f'Fitting {name}: seed={seed} rank={rank} iterations={iters} '
              f'alpha_gene={alpha_gene} zero_weight={zero_weight:.8g}',flush=True)
        fit=zero_aware_nmf_mu_blocked(X,S_cell=Sc,degree_cell=dc,alpha_cell=10.0,
                                  S_gene=Sg,degree_gene=dg,alpha_gene=alpha_gene,
                                  lambda_h_gene=penalty,zero_weight=zero_weight,rank=rank,
                                  max_iter=iters,tol=1e-4,random_state=seed,
                                  lambda_reg=.001,lambda_w_l1=0.0,verbose=True,
                                  tag=f'BIO-{name}',
                                  zero_penalty_exclude_rows=heldout_rows,
                                  zero_penalty_exclude_cols=heldout_cols, block_size=500)
        W=np.asarray(fit.W,dtype=np.float32);H=np.asarray(fit.H,dtype=np.float32)
        np.savez_compressed(cache,W=W,H=H)
        expected.update({'zero_weight':zero_weight,'alpha_gene':alpha_gene,
                         'n_iter':int(fit.n_iter),'final_loss':float(fit.losses[-1]),
                         'n_penalized_zero':int(n_penalized_zero)})
        meta.write_text(json.dumps(expected,indent=2),encoding='utf-8')
        cached[name]=(W,H)
        print('Saved ablation factor cache:',cache,flush=True)
    return cached



def main():
    a=parse_args()
    if a.label_key is None:
        a.label_key='majority_voting' if a.dataset in {'C11','C01'} else 'celltype'
    if a.min_pathway_genes<3 or a.max_pathway_genes<a.min_pathway_genes:
        raise ValueError('Invalid pathway minimum/maximum gene count')
    if a.expression_threshold<=0:
        raise ValueError('Expression threshold must exceed zero')
    requests=[x.strip() for x in a.fit_ablations.split(',') if x.strip()]
    if len(requests)!=len(set(requests)) or any(x not in {'no_zero','no_gene'} for x in requests):
        raise ValueError('--fit-ablations may contain no_zero,no_gene only')
    from pgnmf_revision.pathway_sparsity import get_gene_symbols
    from pgnmf_revision.masking import MaskPlan
    import scanpy as sc
    wdir=ROOT/'work'/a.dataset
    m_path=wdir/'masked_logcp10k.h5ad';r_path=wdir/'reference_logcp10k.h5ad'
    mask_path=ROOT/'audit'/a.dataset/'stage2cpre_gene_set_diagnostic'/'gene_set_masks.npz'
    for p in (m_path,r_path,mask_path,wdir/'mask_plan.npz'):
        if not p.is_file():raise FileNotFoundError(p)
    out=fullpath(a.output_dir,ROOT/'audit'/a.dataset/'stage2c32_biological_validation')
    cache_dir=fullpath(a.cache_dir,ROOT/'audit'/a.dataset/'stage2c27_marker_figures')
    out.mkdir(parents=True,exist_ok=True)
    gmt=choose_gmt(a.gmt)
    print('\n=== PGNMF BIOLOGICAL VALIDATION ===')
    print(f'dataset={a.dataset} label={a.label_key} mode={a.mode} seed={a.seed}')
    print(f'Cache: {cache_dir}\nOutput: {out}\nKEGG GMT: {gmt}',flush=True)
    mask_ad=sc.read_h5ad(m_path,backed='r')
    ref_ad=sc.read_h5ad(r_path,backed='r')
    if mask_ad.shape!=ref_ad.shape or not np.array_equal(mask_ad.obs_names,ref_ad.obs_names) or not np.array_equal(mask_ad.var_names,ref_ad.var_names):
        raise ValueError('Masked and Reference matrices not aligned')
    if a.label_key not in mask_ad.obs.columns:raise KeyError(f'{a.label_key} absent from masked obs')
    labels=mask_ad.obs[a.label_key].astype(str).to_numpy()
    feature_mask=load_mask(mask_path,a.gene_set_key,mask_ad.n_vars)
    symbols=np.asarray(get_gene_symbols(mask_ad,a.gene_symbol_column)[0],dtype=str)
    sym_to_global={}
    for i,sym in enumerate(symbols):
        sym_to_global.setdefault(sym.upper(),i)
    global_to_local=np.full(mask_ad.n_vars,-1,dtype=np.int64)
    global_to_local[np.flatnonzero(feature_mask)]=np.arange(int(feature_mask.sum()))
    selected_symbols={g for g,i in sym_to_global.items() if feature_mask[i]}
    targets={g:tuple(x for x in grp if x in set(labels)) for g,grp in MARKER_TARGETS.items()
             if g in selected_symbols}
    targets={g:t for g,t in targets.items() if t}
    marker_global={g:sym_to_global[g] for g in targets}
    paths=choose_pathways(gmt,a.pathways,selected_symbols,a.min_pathway_genes)
    print('Markers:', ', '.join(marker_global),'\nReference-zero threshold:',a.expression_threshold)
    print('Selected KEGG pathways:',[(x,len(g)) for x,g in paths],flush=True)
    if a.mode=='inspect':
        print('\nINSPECTION COMPLETE: no model fit, no plots. To produce figures use --mode plot.',flush=True)
        mask_ad.file.close();ref_ad.file.close();return
    if len(marker_global)<3 or not paths:
        raise RuntimeError('Insufficient marker genes/pathways. Review --mode inspect and KEGG GMT before plotting.')
    plan=MaskPlan.load(wdir/'mask_plan.npz')
    mask_positions=selected_mask_positions(plan,feature_mask)
    test=heldout_splits(plan,feature_mask,a.validation_fraction,a.validation_seed)
    heldout_hash=heldout_position_hash(test[0],test[1])
    test_cols_local=global_to_local[test[1]]
    if np.any(test_cols_local < 0):
        raise RuntimeError('A held-out global gene column is absent from the model feature set')

    full, fullmeta=valid_full_factor(
        cache_dir,a.dataset,feature_mask,a.seed,a.rank,a.max_iter,
        mask_ad.n_obs,m_path,a.validation_seed,a.validation_fraction,
        heldout_hash,len(test[0])
    )
    if fullmeta['configuration'].get('gene_set_key')!=a.gene_set_key:
        raise ValueError('Full cache gene-set key mismatch')
    # Ensure the cache matches the same KEGG GMT as the project input.
    if Path(fullmeta['configuration'].get('gmt','')).resolve()!=gmt:
        raise ValueError('Full cache used a different GMT from the current run')
    print('Reusing FROZEN Full PGNMF factor cache, no Full refit.',flush=True)
    views={'PGNMF':full}
    if requests:
        other=fit_missing_ablations(
            requests,out,mask_ad,feature_mask,symbols,gmt,a.seed,a.rank,a.max_iter,
            fullmeta['configuration'],test[0],test_cols_local,heldout_hash,
            a.validation_seed,a.validation_fraction,
            a.gene_symbol_column,a.force_refit
        )
        if 'no_zero' in other:views['w/o Zero-aware']=other['no_zero']
        if 'no_gene' in other:views['w/o Gene graph']=other['no_gene']
    marker_views={k:v for k,v in views.items() if k!='w/o Gene graph'}
    marker_table=compute_marker_metrics(marker_global,targets,labels,ref_ad,mask_ad,marker_views,
                                        test,global_to_local,a.expression_threshold)
    if marker_table.empty:raise RuntimeError('No test markers available')
    marker_table.to_csv(out/'marker_recovery_reference_zero_metrics.csv',index=False,encoding='utf-8-sig')
    style();draw_marker_tradeoff(marker_table,out,a.expression_threshold,a.min_marker_heldout)
    print('Marker results:')
    print(marker_table[['marker_gene','method','n_heldout','heldout_rmse','non_target_refzero_gt_threshold']].to_string(index=False),flush=True)
    pview={k:v for k,v in views.items() if k!='w/o Zero-aware'}

    path_results = compute_pathways(
        paths, sym_to_global, global_to_local,
        mask_ad, ref_ad, pview, labels, mask_positions, test,
        a.max_pathway_genes, a.min_pathway_genes
    )

    if path_results:
        pm = pd.DataFrame([
            record for pathway in path_results for record in pathway['metrics']
        ])
        pm.to_csv(
            out / 'pathway_coexpression_recovery_metrics.csv',
            index=False, encoding='utf-8-sig'
        )

        # Save each scope/pathway correlation matrix under a unique filename.
        for j, pathway in enumerate(path_results):
            safe_scope = ''.join(
                ch if ch.isalnum() or ch in '-_' else '_'
                for ch in pathway['scope']
            )[:40]
            safe_name = ''.join(
                ch if ch.isalnum() or ch in '-_' else '_'
                for ch in pathway['name']
            )[:90]

            corr_payload = {
                'corr_' + key.lower().replace(' ', '_').replace('/', ''): value
                for key, value in pathway['matrices'].items()
            }

            np.savez_compressed(
                out / f'pathway_correlation_{j+1}_{safe_scope}_{safe_name}.npz',
                genes=np.asarray(pathway['genes'], dtype=str),
                **corr_payload
            )

        draw_pathways(path_results, out)
        print('\nKEGG pathway correlation recovery by cell group:')
        print(
            pm[[
                'scope', 'n_cells', 'pathway', 'method',
                'n_genes', 'corr_MAE_to_reference'
            ]].to_string(index=False),
            flush=True
        )
    else:
        print(
            'WARNING: no pathways with sufficient variable gene coverage to plot.',
            flush=True
        )

    manifest={'dataset':a.dataset,'label_key':a.label_key,'seed':a.seed,
              'rank':a.rank,'max_iter':a.max_iter,'input_gmt':str(gmt),
              'gene_set_key':a.gene_set_key,'n_model_genes':int(feature_mask.sum()),
              'reference_zero_caution':'Reference sequencing zeros are not experimentally confirmed biological zeros.',
              'target_labels_caution':'C11 majority_voting cell-type labels were inferred using CellTypist.',
              'heldout_positive_definition':'Same masked-positive test split as Stage2C27; held-out coordinates were excluded from the zero-aware penalty during model fitting.',
              'heldout_protocol':{
                  'validation_seed':int(a.validation_seed),
                  'validation_fraction':float(a.validation_fraction),
                  'n_heldout_positions':int(len(test[0])),
                  'heldout_split_sha256':heldout_hash,
                  'excluded_from_zero_penalty_during_fit':True,
                  'heldout_reference_values_used_only_after_fit':True
              },
              'pathway_views':{
                  'PGNMF':'Preserve observed nonzero values and fill all zeros in selected pathway genes.',
                  'PGNMF-mask-only':'Fill all artificial mask-plan positions using W@H; original sequencing zeros remain unchanged.',
                  'PGNMF-heldout':'Fill only the predeclared held-out test subset of artificial mask positions using W@H; all other entries remain unchanged.'
              },
              'pathway_selection':'Predesignated KEGG T/B receptor pathways; top genes selected by masked-input variance.',
              'marker_target_definitions':{g:list(t) for g,t in targets.items()},
              'methods':list(views),'threshold_logcp10k':a.expression_threshold,
              'min_marker_heldout':a.min_marker_heldout,
              'pathways':[{'name':x['name'],'genes':x['genes']} for x in path_results],
              'no_labels_used_in_fitting':True,
              'warning':'Visual or coexpression improvements are NOT guaranteed; report all generated results.'}
    (out/'stage2c32_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    mask_ad.file.close();ref_ad.file.close()
    print('\n=== COMPLETE ===\nFigures and CSVs saved in:',out,flush=True)


if __name__=='__main__':
    main()
