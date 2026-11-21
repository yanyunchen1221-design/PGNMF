#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PGNMF Stage 2C-27: marker validation with an independently held-out test split.

Run from PGNMF repository root with the file in scripts/.

  python -u scripts/run_marker_preservation_analysis.py --dataset C11 --label-key majority_voting --mode inspect
  python -u scripts/run_marker_preservation_analysis.py --dataset C11 --label-key majority_voting --mode plot

The program uses the same HVG1000+PATHWAY feature mask and frozen graph-building
protocol as Stage2C20. The held-out masked-positive test positions are selected
BEFORE fitting and excluded from the zero-aware zero penalty. Their reference
expression values are used only after fitting for evaluation. A new W/H cache
is written for this exact held-out split; use a new output directory.
No cell labels or held-out truths influence model fitting or stopping.
No fictional data are generated. Reference labels may be CellTypist-derived.
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
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))  # maintained public PGNMF package
sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))

IMMUNE_MARKERS = {
    'T cells': ['CD3D', 'CD3E', 'TRAC'],
    'B cells': ['MS4A1', 'CD79A', 'CD79B'],
    'NK / cytotoxic': ['NKG7', 'GNLY'],
    'Monocytes': ['LYZ', 'CD14', 'S100A8'],
    'Dendritic cells': ['FCER1A'],
    'Platelets': ['PPBP', 'PF4'],
}
PANCREAS_MARKERS = {
    'Beta': ['INS', 'IAPP'],
    'Alpha': ['GCG', 'TTR'],
    'Delta': ['SST'],
    'PP': ['PPY'],
    'Acinar': ['PRSS1', 'CPA1', 'REG1A'],
    'Ductal': ['KRT19', 'KRT7'],
}


def parse_args():
    p = argparse.ArgumentParser(description='Marker preservation figures for frozen PGNMF')
    p.add_argument('--dataset', required=True)
    p.add_argument('--label-key', default=None)
    p.add_argument('--mode', choices=['inspect','plot'], default='inspect')
    p.add_argument('--panel', choices=['auto','immune','pancreas'], default='auto')
    p.add_argument('--markers', default=None, help='Optional comma-separated PREDEFINED marker gene symbols')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--gmt', default=None)
    p.add_argument('--gene-set-key', default='HVG1000+PATHWAY')
    p.add_argument('--gene-symbol-column', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rank', type=int, default=80)
    p.add_argument('--max-iter', type=int, default=100)
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--alpha-cell', type=float, default=10.0)
    p.add_argument('--alpha-gene', type=float, default=2.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--lambda-reg', type=float, default=0.001)
    p.add_argument('--sparsity-strength', type=float, default=10.0)
    p.add_argument('--pathway-weight', type=float, default=0.5)
    p.add_argument('--nonpathway-weight', type=float, default=1.0)
    p.add_argument('--expression-threshold', type=float, default=1.0,
                   help='Expression level threshold for dot sizes, in log-CP10K units; NOT >0')
    p.add_argument('--min-cells', type=int, default=20)
    p.add_argument('--max-groups', type=int, default=18)
    p.add_argument('--min-heldout-per-marker', type=int, default=25)
    p.add_argument('--validation-seed', type=int, default=2026)
    p.add_argument('--validation-fraction', type=float, default=0.5)
    p.add_argument('--rebuild-cache', action='store_true')
    return p.parse_args()


def dense32(x):
    return x.toarray().astype(np.float32, copy=False) if sparse.issparse(x) else np.asarray(x, dtype=np.float32)


def chosen_panel(labels, explicit):
    if explicit != 'auto':
        return explicit
    s = ' '.join(str(x).lower() for x in labels)
    n_immune = sum(s.count(k) for k in ['t cell','b cell','nk cell','monocyte','dendritic','macrophage','platelet'])
    n_pancreas = sum(s.count(k) for k in ['beta cell','alpha cell','acinar','ductal','delta cell','pancrea'])
    if n_pancreas > n_immune:
        return 'pancreas'
    if n_immune > 0:
        return 'immune'
    raise ValueError('Cannot infer a relevant marker panel from cell type names. Pass --panel immune/pancreas or --markers GENE1,GENE2,...')


def summary_metadata(args, ad_masked, gene_mask, gene_symbols, obs_labels):
    avail = {str(g).upper(): i for i,g in enumerate(gene_symbols) if gene_mask[i]}
    if args.markers:
        requested = [g.strip().upper() for g in args.markers.split(',') if g.strip()]
        panel = 'custom'
        categories = {'Predefined markers': requested}
    else:
        panel = chosen_panel(obs_labels, args.panel)
        categories = IMMUNE_MARKERS if panel == 'immune' else PANCREAS_MARKERS
        requested = [g for genes in categories.values() for g in genes]
    requested = list(dict.fromkeys(requested))
    found = [g for g in requested if g.upper() in avail]
    absent = [g for g in requested if g.upper() not in avail]
    model_cols = np.flatnonzero(gene_mask)
    global_to_local = {int(g): j for j,g in enumerate(model_cols)}
    selected_global = [avail[g.upper()] for g in found]
    selected_local = [global_to_local[i] for i in selected_global]
    labels = pd.Series(ad_masked.obs[args.label_key].astype(str).to_numpy())
    counts = labels.value_counts()
    kept = counts[counts >= args.min_cells].head(args.max_groups).index.tolist()
    print('\n=== MARKER FIGURE INSPECTION ===', flush=True)
    print(f'dataset={args.dataset} label_key={args.label_key} panel={panel}', flush=True)
    print(f'matrix={ad_masked.shape} model_features={gene_mask.sum()} classes={len(counts)}', flush=True)
    print(f'markers AVAILABLE ({len(found)}): {", ".join(found) if found else "NONE"}', flush=True)
    print(f'markers ABSENT ({len(absent)}): {", ".join(absent) if absent else "NONE"}', flush=True)
    print('cell types (number of cells):', flush=True)
    for grp in kept:
        print(f'  {grp}: {counts[grp]}', flush=True)
    if len(found) < 3:
        print('WARNING: fewer than 3 predefined markers are available in the MODEL gene set.', flush=True)
    return {
        'panel': panel, 'marker_categories': categories, 'markers': found,
        'missing_markers': absent, 'global_cols': selected_global,
        'local_cols': selected_local, 'groups': kept,
        'all_group_counts': {str(k):int(v) for k,v in counts.items()},
    }


def ensure_path(path, default):
    out = Path(path) if path else default
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True,exist_ok=True)
    return out.resolve()


def split_indices(n, frac, seed):
    rng = np.random.default_rng(seed)
    order=rng.permutation(n)
    nv=min(max(1,int(round(n*frac))),n-1)
    return np.sort(order[:nv]),np.sort(order[nv:])


def heldout_test_positions(mask_plan, model_mask, frac, seed):
    """Return the exact held-out masked-positive positions used for scoring.

    Only row/column coordinates are passed into model fitting. The reference
    values are retained separately for post-fit evaluation and never enter the
    solver or any model-construction step.
    """
    relevant = np.asarray(model_mask, dtype=bool)[
        np.asarray(mask_plan.cols, dtype=np.int64)
    ]
    rows = np.asarray(mask_plan.rows[relevant], dtype=np.int64)
    cols = np.asarray(mask_plan.cols[relevant], dtype=np.int64)
    truth = np.asarray(mask_plan.reference_values[relevant], dtype=np.float64)
    if len(truth) < 2:
        raise RuntimeError('Too few masked-positive positions in the selected model genes')
    _, test_idx = split_indices(len(truth), frac, seed)
    test_rows = rows[test_idx]
    test_cols = cols[test_idx]
    test_truth = truth[test_idx]
    if len(test_truth) == 0:
        raise RuntimeError('Held-out test split is empty')
    return test_rows, test_cols, test_truth


def heldout_position_hash(rows, cols):
    """Order-independent SHA256 of held-out matrix coordinates (not expression values)."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    if rows.size != cols.size:
        raise ValueError('Held-out row/column lengths differ')
    order = np.lexsort((cols, rows))
    coordinates = np.column_stack((rows[order], cols[order])).astype(np.int64, copy=False)
    return hashlib.sha256(coordinates.tobytes()).hexdigest()


def fitted_factors(args, out, X, ad_masked, gene_mask, symbols_all, mask_path, heldout_rows, heldout_cols_local, heldout_hash):
    from pgnmf_revision.pathway_sparsity import pathway_membership_from_gmt, make_two_level_penalty
    from pgnmf_revision.gene_graph_stage2b2 import build_pathway_rank_gene_graph
    from pgnmf_revision.cell_graph_stage2b3 import build_sparse_knn_cell_graph
    from pgnmf.solver import zero_aware_nmf_mu_blocked
    from pgnmf_revision.pathway_sparsity import get_gene_symbols

    preferred=ROOT/'data'/'c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt'
    if args.gmt:
        gmt=Path(args.gmt)
    elif preferred.is_file():
        gmt=preferred
    else:
        gs=sorted((ROOT/'data').rglob('*.gmt'))
        gs=[p for p in gs if 'kegg' in p.name.lower()] or gs
        if len(gs)!=1:
            raise FileNotFoundError('Cannot find unique KEGG GMT; pass --gmt.')
        gmt=gs[0]
    if not gmt.is_file():
        raise FileNotFoundError(f'GMT missing: {gmt}')
    heldout_rows = np.asarray(heldout_rows, dtype=np.int64).reshape(-1)
    heldout_cols_local = np.asarray(heldout_cols_local, dtype=np.int64).reshape(-1)
    if heldout_rows.size != heldout_cols_local.size or heldout_rows.size == 0:
        raise ValueError('Held-out row/column coordinates are empty or misaligned')
    if np.any(heldout_rows < 0) or np.any(heldout_rows >= X.shape[0]):
        raise IndexError('Held-out row index out of bounds')
    if np.any(heldout_cols_local < 0) or np.any(heldout_cols_local >= X.shape[1]):
        raise IndexError('Held-out local model-gene column index out of bounds')
    if np.any(X[heldout_rows, heldout_cols_local] != 0):
        raise ValueError('Held-out masked-positive positions must be zero in model input X')
    if not isinstance(heldout_hash, str) or len(heldout_hash) != 64:
        raise ValueError('A SHA256 hash of the GLOBAL held-out coordinates is required')

    config={'dataset':args.dataset,'seed':args.seed,'rank':args.rank,
            'max_iter':args.max_iter,'tol':args.tol,
            'alpha_cell':args.alpha_cell,'alpha_gene':args.alpha_gene,
            'beta':args.beta,'lambda_reg':args.lambda_reg,
            'sparsity_strength':args.sparsity_strength,
            'pathway_weight':args.pathway_weight,
            'nonpathway_weight':args.nonpathway_weight,
            'gmt':str(gmt.resolve()),
            'gene_set_key':args.gene_set_key,
            'gene_mask_sha256':hashlib.sha256(np.asarray(gene_mask,dtype=np.uint8).tobytes()).hexdigest(),
            'masked_file_size':(ROOT/'work'/args.dataset/'masked_logcp10k.h5ad').stat().st_size,
            'masked_file_mtime_ns':(ROOT/'work'/args.dataset/'masked_logcp10k.h5ad').stat().st_mtime_ns,
            'gmt_file_mtime_ns':gmt.stat().st_mtime_ns,
            'zero_penalty_exclusion':True,
            'heldout_protocol_version':'zero-penalty-exclusion-v1',
            'validation_seed':int(args.validation_seed),
            'validation_fraction':float(args.validation_fraction),
            'n_heldout_excluded':int(heldout_rows.size),
            'heldout_split_sha256':heldout_hash}
    cache=out/f'pgnmf_factors_seed{args.seed}_iter{args.max_iter}.npz'
    meta=out/f'pgnmf_factors_seed{args.seed}_iter{args.max_iter}.json'
    if cache.exists() and meta.exists() and not args.rebuild_cache:
        old=json.loads(meta.read_text(encoding='utf-8'))
        if old.get('configuration')!=config:
            raise ValueError(f'Cached factors have a DIFFERENT model configuration. Use a NEW --output-dir or --rebuild-cache.\n{cache}')
        with np.load(cache, allow_pickle=False) as z:
            W=z['W']; H=z['H']
        if W.shape==(X.shape[0],args.rank) and H.shape==(args.rank,X.shape[1]):
            print(f'Reusing cached PGNMF factors: {cache}',flush=True)
            return W,H,old
        raise ValueError('Cached factors have invalid shape; use --rebuild-cache')

    pathway_all,_=pathway_membership_from_gmt(str(gmt),symbols_all,
        min_pathway_genes=10,max_pathway_genes=500,gene_symbol_source=get_gene_symbols(ad_masked,args.gene_symbol_column)[1])
    pathway_model=np.asarray(pathway_all[gene_mask],dtype=bool)
    penalty=make_two_level_penalty(pathway_model,
        strength=args.sparsity_strength,pathway_weight=args.pathway_weight,
        nonpathway_weight=args.nonpathway_weight).astype(np.float32,copy=False)
    npos=int(np.count_nonzero(X>0)); nzero=int(np.count_nonzero(X==0))
    n_penalized_zero = nzero - int(heldout_rows.size)
    if n_penalized_zero <= 0:
        raise ValueError('No zero positions remain for the zero-aware penalty after held-out exclusion')
    q0=float(args.beta*npos/n_penalized_zero)
    if q0>1:
        raise ValueError(f'Unexpected zero weight >1: {q0}')
    print('Building frozen KEGG gene graph...',flush=True)
    Sg_raw,dg_raw,_,_=build_pathway_rank_gene_graph(X,symbols_all,symbols_all[gene_mask],str(gmt),
        min_pathway_genes=10,max_pathway_genes=500,edge_batch_size=256)
    pos=np.asarray(dg_raw)[np.asarray(dg_raw)>0]
    if not len(pos):
        raise RuntimeError('Empty KEGG gene graph')
    scale=float(np.median(pos))
    Sg=(Sg_raw/np.float32(scale)).tocsr()
    dg=np.asarray(dg_raw,dtype=np.float32)/np.float32(scale)
    print('Building frozen union-kNN cell graph...',flush=True)
    Sc,dc,_,_,_=build_sparse_knn_cell_graph(X,n_pcs=30,n_neighbors=15,
        metric='euclidean',random_state=args.seed)
    print(f'Refitting frozen PGNMF once ({args.max_iter} iterations, seed {args.seed}, q0 {q0:.8g})...',flush=True)
    result=zero_aware_nmf_mu_blocked(X,S_cell=Sc,degree_cell=dc,alpha_cell=args.alpha_cell,
        S_gene=Sg,degree_gene=dg,alpha_gene=args.alpha_gene,
        lambda_h_gene=penalty,zero_weight=q0,rank=args.rank,
        max_iter=args.max_iter,tol=args.tol,random_state=args.seed,
        lambda_reg=args.lambda_reg,lambda_w_l1=0.0,verbose=True,tag=f'MARKER-{args.dataset}',
        zero_penalty_exclude_rows=heldout_rows,
        zero_penalty_exclude_cols=heldout_cols_local, block_size=500)
    W=np.asarray(result.W,dtype=np.float32)
    H=np.asarray(result.H,dtype=np.float32)
    np.savez_compressed(cache,W=W,H=H)
    metadata={'configuration':config,'n_iter':int(result.n_iter),
              'converged':bool(result.converged),'final_loss':float(result.losses[-1]),
              'q0':q0,'n_penalized_zero':int(n_penalized_zero),
              'n_heldout_excluded_from_zero_penalty':int(heldout_rows.size),
              'cache_path':str(cache)}
    meta.write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved model factor cache:',cache,flush=True)
    return W,H,metadata


def make_summary(y, arrays, group_names, markers, threshold):
    out=[]
    for view, A in arrays.items():
        for group in group_names:
            m=(y==group)
            vec=A[m]
            means=vec.mean(axis=0)
            rate=(vec>threshold).mean(axis=0)
            for j,gene in enumerate(markers):
                out.append({'view':view,'cell_type':group,'marker_gene':gene,
                            'n_cells':int(m.sum()),'mean_logcp10k':float(means[j]),
                            'fraction_above_threshold':float(rate[j]),
                            'threshold_logcp10k':threshold})
    return pd.DataFrame(out)


def draw_dotplot(stats, groups, genes, threshold, out):
    plt.rcParams.update({'font.family':'Times New Roman','pdf.fonttype':42,'ps.fonttype':42,
                         'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    data=stats.copy()
    vmax=float(np.quantile(data['mean_logcp10k'].to_numpy(),0.985))
    vmax=max(vmax,0.1)
    norm=Normalize(vmin=0,vmax=vmax)
    fig,axes=plt.subplots(1,3,figsize=(max(13.6,3.1+0.36*len(genes)*3),max(5.5,2.9+0.34*len(groups))),sharey=True,dpi=180)
    views=[('Reference','(a) Reference'),('Masked input','(b) Masked input'),('PGNMF','(c) PGNMF')]
    for ax,(view,title) in zip(axes,views):
        sub=data[data.view==view].copy().set_index(['cell_type','marker_gene'])
        xx=[];yy=[];sizes=[];colors=[]
        for iy,group in enumerate(groups):
            for ix,gene in enumerate(genes):
                rec=sub.loc[(group,gene)]
                xx.append(ix);yy.append(iy)
                rate=float(rec.fraction_above_threshold)
                sizes.append(6+220*rate)
                colors.append(float(rec.mean_logcp10k))
        ax.scatter(xx,yy,s=sizes,c=colors,cmap='viridis',norm=norm,
                   marker='o',edgecolors='white',linewidths=0.4)
        ax.set_title(title,fontsize=11,pad=10,fontweight='bold')
        ax.set_xlim(-0.7,len(genes)-0.3)
        ax.set_ylim(len(groups)-0.55,-0.55)
        ax.set_xticks(range(len(genes)),labels=genes,rotation=70,ha='right')
        ax.set_yticks(range(len(groups)),labels=groups)
        ax.tick_params(axis='both',length=0)
        ax.set_axisbelow(True)
        ax.grid(color='#e7e7e7',alpha=0.9,linewidth=0.65)
    fig.subplots_adjust(bottom=0.20,right=0.87,wspace=0.12,left=0.22,top=0.89)
    cax=fig.add_axes([0.89,0.36,0.012,0.43])
    sm=plt.cm.ScalarMappable(cmap='viridis',norm=norm)
    cbar=fig.colorbar(sm,cax=cax)
    cbar.set_label('Mean expression (log-CP10K)',fontsize=9)
    lax=fig.add_axes([0.885,0.12,0.10,0.13]); lax.axis('off')
    for j,frac in enumerate([0.1,0.3,0.6,0.9]):
        x=0.18+(j%2)*0.49;y=0.78-(j//2)*0.48
        lax.scatter([x],[y],s=6+220*frac,c='#777777',edgecolors='white')
        lax.text(x+0.12,y,f'{int(frac*100)}%',va='center',fontsize=8)
    lax.set_xlim(0,1.25);lax.set_ylim(-0.02,1.1)
    fig.text(0.62,0.045,f'Dot size = fraction of cells with expression > {threshold:g} log-CP10K (same cutoff in all panels)',
             ha='center',fontsize=9)
    save_figure(fig,out,'Figure_marker_preservation_dotplot')


def save_figure(fig,out,stem):
    for ext in ['png','pdf','svg']:
        path=out/f'{stem}.{ext}'
        fig.savefig(path, dpi=600 if ext=='png' else None,
                    facecolor='white',bbox_inches='tight')
        print('Saved',path,flush=True)
    plt.close(fig)


def marker_test_rmse(selected_global, selected_local, W, H,
                     test_rows, test_cols, test_truth):
    test_rows = np.asarray(test_rows, dtype=np.int64)
    test_cols = np.asarray(test_cols, dtype=np.int64)
    test_truth = np.asarray(test_truth, dtype=np.float64)
    if not (len(test_rows) == len(test_cols) == len(test_truth)):
        raise ValueError('Held-out test arrays have inconsistent lengths')
    records=[]
    for global_col,local_col in zip(selected_global,selected_local):
        m=(test_cols==global_col)
        rows=test_rows[m]; t=test_truth[m]
        if not len(t):
            records.append({'global_col':global_col,'local_col':local_col,'n_test':0,
                            'masked_rmse':np.nan,'pgnmf_rmse':np.nan,'masked_mae':np.nan,'pgnmf_mae':np.nan})
            continue
        pred=np.sum(W[rows,:].astype(np.float64)*H[:,local_col].astype(np.float64)[None,:],axis=1)
        records.append({'global_col':global_col,'local_col':local_col,'n_test':len(t),
                        'masked_rmse':float(np.sqrt(np.mean(t**2))),
                        'pgnmf_rmse':float(np.sqrt(np.mean((t-pred)**2))),
                        'masked_mae':float(np.mean(np.abs(t))),
                        'pgnmf_mae':float(np.mean(np.abs(t-pred)))})
    return pd.DataFrame(records)


def plot_rmse(df,out,min_n):
    kept=df[df.n_test>=min_n]
    if len(kept)<2:
        print(f'Not enough markers with {min_n}+ held-out TEST values to draw RMSE panel; CSV still saved.',flush=True)
        return
    plt.rcParams.update({'font.family':'Times New Roman','pdf.fonttype':42,'ps.fonttype':42,'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.2,max(3.7,len(kept)*0.45+1.7)),dpi=180)
    yy=np.arange(len(kept))
    ax.barh(yy-0.17,kept.masked_rmse,height=0.32,label='Masked input',color='#9e9e9e')
    ax.barh(yy+0.17,kept.pgnmf_rmse,height=0.32,label='PGNMF',color='#366caa')
    ax.set_yticks(yy,labels=kept.marker_gene)
    ax.set_xlabel('Held-out positive-entry RMSE (log-CP10K; lower is better)')
    ax.invert_yaxis();ax.legend(frameon=False,ncol=2,loc='upper right')
    ax.grid(axis='x',alpha=0.18);ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig,out,'Supplementary_marker_heldout_rmse')


def main():
    a=parse_args()
    if a.label_key is None:
        a.label_key='majority_voting' if a.dataset in ('C11','C01') else 'celltype'
    if a.expression_threshold<=0:
        raise ValueError('--expression-threshold must be >0 to avoid treating dense low-level predictions as universal detection')
    work=ROOT/'work'/a.dataset
    masked=work/'masked_logcp10k.h5ad'
    ref=work/'reference_logcp10k.h5ad'
    masks=ROOT/'audit'/a.dataset/'stage2cpre_gene_set_diagnostic'/'gene_set_masks.npz'
    for file in [masked,ref,masks]:
        if not file.is_file():
            raise FileNotFoundError(f'Expected project input missing: {file}')
    import scanpy as sc
    from pgnmf_revision.pathway_sparsity import get_gene_symbols
    from pgnmf_revision.masking import MaskPlan

    ad_masked=sc.read_h5ad(masked,backed='r' if a.mode=='inspect' else None)
    if a.label_key not in ad_masked.obs:
        raise KeyError(f'Unknown label {a.label_key}, columns={ad_masked.obs.columns.tolist()}')
    with np.load(masks,allow_pickle=False) as z:
        if a.gene_set_key not in z.files:
            raise KeyError(f'{a.gene_set_key} not found in {masks}; choices={z.files}')
        gene_mask=np.asarray(z[a.gene_set_key],dtype=bool)
    if gene_mask.shape!=(ad_masked.n_vars,):
        raise ValueError('Gene mask dimensions differ from h5ad')
    all_symbols,_=get_gene_symbols(ad_masked,a.gene_symbol_column)
    all_symbols=np.asarray(all_symbols,dtype=str)
    labels=ad_masked.obs[a.label_key].astype(str).to_numpy()
    info=summary_metadata(a,ad_masked,gene_mask,all_symbols,labels)
    if a.mode=='inspect':
        if getattr(ad_masked,'file',None) is not None:
            ad_masked.file.close()
        print('\nInspect completed: NO refitting and NO figures. For selected markers use --mode plot.',flush=True)
        return
    if len(info['markers'])<3:
        raise ValueError('Fewer than 3 markers available. Supply --markers from genes in the model feature set.')
    out=ensure_path(a.output_dir,ROOT/'audit'/a.dataset/'stage2c27_marker_figures')
    if out.resolve()==(ROOT/'audit'/a.dataset/'stage2c20_final_label_free_protocol').resolve():
        raise ValueError('Output directory must not overwrite official benchmark')
    X=dense32(ad_masked[:,gene_mask].X)
    ref_ad=sc.read_h5ad(ref,backed='r')
    if ref_ad.shape!=ad_masked.shape or not np.array_equal(ref_ad.obs_names,ad_masked.obs_names):
        raise ValueError('Reference/masked cell alignment mismatch')
    if not np.array_equal(ref_ad.var_names,ad_masked.var_names):
        raise ValueError('Reference/masked gene alignment mismatch')

    # Freeze the held-out split BEFORE fitting. Only coordinates enter the solver;
    # reference expression values are kept solely for post-fit scoring.
    plan=MaskPlan.load(work/'mask_plan.npz')
    test_rows,test_cols_global,test_truth=heldout_test_positions(
        plan,gene_mask,a.validation_fraction,a.validation_seed
    )
    global_to_local=np.full(gene_mask.size,-1,dtype=np.int64)
    global_to_local[np.flatnonzero(gene_mask)]=np.arange(int(gene_mask.sum()),dtype=np.int64)
    test_cols_local=global_to_local[test_cols_global]
    if np.any(test_cols_local < 0):
        raise RuntimeError('A held-out global gene column is absent from the model feature set')
    heldout_hash=heldout_position_hash(test_rows,test_cols_global)
    W,H,cache_meta=fitted_factors(
        a,out,X,ad_masked,gene_mask,all_symbols,work/'mask_plan.npz',
        test_rows,test_cols_local,heldout_hash
    )
    global_cols=info['global_cols'];local_cols=info['local_cols']
    ref_sel=dense32(ref_ad[:,global_cols].X)
    mask_sel=np.asarray(X[:,local_cols],dtype=np.float32)
    model_sel=np.asarray(W@H[:,local_cols],dtype=np.float32)
    imputed=np.where(mask_sel>0,mask_sel,model_sel)
    views={'Reference':ref_sel,'Masked input':mask_sel,'PGNMF':imputed}
    stat=make_summary(labels,views,info['groups'],info['markers'],a.expression_threshold)
    stat.to_csv(out/'marker_dotplot_statistics.csv',index=False,encoding='utf-8-sig')
    draw_dotplot(stat,info['groups'],info['markers'],a.expression_threshold,out)

    errs=marker_test_rmse(
        global_cols,local_cols,W,H,test_rows,test_cols_global,test_truth
    )
    errs.insert(0,'marker_gene',info['markers'])
    errs.to_csv(out/'marker_heldout_error.csv',index=False,encoding='utf-8-sig')
    plot_rmse(errs,out,a.min_heldout_per_marker)

    manifest={'dataset':a.dataset,'label_key':a.label_key,'seed':a.seed,
              'cell_label_source':'Use caution if cell labels were computationally inferred (e.g., CellTypist)',
              'marker_panel':info['panel'],'predefined_requested':list(dict.fromkeys([
                   g for sub in info['marker_categories'].values() for g in sub])),
              'marker_genes':info['markers'],'markers_absent':info['missing_markers'],
              'cell_groups':info['groups'],'expression_threshold_logcp10k':a.expression_threshold,
              'n_test_masked_entries_per_marker':errs.set_index('marker_gene')['n_test'].to_dict(),
              'comparison':'Same cells, same marker genes, common color scale and expression threshold',
              'test_rmse':'Only simulated masked positive entries in the same heldout test split, not original zeros',
              'heldout_protocol':{
                  'validation_seed':int(a.validation_seed),
                  'validation_fraction':float(a.validation_fraction),
                  'n_heldout_positions':int(len(test_rows)),
                  'heldout_split_sha256':heldout_hash,
                  'excluded_from_zero_penalty_during_fit':True,
                  'heldout_reference_values_used_only_after_fit':True
              },
              'model_cache_meta':cache_meta}
    (out/'marker_figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if getattr(ref_ad,'file',None) is not None:
        ref_ad.file.close()
    print('\n=== COMPLETED ===\nOutput:',out,flush=True)
    print('IMPORTANT: inspect dotplots for non-target expression and compare RMSE before claiming marker preservation.',flush=True)


if __name__=='__main__':
    main()
