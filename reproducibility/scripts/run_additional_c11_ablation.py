#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PGNMF Stage 2C-25: finish C11 ablations, with true pathway removal.

Run from PGNMF repository root:
    python -u reproducibility/scripts/run_additional_c11_ablation.py --dataset C11 --seeds 42 --label-key majority_voting

New experimental variants (each trained once for each seed):
  baseline_nmf       : standard Frobenius NMF (zero_weight=1; no graphs, L1, L2).
  no_pathway_all     : no KEGG-enriched feature selection, no KEGG gene graph,
                       no pathway-differential sparsity. Use the same number
                       of top-HVG genes selected *only* from masked input;
                       retain zero-aware term, cell graph, same rank/iterations,
                       and a uniform L1 coefficient (strength=10).
  full_common_refit  : full PGNMF rerun to evaluate *exactly the same* masked
                       entries on the intersection of the two feature sets.

'no_pathway_all' changes the feature set. Consequently, only matched-common-
feature RMSE is directly comparable with Full. The ordinary test_rmse on each
variant's own feature set is NOT directly comparable across feature sets.
ARI/NMI can be compared on the same cells and the frozen Leiden protocol.

The program never touches the original stage2c20 results nor existing stage2c22
and stage2c23 results. Writes a manifest and checkpoints after each fit. On
restart with the same configuration, completed variants are skipped.

The optional combined table merges existing per-seed output tables from
stage2c22/23 with the new rows, without treating per-seed values as multi-seed
means. Prior original 'uniform_sparsity' (weight 1) is kept distinct from
'uniform_matched' (matched average L1 strength).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))  # maintained public PGNMF package
sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))

from pgnmf_revision.masking import MaskPlan
from pgnmf_revision.clustering import select_common_hvgs
from pgnmf_revision.pathway_sparsity import (
    get_gene_symbols, pathway_membership_from_gmt, make_two_level_penalty,
)
from pgnmf_revision.gene_graph_stage2b2 import build_pathway_rank_gene_graph
from pgnmf_revision.cell_graph_stage2b3 import (
    build_sparse_knn_cell_graph, knn_indices_from_expression,
    neighborhood_preservation,
)
from pgnmf.solver import zero_aware_nmf_mu_blocked

VARIANTS = ('baseline_nmf', 'no_pathway_all', 'full_common_refit')
METRICS = ('test_rmse','validation_rmse','common_test_rmse','common_validation_rmse',
           'ari','nmi','n_clusters','knn_overlap_vs_reference','xhat_zero_rms',
           'fit_seconds','n_iter')


def args_parser():
    p = argparse.ArgumentParser(description='Finish PGNMF C11 ablations: standard NMF + full pathway-removal + matched RMSE')
    p.add_argument('--dataset', default='C11')
    p.add_argument('--seeds', default='42')
    p.add_argument('--label-key', default='majority_voting')
    p.add_argument('--variants', default=','.join(VARIANTS), help='Comma-separated subset: '+','.join(VARIANTS))
    p.add_argument('--output-dir', default=None)
    p.add_argument('--gmt', default=None)
    p.add_argument('--gene-set-masks', default=None)
    p.add_argument('--gene-set-key', default='HVG1000+PATHWAY')
    p.add_argument('--gene-symbol-column', default=None)
    p.add_argument('--min-pathway-genes', type=int, default=10)
    p.add_argument('--max-pathway-genes', type=int, default=500)
    p.add_argument('--gene-edge-batch-size', type=int, default=256)
    p.add_argument('--resolution', type=float, default=None)
    p.add_argument('--rank', type=int, default=80)
    p.add_argument('--max-iter', type=int, default=100)
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--lambda-reg', type=float, default=0.001)
    p.add_argument('--alpha-cell', type=float, default=10)
    p.add_argument('--alpha-gene', type=float, default=2)
    p.add_argument('--beta', type=float, default=1)
    p.add_argument('--sparsity-strength', type=float, default=10)
    p.add_argument('--pathway-weight', type=float, default=0.5)
    p.add_argument('--nonpathway-weight', type=float, default=1)
    p.add_argument('--validation-seed', type=int, default=2026)
    p.add_argument('--validation-fraction', type=float, default=0.5)
    p.add_argument('--n-pcs', type=int, default=30)
    p.add_argument('--n-neighbors', type=int, default=15)
    p.add_argument('--metric', default='cosine')
    p.add_argument('--graph-pcs', type=int, default=30)
    p.add_argument('--graph-neighbors', type=int, default=15)
    p.add_argument('--graph-metric', default='euclidean')
    p.add_argument('--no-merge', action='store_true', help='Skip merging older stage2c22/23 tables')
    return p.parse_args()


def frozen_resolution(dataset, explicit):
    if explicit is not None:
        return float(explicit)
    f = ROOT / 'audit' / dataset / 'stage2c19_label_free_resolution' / 'stage2c19_selected_resolution.txt'
    if not f.exists():
        raise FileNotFoundError(f'Frozen resolution file absent: {f}; use --resolution')
    return float(f.read_text(encoding='utf-8-sig').strip())


def choose_gmt(explicit):
    p = Path(explicit) if explicit else ROOT / 'data' / 'c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt'
    if not p.is_file():
        raise FileNotFoundError(f'KEGG GMT missing: {p}; use --gmt')
    return p.resolve()


def dense32(x):
    return x.toarray().astype(np.float32, copy=False) if sparse.issparse(x) else np.asarray(x, dtype=np.float32)


def split_indices(n, fraction, seed):
    if n < 2:
        raise ValueError('Need at least two held-out observations')
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    nv = min(max(1,int(round(n*fraction))), n-1)
    return np.sort(order[:nv]),np.sort(order[nv:])


def load_gene_mask(path,key,n):
    with np.load(path,allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f'{key!r} not found in {path}; available={z.files}')
        m = np.asarray(z[key],dtype=bool)
    if m.shape != (n,):
        raise ValueError(f'Bad feature mask shape {m.shape}; expected {(n,)}')
    return m


def cluster_metrics(X,obs,label_key,resolution,seed,a):
    tmp=ad.AnnData(X=np.asarray(X,dtype=np.float32),obs=obs.copy())
    n_pcs=min(a.n_pcs,tmp.n_vars-1,tmp.n_obs-1)
    sc.tl.pca(tmp,n_comps=n_pcs,random_state=seed)
    sc.pp.neighbors(tmp,n_neighbors=min(a.n_neighbors,max(2,tmp.n_obs-1)),
                    n_pcs=n_pcs,use_rep='X_pca',metric=a.metric,random_state=seed)
    sc.tl.leiden(tmp,resolution=resolution,key_added='leiden',random_state=seed)
    labels=tmp.obs[label_key].astype(str).to_numpy()
    pred=tmp.obs['leiden'].astype(str).to_numpy()
    result=dict(ari=float(adjusted_rand_score(labels,pred)),
                nmi=float(normalized_mutual_info_score(labels,pred)),
                n_clusters=int(pd.Series(pred).nunique()))
    del tmp
    return result


def rmse(y,p):
    if not len(y): return float('nan')
    d=np.asarray(y,dtype=np.float64)-np.asarray(p,dtype=np.float64)
    return float(np.sqrt(np.mean(d*d)))


def features_from_masked(ad_masked, n):
    """Use exact project helper employed for masked-only HVG selection."""
    m=np.asarray(select_common_hvgs(ad_masked,n_top_genes=n),dtype=bool)
    if m.shape!=(ad_masked.n_vars,) or int(m.sum())!=n:
        raise RuntimeError(f'HVG selector returned {m.sum()} features instead of {n}')
    return m


def evaluate(X,Xref,mask,ad_masked,plan,truth_val_seed,truth_val_frac,
             common_mask, common_rows, common_true, common_val, common_test,
             seed,a,resolution,variant,fit):
    """Evaluate under the frozen DENSE_PRESERVE_OBS protocol."""
    local=np.full(ad_masked.n_vars,-1,dtype=np.int64)
    local[np.flatnonzero(mask)]=np.arange(int(mask.sum()),dtype=np.int64)
    keep=mask[np.asarray(plan.cols,dtype=np.int64)]
    rows=np.asarray(plan.rows[keep],dtype=np.int64)
    cols=local[np.asarray(plan.cols[keep],dtype=np.int64)]
    truth=np.asarray(plan.reference_values[keep],dtype=np.float64)
    vi,ti=split_indices(len(truth),truth_val_frac,truth_val_seed)
    zmask=X<=0
    Xhat=(fit.W@fit.H).astype(np.float32,copy=False)
    Xdense=X.copy()
    Xdense[zmask]=Xhat[zmask]
    pred=Xdense[rows,cols]
    refn=knn_indices_from_expression(Xref,n_pcs=a.n_pcs,
            n_neighbors=a.n_neighbors,metric=a.metric,random_state=seed)
    impn=knn_indices_from_expression(Xdense,n_pcs=a.n_pcs,
            n_neighbors=a.n_neighbors,metric=a.metric,random_state=seed)
    neigh=float(neighborhood_preservation(refn,impn)['mean_knn_overlap'])
    cl=cluster_metrics(Xdense,ad_masked.obs,a.label_key,resolution,seed,a)
    common_cols=local[np.asarray(plan.cols[common_mask],dtype=np.int64)]
    if np.any(common_cols < 0):
        raise RuntimeError('Common masked indices not present in model feature mask')
    cpred=np.asarray(Xdense[common_rows,common_cols],dtype=np.float64)
    vals=dict(test_rmse=rmse(truth[ti],pred[ti]),
              validation_rmse=rmse(truth[vi],pred[vi]),
              common_test_rmse=rmse(common_true[common_test],cpred[common_test]),
              common_validation_rmse=rmse(common_true[common_val],cpred[common_val]),
              knn_overlap_vs_reference=neigh,
              xhat_zero_rms=float(np.sqrt(np.mean(np.square(Xhat[zmask],dtype=np.float64)))),
              **cl)
    del Xhat,Xdense,impn,refn
    return vals


def atomic_save(out,records,seeds,variants):
    df=pd.DataFrame(list(records.values())).sort_values(['variant','seed'])
    fn=out/'ablation_per_seed.csv'
    tmp=out/'ablation_per_seed.tmp.csv'
    df.to_csv(tmp,index=False); tmp.replace(fn)
    summ=[]
    for v in variants:
        d=df[df.variant==v]
        if d.empty: continue
        row={'variant':v,'completed_seeds':len(d),'planned_seeds':len(seeds)}
        for m in METRICS:
            if m in d.columns:
                x=pd.to_numeric(d[m],errors='coerce').dropna().to_numpy(dtype=float)
                row[m+'_mean']=float(np.mean(x)) if len(x) else np.nan
                row[m+'_sd']=float(np.std(x,ddof=1)) if len(x)>1 else np.nan
        summ.append(row)
    sm=pd.DataFrame(summ)
    sm.to_csv(out/'ablation_summary_mean_sd.csv',index=False)
    cols=[x for x in ['variant','completed_seeds','test_rmse_mean','common_test_rmse_mean','ari_mean','nmi_mean'] if x in sm]
    print('\n=== SAVED PROGRESS ===\n'+sm[cols].to_string(index=False),flush=True)


def merge_prior(out, dataset, seed_list, new_records):
    """Merge only existing seed rows; no refitting or cherry-picking."""
    basedir=ROOT/'audit'/dataset
    sources=[
        basedir/'stage2c22_ablation_seed42'/'ablation_per_seed.csv',
        basedir/'stage2c23_pathway_sparsity_seed42'/'ablation_per_seed.csv',
        basedir/'stage2c24_baseline_nmf_seed42'/'ablation_per_seed.csv',
    ]
    merged={}
    for f in sources:
        if not f.exists():
            continue
        df=pd.read_csv(f)
        for row in df.to_dict(orient='records'):
            if int(row.get('seed',-1)) not in seed_list: continue
            name=str(row.get('variant',''))
            key=(int(row['seed']),name)
            if key not in merged:
                merged[key]=dict(row)
            else:
                prev=merged[key]
                for m in ('test_rmse','ari','nmi'):
                    if m in row and m in prev and pd.notna(row[m]) and pd.notna(prev[m]):
                        if not np.isclose(float(row[m]),float(prev[m]),atol=1e-5,rtol=0):
                            raise RuntimeError(f'Inconsistent repeated {name} seed={key[0]} {m}: {prev[m]} vs {row[m]}')
    for key,row in new_records.items():
        if key[1]=='full_common_refit':
            # Use this row to supply a matched-common RMSE for reused Full.
            fullkey=(key[0],'full')
            if fullkey in merged:
                for m in ('test_rmse','ari','nmi'):
                    if not np.isclose(float(row[m]),float(merged[fullkey][m]),atol=2e-5,rtol=0):
                        raise RuntimeError(f'Full PGNMF refit mismatch on {m}: {row[m]} vs {merged[fullkey][m]}')
                merged[fullkey]['common_test_rmse']=row['common_test_rmse']
                merged[fullkey]['common_validation_rmse']=row['common_validation_rmse']
            else:
                merged[fullkey]={**row,'variant':'full','source':'full_common_refit'}
        else:
            merged[key]=dict(row)
    # Include official Full result if not found in the previous stage2c22/23 CSVs.
    official=basedir/'stage2c20_final_label_free_protocol'/'stage2c8_graph_view_seed_results.csv'
    if official.is_file():
        df=pd.read_csv(official)
        sub=df[(df['view'].astype(str)=='DENSE_PRESERVE_OBS') &
               np.isclose(pd.to_numeric(df['alpha_cell']),10) &
               np.isclose(pd.to_numeric(df['alpha_gene']),2) &
               np.isclose(pd.to_numeric(df['beta']),1)]
        for seed in seed_list:
            key=(seed,'full')
            if key not in merged:
                row=sub[pd.to_numeric(sub['seed'])==seed]
                if len(row)==1:
                    rr=row.iloc[0]
                    merged[key]={'variant':'full','seed':seed,'test_rmse':float(rr['test_rmse']),
                          'ari':float(rr['ari']),'nmi':float(rr['nmi']),
                          'knn_overlap_vs_reference':float(rr['knn_overlap_vs_reference']),
                          'source':str(official)}
    # Do not include the technical 'full_common_refit' as an extra model.
    combined=pd.DataFrame(merged.values()).sort_values(['variant','seed']).reset_index(drop=True)
    if not combined.empty:
        combined['gene_universe']=np.where(combined['variant']=='no_pathway_all',
                                           'masked-only matched-size HVG',
                                           'HVG1000+PATHWAY')
        combined['rmse_comparability_note']=np.where(combined['variant']=='no_pathway_all',
              'Use common_test_rmse against Full; own-set test_rmse is NOT directly comparable',
              'Own-set test_rmse comparable with other frozen-gene-universe ablations')
        f=out/'C11_all_ablation_seed42_combined.csv'
        combined.to_csv(f,index=False)
        print('Combined historical + new single-seed table:',f,flush=True)
        focus=combined[['variant','test_rmse','common_test_rmse','ari','nmi']]
        print('\n=== ALL AVAILABLE ABLATION RESULTS ===\n'+focus.to_string(index=False),flush=True)


def main():
    a=args_parser()
    seeds=[int(s.strip()) for s in a.seeds.split(',') if s.strip()]
    if not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError('--seeds must be a unique comma-separated integer list')
    variants=[s.strip() for s in a.variants.split(',') if s.strip()]
    if len(variants)!=len(set(variants)) or not variants or any(v not in VARIANTS for v in variants):
        raise ValueError('Unsupported --variants; choices='+','.join(VARIANTS))
    res=frozen_resolution(a.dataset,a.resolution)
    work=ROOT/'work'/a.dataset
    paths={
        'masked':work/'masked_logcp10k.h5ad',
        'reference':work/'reference_logcp10k.h5ad',
        'plan':work/'mask_plan.npz',
        'gene_set_masks':Path(a.gene_set_masks) if a.gene_set_masks else ROOT/'audit'/a.dataset/'stage2cpre_gene_set_diagnostic'/'gene_set_masks.npz',
    }
    for k,p in paths.items():
        if not p.is_file(): raise FileNotFoundError(f'{k} not found: {p}')
    gmt=choose_gmt(a.gmt) if 'full_common_refit' in variants else None
    out=Path(a.output_dir) if a.output_dir else ROOT/'audit'/a.dataset/'stage2c25_C11_remaining_ablation_seed42'
    if not out.is_absolute(): out=(Path.cwd()/out).resolve()
    official=(ROOT/'audit'/a.dataset/'stage2c20_final_label_free_protocol').resolve()
    if out==official or official in out.parents:
        raise ValueError('Refusing to overwrite official benchmark results')
    out.mkdir(parents=True,exist_ok=True)
    config={'stage':'2C-25','dataset':a.dataset,'seeds':seeds,'variants':variants,
            'label_key':a.label_key,'resolution':res,
            'files':{k:str(v.resolve()) for k,v in paths.items()},
            'gmt':str(gmt) if gmt else None,'gene_set_key':a.gene_set_key,
            'rank':a.rank,'max_iter':a.max_iter,'tol':a.tol,'lambda_reg':a.lambda_reg,
            'alpha_cell':a.alpha_cell,'alpha_gene':a.alpha_gene,'beta':a.beta,
            'sparsity_strength':a.sparsity_strength,
            'pathway_weight':a.pathway_weight,'nonpathway_weight':a.nonpathway_weight,
            'validation_seed':a.validation_seed,'validation_fraction':a.validation_fraction,
            'n_pcs':a.n_pcs,'n_neighbors':a.n_neighbors,'metric':a.metric,
            'graph_pcs':a.graph_pcs,'graph_neighbors':a.graph_neighbors,'graph_metric':a.graph_metric,
            'min_pathway_genes':a.min_pathway_genes,'max_pathway_genes':a.max_pathway_genes,
            'gene_edge_batch_size':a.gene_edge_batch_size,'gene_symbol_column':a.gene_symbol_column}
    manifest=out/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf8'))
        if old!=config:
            raise ValueError('Existing output manifest differs from this run; pick a fresh --output-dir')
    else:
        if (out/'ablation_per_seed.csv').exists():
            raise RuntimeError('Existing CSV without manifest: use new output dir')
        manifest.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    saved={}
    previous=out/'ablation_per_seed.csv'
    if previous.is_file():
        for row in pd.read_csv(previous).to_dict(orient='records'):
            saved[(int(row['seed']),str(row['variant']))]=row
    todo=[(seed,v) for seed in seeds for v in variants if (seed,v) not in saved]
    print('\n=== PGNMF STAGE2C25 REMAINING C11 ABLATIONS ===',flush=True)
    print('dataset=',a.dataset,'seeds=',seeds,'label=',a.label_key,'resolution=',res,flush=True)
    print('variants=',variants,'remaining=',len(todo),'max_iter=',a.max_iter,flush=True)
    print('output=',out,flush=True)
    if not todo:
        print('All selected variants already completed. No refitting.',flush=True)
        if not a.no_merge: merge_prior(out,a.dataset,seeds,saved)
        return

    masked=sc.read_h5ad(paths['masked']); reference=sc.read_h5ad(paths['reference'])
    if masked.shape!=reference.shape or not np.array_equal(masked.var_names,reference.var_names):
        raise ValueError('Input and reference shapes/gene order differ')
    plan=MaskPlan.load(paths['plan'])
    if tuple(plan.shape)!=tuple(masked.shape):
        raise ValueError('mask_plan.shape differs from inputs')
    if a.label_key not in masked.obs:
        raise KeyError(f'{a.label_key} not found; available={list(masked.obs.columns)}')
    fullmask=load_gene_mask(paths['gene_set_masks'],a.gene_set_key,masked.n_vars)
    nfull=int(fullmask.sum())
    print('Selecting',nfull,'top HVGs using project select_common_hvgs() on MASKED input only ...',flush=True)
    hvgmask=features_from_masked(masked,nfull)
    overlap=fullmask & hvgmask
    info={'n_full':nfull,'n_HVG_only':int(hvgmask.sum()),'n_overlap':int(overlap.sum()),
          'overlap_fraction_of_full':float(np.mean(hvgmask[fullmask])),
          'full_selection':'HVG1000+PATHWAY','no_pathway_selection':'top matched-size HVGs on masked input only',
          'note':'common test/validation splits are identical between no_pathway_all and full_common_refit'}
    print('Feature universe:',info,flush=True)
    (out/'feature_overlap_summary.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    pd.DataFrame({'gene_name':masked.var_names.astype(str),
                  'frozen_full':fullmask,'HVG_only':hvgmask,'common':overlap}).to_csv(
                     out/'feature_selection_comparison.csv',index=False)
    if int(overlap.sum())<2:
        raise RuntimeError('Too few common genes to calculate common RMSE')

    # This is intentionally a single fixed, shared split in the original
    # mask plan. Do not select the overlap after inspecting model performance.
    plan_cols=np.asarray(plan.cols,dtype=np.int64)
    common_mask=overlap[plan_cols]
    common_rows=np.asarray(plan.rows[common_mask],dtype=np.int64)
    common_truth=np.asarray(plan.reference_values[common_mask],dtype=np.float64)
    common_val,common_test=split_indices(len(common_truth),a.validation_fraction,a.validation_seed)
    print(f'Common masked positions={len(common_truth)}; common test={len(common_test)}',flush=True)

    # For reproducibility and independent resume of each variant, use a fresh
    # unchanged feature matrix for each distinct universe.
    for seed in seeds:
        pending=[v for v in variants if (seed,v) not in saved]
        if not pending: continue
        for variant in pending:
            is_hvg=variant=='no_pathway_all'
            featmask=hvgmask if is_hvg else fullmask
            X=dense32(masked[:,featmask].X)
            Xref=dense32(reference[:,featmask].X)
            nzero=int(np.count_nonzero(X<=0)); npos=int(np.count_nonzero(X>0))
            if not nzero or not npos: raise ValueError('All-positive or all-zero selected matrix')
            q0=a.beta*npos/nzero
            ncell,ngene=X.shape
            Sc=sparse.csr_matrix((ncell,ncell),dtype=np.float32)
            dc=np.zeros(ncell,dtype=np.float32)
            Sg=sparse.csr_matrix((ngene,ngene),dtype=np.float32)
            dg=np.zeros(ngene,dtype=np.float32)
            if variant=='baseline_nmf':
                ac=ag=reg=0.0; penalty=np.zeros(ngene,dtype=np.float32); q0=1.0
            elif variant=='no_pathway_all':
                ac=a.alpha_cell;ag=0.0;reg=a.lambda_reg
                # STRICT NO KEGG: uniform penalty independent of pathway labels.
                penalty=np.full(ngene,a.sparsity_strength*a.nonpathway_weight,dtype=np.float32)
            elif variant=='full_common_refit':
                ac=a.alpha_cell;ag=a.alpha_gene;reg=a.lambda_reg
                gene_symbols,source=get_gene_symbols(masked,a.gene_symbol_column)
                p_all,_=pathway_membership_from_gmt(str(gmt),gene_symbols,
                      min_pathway_genes=a.min_pathway_genes,
                      max_pathway_genes=a.max_pathway_genes,gene_symbol_source=source)
                penalty=make_two_level_penalty(np.asarray(p_all[fullmask],dtype=bool),
                     strength=a.sparsity_strength,pathway_weight=a.pathway_weight,
                     nonpathway_weight=a.nonpathway_weight).astype(np.float32,copy=False)
                print('Building frozen KEGG gene graph for FULL matched-RMSE refit ...',flush=True)
                Sg_raw,dg_raw,_,_=build_pathway_rank_gene_graph(
                    X,gene_symbols,gene_symbols[fullmask],str(gmt),
                    min_pathway_genes=a.min_pathway_genes,
                    max_pathway_genes=a.max_pathway_genes,
                    edge_batch_size=a.gene_edge_batch_size)
                deg=np.asarray(dg_raw)[np.asarray(dg_raw)>0]
                if not len(deg):raise RuntimeError('Empty Full gene graph')
                scale=float(np.median(deg))
                Sg=(Sg_raw/np.float32(scale)).tocsr()
                dg=np.asarray(dg_raw,dtype=np.float32)/np.float32(scale)
                del Sg_raw,dg_raw,deg
            else:
                raise ValueError(variant)
            if ac>0:
                Sc,dc,_,_,_=build_sparse_knn_cell_graph(X,n_pcs=a.graph_pcs,
                    n_neighbors=a.graph_neighbors,metric=a.graph_metric,random_state=seed)
            if not 0<=q0<=1: raise ValueError('Zero weight out of range')
            print(f'\n--- {variant} seed={seed} X={X.shape} q0={q0:.8g} ac={ac} ag={ag} sparse_mean={float(np.mean(penalty)):.7g} ---',flush=True)
            fit=zero_aware_nmf_mu_blocked(X,S_cell=Sc,degree_cell=dc,alpha_cell=ac,
                S_gene=Sg,degree_gene=dg,alpha_gene=ag,lambda_h_gene=penalty,
                zero_weight=q0,rank=a.rank,max_iter=a.max_iter,tol=a.tol,
                random_state=seed,lambda_reg=reg,lambda_w_l1=0.0,verbose=False,
                tag=f'Stage2C25-{variant}-{seed}', block_size=500)
            met=evaluate(X,Xref,featmask,masked,plan,a.validation_seed,
                a.validation_fraction,common_mask,common_rows,common_truth,
                common_val,common_test,seed,a,res,variant,fit)
            result={'variant':variant,'seed':int(seed),'dataset':a.dataset,
                'feature_set':'HVG_only_matched_size' if is_hvg else 'HVG1000+PATHWAY',
                'n_model_genes':ngene,'n_common_genes':int(overlap.sum()),
                'n_common_masked_test':int(len(common_test)),
                'label_key':a.label_key,'resolution':float(res),
                'view':'DENSE_PRESERVE_OBS',**met,
                'zero_weight_effective':float(q0),'alpha_cell':ac,'alpha_gene':ag,
                'lambda_reg_effective':reg,'lambda_h_mean':float(np.mean(penalty)),
                'fit_seconds':float(fit.elapsed_seconds),'n_iter':int(fit.n_iter),
                'converged':bool(fit.converged),'source':'stage2c25_new_fitting'}
            saved[(seed,variant)]=result
            print(f'RESULT {variant} RMSE_own={met["test_rmse"]:.6f} '
                  f'RMSE_common={met["common_test_rmse"]:.6f} ARI={met["ari"]:.5f} '
                  f'NMI={met["nmi"]:.5f} kNN={met["knn_overlap_vs_reference"]:.5f}',flush=True)
            atomic_save(out,saved,seeds,variants)
            del X,Xref,Sc,dc,Sg,dg,penalty,fit
            gc.collect()

    if not a.no_merge: merge_prior(out,a.dataset,seeds,saved)
    print('\n=== COMPLETE ===',flush=True)
    print('New per-seed:',out/'ablation_per_seed.csv',flush=True)
    print('Combined all historical + new:',out/'C11_all_ablation_seed42_combined.csv',flush=True)
    print('Feature selection:',out/'feature_selection_comparison.csv',flush=True)
    print('IMPORTANT: compare no_pathway_all and full only on common_test_rmse; not on own-set RMSE.',flush=True)


if __name__=='__main__':
    main()
