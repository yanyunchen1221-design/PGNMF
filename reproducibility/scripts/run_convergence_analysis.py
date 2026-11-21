#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PGNMF final-model convergence diagnostic, Stage 2C-21.

Drop this ONE file into reproducibility/scripts/ and execute from the
project root. It reuses the original model-gene universe, masked/reference
inputs, mask plan, pathway-rank gene graph and frozen downstream Leiden
resolution. DOES NOT alter original source files or benchmark outputs.

The maintained exact gene-block solver is used for the convergence trace;
the ONLY addition is a post-update observer that records test RMSE/ARI/NMI.
Observation of metrics does not change W/H, training, or stopping.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))  # maintained public PGNMF package
sys.path.insert(0, str(ROOT / "reproducibility" / "legacy"))

from pgnmf_revision.masking import MaskPlan
from pgnmf_revision.pathway_sparsity import (
    get_gene_symbols, pathway_membership_from_gmt, make_two_level_penalty,
)
from pgnmf_revision.gene_graph_stage2b2 import build_pathway_rank_gene_graph
from pgnmf_revision.cell_graph_stage2b3 import build_sparse_knn_cell_graph
from pgnmf.solver import zero_aware_nmf_mu_blocked

def dense32(x):
    return x.toarray().astype(np.float32, copy=False) if sparse.issparse(x) else np.asarray(x, dtype=np.float32)


def choose_gmt(gmt):
    if gmt:
        p=Path(gmt)
        if not p.is_file():
            raise FileNotFoundError(f"GMT does not exist: {p}")
        return p.resolve()
    candidates = sorted(p for p in ROOT.rglob("*.gmt") if p.is_file())
    candidates = [p for p in candidates if "kegg" in p.name.lower()] or candidates
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("No GMT found in project. Pass --gmt with your original KEGG GMT file path.")
    raise ValueError("Multiple GMT files found. Pass --gmt explicitly:\n" + "\n".join(map(str,candidates[:30])))


def read_resolution(dataset, explicit):
    if explicit is not None:
        return float(explicit)
    p=ROOT/"audit"/dataset/"stage2c19_label_free_resolution"/"stage2c19_selected_resolution.txt"
    if not p.is_file():
        raise FileNotFoundError(f"Frozen resolution file missing: {p}. Pass --resolution explicitly (C11=0.30)")
    return float(p.read_text(encoding="utf-8-sig").strip())


def load_gene_mask(path, key, n):
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"Gene set {key!r} not found in {path}, keys={z.files}")
        mask=np.asarray(z[key],dtype=bool)
    if mask.shape!=(n,):
        raise ValueError(f"Gene set shape {mask.shape} != ({n},)")
    return mask


def split_indices(n, frac, seed):
    if n < 2:
        raise ValueError("Need 2+ held-out entries")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = min(max(1, int(round(n * frac))), n-1)
    return np.sort(order[:n_val]), np.sort(order[n_val:])


def leiden_metrics(X, obs, key, *, resolution, seed, n_pcs=30, n_neighbors=15, metric="cosine"):
    tmp=ad.AnnData(X=np.asarray(X,dtype=np.float32),obs=obs.copy())
    n_pc=min(int(n_pcs),tmp.n_vars-1,tmp.n_obs-1)
    sc.tl.pca(tmp,n_comps=n_pc,random_state=seed)
    sc.pp.neighbors(tmp,n_neighbors=min(n_neighbors,max(2,tmp.n_obs-1)),
                    n_pcs=n_pc,use_rep="X_pca",metric=metric,random_state=seed)
    sc.tl.leiden(tmp,resolution=resolution,key_added="leiden",random_state=seed)
    pred=tmp.obs["leiden"].astype(str).to_numpy()
    labels=tmp.obs[key].astype(str).to_numpy()
    return (float(adjusted_rand_score(labels,pred)),
            float(normalized_mutual_info_score(labels,pred)))


def parse_args():
    p=argparse.ArgumentParser(description="Final PGNMF 300-iteration convergence diagnostic")
    p.add_argument("--dataset",default="C11",help="e.g. C11 or SCIB")
    p.add_argument("--gmt",default=None,help="original KEGG GMT; auto-detect if unique")
    p.add_argument("--resolution",type=float,default=None,help="auto-read frozen stage2c19 resolution")
    p.add_argument("--output-dir",default=None)
    p.add_argument("--label-key",default="celltype")
    p.add_argument("--gene-symbol-column",default=None)
    p.add_argument("--gene-set-key",default="HVG1000+PATHWAY")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--rank",type=int,default=80)
    p.add_argument("--max-iter",type=int,default=300)
    p.add_argument("--tol",type=float,default=1e-4)
    p.add_argument("--alpha-cell",type=float,default=10.0)
    p.add_argument("--alpha-gene",type=float,default=2.0)
    p.add_argument("--beta",type=float,default=1.0)
    p.add_argument("--lambda-reg",type=float,default=1e-3)
    p.add_argument("--sparsity-strength",type=float,default=10.0)
    p.add_argument("--pathway-weight",type=float,default=0.5)
    p.add_argument("--nonpathway-weight",type=float,default=1.0)
    p.add_argument("--zero-weight-mode",choices=["balanced","per_entry"],default="balanced")
    p.add_argument("--min-pathway-genes",type=int,default=10)
    p.add_argument("--max-pathway-genes",type=int,default=500)
    p.add_argument("--gene-edge-batch-size",type=int,default=256)
    p.add_argument("--graph-pcs",type=int,default=30)
    p.add_argument("--graph-neighbors",type=int,default=15)
    p.add_argument("--graph-metric",default="euclidean")
    p.add_argument("--n-pcs",type=int,default=30)
    p.add_argument("--n-neighbors",type=int,default=15)
    p.add_argument("--metric",default="cosine")
    p.add_argument("--validation-seed",type=int,default=2026)
    p.add_argument("--validation-fraction",type=float,default=0.5)
    p.add_argument("--checkpoints",default="1,10,20,30,40,50,60,80,100,120,150,180,200,250,300")
    p.add_argument("--no-clustering-at-checkpoints",action="store_true",
                   help="faster diagnostic: only evaluate RMSE, skip ARI/NMI checkpoints")
    return p.parse_args()


def make_figure(lossdf, metricdf, outfile, dataset, tol):
    fig,axes=plt.subplots(1,3,figsize=(15.6,4.5),dpi=180)
    ax=axes[0]
    ax.plot(lossdf.iteration,lossdf.objective,color="#0000FF",lw=2)
    ax.set_yscale("log")
    ax.set(xlabel="Iteration",ylabel="Full objective",title="(a) Objective")
    ax.grid(alpha=.25,ls="--")
    ax=axes[1]
    a=lossdf.dropna(subset=["relative_change"])
    ax.plot(a.iteration,a.relative_change,color="#7E57C2",lw=2)
    ax.axhline(tol,color="#CC3333",ls="--",lw=1.5,label=f"Tolerance={tol:g}")
    ax.set_yscale("log")
    ax.set(xlabel="Iteration",ylabel="Relative objective change",title="(b) Stopping criterion")
    ax.grid(alpha=.25,ls="--"); ax.legend(fontsize=9)
    ax=axes[2]
    ax.plot(metricdf.iteration,metricdf.test_rmse,color="#FFA500",marker="o",lw=1.8,label="Test RMSE")
    if "ari" in metricdf and metricdf.ari.notna().any():
        ax.plot(metricdf.iteration,metricdf.ari,color="#FF0000",marker="^",lw=1.6,label="ARI")
        ax.plot(metricdf.iteration,metricdf.nmi,color="#008000",marker="x",lw=1.6,label="NMI")
    ax.set(xlabel="Iteration",ylabel="Metric value",title="(c) Held-out/downstream metrics")
    ax.grid(alpha=.25,ls="--"); ax.legend(fontsize=9)
    fig.suptitle(f"PGNMF convergence diagnostic — {dataset} (seed 42)",fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile,dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    a=parse_args()
    D=a.dataset
    work=ROOT/"work"/D
    masked=work/"masked_logcp10k.h5ad"
    reference=work/"reference_logcp10k.h5ad"
    mask_plan=work/"mask_plan.npz"
    geneset=ROOT/"audit"/D/"stage2cpre_gene_set_diagnostic"/"gene_set_masks.npz"
    required=[masked,reference,mask_plan,geneset]
    missing=[str(x) for x in required if not x.is_file()]
    if missing:
        raise FileNotFoundError("Missing required PGNMF inputs:\n" + "\n".join(missing))
    gmt=choose_gmt(a.gmt)
    resolution=read_resolution(D,a.resolution)
    out=Path(a.output_dir) if a.output_dir else ROOT/"audit"/D/"stage2c21_convergence_300"
    if out.resolve() == (ROOT/"audit"/D/"stage2c20_final_label_free_protocol").resolve():
        raise ValueError("Refusing to overwrite official stage2c20 benchmark results")
    out.mkdir(parents=True,exist_ok=True)
    print(f"\n=== PGNMF CONVERGENCE DIAGNOSTIC ===",flush=True)
    print(f"dataset={D} seed={a.seed} max_iter={a.max_iter} tol={a.tol}",flush=True)
    print(f"masked={masked}\nreference={reference}\nGMT={gmt}\nresolution={resolution}",flush=True)
    print(f"output={out}",flush=True)
    ad_masked=sc.read_h5ad(masked)
    ad_ref=sc.read_h5ad(reference)
    plan=MaskPlan.load(mask_plan)
    if ad_masked.shape!=ad_ref.shape or tuple(plan.shape)!=tuple(ad_masked.shape):
        raise ValueError("masked/reference/mask_plan shapes disagree")
    if not np.array_equal(ad_masked.var_names.astype(str),ad_ref.var_names.astype(str)):
        raise ValueError("gene ordering mismatch")
    if a.label_key not in ad_masked.obs.columns and not a.no_clustering_at_checkpoints:
        raise KeyError(f"label key {a.label_key!r} missing; pass --label-key or --no-clustering-at-checkpoints")
    gene_mask=load_gene_mask(geneset,a.gene_set_key,ad_masked.n_vars)
    model_idx=np.flatnonzero(gene_mask)
    X=dense32(ad_masked[:,gene_mask].X)
    symbols_all,symbol_source=get_gene_symbols(ad_masked,a.gene_symbol_column)
    pathway_all,_=pathway_membership_from_gmt(
        str(gmt),symbols_all,min_pathway_genes=a.min_pathway_genes,
        max_pathway_genes=a.max_pathway_genes,gene_symbol_source=symbol_source)
    pathway_model=np.asarray(pathway_all[gene_mask],dtype=bool)
    penalty=make_two_level_penalty(
        pathway_model,strength=a.sparsity_strength,
        pathway_weight=a.pathway_weight,nonpathway_weight=a.nonpathway_weight).astype(np.float32,copy=False)
    global_to_local=np.full(ad_masked.n_vars,-1,dtype=np.int64)
    global_to_local[model_idx]=np.arange(model_idx.size,dtype=np.int64)
    keep=gene_mask[plan.cols]
    held_rows=np.asarray(plan.rows[keep],dtype=np.int64)
    held_cols=global_to_local[np.asarray(plan.cols[keep],dtype=np.int64)]
    held_truth=np.asarray(plan.reference_values[keep],dtype=np.float64)
    if len(held_truth)==0:
        raise ValueError("No held-out entries in model-gene universe")
    validation_idx,test_idx=split_indices(len(held_truth),a.validation_fraction,a.validation_seed)
    observed=X>0
    n_pos=int(observed.sum());n_zero=int((~observed).sum())
    q0=float(a.beta*n_pos/n_zero) if a.zero_weight_mode=="balanced" else float(a.beta)
    if not (0<=q0<=1):
        raise ValueError(f"q0 must be in [0,1], got {q0}")
    print(f"X_model={X.shape}, heldout={len(held_truth)}, test={len(test_idx)}, q0={q0:.8g}",flush=True)

    print("Building pathway-rank gene graph...",flush=True)
    Sg_raw,dg_raw,_,_=build_pathway_rank_gene_graph(
        X,symbols_all,symbols_all[gene_mask],str(gmt),
        min_pathway_genes=a.min_pathway_genes,max_pathway_genes=a.max_pathway_genes,
        edge_batch_size=a.gene_edge_batch_size)
    pos_deg=np.asarray(dg_raw)[np.asarray(dg_raw)>0]
    if pos_deg.size==0:
        raise RuntimeError("Gene graph has no positive degree")
    graph_scale=float(np.median(pos_deg))
    Sg=(Sg_raw/np.float32(graph_scale)).tocsr()
    dg=np.asarray(dg_raw,dtype=np.float32)/np.float32(graph_scale)
    print("Building union-kNN Euclidean cell graph...",flush=True)
    Sc,dc,_,_,_=build_sparse_knn_cell_graph(
        X,n_pcs=a.graph_pcs,n_neighbors=a.graph_neighbors,
        metric=a.graph_metric,random_state=a.seed)

    checkpoints={int(s) for s in a.checkpoints.split(",") if s.strip()}
    checkpoints={t for t in checkpoints if 1<=t<=a.max_iter}
    checkpoint_rows=[]

    def observe_iteration(it,loss,W,H):
        if it not in checkpoints:
            return
        # Exactly like original Stage2C8 "DENSE_PRESERVE_OBS": keep observed X.
        xhat=np.asarray(W@H,dtype=np.float32)
        xdense=X.copy()
        zero=~observed
        xdense[zero]=xhat[zero]
        pred=np.asarray(xdense[held_rows,held_cols],dtype=np.float64)
        rmse=float(np.sqrt(np.mean((pred[test_idx]-held_truth[test_idx])**2)))
        ari=nmi=float("nan")
        if not a.no_clustering_at_checkpoints:
            ari,nmi=leiden_metrics(
                xdense,ad_masked.obs,a.label_key,resolution=resolution,
                seed=a.seed,n_pcs=a.n_pcs,n_neighbors=a.n_neighbors,metric=a.metric)
        checkpoint_rows.append({"iteration":it,"objective":float(loss),
                                "test_rmse":rmse,"ari":ari,"nmi":nmi})
        print(f"  checkpoint={it:3d}: test_RMSE={rmse:.6f} ARI={ari:.5f} NMI={nmi:.5f}",flush=True)

    print("Starting traced PGNMF fitting...",flush=True)
    t0=time.perf_counter()
    result=zero_aware_nmf_mu_blocked(
        X,S_cell=Sc,degree_cell=dc,alpha_cell=a.alpha_cell,
        S_gene=Sg,degree_gene=dg,alpha_gene=a.alpha_gene,
        lambda_h_gene=penalty,zero_weight=q0,rank=a.rank,
        max_iter=a.max_iter,tol=a.tol,random_state=a.seed,
        lambda_reg=a.lambda_reg,lambda_w_l1=0.0,
        verbose=True,tag=f"PGNMF-{D}",block_size=500,iteration_callback=observe_iteration)
    if not checkpoint_rows or checkpoint_rows[-1]["iteration"] != result.n_iter:
        observe_iteration_at_final = result.n_iter
        checkpoints.add(result.n_iter)
        observe_iteration(result.n_iter,result.losses[-1],result.W,result.H)
    objective=np.asarray(result.losses,dtype=np.float64)
    rel=np.full(len(objective),np.nan)
    rel[1:]=np.abs(np.diff(objective))/np.maximum(np.abs(objective[:-1]),1e-12)
    lossdf=pd.DataFrame({"iteration":np.arange(1,len(objective)+1),
                         "objective":objective,"relative_change":rel})
    metricdf=pd.DataFrame(checkpoint_rows).sort_values("iteration")
    lossdf.to_csv(out/"convergence_objective_trace.csv",index=False)
    metricdf.to_csv(out/"convergence_checkpoint_metrics.csv",index=False)
    last_rel=float(rel[-1]) if len(rel)>1 else float("nan")
    summary={
        "dataset":D,"model_seed":a.seed,"rank":a.rank,"max_iter":a.max_iter,
        "tol":a.tol,"n_iter":result.n_iter,"converged":result.converged,
        "initial_loss":float(objective[0]),"final_loss":float(objective[-1]),
        "last_relative_change":last_rel,
        "loss_nonincreasing":bool(np.all(np.diff(objective)<=1e-6*np.abs(objective[:-1])+1e-7)),
        "fit_seconds":float(result.elapsed_seconds),"wall_seconds":time.perf_counter()-t0,
        "effective_zero_weight":q0,"alpha_cell":a.alpha_cell,"alpha_gene":a.alpha_gene,
        "resolution":resolution,"gene_set_key":a.gene_set_key,"n_model_genes":int(model_idx.size),
        "checkpoints":sorted(checkpoints),"gmt":str(gmt),
    }
    for t in [50,100,150,200,250,300]:
        if result.n_iter>=t:
            summary[f"objective_iter{t}"]=float(objective[t-1])
            summary[f"relative_change_iter{t}"]=float(rel[t-1])
    if result.n_iter>=200:
        summary["loss_reduction_iter100_to200"] = float((objective[99]-objective[199])/objective[99])
    if result.n_iter>=300:
        summary["loss_reduction_iter200_to300"] = float((objective[199]-objective[299])/objective[199])
    (out/"convergence_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    make_figure(lossdf,metricdf,out/"convergence_figure6_draft.png",D,a.tol)
    print("\n====== RESULTS ======",flush=True)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    print("\nSaved:\n "+str(out/"convergence_objective_trace.csv")+
          "\n "+str(out/"convergence_checkpoint_metrics.csv")+
          "\n "+str(out/"convergence_summary.json")+
          "\n "+str(out/"convergence_figure6_draft.png"),flush=True)


if __name__=="__main__":
    main()
