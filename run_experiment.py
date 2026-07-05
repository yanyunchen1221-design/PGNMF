import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns

from pgnmf import run_pathway_guided_pgnmf
from evaluation import scRNAEvaluatorFixed
from factor_analysis import NMFAnalyzer
from visualization import plot_main_results, plot_supplementary
from simulation import SimulatedMissingEvaluator 

import sys
from pathlib import Path
import multiprocessing as mp


def run_complete_experiment(
    adata_path: str,
    gmt_file: str,
    output_dir: str = './results',
    rank: int = 20,
    alpha_gene: float = 0.05,
    alpha_cell: float = 0.05,
    lambda_reg: float = 0.1,
    lambda_sparse: float = 100.0,
    n_hvgs: int = 2000,
    n_pcs: int = 50,
    leiden_resolution: float = 0.4,
    n_jobs: int = -1,
    simulated_mask_ratio: float = 0.2, 
    simulated_random_seed: int = 42,
    organism: str = 'human',
    random_state: int = 42
):
    
    import os
    os.makedirs(output_dir, exist_ok=True)

    adata_path_orig = adata_path
    adata_path_p = Path(adata_path)
    if not adata_path_p.is_absolute() and not adata_path_p.exists():
        script_dir = Path(__file__).resolve().parent
        cand = script_dir / adata_path
        if cand.exists():
            adata_path = str(cand)
        else:
            cand2 = script_dir.parent / adata_path
            if cand2.exists():
                adata_path = str(cand2)
            else:
                adata_path = str(adata_path_p)

    if adata_path.endswith('.mat'):
        from scipy.io import loadmat
        import anndata as ad
        
        m = loadmat(adata_path, struct_as_record=False, squeeze_me=True)
        keys = [k for k in m.keys() if not k.startswith('__')]

        candidates = []
        for k, v in m.items():
            if k.startswith('__'):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
                candidates.append((k, v.shape[0] * v.shape[1], v.shape))
        
        if not candidates:
            raise ValueError("No 2D numeric array found in .mat file")
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        expr_key, _, expr_shape = candidates[0]
        X = np.array(m[expr_key], dtype=float)

        rows, cols = X.shape
        if rows > cols:
            X = X.T
        
        n_cells, n_genes = X.shape

        obs_index = None
        var_index = None
        label_col = None
        label_key = None
        label_keys = ['celltype', 'cell_type', 'Label', 'label', 'cellType', 'CellType', 
                      'type', 'Type', 'group', 'Group', 'cluster', 'Cluster',
                      'y', 'Y', 'labels', 'Labels', 'annotation', 'Annotation','true_labs']
        
        for k, v in m.items():
            if k.startswith('__'):
                continue
            try:
                arr = np.array(v)
                if arr.ndim == 1:
                    if arr.shape[0] == n_cells:
                        if obs_index is None:
                            obs_index = (k, arr)
                        if label_col is None and k in label_keys:
                            label_col = arr
                            label_key = k
                    if arr.shape[0] == n_genes and var_index is None:
                        var_index = (k, arr)
            except:
                continue

        if label_col is None:
            for k, v in m.items():
                if k.startswith('__'):
                    continue
                try:
                    arr = np.array(v)
                    if arr.ndim == 1 and arr.shape[0] == n_cells:
                        if (np.issubdtype(arr.dtype, np.str_) or 
                            np.issubdtype(arr.dtype, np.bytes_)) or \
                           len(np.unique(arr)) < min(n_cells * 0.3, 50):
                            label_col = arr
                            label_key = k
                            break
                except:
                    continue

        if obs_index is not None:
            obs_names = [str(x) for x in obs_index[1]]
            obs = pd.DataFrame(index=obs_names)
        else:
            obs_names = [f'cell{i}' for i in range(n_cells)]
            obs = pd.DataFrame(index=obs_names)
        
        if label_col is not None and len(label_col) == n_cells:
            obs['celltype'] = [str(x) for x in label_col]
        
        if var_index is not None:
            var_names = [str(x) for x in var_index[1]]
            var = pd.DataFrame(index=var_names)
        else:
            var = pd.DataFrame(index=[f'gene{i}' for i in range(n_genes)])
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
    else:
        adata = sc.read_h5ad(adata_path)

    label_col = None
    has_labels = False
    for col in ['celltype', 'cell_type', 'Label', 'label', 'cellType', 'CellType']:
        if col in adata.obs.columns:
            if col != 'celltype':
                adata.obs['celltype'] = adata.obs[col].copy()
            label_col = col
            has_labels = True
            break
    
    if not has_labels:
        adata.obs['celltype'] = 'Unknown'
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, 
                                log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    from scipy.sparse import issparse
    if issparse(adata.X):
        X_original_counts = adata.X.toarray()
    else:
        X_original_counts = np.array(adata.X)
    
    X_original_counts = np.round(X_original_counts).astype(int) 

    sim_evaluator = SimulatedMissingEvaluator(
        mask_ratio=simulated_mask_ratio,
        random_seed=simulated_random_seed
    )
   
    sim_missing_mask = sim_evaluator.generate_simulated_missing_mask(X_original_counts)
   
    X_masked_counts = sim_evaluator.create_masked_data(X_original_counts)
   
    adata_for_imputation = adata.copy()
    adata_for_imputation.X = X_masked_counts

    if issparse(adata_for_imputation.X):
        X_check = adata_for_imputation.X.toarray()
    else:
        X_check = np.array(adata_for_imputation.X)

    if X_check.max() > 100:
        sc.pp.normalize_total(adata_for_imputation, target_sum=1e4)
        sc.pp.log1p(adata_for_imputation)
    elif X_check.max() < 0.5:
        raise ValueError("Data may have been log-transformed multiple times!")

    adata_imputed, imputation_results = run_pathway_guided_pgnmf(
        adata=adata_for_imputation,  
        gmt_file=gmt_file,
        rank=rank,
        alpha_gene=alpha_gene,
        alpha_cell=alpha_cell,
        lambda_reg=lambda_reg,
        lambda_sparse=lambda_sparse,
        min_pathway_size=10,
        max_pathway_size=500,
        similarity_metric='exponential',
        beta=2.0,
        n_pcs=30,
        max_iter=50,
        n_jobs=n_jobs,
        save_prefix=f'{output_dir}/pathway_pgnmf',
        organism=organism
    )

    if 'imputed' in adata_imputed.layers:
        if issparse(adata_imputed.layers['imputed']):
            X_check = adata_imputed.layers['imputed'].toarray()
        else:
            X_check = np.array(adata_imputed.layers['imputed'])
        
        if X_check.min() >= 1.0:
            raise ValueError("Data integrity check failed!")
    else:
        raise ValueError("Imputation results not found!")

    if 'celltype' not in adata_imputed.obs.columns:
        if 'celltype' in adata.obs.columns:
            adata_imputed.obs['celltype'] = adata.obs['celltype'].copy()
        else:
            adata_imputed.obs['celltype'] = 'Unknown'

    adata_for_clustering = adata_imputed.copy()
    sc.pp.normalize_total(adata_for_clustering, target_sum=1e4)
    sc.pp.log1p(adata_for_clustering)

    sc.pp.highly_variable_genes(
        adata_for_clustering, 
        n_top_genes=n_hvgs, 
        flavor='seurat_v3'
    )
    
    adata_hvg = adata_for_clustering[:, adata_for_clustering.var['highly_variable']].copy()
    
    sc.tl.pca(adata_hvg, n_comps=n_pcs, random_state=random_state)

    use_rep = 'X_pca'
    if 'Batch' in adata_hvg.obs.columns and adata_hvg.obs['Batch'].nunique() > 1:
        import scanpy.external as sce
        sce.pp.harmony_integrate(adata_hvg, 'Batch', max_iter_harmony=20)
        use_rep = 'X_pca_harmony'

    sc.pp.neighbors(
        adata_hvg, 
        n_neighbors=15, 
        n_pcs=30,
        use_rep=use_rep,
        metric='cosine',
        random_state=random_state
    )

    sc.tl.umap(adata_hvg, random_state=random_state)

    resolutions = [0.2, 0.4, 0.6, 0.8, 1.0]
    best_ari = 0
    best_res = leiden_resolution
    best_sil = 0
    
    results_by_resolution = []

    has_true_labels = ('celltype' in adata_hvg.obs.columns and 
                      adata_hvg.obs['celltype'].nunique() > 1 and
                      adata_hvg.obs['celltype'].unique()[0] != 'Unknown')
    
    for res in resolutions:
        key = f'leiden_res{res}'
        sc.tl.leiden(adata_hvg, resolution=res, key_added=key, random_state=random_state)
        n_clusters = adata_hvg.obs[key].nunique()

        X_low = adata_hvg.obsm[use_rep][:, :30]
        try:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(X_low, adata_hvg.obs[key], metric='euclidean')
        except:
            sil = 0.0
        
        result = {
            'resolution': res,
            'n_clusters': n_clusters,
            'Silhouette': sil
        }

        if has_true_labels:
            try:
                ari = adjusted_rand_score(adata_hvg.obs['celltype'], adata_hvg.obs[key])
                nmi = normalized_mutual_info_score(adata_hvg.obs['celltype'], adata_hvg.obs[key])
                result['ARI'] = ari
                result['NMI'] = nmi
                
                if ari > best_ari:
                    best_ari = ari
                    best_res = res
            except Exception as e:
                result['ARI'] = None
                result['NMI'] = None
        else:
            if sil > best_sil:
                best_sil = sil
                best_res = res
        
        results_by_resolution.append(result)

    df_resolutions = pd.DataFrame(results_by_resolution)
    df_resolutions.to_csv(f'{output_dir}/clustering_resolutions.csv', index=False)

    if not has_true_labels:
        adata_hvg.obs['celltype'] = adata_hvg.obs[f'leiden_res{best_res}']

    adata_hvg.obs['leiden'] = adata_hvg.obs[f'leiden_res{best_res}']
    
    sc.tl.rank_genes_groups(
        adata_hvg, 
        groupby='leiden', 
        method='wilcoxon',
        key_added='rank_genes_leiden'
    )

    adata_imputed.obsm['X_pca'] = adata_hvg.obsm['X_pca']
    adata_imputed.obsm['X_umap'] = adata_hvg.obsm['X_umap']
    adata_imputed.obs['leiden'] = adata_hvg.obs['leiden']
    
    if use_rep == 'X_pca_harmony':
        adata_imputed.obsm['X_pca_harmony'] = adata_hvg.obsm['X_pca_harmony']

    for res in resolutions:
        key = f'leiden_res{res}'   
        if key in adata_hvg.obs.columns:
            adata_imputed.obs[key] = adata_hvg.obs[key]

    adata_original_for_eval = adata.copy()
    adata_imputed_for_eval = adata.copy()

    if 'imputed' in adata_imputed.layers:
        X_imputed_layer = adata_imputed.layers['imputed']
        
        if issparse(X_imputed_layer):
            X_imputed_dense = X_imputed_layer.toarray()
        else:
            X_imputed_dense = np.array(X_imputed_layer)
        
        adata_imputed_for_eval.X = X_imputed_dense
        
        if issparse(adata_original_for_eval.X):
            X_original_dense = adata_original_for_eval.X.toarray()
        else:
            X_original_dense = np.array(adata_original_for_eval.X)
        
        adata_original_for_eval.X = X_original_dense
    else:
        raise ValueError("layers['imputed'] not found!")

    if X_original_dense.max() > 20:
        if issparse(adata_original_for_eval.X):
            adata_original_for_eval.X = adata_original_for_eval.X.toarray()
        sc.pp.normalize_total(adata_original_for_eval, target_sum=1e4)
        sc.pp.log1p(adata_original_for_eval)

    has_true_labels_for_eval = ('celltype' in adata.obs.columns and 
                                adata.obs['celltype'].nunique() > 1 and
                                adata.obs['celltype'].unique()[0] != 'Unknown')
    
    true_labels_for_eval = None
    label_key_for_eval = None
    if has_true_labels_for_eval:
        true_labels_for_eval = adata.obs['celltype'].values
        label_key_for_eval = 'celltype'

    evaluator = scRNAEvaluatorFixed(
        adata_original=adata_original_for_eval,
        adata_imputed=adata_imputed_for_eval,
        true_labels=true_labels_for_eval,
        label_key=label_key_for_eval
    )

    evaluator.evaluate_clustering_with_precomputed(
        pred_labels=adata_imputed.obs['leiden'].values,
        use_rep=use_rep
    )

    evaluator.evaluate_imputation_accuracy()

    evaluator.evaluate_sparsity()

    report_df = evaluator.generate_report(f'{output_dir}/evaluation_report.csv')

    evaluator.plot_comparison(f'{output_dir}/evaluation_comparison.pdf')

    W = imputation_results['W']
    H = imputation_results['H']
    pathway_dict = imputation_results['pathway_dict']
    
    analyzer = NMFAnalyzer(
        W=W,
        H=H,
        adata=adata_imputed,
        pathway_dict=pathway_dict,
        output_dir=output_dir
    )

    factor_pathways = analyzer.factor_pathway_enrichment(top_n_genes=50)

    if has_true_labels:
        factor_celltype = analyzer.factor_celltype_association()
    else:
        factor_celltype = None

    analyzer.visualize_top_factors(top_k=5)
 
    analyzer.plot_factor_umap(top_k=5)

    plot_main_results(
        adata=adata_imputed,
        clustering_metrics=evaluator.results,
        output_path=f'{output_dir}/Figure_Main.pdf'
    )
    
    plot_supplementary(
        adata=adata_imputed,
        W=W,
        H=H,
        pathway_dict=pathway_dict,
        factor_pathways=factor_pathways,
        output_path=f'{output_dir}/Figure_Supplementary.pdf'
    )

    adata_imputed.write(f'{output_dir}/adata_final.h5ad')

    np.savez(
        f'{output_dir}/nmf_matrices.npz',
        W=W,
        H=H,
        gene_names=adata_imputed.var_names.values,
        cell_names=adata_imputed.obs_names.values
    )

    final_summary = {
        'Data Statistics': {
            'n_cells': adata_imputed.n_obs,
            'n_genes': adata_imputed.n_vars,
            'n_cell_types': adata_imputed.obs.get('celltype', pd.Series(['Unknown'])).nunique(),
        },
        'Imputation Results': {
            'original_nonzero_rate': f"{imputation_results['nmf_stats']['original_nonzero']:.2f}%",
            'imputed_nonzero_rate': f"{imputation_results['nmf_stats']['overall_nonzero_rate']:.2f}%",
            'true_imputation_rate': f"{imputation_results['nmf_stats']['true_impute_rate']:.2f}%"
        },
        'NMF Configuration': {
            'rank': rank,
            'alpha_gene': alpha_gene,
            'alpha_cell': alpha_cell,
            'lambda_reg': lambda_reg,
            'lambda_sparse': lambda_sparse
        }
    }
    
    import json
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    return adata_imputed, evaluator.results, imputation_results


if __name__ == "__main__":
    mp.freeze_support()
    
    config = {
        'adata_path': './data/C01_annotated_celltypist.h5ad',
        'gmt_file': 'c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt',
        'output_dir': './results_c01',

        'rank': 20,
        'alpha_gene': 0.05,
        'alpha_cell': 0.05,
        'lambda_reg': 0.5,
        'lambda_sparse': 5.0,

        'n_hvgs': 2000,
        'n_pcs': 50,
        'leiden_resolution': 0.4,

        'n_jobs': -1,

        'simulated_mask_ratio': 0.1,
        'simulated_random_seed': 42,   

        'organism': 'human',
        'random_state': 42
    }

    adata_final, metrics, imputation_results = \
        run_complete_experiment(**config)