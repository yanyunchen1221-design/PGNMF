import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings('ignore')


def load_and_filter_pathways(
    gmt_file: str, 
    gene_list: List[str], 
    min_size: int = 15, 
    max_size: int = 500,
    organism: str = 'human'
) -> Tuple[Dict, Dict]:
    
    pathway_dict = gp.parser.read_gmt(gmt_file)
    
    if organism.lower() == 'mouse':
        gene_list = [g.upper() for g in gene_list]
    
    gene_set = set(gene_list)
    
    filtered_pathways = {}
    for name, genes in pathway_dict.items():
        genes_in_data = [g for g in genes if g in gene_set]
        if min_size <= len(genes_in_data) <= max_size:
            filtered_pathways[name] = genes_in_data

    gene_to_pathways = {}
    for pathway, genes in filtered_pathways.items():
        for gene in genes:
            if gene not in gene_to_pathways:
                gene_to_pathways[gene] = []
            gene_to_pathways[gene].append(pathway)
    
    return filtered_pathways, gene_to_pathways


def construct_pathway_gene_pairs(
    pathway_dict: Dict[str, List[str]], 
    gene_list: List[str]
) -> Tuple[List[Tuple], Dict]:
    
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    gene_pairs = []
    pair_to_pathways = {}
    
    for pathway_name, genes in pathway_dict.items():
        genes_in_data = [g for g in genes if g in gene_to_idx]
        for i, gene_i in enumerate(genes_in_data):
            for gene_j in genes_in_data[i+1:]:
                pair = tuple(sorted([gene_i, gene_j]))
                gene_pairs.append((pair[0], pair[1], pathway_name))
                if pair not in pair_to_pathways:
                    pair_to_pathways[pair] = []
                pair_to_pathways[pair].append(pathway_name)
    
    return gene_pairs, pair_to_pathways


def calculate_gene_ranks(expr_matrix: np.ndarray) -> np.ndarray:
    n_cells, n_genes = expr_matrix.shape
    rank_matrix = np.zeros_like(expr_matrix, dtype=int)
    for i in range(n_cells):
        rank_matrix[i, :] = expr_matrix[i, :].argsort().argsort()
    return rank_matrix


def calculate_pathway_guided_rank_differential(
    adata: sc.AnnData,
    pathway_dict: Dict[str, List[str]],
    similarity_metric: str = 'exponential',
    beta: float = 2.0,
    weight_by_pathway_count: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    
    if hasattr(adata.X, 'toarray'):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = np.array(adata.X)
    
    gene_list = adata.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)
    n_cells = expr_matrix.shape[0]
    
    rank_matrix = calculate_gene_ranks(expr_matrix)
    gene_pairs, pair_to_pathways = construct_pathway_gene_pairs(pathway_dict, gene_list)
    
    S_pathway = np.zeros((n_genes, n_genes))
    rank_diff_stats = []
    
    for gene_i, gene_j, pathway in gene_pairs:
        idx_i = gene_to_idx[gene_i]
        idx_j = gene_to_idx[gene_j]
        if S_pathway[idx_i, idx_j] > 0:
            continue
        
        rank_diff_count = np.sum(rank_matrix[:, idx_i] > rank_matrix[:, idx_j])
        rank_diff_ratio = rank_diff_count / n_cells
        rank_diff_stats.append(rank_diff_ratio)
        
        if similarity_metric == 'exponential':
            similarity = np.exp(-beta * abs(rank_diff_ratio - 0.5))
        elif similarity_metric == 'linear':
            similarity = 1 - 2 * abs(rank_diff_ratio - 0.5)
        elif similarity_metric == 'threshold':
            threshold = 0.3
            similarity = 1.0 if abs(rank_diff_ratio - 0.5) < threshold else 0.0
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
  
        if weight_by_pathway_count:
            pair = tuple(sorted([gene_i, gene_j]))
            n_shared_pathways = len(pair_to_pathways.get(pair, []))
            similarity *= np.log1p(n_shared_pathways)
   
        S_pathway[idx_i, idx_j] = similarity
        S_pathway[idx_j, idx_i] = similarity
    
    if normalize:
        max_val = S_pathway.max()
        if max_val > 0:
            S_pathway = S_pathway / max_val
    
    D = np.diag(S_pathway.sum(axis=1))
    L_pathway = D - S_pathway
    D_sqrt_inv = np.diag(1.0 / np.sqrt(D.diagonal() + 1e-10))
    L_pathway_norm = D_sqrt_inv @ L_pathway @ D_sqrt_inv

    n_edges = (S_pathway > 0).sum() / 2
    n_possible_edges = n_genes * (n_genes - 1) / 2
    coverage = n_edges / n_possible_edges * 100
    
    stats = {
        'n_edges': int(n_edges),
        'coverage': coverage,
        'rank_diff_mean': np.mean(rank_diff_stats),
        'rank_diff_std': np.std(rank_diff_stats),
        'similarity_mean': S_pathway[S_pathway > 0].mean(),
        'similarity_max': S_pathway.max()
    }
    
    return L_pathway_norm, S_pathway, stats


def construct_cell_similarity_network(
    adata: sc.AnnData,
    n_pcs: int = 50,
    use_rep: str = 'X_pca'
) -> Tuple[np.ndarray, np.ndarray]:
    
    if use_rep not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_pcs)
    
    X_reduced = adata.obsm[use_rep][:, :n_pcs]
    n_cells = X_reduced.shape[0]

    dist_matrix = squareform(pdist(X_reduced, metric='euclidean'))
    mst = minimum_spanning_tree(csr_matrix(dist_matrix))
    mst = mst.toarray()
    mst = mst + mst.T 

    G = nx.from_numpy_array(mst)
    geodesic_dist = np.zeros((n_cells, n_cells))
    
    for i in range(n_cells):
        lengths = nx.single_source_dijkstra_path_length(G, i)
        for j, length in lengths.items():
            geodesic_dist[i, j] = length

    A_cell = np.zeros_like(geodesic_dist)
    mask = geodesic_dist > 0
    A_cell[mask] = 1.0 / geodesic_dist[mask]
    np.fill_diagonal(A_cell, 0)

    D_cell = np.diag(A_cell.sum(axis=1))
    L_cell = D_cell - A_cell
    
    D_sqrt_inv = np.diag(1.0 / np.sqrt(D_cell.diagonal() + 1e-10))
    L_cell_norm = D_sqrt_inv @ L_cell @ D_sqrt_inv
    
    return A_cell, L_cell_norm


def build_cell_similarity_network_scalable(
    adata: sc.AnnData,
    n_neighbors: int = 30,
    use_rep: str = 'X_pca',
    n_pcs: int = 50
) -> Tuple[np.ndarray, Dict]:
    
    from scipy.sparse import csr_matrix, diags
    from sklearn.neighbors import NearestNeighbors
    
    if use_rep not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_pcs)
        use_rep = 'X_pca'
    
    X = adata.obsm[use_rep][:, :n_pcs]
    n_cells = X.shape[0]
    
    k = min(n_neighbors, n_cells - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1)
    nbrs.fit(X)
    
    distances, indices = nbrs.kneighbors(X)
    
    row_ind = np.repeat(np.arange(n_cells), k+1)
    col_ind = indices.flatten()

    sigma = np.median(distances[:, 1:]) 
    if sigma == 0:
        sigma = 1.0
    similarities = np.exp(-(distances.flatten() ** 2) / (2 * sigma ** 2))

    S = csr_matrix((similarities, (row_ind, col_ind)), shape=(n_cells, n_cells))
    S = (S + S.T) / 2

    degrees = np.array(S.sum(axis=1)).flatten()
    D = diags(degrees)
    L_cell = D - S

    d_inv_sqrt = 1.0 / np.sqrt(degrees + 1e-10)
    D_inv_sqrt = diags(d_inv_sqrt)
    L_cell_norm = D_inv_sqrt @ L_cell @ D_inv_sqrt

    if n_cells > 50000:
        L_cell_final = L_cell_norm
    else:
        L_cell_final = L_cell_norm.toarray()

    stats = {
        'n_cells': n_cells,
        'k_neighbors': k,
        'sparsity': S.nnz / (n_cells ** 2),
        'avg_degree': np.mean(degrees),
        'sparse': hasattr(L_cell_final, 'toarray')
    }
    
    return L_cell_final, stats


def _update_H_block_worker(args: tuple) -> np.ndarray:
    (block_idx, H_block, W, X_block, M_block, L_block, 
     sparse_mask_block, learning_rate, alpha_gene, lambda_reg) = args
    
    eps = 1e-10
    
    WH = W @ H_block
    numerator = W.T @ (M_block * X_block) + eps
    denominator_base = W.T @ (M_block * WH) + eps
    
    if L_block is not None and L_block.size > 0:
        graph_term = alpha_gene * (H_block @ L_block)
        graph_term = np.maximum(graph_term, 0)
        denominator_base = denominator_base + graph_term
    
    l2_term = lambda_reg * H_block + eps
    H_sign = np.where(H_block > eps, 1.0, 0.0)
    sparse_term = sparse_mask_block.reshape(1, -1) * np.ones((H_block.shape[0], 1)) * H_sign
    sparse_term = np.maximum(sparse_term, 0)
    
    denominator = denominator_base + l2_term + sparse_term
    H_new = H_block * (numerator / denominator)
    H_new = learning_rate * H_new + (1 - learning_rate) * H_block
    H_new = np.maximum(H_new, 0)
    
    return H_new


def nmf_pathway_aware_fixed(
    X: np.ndarray,
    M: np.ndarray,
    L_gene: np.ndarray,
    L_cell: np.ndarray,
    pathway_dict: Dict[str, List[str]],
    gene_list: List[str],
    rank: int = 20,
    alpha_gene: float = 0.05,
    alpha_cell: float = 0.05,
    lambda_reg: float = 0.1,
    lambda_sparse_pathway: float = 50.0,
    lambda_sparse_nonpathway: float = 150.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    target_impute_rate: float = 0.35,
    n_jobs: int = -1,
    parallel_threshold: int = 50000
) -> Tuple[np.ndarray, np.ndarray, List[float], Dict]:
    
    n_cells, n_genes = X.shape
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    eps = 1e-10
    
    use_parallel = False
    actual_n_jobs = 1
    
    if n_jobs == -1:
        if n_cells >= parallel_threshold:
            use_parallel = True
            actual_n_jobs = mp.cpu_count()
    elif n_jobs > 1:
        use_parallel = True
        actual_n_jobs = min(n_jobs, mp.cpu_count())

    if X.max() > 20:
        raise ValueError("Data must be in log space!")
    
    X_original = X.copy()
    
    gene_to_block = {}
    pathway_blocks = []
    genes_in_pathways = set()
    
    for block_idx, (pathway_name, genes) in enumerate(pathway_dict.items()):
        gene_indices = []
        for gene in genes:
            if gene in gene_to_idx:
                idx = gene_to_idx[gene]
                gene_indices.append(idx)
                gene_to_block[idx] = block_idx
                genes_in_pathways.add(idx)
        
        if len(gene_indices) >= 5:
            pathway_blocks.append(gene_indices)
    
    all_gene_indices = set(range(len(gene_list)))
    non_pathway_genes = list(all_gene_indices - genes_in_pathways)
    
    if len(non_pathway_genes) > 0:
        chunk_size = 500
        for i in range(0, len(non_pathway_genes), chunk_size):
            chunk = non_pathway_genes[i:i+chunk_size]
            pathway_blocks.append(chunk)
            for idx in chunk:
                gene_to_block[idx] = len(pathway_blocks) - 1

    sparse_mask = np.ones(n_genes) * lambda_sparse_nonpathway
    
    for block in pathway_blocks:
        if block[0] not in non_pathway_genes:
            for idx in block:
                sparse_mask[idx] = lambda_sparse_pathway
    
    L_gene = np.array(L_gene)
    block_laplacians = []
    for block_idx, gene_indices in enumerate(pathway_blocks):
        L_block = L_gene[np.ix_(gene_indices, gene_indices)]
        max_val = np.abs(L_block).max()
        if max_val > 0:
            L_block = L_block / max_val * 0.01
        block_laplacians.append(L_block)
    
    L_gene = np.array(L_gene) / (np.abs(L_gene).max() + eps) * 0.01
    
    if L_cell is not None:
        if hasattr(L_cell, 'toarray'):
            max_val = np.abs(L_cell.data).max() if len(L_cell.data) > 0 else 1.0
            L_cell = L_cell / (max_val + eps) * 0.01
        else:
            L_cell = np.array(L_cell) / (np.abs(L_cell).max() + eps) * 0.01
    
    np.random.seed(42)
    W = np.abs(np.random.randn(n_cells, rank)) * 0.01 + eps
    
    H_blocks = []
    for gene_indices in pathway_blocks:
        H_block = np.abs(np.random.randn(rank, len(gene_indices))) * 0.01 + eps
        H_blocks.append(H_block)
    
    losses = []
    learning_rate = 0.3
    lr_decay = 0.95
    
    time_start = time.time()
    
    pool = None
    if use_parallel:
        pool = mp.Pool(actual_n_jobs)
    
    try:
        for iteration in range(max_iter):
            if use_parallel and pool is not None:
                tasks_with_params = []
                for block_idx, gene_indices in enumerate(pathway_blocks):
                    X_block = X[:, gene_indices]
                    M_block = M[:, gene_indices]
                    L_block = block_laplacians[block_idx]
                    mask_block = sparse_mask[gene_indices]
                    
                    tasks_with_params.append((
                        block_idx, H_blocks[block_idx], W,
                        X_block, M_block, L_block, mask_block, learning_rate,
                        alpha_gene, lambda_reg
                    ))
                
                H_blocks = pool.map(_update_H_block_worker, tasks_with_params)
            else:
                for block_idx, gene_indices in enumerate(pathway_blocks):
                    X_block = X[:, gene_indices]
                    M_block = M[:, gene_indices]
                    L_block = block_laplacians[block_idx]
                    mask_block = sparse_mask[gene_indices]
                    
                    H_blocks[block_idx] = _update_H_block_worker((
                        block_idx, H_blocks[block_idx], W,
                        X_block, M_block, L_block, mask_block, learning_rate,
                        alpha_gene, lambda_reg
                    ))
            
            H = np.zeros((rank, n_genes))
            for block_idx, gene_indices in enumerate(pathway_blocks):
                H[:, gene_indices] = H_blocks[block_idx]
            
            WH = W @ H
            numerator_W = (M * X) @ H.T + eps
            
            if L_cell is not None:
                if hasattr(L_cell, 'toarray'):
                    cell_reg = alpha_cell * (L_cell @ W)
                else:
                    cell_reg = alpha_cell * (L_cell @ W)
                
                denominator_W = (
                    (M * WH) @ H.T +
                    cell_reg +
                    lambda_reg * W +
                    eps
                )
            else:
                denominator_W = (
                    (M * WH) @ H.T +
                    lambda_reg * W +
                    eps
                )
            
            W_new = W * (numerator_W / denominator_W)
            W = learning_rate * W_new + (1 - learning_rate) * W
            W = np.maximum(W, eps)
           
            WH = W @ H
            reconstruction_loss = np.sum((M * (X - WH)) ** 2)
            
            if L_cell is not None:
                cell_reg = np.trace(W.T @ L_cell @ W)
                cell_loss = alpha_cell * cell_reg
            else:
                cell_loss = 0
            
            gene_loss = 0
            for block_idx, L_block in enumerate(block_laplacians):
                if L_block is not None and L_block.size > 0:
                    H_block = H_blocks[block_idx]
                    gene_loss += alpha_gene * np.trace(H_block @ L_block @ H_block.T)
            
            l2_loss = lambda_reg * (np.sum(W**2) + np.sum(H**2))
            
            l1_loss = 0
            for block_idx, gene_indices in enumerate(pathway_blocks):
                H_block = H_blocks[block_idx]
                mask_block = sparse_mask[gene_indices]
                l1_loss += np.sum(mask_block * np.abs(H_block))
            
            total_loss = reconstruction_loss + cell_loss + gene_loss + l2_loss + l1_loss
            losses.append(total_loss)
            
            if len(losses) > 1:
                rel_change = abs(losses[-1] - losses[-2]) / (abs(losses[-2]) + eps)
                if rel_change < tol:
                    break
            
            learning_rate *= lr_decay
    
    finally:
        if pool is not None:
            pool.close()
            pool.join()
  
    H = np.zeros((rank, n_genes))
    for block_idx, gene_indices in enumerate(pathway_blocks):
        H[:, gene_indices] = H_blocks[block_idx]
  
    X_imputed = W @ H
    orig_zero_mask = (M == 0)
    
    imputed_values_at_zeros = X_imputed[orig_zero_mask]
    
    if len(imputed_values_at_zeros) > 0:
        n_to_keep = int(orig_zero_mask.sum() * target_impute_rate)
        if n_to_keep > 0 and n_to_keep < len(imputed_values_at_zeros):
            sorted_values = np.sort(imputed_values_at_zeros)[::-1]
            threshold = sorted_values[n_to_keep - 1]
            X_imputed[orig_zero_mask & (X_imputed < threshold)] = 0
    
    observed_mask = (M > 0)
    
    imputed_and_kept_mask = (M == 0) & (X_imputed > eps)
    
    if imputed_and_kept_mask.sum() > 0:
        obs_mean = X_original[observed_mask].mean()
        obs_std = X_original[observed_mask].std()
        imp_mean = X_imputed[imputed_and_kept_mask].mean()
        imp_std = X_imputed[imputed_and_kept_mask].std()
        
        if imp_std > 0:
            scale_factor = 0.8  
            X_imputed[imputed_and_kept_mask] = (
                (X_imputed[imputed_and_kept_mask] - imp_mean) / imp_std 
                * obs_std * scale_factor
                + obs_mean
            )
    
    observed_mask = (M > 0)
    max_val = np.percentile(X_original[observed_mask], 99)

    imputed_mask = (M == 0) & (X_imputed > eps)
    X_imputed[imputed_mask] = np.clip(
        X_imputed[imputed_mask], 
        0, 
        max_val
    )
    
    zero_mask = (M == 0) & (X_imputed < eps)
    X_imputed[zero_mask] = 0.0 

    final_zero_mask = (M == 0)
    final_imputed = (X_imputed[final_zero_mask] > eps).sum()
    final_rate = final_imputed / final_zero_mask.sum() * 100 if final_zero_mask.sum() > 0 else 0

    observed_mask = (M > 0)
    diff = X_original[observed_mask] - X_imputed[observed_mask]
    rmse_obs = np.sqrt(np.mean(diff**2))

    time_total = time.time() - time_start

    observed_mask = (M > 0)
    if observed_mask.sum() > 0:
        diff_observed = X_original[observed_mask] - X_imputed[observed_mask]
        rmse_observed = np.sqrt(np.mean(diff_observed**2))
    else:
        rmse_observed = 0.0
    
    H_positive = H[H > 0]
    if len(H_positive) > 0:
        threshold_sparse = np.percentile(H_positive, 1)
        h_sparsity = (H < threshold_sparse).sum() / H.size * 100
    else:
        h_sparsity = 0.0
    
    stats = {
        'original_nonzero': (M > 0).sum() / M.size * 100,
        'overall_nonzero_rate': (X_imputed > eps).sum() / X_imputed.size * 100,
        'true_impute_rate': final_rate,
        'final_sparsity': h_sparsity,
        'rmse_observed': rmse_observed,
        'total_time': time_total,
        'n_blocks': len(pathway_blocks),
        'parallel_used': use_parallel,
        'n_jobs': actual_n_jobs,
        'X_imputed_processed': X_imputed
    }
    
    return W, H, losses, stats


def perform_clustering_and_trajectory(
    adata: sc.AnnData,
    W: np.ndarray,
    n_clusters: int = None,
    resolution: float = 0.9  
):
    adata.obsm['X_nmf'] = W
    
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42, 
        n_neighbors=10,    
        min_dist=0.2,      
        spread=1.0,
        metric='euclidean'
    )
    adata.obsm['X_umap_nmf'] = reducer.fit_transform(W)

    sc.pp.neighbors(
        adata, 
        use_rep='X_nmf', 
        n_neighbors=10,     
        metric='euclidean'  
    )
    
    sc.tl.leiden(adata, resolution=1.2, key_added='leiden_nmf')
    
    current_n = len(adata.obs['leiden_nmf'].unique())
    
    if n_clusters is not None:
        max_attempts = 5
        for attempt in range(max_attempts):
            if current_n == n_clusters:
                break
            elif current_n < n_clusters:
                resolution *= 1.3
            else:
                resolution *= 0.7
            
            sc.tl.leiden(adata, resolution=resolution, key_added='leiden_nmf')
            current_n = len(adata.obs['leiden_nmf'].unique())

    sc.tl.paga(adata, groups='leiden_nmf')
    
    if 'iroot' not in adata.uns:
        if hasattr(adata.X, 'toarray'):
            expr_sum = np.array(adata.X.sum(axis=1)).flatten()
        else:
            expr_sum = adata.X.sum(axis=1)
            if hasattr(expr_sum, 'A1'):
                expr_sum = expr_sum.A1
        root_idx = int(np.argmin(expr_sum))
        adata.uns['iroot'] = root_idx
    
    sc.tl.diffmap(adata, n_comps=min(10, adata.n_obs-1))
    sc.tl.dpt(adata)


def visualize_results(
    adata: sc.AnnData,
    S_pathway: np.ndarray,
    pathway_dict: Dict,
    save_prefix: str = 'results'
):
    if 'X_umap_nmf' in adata.obsm and 'X_umap' not in adata.obsm:
        adata.obsm['X_umap'] = adata.obsm['X_umap_nmf']
    
    fig = plt.figure(figsize=(20, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    sc.pl.umap(adata, color='leiden_nmf', ax=ax1, show=False, title='Clusters')
    
    ax2 = plt.subplot(2, 3, 2)
    sc.pl.umap(adata, color='dpt_pseudotime', ax=ax2, show=False, 
               title='Pseudotime', cmap='viridis')
    
    ax3 = plt.subplot(2, 3, 3)
    sc.pl.paga(adata, ax=ax3, show=False, title='PAGA Trajectory')

    ax4 = plt.subplot(2, 3, 4)
    degrees = (S_pathway > 0).sum(axis=1)
    ax4.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Node Degree')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Pathway Network Degree Distribution')
    ax4.grid(alpha=0.3)
    
    ax5 = plt.subplot(2, 3, 5)
    similarities = S_pathway[S_pathway > 0].flatten()
    ax5.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Similarity Score')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Edge Weight Distribution')
    ax5.grid(alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    pathway_sizes = {name: len(genes) for name, genes in pathway_dict.items()}
    sorted_pathways = sorted(pathway_sizes.items(), key=lambda x: x[1], reverse=True)[:15]
    names, sizes = zip(*sorted_pathways)
    names = [n.split('_')[-1][:30] for n in names]
    ax6.barh(range(len(names)), sizes, color='steelblue')
    ax6.set_yticks(range(len(names)))
    ax6.set_yticklabels(names, fontsize=8)
    ax6.set_xlabel('Number of Genes')
    ax6.set_title('Top 15 Pathways by Size')
    ax6.invert_yaxis()
    ax6.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_overview.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_pathway_network(
    S_pathway: np.ndarray,
    gene_list: List[str],
    pathway_dict: Dict,
    top_k: int = 3,
    threshold: float = 0.2,
    save_prefix: str = 'results'
):
    pathway_sizes = {name: len(genes) for name, genes in pathway_dict.items()}
    top_pathways = sorted(pathway_sizes.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    fig, axes = plt.subplots(1, top_k, figsize=(top_k * 7, 6))
    if top_k == 1:
        axes = [axes]
    
    for idx, (pathway_name, _) in enumerate(top_pathways):
        genes_in_pathway = pathway_dict[pathway_name]
        gene_indices = [gene_list.index(g) for g in genes_in_pathway if g in gene_list]
        
        sub_matrix = S_pathway[np.ix_(gene_indices, gene_indices)]
        
        G = nx.Graph()
        for i, idx_i in enumerate(gene_indices):
            for j, idx_j in enumerate(gene_indices):
                if i < j and sub_matrix[i, j] > threshold:
                    G.add_edge(
                        gene_list[idx_i],
                        gene_list[idx_j],
                        weight=sub_matrix[i, j]
                    )

        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

            node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]
            
            nx.draw_networkx_nodes(
                G, pos, ax=axes[idx],
                node_size=node_sizes,
                node_color='lightblue',
                alpha=0.8
            )
            
            nx.draw_networkx_labels(
                G, pos, ax=axes[idx],
                font_size=7,
                font_weight='bold'
            )

            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(
                G, pos, ax=axes[idx],
                width=[w * 3 for w in weights],
                alpha=0.5,
                edge_color='gray'
            )
            
            axes[idx].set_title(f"{pathway_name.split('_')[-1][:40]}\n({len(G.nodes())} genes, {len(G.edges())} edges)", 
                              fontsize=10, fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, 'No edges above threshold', 
                          ha='center', va='center', transform=axes[idx].transAxes)
        
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_pathway_networks.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def identify_trajectory_markers(
    adata: sc.AnnData,
    pathway_dict: Dict,
    group_key: str = 'leiden_nmf',
    n_top_genes: int = 50
) -> pd.DataFrame:
    sc.tl.rank_genes_groups(adata, groupby=group_key, method='wilcoxon')
   
    results = []
    for cluster in adata.obs[group_key].unique():
        marker_genes = sc.get.rank_genes_groups_df(adata, group=cluster)
        top_genes = marker_genes.head(n_top_genes)['names'].tolist()

        enriched_pathways = []
        for pathway_name, genes_in_pathway in pathway_dict.items():
            overlap = set(top_genes) & set(genes_in_pathway)
            if len(overlap) >= 3:  
                enriched_pathways.append({
                    'pathway': pathway_name,
                    'n_overlap': len(overlap),
                    'overlap_genes': ','.join(overlap)
                })
        
        enriched_df = pd.DataFrame(enriched_pathways)
        if len(enriched_df) > 0:
            enriched_df = enriched_df.sort_values('n_overlap', ascending=False)
        
        results.append({
            'cluster': cluster,
            'n_cells': (adata.obs[group_key] == cluster).sum(),
            'top_genes': ','.join(top_genes[:10]),
            'top_pathway': enriched_df.iloc[0]['pathway'] if len(enriched_df) > 0 else 'None'
        })
    
    results_df = pd.DataFrame(results)
    return results_df


def compare_with_correlation_network(
    adata: sc.AnnData,
    S_pathway: np.ndarray,
    corr_threshold: float = 0.3
) -> Dict:
    if hasattr(adata.X, 'toarray'):
        expr_matrix = adata.X.toarray().T
    else:
        expr_matrix = np.array(adata.X).T
    
    corr_matrix = np.corrcoef(expr_matrix)
    corr_matrix = np.abs(corr_matrix)

    S_corr = (corr_matrix > corr_threshold).astype(float)
    np.fill_diagonal(S_corr, 0)
 
    pathway_edges = (S_pathway > 0)
    corr_edges = (S_corr > 0)
    
    overlap = (pathway_edges & corr_edges).sum() / 2
    pathway_only = (pathway_edges & ~corr_edges).sum() / 2
    corr_only = (~pathway_edges & corr_edges).sum() / 2
    
    stats = {
        'pathway_edges': pathway_edges.sum() / 2,
        'corr_edges': corr_edges.sum() / 2,
        'overlap': overlap,
        'pathway_only': pathway_only,
        'corr_only': corr_only,
        'jaccard': overlap / (pathway_edges.sum() / 2 + corr_edges.sum() / 2 - overlap)
    }
    
    return stats


def run_pathway_guided_pgnmf(
    adata: sc.AnnData,
    gmt_file: str,
    rank: int = 20,
    alpha_gene: float = 0.001,
    alpha_cell: float = 0.001,
    lambda_reg: float = 0.001,
    lambda_sparse: float = 100, 
    min_pathway_size: int = 15,
    max_pathway_size: int = 500,
    similarity_metric: str = 'exponential',
    beta: float = 2.0,
    n_pcs: int = 50,
    max_iter: int = 100,
    n_jobs: int = -1, 
    save_prefix: str = 'pbmc3k_pathway_pgnmf',
    organism: str = 'human'
) -> Tuple[sc.AnnData, Dict]:
    
    adata_copy = adata.copy()

    if hasattr(adata_copy.X, 'toarray'):
        X_max = adata_copy.X.max()
    else:
        X_max = np.array(adata_copy.X).max()
    
    if X_max > 100:
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        sc.pp.log1p(adata_copy)

    if hasattr(adata_copy.X, 'toarray'):
        X_check = adata_copy.X.toarray()
    else:
        X_check = np.array(adata_copy.X)

    if X_check.max() > 100:
        M_original = (X_check > 0).astype(float) 
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        sc.pp.log1p(adata_copy)
        
        if hasattr(adata_copy.X, 'toarray'):
            X = adata_copy.X.toarray()
        else:
            X = np.array(adata_copy.X)
        
        M = M_original 
    else:
        if hasattr(adata_copy.X, 'toarray'):
            X = adata_copy.X.toarray()
        else:
            X = np.array(adata_copy.X)
        
        M = (X > 0.01).astype(float)

    pathway_dict, gene_to_pathways = load_and_filter_pathways(
        gmt_file=gmt_file,
        gene_list=adata_copy.var_names.tolist(),
        min_size=min_pathway_size,
        max_size=max_pathway_size,
        organism=organism
    )
    
    L_gene, S_pathway, pathway_stats = calculate_pathway_guided_rank_differential(
        adata=adata_copy,
        pathway_dict=pathway_dict,
        similarity_metric=similarity_metric,
        beta=beta,
        weight_by_pathway_count=True,
        normalize=True
    )

    n_cells = adata_copy.n_obs

    if n_cells < 50000:
        A_cell, L_cell = construct_cell_similarity_network(
            adata=adata_copy,
            n_pcs=n_pcs,
            use_rep='X_pca'
        )
        cell_stats = {'method': 'full', 'n_cells': n_cells}
        
    elif n_cells < 150000:
        L_cell, cell_stats = build_cell_similarity_network_scalable(
            adata=adata_copy,
            n_neighbors=30,
            use_rep='X_pca',
            n_pcs=n_pcs
        )
        cell_stats['method'] = 'knn'
        
    else:
        L_cell = None
        cell_stats = {
            'method': 'skipped',
            'n_cells': n_cells,
            'reason': 'memory_optimization'
        }

    W, H, losses, nmf_stats = nmf_pathway_aware_fixed(
        X=X,
        M=M,
        L_gene=L_gene,
        L_cell=L_cell,
        pathway_dict=pathway_dict,
        gene_list=adata_copy.var_names.tolist(),
        rank=rank, 
        alpha_gene=alpha_gene,
        alpha_cell=alpha_cell,
        lambda_reg=lambda_reg,
        lambda_sparse_pathway=lambda_sparse * 0.3,   
        lambda_sparse_nonpathway=lambda_sparse * 0.8, 
        max_iter=max_iter,
        tol=1e-4,
        n_jobs=n_jobs  
    )

    if 'X_imputed_processed' in nmf_stats:
        X_imputed = nmf_stats['X_imputed_processed']
    else:
        X_imputed = W @ H

    X_imputed = np.clip(X_imputed, 0, X.max() * 1.1)
    
    adata_copy.layers['imputed'] = X_imputed.copy()
    adata_copy.obsm['X_nmf_cell'] = W
    adata_copy.varm['X_nmf_gene'] = H.T

    perform_clustering_and_trajectory(
        adata=adata_copy,
        W=W,
        n_clusters=None,
        resolution=0.9
    )

    visualize_results(
        adata=adata_copy,
        S_pathway=S_pathway,
        pathway_dict=pathway_dict,
        save_prefix=save_prefix
    )
    
    visualize_pathway_network(
        S_pathway=S_pathway,
        gene_list=adata_copy.var_names.tolist(),
        pathway_dict=pathway_dict,
        top_k=3,
        threshold=0.2,
        save_prefix=save_prefix
    )
    
    marker_results = identify_trajectory_markers(
        adata=adata_copy,
        pathway_dict=pathway_dict,
        group_key='leiden_nmf',
        n_top_genes=50
    )
    
    comparison_stats = compare_with_correlation_network(
        adata=adata_copy,
        S_pathway=S_pathway,
        corr_threshold=0.3
    )

    results = {
        'W': W,
        'H': H,
        'losses': losses,
        'nmf_stats': nmf_stats,  
        'S_pathway': S_pathway,
        'L_gene': L_gene,
        'L_cell': L_cell,
        'pathway_dict': pathway_dict,
        'pathway_stats': pathway_stats,
        'marker_results': marker_results,
        'comparison_stats': comparison_stats
    }

    output_file = f'{save_prefix}_adata.h5ad'
    adata_copy.write(output_file)

    np.savez(f'{save_prefix}_networks.npz',
             S_pathway=S_pathway,
             L_gene=L_gene,
             L_cell=L_cell,
             W=W,
             H=H)
    
    return adata_copy, results


if __name__ == "__main__":
    adata = sc.datasets.pbmc3k()
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    gmt_file = "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"

    adata_result, results = run_pathway_guided_pgnmf(
        adata=adata,
        gmt_file=gmt_file,
        rank=20,                    
        alpha_gene=0.5,              
        alpha_cell=0.5,             
        lambda_reg=0.01,           
        lambda_sparse=0.25,          
        min_pathway_size=15,       
        max_pathway_size=500,       
        similarity_metric='exponential',
        beta=2.0,
        n_pcs=30,
        max_iter=50,
        save_prefix='pbmc3k_pathway_pgnmf',
        organism='human'
    )  

    plt.figure(figsize=(10, 6))
    plt.plot(results['losses'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('NMF Optimization Loss Curve')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('nmf_loss_curve.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    sc.pl.umap(adata_result, color=['leiden_nmf', 'dpt_pseudotime'], 
               ncols=2, save='_final.pdf')