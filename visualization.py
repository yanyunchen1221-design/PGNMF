import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from matplotlib.gridspec import GridSpec


def plot_main_results(adata, clustering_metrics, output_path='Figure_Main.pdf'):
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    if 'X_umap' in adata.obsm and 'celltype' in adata.obs.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        celltype_numeric = le.fit_transform(adata.obs['celltype'])
        
        scatter = ax1.scatter(
            adata.obsm['X_umap'][:, 0],
            adata.obsm['X_umap'][:, 1],
            c=celltype_numeric,
            cmap='tab20',
            s=1,
            alpha=0.7
        )
        
        unique_types = adata.obs['celltype'].unique()
        if len(unique_types) <= 12:
            handles = []
            labels = []
            for i, ct in enumerate(unique_types):
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=plt.cm.tab20(le.transform([ct])[0] / len(unique_types)), 
                              markersize=8))
                labels.append(ct)
            ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
                      fontsize=8, frameon=False)
        
        ax1.set_title('A) True Cell Types', fontsize=14, fontweight='bold')
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    if 'leiden' in adata.obs.columns:
        leiden_numeric = adata.obs['leiden'].astype('category').cat.codes
        
        scatter = ax2.scatter(
            adata.obsm['X_umap'][:, 0],
            adata.obsm['X_umap'][:, 1],
            c=leiden_numeric,
            cmap='tab20',
            s=1,
            alpha=0.7
        )
        
        ax2.set_title('B) Predicted Clusters (Leiden)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('UMAP1')
        ax2.set_ylabel('UMAP2')
        ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2:])
    
    if 'celltype' in adata.obs.columns and 'leiden' in adata.obs.columns:
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(adata.obs['celltype'], adata.obs['leiden'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        n_types = len(adata.obs['celltype'].unique())
        if n_types > 15:
            pass
        
        sns.heatmap(
            cm_normalized,
            cmap='Blues',
            annot=False,
            fmt='.2f',
            cbar_kws={'label': 'Proportion'},
            ax=ax3,
            xticklabels=adata.obs['leiden'].cat.categories,
            yticklabels=adata.obs['celltype'].cat.categories
        )
        
        ax3.set_title('C) Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Predicted Cluster')
        ax3.set_ylabel('True Cell Type')
        ax3.tick_params(axis='both', labelsize=8)
    
    ax4 = fig.add_subplot(gs[1, :2])
    
    metrics = ['ARI', 'NMI', 'Silhouette']
    values = [clustering_metrics.get(m, 0) for m in metrics]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax4.set_ylim([0, 1.1])
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('D) Clustering Performance Metrics', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Good threshold')
    ax4.legend(fontsize=10)
    
    ax5 = fig.add_subplot(gs[1, 2:])
    
    impute_metrics = {
        'RMSE': clustering_metrics.get('RMSE', 0),
        'Cell Corr': clustering_metrics.get('Cell_Corr_Mean', 0),
        'Gene Corr': clustering_metrics.get('Gene_Corr_Mean', 0),
        'Impute Rate': clustering_metrics.get('Actual_Imputation_Rate', 0) / 100
    }
    
    metric_names = list(impute_metrics.keys())
    metric_values = list(impute_metrics.values())
    
    for i, name in enumerate(metric_names[:3]):
        if name == 'RMSE':
            metric_values[i] = 1 / (1 + metric_values[i])
    
    colors = ['#9b59b6', '#1abc9c', '#f39c12', '#e67e22']
    bars = ax5.barh(metric_names, metric_values, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        width = bar.get_width()
        if metric_names[i] == 'Impute Rate':
            label = f'{val*100:.1f}%'
        else:
            label = f'{val:.3f}'
        
        ax5.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                label,
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax5.set_xlim([0, 1.2])
    ax5.set_xlabel('Score', fontsize=12)
    ax5.set_title('E) Imputation Quality Metrics', fontsize=14, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3, linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_supplementary(adata, W, H, pathway_dict, factor_pathways, 
                       output_path='Figure_Supplementary.pdf'):
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, :])
    
    if 'celltype' in adata.obs.columns:
        cell_types = adata.obs['celltype'].unique()
        n_factors = W.shape[1]
        
        factor_activity = np.zeros((len(cell_types), min(n_factors, 20)))
        
        for i, ct in enumerate(cell_types):
            mask = adata.obs['celltype'] == ct
            factor_activity[i, :] = W[mask, :min(n_factors, 20)].mean(axis=0)
        
        factor_activity_zscore = (factor_activity - factor_activity.mean(axis=0)) / factor_activity.std(axis=0)
        
        sns.heatmap(
            factor_activity_zscore.T,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Z-score'},
            xticklabels=cell_types,
            yticklabels=[f'F{i}' for i in range(factor_activity_zscore.shape[1])],
            ax=ax1,
            linewidths=0.5
        )
        
        ax1.set_title('A) NMF Factor Activity Across Cell Types', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cell Type')
        ax1.set_ylabel('Factor')
    
    axes_b = [fig.add_subplot(gs[1, i]) for i in range(3)]
    
    mean_activity = W.mean(axis=0)
    top_factors = np.argsort(mean_activity)[::-1][:3]
    gene_names = adata.var_names.tolist()
    
    for idx, factor_id in enumerate(top_factors):
        ax = axes_b[idx]
        
        factor_scores = H[factor_id, :]
        top_indices = np.argsort(factor_scores)[::-1][:15]
        top_genes = [gene_names[i] for i in top_indices]
        top_scores = factor_scores[top_indices]
        
        y_pos = np.arange(len(top_genes))
        ax.barh(y_pos, top_scores, color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_genes, fontsize=9)
        ax.set_xlabel('Factor Score', fontsize=10)
        ax.set_title(f'Factor {factor_id}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    fig.text(0.5, 0.635, 'B) Top Genes in Key Factors', 
            ha='center', fontsize=14, fontweight='bold')
    
    if 'X_umap' in adata.obsm:
        axes_c = [fig.add_subplot(gs[2, i]) for i in range(3)]
        umap_coords = adata.obsm['X_umap']
        
        for idx, factor_id in enumerate(top_factors):
            ax = axes_c[idx]
            
            scatter = ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=W[:, factor_id],
                cmap='viridis',
                s=1,
                alpha=0.7
            )
            
            ax.set_title(f'Factor {factor_id}', fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        
        fig.text(0.5, 0.285, 'C) Factor Activity Distribution on UMAP', 
                ha='center', fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_marker_genes_heatmap(adata, output_path='marker_genes_heatmap.pdf'):
    if 'rank_genes_leiden' not in adata.uns:
        return
    
    sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=10,
        groupby='leiden',
        show_gene_labels=True,
        save='_marker_genes.pdf'
    )


def plot_pathway_network_example(S_pathway, gene_list, pathway_dict, 
                                 pathway_name, output_path='pathway_network.pdf'):
    import networkx as nx
    
    if pathway_name not in pathway_dict:
        return
    
    genes_in_pathway = pathway_dict[pathway_name]
    gene_indices = [gene_list.index(g) for g in genes_in_pathway if g in gene_list]
    
    if len(gene_indices) < 5:
        return
    
    sub_matrix = S_pathway[np.ix_(gene_indices, gene_indices)]
    
    G = nx.Graph()
    threshold = 0.2
    
    for i, idx_i in enumerate(gene_indices):
        for j, idx_j in enumerate(gene_indices):
            if i < j and sub_matrix[i, j] > threshold:
                G.add_edge(
                    gene_list[idx_i],
                    gene_list[idx_j],
                    weight=sub_matrix[i, j]
                )
    
    if len(G.nodes()) == 0:
        return
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    node_sizes = [500 * (1 + G.degree(node)) for node in G.nodes()]
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='lightblue',
        alpha=0.8,
        edgecolors='black',
        linewidths=2
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight='bold'
    )
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(
        G, pos,
        width=[w * 5 for w in weights],
        alpha=0.5,
        edge_color='gray'
    )
    
    plt.title(f'Pathway: {pathway_name}\n({len(G.nodes())} genes, {len(G.edges())} edges)',
             fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_figures(adata, W, H, pathway_dict, clustering_metrics,
                        factor_pathways, output_dir='./results'):
    plot_main_results(
        adata=adata,
        clustering_metrics=clustering_metrics,
        output_path=f'{output_dir}/Figure_Main.pdf'
    )
    
    plot_supplementary(
        adata=adata,
        W=W,
        H=H,
        pathway_dict=pathway_dict,
        factor_pathways=factor_pathways,
        output_path=f'{output_dir}/Figure_Supplementary.pdf'
    )
    
    plot_marker_genes_heatmap(
        adata=adata,
        output_path=f'{output_dir}/marker_genes_heatmap.pdf'
    )