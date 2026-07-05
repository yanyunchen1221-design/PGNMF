import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from typing import Dict, List


class NMFAnalyzer:

    def __init__(self, W, H, adata, pathway_dict, output_dir='./results'):
        self.W = W
        self.H = H
        self.adata = adata
        self.pathway_dict = pathway_dict
        self.output_dir = output_dir
        
        self.n_cells, self.rank = W.shape
        _, self.n_genes = H.shape
    
    def factor_pathway_enrichment(self, top_n_genes=50, min_overlap=3):
        gene_names = self.adata.var_names.tolist()
        factor_enrichment = {}
        
        for factor_id in range(self.rank):
            factor_scores = self.H[factor_id, :]
            top_indices = np.argsort(factor_scores)[::-1][:top_n_genes]
            top_genes = [gene_names[i] for i in top_indices]

            enriched = []
            for pathway_name, pathway_genes in self.pathway_dict.items():
                overlap = set(top_genes) & set(pathway_genes)
                if len(overlap) >= min_overlap:
                    enriched.append({
                        'pathway': pathway_name,
                        'n_overlap': len(overlap),
                        'overlap_genes': ','.join(sorted(overlap)),
                        'overlap_ratio': len(overlap) / len(pathway_genes)
                    })

            enriched = sorted(enriched, key=lambda x: x['n_overlap'], reverse=True)
            factor_enrichment[factor_id] = enriched

        self._save_factor_enrichment(factor_enrichment)
        
        return factor_enrichment
    
    def _save_factor_enrichment(self, factor_enrichment):
        all_rows = []
        for factor_id, enriched_list in factor_enrichment.items():
            for item in enriched_list:
                all_rows.append({
                    'Factor': factor_id,
                    'Pathway': item['pathway'],
                    'N_Overlap': item['n_overlap'],
                    'Overlap_Ratio': item['overlap_ratio'],
                    'Genes': item['overlap_genes']
                })
        
        df = pd.DataFrame(all_rows)
        output_path = f'{self.output_dir}/factor_pathway_enrichment.csv'
        df.to_csv(output_path, index=False)

    
    def factor_celltype_association(self, celltype_col='celltype'):
        if celltype_col not in self.adata.obs.columns:
            return None
        
        cell_types = self.adata.obs[celltype_col].unique()
        factor_activity = pd.DataFrame(
            index=cell_types,
            columns=[f'Factor_{i}' for i in range(self.rank)]
        )
        
        for cell_type in cell_types:
            mask = self.adata.obs[celltype_col] == cell_type
            W_celltype = self.W[mask, :]
            mean_activity = W_celltype.mean(axis=0)
            factor_activity.loc[cell_type] = mean_activity
 
        factor_activity_zscore = (factor_activity - factor_activity.mean()) / factor_activity.std()

        self._plot_factor_celltype_heatmap(factor_activity_zscore)

        output_path = f'{self.output_dir}/factor_celltype_activity.csv'
        factor_activity.to_csv(output_path)

        self._identify_celltype_specific_factors(factor_activity_zscore)
        
        return factor_activity
    
    def _plot_factor_celltype_heatmap(self, factor_activity_zscore):
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            factor_activity_zscore.T.astype(float),
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Z-score'},
            xticklabels=True,
            yticklabels=True,
            linewidths=0.5
        )
        plt.xlabel('Cell Type')
        plt.ylabel('NMF Factor')
        plt.title('Factor Activity Across Cell Types')
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/factor_celltype_heatmap.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _identify_celltype_specific_factors(self, factor_activity_zscore, threshold=1.5):
        specific_factors = []
        for col in factor_activity_zscore.columns:
            max_celltype = factor_activity_zscore[col].idxmax()
            max_zscore = factor_activity_zscore[col].max()
            
            if max_zscore > threshold:
                specific_factors.append({
                    'Factor': col,
                    'Cell_Type': max_celltype,
                    'Z-score': max_zscore
                })
        
        return specific_factors

    def visualize_top_factors(self, top_k=5, n_genes_show=20):
        mean_activity = self.W.mean(axis=0)
        top_factor_ids = np.argsort(mean_activity)[::-1][:top_k]
        
        gene_names = self.adata.var_names.tolist()
        
        fig, axes = plt.subplots(1, top_k, figsize=(top_k * 4, 6))
        if top_k == 1:
            axes = [axes]
        
        for idx, factor_id in enumerate(top_factor_ids):
            ax = axes[idx]

            factor_scores = self.H[factor_id, :]
            top_indices = np.argsort(factor_scores)[::-1][:n_genes_show]
            top_genes = [gene_names[i] for i in top_indices]
            top_scores = factor_scores[top_indices]

            y_pos = np.arange(len(top_genes))
            ax.barh(y_pos, top_scores, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_genes, fontsize=8)
            ax.set_xlabel('Factor Score')
            ax.set_title(f'Factor {factor_id}\n(Mean Activity: {mean_activity[factor_id]:.3f})')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/top_factors_genes.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    
    def plot_factor_umap(self, top_k=5):
        if 'X_umap' not in self.adata.obsm:
            return
        
        umap_coords = self.adata.obsm['X_umap']
        
        mean_activity = self.W.mean(axis=0)
        top_factor_ids = np.argsort(mean_activity)[::-1][:top_k]
        
        fig, axes = plt.subplots(1, top_k, figsize=(top_k * 4, 3))
        if top_k == 1:
            axes = [axes]
        
        for idx, factor_id in enumerate(top_factor_ids):
            ax = axes[idx]

            factor_activity = self.W[:, factor_id]
            
            scatter = ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=factor_activity,
                cmap='viridis',
                s=5,
                alpha=0.7
            )
            
            ax.set_title(f'Factor {factor_id}')
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.axis('off')
            
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/factor_umap_distribution.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def factor_correlation_analysis(self):
        corr_matrix = np.corrcoef(self.W.T)
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation'},
            vmin=-1, vmax=1
        )
        
        plt.title('NMF Factor Correlation Matrix')
        plt.xlabel('Factor')
        plt.ylabel('Factor')
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/factor_correlation_matrix.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
       
        high_corr_pairs = []
        for i in range(self.rank):
            for j in range(i+1, self.rank):
                if abs(corr_matrix[i, j]) > 0.7:
                    high_corr_pairs.append({
                        'Factor1': i,
                        'Factor2': j,
                        'Correlation': corr_matrix[i, j]
                    })
        
        return corr_matrix


if __name__ == "__main__":
    pass