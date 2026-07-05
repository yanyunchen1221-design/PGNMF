import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.stats import pearsonr


class scRNAEvaluatorFixed:

    def __init__(self, adata_original, adata_imputed, true_labels=None, label_key=None):
        self.adata_orig = adata_original
        self.adata_imp = adata_imputed
        
        if true_labels is not None:
            self.true_labels = np.array(true_labels)
        elif label_key is not None and label_key in adata_original.obs.columns:
            self.true_labels = adata_original.obs[label_key].values
        else:
            self.true_labels = self._auto_detect_labels(adata_original)
        
        self.X_orig = self._get_array(adata_original.X)
        self.X_imp = self._get_array(adata_imputed.X)
        
        self._ensure_log_space()
        
        self.results = {}
    
    def _get_array(self, X):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return np.array(X)
    
    def _ensure_log_space(self):
        if self.X_orig.max() > 20:
            self.X_orig = np.log1p(self.X_orig)
        
        if self.X_imp.max() > 20:
            self.X_imp = np.log1p(self.X_imp)
    
    def _auto_detect_labels(self, adata):
        candidates = ['celltype', 'cell_type', 'Label', 'label', 'leiden', 'louvain']
        
        for col in candidates:
            if col in adata.obs.columns:
                values = adata.obs[col].values
                if len(np.unique(values)) > 1:
                    return values
        
        return None
    
    def evaluate_clustering_with_precomputed(self, pred_labels, use_rep='X_pca', metric='euclidean'):
        pred_labels = np.array(pred_labels)
        
        if use_rep in self.adata_imp.obsm:
            X_low = self.adata_imp.obsm[use_rep][:, :30]
        else:
            sc.tl.pca(self.adata_imp, n_comps=30)
            X_low = self.adata_imp.obsm['X_pca'][:, :30]
        
        if self.true_labels is not None:
            ari = adjusted_rand_score(self.true_labels, pred_labels)
            nmi = normalized_mutual_info_score(self.true_labels, pred_labels)
            
            self.results['ARI'] = ari
            self.results['NMI'] = nmi
        
        try:
            sil = silhouette_score(X_low, pred_labels, metric=metric)
            self.results['Silhouette'] = sil
        except:
            self.results['Silhouette'] = None
        
        return self.results
    
    def evaluate_imputation_accuracy(self):
        mask_observed = (self.X_orig > 0)
        mask_imputed = (self.X_orig == 0) & (self.X_imp > 0)
        
        if mask_observed.sum() > 0:
            rmse_obs = np.sqrt(np.mean(
                (self.X_orig[mask_observed] - self.X_imp[mask_observed])**2
            ))
            self.results['RMSE_Observed'] = rmse_obs
        
        if mask_imputed.sum() > 0:
            obs_mean = self.X_orig[mask_observed].mean()
            imp_mean = self.X_imp[mask_imputed].mean()
            
            ratio = imp_mean / obs_mean
            self.results['Imputed_Mean_Ratio'] = ratio
        
        cell_corrs = []
        n_sample = min(500, self.X_orig.shape[0])
        for i in np.random.choice(self.X_orig.shape[0], n_sample, replace=False):
            mask = self.X_orig[i, :] > 0
            if mask.sum() > 10:
                corr, _ = pearsonr(self.X_orig[i, mask], 
                                   self.X_imp[i, mask])
                if not np.isnan(corr):
                    cell_corrs.append(corr)
        
        if len(cell_corrs) > 0:
            cell_corr_mean = np.mean(cell_corrs)
            self.results['Cell_Corr_Mean'] = cell_corr_mean
        
        return self.results
    
    def evaluate_sparsity(self):
        threshold = 0.01 if self.X_imp.max() < 20 else 0
        
        mask_orig_zero = (self.X_orig == 0)
        mask_imp_nonzero = (self.X_imp > threshold)
        
        imputed_count = (mask_orig_zero & mask_imp_nonzero).sum()
        total_zeros = mask_orig_zero.sum()
        
        actual_impute_rate = (imputed_count / total_zeros) * 100 if total_zeros > 0 else 0
        
        orig_sparsity = mask_orig_zero.sum() / self.X_orig.size * 100
        imp_sparsity = (self.X_imp <= threshold).sum() / self.X_imp.size * 100
        
        self.results['Original_Sparsity'] = orig_sparsity
        self.results['Imputed_Sparsity'] = imp_sparsity
        self.results['Actual_Imputation_Rate'] = actual_impute_rate
    
    def generate_report(self, save_path='evaluation_report.csv'):
        df = pd.DataFrame([self.results])
        df.to_csv(save_path, index=False)
        return df
    
    def plot_comparison(self, save_path='evaluation_comparison.pdf'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax = axes[0, 0]
        if 'X_umap' in self.adata_imp.obsm:
            umap_coords = self.adata_imp.obsm['X_umap']
        elif 'X_umap_nmf' in self.adata_imp.obsm:
            umap_coords = self.adata_imp.obsm['X_umap_nmf']
        else:
            sc.tl.umap(self.adata_imp)
            umap_coords = self.adata_imp.obsm['X_umap']
        
        if self.true_labels is not None:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels_numeric = le.fit_transform(self.true_labels)
            scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                c=labels_numeric, cmap='tab20', s=5, alpha=0.7)
            ax.set_title('UMAP (True Labels)')
        else:
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=5, alpha=0.7)
            ax.set_title('UMAP')
        ax.axis('off')
        
        ax = axes[0, 1]
        sparsity = [self.results.get('Original_Sparsity', 0),
                   self.results.get('Imputed_Sparsity', 0)]
        ax.bar(['Original', 'Imputed'], sparsity, color=['steelblue', 'coral'])
        ax.set_ylabel('Sparsity (%)')
        ax.set_title('Data Sparsity')
        for i, v in enumerate(sparsity):
            ax.text(i, v+1, f'{v:.1f}%', ha='center')
        
        ax = axes[0, 2]
        metrics = []
        values = []
        for m in ['ARI', 'NMI', 'Silhouette']:
            if self.results.get(m) is not None:
                metrics.append(m)
                values.append(self.results[m])
        
        if len(metrics) > 0:
            ax.bar(metrics, values, color='teal')
            ax.set_ylabel('Score')
            ax.set_title('Clustering Quality')
            ax.set_ylim([0, 1])
            for i, v in enumerate(values):
                ax.text(i, v+0.02, f'{v:.3f}', ha='center')
        
        ax = axes[1, 0]
        errors = [self.results.get('RMSE', 0), self.results.get('MAE', 0)]
        ax.bar(['RMSE', 'MAE'], errors, color='salmon')
        ax.set_ylabel('Error')
        ax.set_title('Imputation Error')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Target<0.5')
        ax.legend()
        
        ax = axes[1, 1]
        corrs = []
        labels = []
        if 'Cell_Corr_Mean' in self.results:
            corrs.append(self.results['Cell_Corr_Mean'])
            labels.append('Cell-wise')
        if 'Gene_Corr_Mean' in self.results:
            corrs.append(self.results['Gene_Corr_Mean'])
            labels.append('Gene-wise')
        
        if len(corrs) > 0:
            ax.bar(labels, corrs, color='mediumseagreen')
            ax.set_ylabel('Pearson Correlation')
            ax.set_title('Correlation Analysis')
            ax.set_ylim([0, 1])
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
        
        ax = axes[1, 2]
        impute_rate = self.results.get('Actual_Imputation_Rate', 0)
        colors = ['orchid' if 30 < impute_rate < 50 else 'orange']
        ax.bar(['Imputation Rate'], [impute_rate], color=colors)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('True Imputation Rate')
        ax.text(0, impute_rate+2, f'{impute_rate:.1f}%', ha='center')
        ax.set_ylim([0, 100])
        ax.axhline(y=35, color='green', linestyle='--', alpha=0.5, label='Target 35%')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Max 50%')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()