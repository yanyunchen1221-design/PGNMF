# PG-NMF: Pathway-Guided Graph Regularized Non-Negative Matrix Factorization

PG-NMF is a pathway-guided graph regularized non-negative matrix factorization framework for single-cell RNA-seq imputation. This method integrates biological pathway knowledge into the NMF framework through graph regularization, improving the accuracy and biological interpretability of imputation results.

## Method Overview

PG-NMF employs three key strategies to enhance single-cell RNA-seq imputation:

1. **Pathway-guided gene graph regularization**: Constructs a gene similarity graph based on shared pathway membership and expression rank differentials, guiding the factorization to preserve biologically meaningful gene relationships.

2. **Cell similarity graph regularization**: Builds a cell similarity network using PCA-based k-nearest neighbors, ensuring cells with similar expression profiles are mapped close together in the latent space.

3. **Adaptive sparsity constraints**: Applies pathway-aware sparsity penalties that are more lenient for genes within known pathways and stricter for unannotated genes, balancing imputation completeness and noise reduction.

## Requirements

- Python 3.11+
- scanpy
- numpy
- scipy
- networkx
- scikit-learn
- matplotlib
- seaborn
- gseapy
- umap-learn
- anndata
- pandas

## Installation

```bash
pip install -r requirements.txt
```

Or install dependencies individually:

```bash
pip install scanpy numpy scipy networkx scikit-learn matplotlib seaborn gseapy umap-learn anndata pandas
```

## Project Structure

```
PGNMF/
├── pgnmf.py               # Core PG-NMF algorithm implementation
│   ├── run_pathway_guided_pgnmf()  # Main function for imputation
│   ├── nmf_pathway_aware_fixed()    # Pathway-aware NMF factorization
│   ├── calculate_pathway_guided_rank_differential()  # Gene similarity graph
│   └── build_cell_similarity_network_scalable()      # Cell similarity network
├── factor_analysis.py     # NMF factor analysis and visualization
│   └── NMFAnalyzer        # Class for factor pathway enrichment and cell type association
├── evaluation.py          # Evaluation metrics for imputation and clustering
│   └── scRNAEvaluatorFixed  # Class for ARI, NMI, RMSE, and sparsity evaluation
├── visualization.py       # Result visualization functions
│   ├── plot_main_results()       # Main figure generation
│   └── plot_supplementary()     # Supplementary figure generation
├── simulation.py          # Simulated missing value generation
│   └── SimulatedMissingEvaluator  # Class for mask generation
├── run_experiment.py      # Complete experiment workflow
│   └── run_complete_experiment()  # End-to-end pipeline
└── requirements.txt       # Dependency list
```

## Usage

### Quick Start

```bash
python run_experiment.py
```

### Custom Configuration

Edit the `run_experiment.py` to customize parameters:

```python
config = {
    'adata_path': './data/C01_annotated_celltypist.h5ad',
    'gmt_file': 'c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt',
    'output_dir': './results_c01',
    
    # NMF parameters
    'rank': 20,              # Number of latent factors
    'alpha_gene': 0.05,      # Gene graph regularization weight
    'alpha_cell': 0.05,      # Cell graph regularization weight
    'lambda_reg': 0.5,       # L2 regularization weight
    'lambda_sparse': 5.0,    # Sparsity penalty weight
    
    # Preprocessing parameters
    'n_hvgs': 2000,          # Number of highly variable genes
    'n_pcs': 50,             # Number of principal components
    'leiden_resolution': 0.4, # Clustering resolution
    
    # Simulation parameters
    'simulated_mask_ratio': 0.1,    # Ratio of values to mask for evaluation
    'simulated_random_seed': 42,
    
    'organism': 'human',
    'n_jobs': -1
}
```

### Using the API Directly

```python
from pgnmf import run_pathway_guided_pgnmf
from factor_analysis import NMFAnalyzer
from evaluation import scRNAEvaluatorFixed

# Run imputation
adata_imputed, results = run_pathway_guided_pgnmf(
    adata=adata,
    gmt_file='c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt',
    rank=20,
    alpha_gene=0.05,
    alpha_cell=0.05,
    lambda_reg=0.5,
    lambda_sparse=5.0
)

# Analyze factors
analyzer = NMFAnalyzer(
    W=results['W'],
    H=results['H'],
    adata=adata_imputed,
    pathway_dict=results['pathway_dict'],
    output_dir='./results'
)
factor_pathways = analyzer.factor_pathway_enrichment(top_n_genes=50)

# Evaluate results
evaluator = scRNAEvaluatorFixed(
    adata_original=adata,
    adata_imputed=adata_imputed,
    true_labels=adata.obs['celltype'].values
)
evaluator.evaluate_imputation_accuracy()
evaluator.evaluate_clustering_with_precomputed(pred_labels=adata_imputed.obs['leiden'].values)
```

## Input Requirements

### Gene Expression Data

- **Format**: `.h5ad` (AnnData) or `.mat` (MATLAB)
- **Content**: Raw count matrix with cells as rows and genes as columns
- **Optional**: Cell type annotations in `adata.obs['celltype']` or similar columns

### Pathway Annotation File

- **Format**: GMT (Gene Matrix Transposed)
- **Source**: MSigDB (Molecular Signatures Database)
- **Download**: https://www.gsea-msigdb.org/gsea/msigdb
- **Example**: `c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt`

## Output

The pipeline generates the following outputs in the specified `output_dir`:

### Data Files
- `adata_final.h5ad` - Imputed AnnData object with clustering results
- `nmf_matrices.npz` - NMF factor matrices (W, H) with gene and cell names
- `evaluation_report.csv` - Quantitative evaluation metrics
- `clustering_resolutions.csv` - Clustering metrics across different resolutions
- `factor_pathway_enrichment.csv` - Pathway enrichment results for each factor
- `summary.json` - Experiment summary statistics

### Figures
- `Figure_Main.pdf` - Main results including UMAP, confusion matrix, and metrics
- `Figure_Supplementary.pdf` - Supplementary figures with factor activity heatmaps
- `evaluation_comparison.pdf` - Visual comparison of imputation quality
- `top_factors_genes.pdf` - Top genes for each factor
- `factor_umap_distribution.pdf` - Factor activity distribution on UMAP
- `factor_celltype_heatmap.pdf` - Factor activity across cell types

### PG-NMF Specific Outputs
- `pathway_pgnmf_overview.pdf` - Method overview visualization
- `pathway_pgnmf_pathway_networks.pdf` - Pathway network visualization
- `pathway_pgnmf_adata.h5ad` - Intermediate results
- `pathway_pgnmf_networks.npz` - Network matrices

## Dataset

All public datasets used in the study are available from the repositories listed in Supplementary Table S1.

## Citation

Coming soon.

## License

This project is licensed under the MIT License.