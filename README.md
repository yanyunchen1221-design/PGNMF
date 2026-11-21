# PGNMF

**Pathway-Guided Hierarchical Sparse Non-Negative Matrix Factorization for scRNA-seq Imputation**

PGNMF is a pathway-guided, graph-regularized non-negative matrix factorization framework for imputing single-cell RNA-sequencing (scRNA-seq) expression data. It is designed to recover missing expression values while retaining cellular heterogeneity and biologically meaningful gene relationships.

The method combines four complementary components:

1. **Pathway-guided gene graph.** Gene relationships are informed by shared pathway membership and expression-rank agreement.
2. **Cell-neighborhood graph.** A sparse cell graph captures local relationships between cells in the expression space.
3. **Hierarchical gene sparsity.** Different sparsity penalties are assigned to pathway-annotated and non-annotated genes, allowing pathway-supported genes to receive a less stringent penalty.
4. **Weak zero-background regularization.** A balanced, weak penalty regulates predictions at zero-valued input positions to limit unsupported background signals.

This repository provides the PGNMF core implementation, graph and pathway utilities, a command-line example for prepared expression matrices, and a small solver smoke test. For the methodological formulation and the full experimental design, please refer to the accompanying manuscript.

## Method overview

The workflow takes a prepared, masked cell-by-gene expression matrix and the corresponding gene symbols and pathway collection. It constructs the pathway-guided gene graph, the cell-neighborhood graph, and gene-specific sparsity penalties, and then estimates non-negative cell and gene factors.

The released solver processes gene updates in computational blocks while retaining the complete gene graph, including cross-block gene relationships. The block size controls computational chunking.

## Repository contents

```text
PGNMF/
├── src/
│   └── pgnmf/
│       ├── __init__.py
│       ├── solver.py             # PGNMF multiplicative-update solver
│       ├── gene_graph.py         # Pathway-guided gene graph construction
│       ├── cell_graph.py         # PCA-based cell kNN graph construction
│       ├── pathway_sparsity.py   # Pathway membership and gene-specific penalties
│       ├── postprocess.py        # Exact top-k post-processing utilities
│       ├── masking.py            # Simulated-missing mask utilities
│       ├── matrix_utils.py       # Matrix helper functions
│       └── metrics.py            # Masked-expression evaluation metrics
├── examples/
│   └── run_from_prepared.py      # Example command-line workflow
├── tests/
│   └── test_solver_smoke.py      # Small synthetic solver test
├── reproducibility/
│   ├── scripts/                  # Selected manuscript analyses
│   ├── legacy/                   # Minimal compatibility helpers
│   ├── configs/                  # Example experiment configuration
│   └── requirements-experiments.txt
├── LICENSE
├── README.md
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

### Scope of this release

This release contains the core algorithm, a prepared-input example workflow, and a curated set of scripts for selected manuscript analyses. It does **not** bundle raw scRNA-seq datasets, controlled-access clinical data, the MSigDB/KEGG pathway collection, intermediate caches, or benchmark result tables. See `reproducibility/README.md` for the included analyses and their reproduction requirements.

## Installation

### Requirements

- Python 3.10 or later
- NumPy
- SciPy
- scikit-learn

The package dependencies are listed in `requirements.txt`.

### Install from GitHub

```bash
git clone https://github.com/yanyunchen1221-design/PGNMF.git
cd PGNMF

python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install PGNMF and its dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

To run the included solver smoke test:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The smoke test uses a small synthetic matrix generated within the test; it does not reproduce the full manuscript benchmarks.

## Input preparation

The example runner expects the following files:

| Input | Format | Description |
|---|---|---|
| `X_model.npy` | NumPy array | Dense, finite, non-negative masked expression matrix. Rows are cells and columns are selected model genes. |
| `all_gene_symbols.npy` | NumPy array | Gene symbols for the complete gene universe before model-gene selection. |
| `model_gene_symbols.npy` | NumPy array | Gene symbols for the selected model genes, in exactly the same order as the columns of `X_model.npy`. |
| `pathways.gmt` | GMT text file | Pathway gene sets used to construct the pathway-guided gene graph and gene-specific sparsity penalties. |

The matrix preprocessing and model-gene selection are dataset- and experiment-specific. For reproducing the manuscript experiments, prepare the expression matrix using the normalization, transformation, and feature-selection procedure described in the manuscript and Supplementary Data. The example runner does not perform raw-count normalization or highly-variable-gene selection.

When evaluating simulated missing values, `X_model.npy` must contain the masked input matrix. The cell graph and pathway-rank gene graph are constructed from this masked matrix; do not construct them from an unmasked reference matrix.

### Preparing NumPy inputs from AnnData

If the prepared data are already available as an AnnData object, the required arrays can be saved as follows. Here, `model_mask` is a Boolean mask identifying the selected model genes.

```python
import numpy as np
from scipy import sparse

X_model = adata_masked[:, model_mask].X
if sparse.issparse(X_model):
    X_model = X_model.toarray()

np.save("X_model.npy", np.asarray(X_model, dtype=np.float32))
np.save(
    "all_gene_symbols.npy",
    adata_masked.var_names.astype(str).to_numpy(),
)
np.save(
    "model_gene_symbols.npy",
    adata_masked.var_names[model_mask].astype(str).to_numpy(),
)
```

The number and order of columns in `X_model.npy` must match `model_gene_symbols.npy` exactly. The model-gene symbols must be a subset of the full gene-symbol universe represented by `all_gene_symbols.npy`.

## Pathway gene-set resource

The runner requires a pathway collection in standard GMT format. The manuscript workflow uses the human MSigDB C2:CP:KEGG_LEGACY collection (for example, `c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt`).

Download the appropriate human gene-set file from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/) and follow the applicable registration, citation, and license terms. The GMT file is not included in this repository because it is a third-party resource and the KEGG_LEGACY gene sets have additional copyright and licensing conditions. The MIT License in this repository applies to the PGNMF software code; it does not relicense third-party pathway data.

See the [MSigDB license terms](https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp), including the additional terms for KEGG-derived gene sets. If redistributing a pathway file, first verify that the specific collection and version may be redistributed under its applicable terms.

## Run PGNMF

After preparing the four input files, run:

```bash
python examples/run_from_prepared.py \
  --x-model data/X_model.npy \
  --all-gene-symbols data/all_gene_symbols.npy \
  --model-gene-symbols data/model_gene_symbols.npy \
  --gmt data/pathways.gmt \
  --output-dir results \
  --rank 80 \
  --alpha-cell 10 \
  --alpha-gene 2 \
  --zero-weight 0.13577193 \
  --sparsity-strength 10 \
  --pathway-weight 0.5 \
  --nonpathway-weight 1.0 \
  --max-iter 100 \
  --tol 1e-4 \
  --seed 42 \
  --block-size 500
```

**Parameter note:** The values above illustrate the confirmed C11 Stage2C8 configuration (`rank=80`, `alpha_cell=10`, `alpha_gene=2`, `beta=1`, and hierarchical sparsity strength `10`). They are not universal defaults for every dataset. Use the frozen settings documented for the dataset and experiment being reproduced.

For simulated-missing evaluation, held-out coordinates can optionally be excluded from the weak zero-background penalty. Supply an `.npz` file containing two arrays named `rows` and `cols`:

```bash
python examples/run_from_prepared.py ... \
  --zero-penalty-exclusions data/heldout_model_coordinates.npz
```

The row and column indices must refer to `X_model.npy` (the selected model-gene matrix), not the full unfiltered gene matrix.

## Parameters

The command-line runner exposes the following parameters.

### Required parameters

| Parameter | Meaning |
|---|---|
| `--x-model` | Path to the masked model-gene expression matrix (`.npy`). |
| `--all-gene-symbols` | Path to the full gene-symbol array (`.npy`). |
| `--model-gene-symbols` | Path to the model-gene-symbol array (`.npy`), ordered to match `X_model`. |
| `--gmt` | Path to the pathway collection in GMT format. |
| `--output-dir` | Directory in which results are written. |
| `--rank` | Number of latent factors, \(r\). |
| `--alpha-cell` | Cell-graph regularization coefficient. |
| `--alpha-gene` | Gene-graph regularization coefficient. |
| `--zero-weight` | Effective weak zero-background weight, \(q_0\), constrained to the interval [0, 1]. |
| `--sparsity-strength` | Overall strength of the hierarchical gene-specific sparsity penalty. |

### Optional parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--pathway-weight` | `0.5` | Relative sparsity weight for pathway-annotated genes. |
| `--nonpathway-weight` | `1.0` | Relative sparsity weight for genes without pathway annotation. It should exceed `pathway-weight` when the sparsity strength is positive. |
| `--max-iter` | `100` | Maximum number of solver iterations. |
| `--tol` | `1e-4` | Relative objective-change tolerance used by the convergence check. |
| `--seed` | `42` | Random seed used for reproducible initialization and graph construction. |
| `--lambda-reg` | `1e-3` | L2 regularization coefficient. |
| `--lambda-w-l1` | `0.0` | L1 regularization coefficient for the cell-factor matrix \(W\). |
| `--block-size` | `500` | Number of genes processed per computational block. This is a computational setting, not a model hyperparameter. |
| `--min-pathway-genes` | `10` | Minimum number of dataset-overlapping genes required to retain a pathway. |
| `--max-pathway-genes` | `500` | Maximum number of dataset-overlapping genes allowed for a retained pathway. |
| `--gene-edge-batch-size` | `256` | Number of gene-pair edges processed per batch during gene-graph construction. |
| `--graph-pcs` | `30` | Number of principal components used to construct the cell graph, subject to the matrix dimensions. |
| `--graph-neighbors` | `15` | Number of nearest neighbors used to construct the cell graph. |
| `--graph-metric` | `euclidean` | Distance metric used for cell-neighbor search. |
| `--zero-penalty-exclusions` | None | Optional `.npz` file containing `rows` and `cols` arrays of local coordinates excluded from the zero-background penalty. |
| `--save-reconstruction` | Off | If specified, saves the reconstructed model-gene matrix as `X_hat_model.npy`. |

The pathway sparsity penalty is gene-specific: pathway-annotated genes receive `sparsity-strength × pathway-weight`, whereas non-annotated genes receive `sparsity-strength × nonpathway-weight`.

`--zero-weight` is the effective zero-background weight passed to the solver. When using the balanced weighting scheme described in the manuscript, calculate this value from the masked model-gene matrix according to the manuscript's definition before running the command. The resulting value must lie in [0, 1].

## Outputs

The example runner writes the following files to `--output-dir`:

| Output | Description |
|---|---|
| `pgnmf_factors.npz` | Contains the learned factor matrices `W` and `H`, the objective trace (`losses`), and the model-gene symbols. |
| `run_summary.json` | Records the input matrix shape, parameter settings, solver summary, pathway-membership information, and gene- and cell-graph diagnostics. |
| `X_hat_model.npy` | Optional reconstructed model-gene expression matrix (`W @ H`); generated only when `--save-reconstruction` is specified. |

The factor matrices have dimensions `W: (n_cells, rank)` and `H: (rank, n_model_genes)`.

The `pgnmf.postprocess` module also provides an exact top-k post-processing utility that preserves observed entries and selects a specified number of zero-valued positions for filling. This utility is separate from the example runner's factor-fitting step.

## Python API

The core solver can also be called directly after constructing the cell graph, gene graph, and gene-specific sparsity penalty:

```python
from pgnmf.solver import zero_aware_nmf_mu_blocked

fit = zero_aware_nmf_mu_blocked(
    X_model,
    S_cell=S_cell,
    degree_cell=degree_cell,
    alpha_cell=alpha_cell,
    S_gene=S_gene,
    degree_gene=degree_gene,
    alpha_gene=alpha_gene,
    lambda_h_gene=lambda_h_gene,
    zero_weight=zero_weight,
    rank=rank,
    max_iter=100,
    tol=1e-4,
    random_state=42,
    lambda_reg=1e-3,
    block_size=500,
)

W = fit.W
H = fit.H
X_hat_model = W @ H
```

The graph and penalty inputs can be prepared with the utilities in `pgnmf.cell_graph`, `pgnmf.gene_graph`, and `pgnmf.pathway_sparsity`.

## Reproducing manuscript analyses

Selected scripts for the main single-dataset workflow, ablations, convergence,
parameter sensitivity, and biological analyses are provided under
[`reproducibility/`](reproducibility/). Standalone plot-only scripts and
temporary visualization experiments are intentionally omitted; some analysis
scripts generate figures as part of their outputs.

See [`reproducibility/README.md`](reproducibility/README.md) for the script
index, installation instructions, required inputs, and current limitations.
The public release does not include datasets, GMT pathway files, masks, or
intermediate experiment caches. The complete manuscript analyses have not
been rerun end-to-end from this release.

## Citation

If you use PGNMF in academic work, please cite the associated manuscript.
Full bibliographic information and a persistent identifier will be added
here when they are available.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
the full license text.
