import numpy as np
from scipy import sparse

from pgnmf.solver import zero_aware_nmf_mu_blocked


def _symmetric_graph(n, rng):
    a = rng.random((n, n), dtype=np.float32)
    a = np.triu((a > 0.75).astype(np.float32), 1)
    a = a + a.T
    graph = sparse.csr_matrix(a)
    degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    return graph, degree


def test_exact_gene_block_solver_smoke():
    rng = np.random.default_rng(7)
    n_cells, n_genes = 12, 17
    X = rng.random((n_cells, n_genes), dtype=np.float32)
    X[rng.random(X.shape) < 0.35] = 0.0

    S_cell, d_cell = _symmetric_graph(n_cells, rng)
    S_gene, d_gene = _symmetric_graph(n_genes, rng)
    penalty = np.full(n_genes, 0.05, dtype=np.float32)

    result = zero_aware_nmf_mu_blocked(
        X,
        S_cell=S_cell,
        degree_cell=d_cell,
        alpha_cell=0.1,
        S_gene=S_gene,
        degree_gene=d_gene,
        alpha_gene=0.1,
        lambda_h_gene=penalty,
        zero_weight=0.05,
        rank=3,
        max_iter=5,
        tol=0.0,
        random_state=42,
        lambda_reg=1e-3,
        block_size=6,
        verbose=False,
    )

    assert result.W.shape == (n_cells, 3)
    assert result.H.shape == (3, n_genes)
    assert np.all(np.isfinite(result.W))
    assert np.all(np.isfinite(result.H))
    assert np.all(result.W >= 0)
    assert np.all(result.H >= 0)
    assert len(result.losses) == result.n_iter



def test_iteration_callback_reports_completed_exact_block_updates():
    rng = np.random.default_rng(9)
    n_cells, n_genes = 8, 11
    X = rng.random((n_cells, n_genes), dtype=np.float32)
    X[rng.random(X.shape) < 0.25] = 0.0
    S_cell, d_cell = _symmetric_graph(n_cells, rng)
    S_gene, d_gene = _symmetric_graph(n_genes, rng)
    seen = []

    def observe(iteration, loss, W, H):
        # Observer is read-only; snapshots confirm dimensions and finite state.
        seen.append((iteration, float(loss), W.shape, H.shape,
                     bool(np.all(np.isfinite(W))), bool(np.all(np.isfinite(H)))))

    result = zero_aware_nmf_mu_blocked(
        X, S_cell=S_cell, degree_cell=d_cell, alpha_cell=0.1,
        S_gene=S_gene, degree_gene=d_gene, alpha_gene=0.1,
        lambda_h_gene=np.full(n_genes, 0.02, dtype=np.float32),
        zero_weight=0.05, rank=3, max_iter=4, tol=0.0,
        random_state=17, block_size=4, verbose=False,
        iteration_callback=observe,
    )
    assert len(seen) == result.n_iter
    assert [x[0] for x in seen] == list(range(1, result.n_iter + 1))
    assert all(x[2] == (n_cells, 3) and x[3] == (3, n_genes) for x in seen)
    assert all(x[4] and x[5] for x in seen)
