from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class PathwayMembershipInfo:
    n_input_genes: int
    n_retained_pathways: int
    n_pathway_genes: int
    n_nonpathway_genes: int
    min_pathway_genes: int
    max_pathway_genes: int
    gene_symbol_source: str

    def to_dict(self):
        return asdict(self)


def get_gene_symbols(adata, gene_symbol_column: str | None = None) -> tuple[np.ndarray, str]:
    """Return one gene symbol per adata.var row without changing the AnnData object."""
    if gene_symbol_column:
        if gene_symbol_column not in adata.var.columns:
            raise KeyError(
                f"gene symbol column {gene_symbol_column!r} not found in adata.var; "
                f"available columns={list(adata.var.columns)}"
            )
        symbols = adata.var[gene_symbol_column].astype(str).to_numpy()
        source = f"var[{gene_symbol_column!r}]"
    else:
        symbols = adata.var_names.astype(str).to_numpy()
        source = "var_names"
    return symbols, source


def pathway_membership_from_gmt(
    gmt_path: str | Path,
    dataset_gene_symbols: np.ndarray,
    min_pathway_genes: int = 10,
    max_pathway_genes: int = 500,
    gene_symbol_source: str = "var_names",
) -> tuple[np.ndarray, PathwayMembershipInfo]:
    """
    Mark dataset genes that occur in at least one retained GMT pathway.

    Pathways are retained after intersection with the dataset gene universe, using
    the same 10--500 retained-gene logic described in the manuscript by default.
    Matching is case-insensitive, which is appropriate for the supplied human
    *.Hs.symbols.gmt file and avoids harmless capitalization mismatches.
    """
    gmt_path = Path(gmt_path)
    if not gmt_path.exists():
        raise FileNotFoundError(gmt_path)
    if min_pathway_genes < 1 or max_pathway_genes < min_pathway_genes:
        raise ValueError("invalid pathway-size bounds")

    symbols = np.asarray(dataset_gene_symbols, dtype=str)
    upper_to_indices: dict[str, list[int]] = {}
    for i, g in enumerate(symbols):
        key = g.strip().upper()
        if key:
            upper_to_indices.setdefault(key, []).append(i)
    dataset_universe = set(upper_to_indices)

    pathway_gene_upper: set[str] = set()
    retained = 0
    with gmt_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            overlap = genes & dataset_universe
            if min_pathway_genes <= len(overlap) <= max_pathway_genes:
                retained += 1
                pathway_gene_upper.update(overlap)

    membership = np.zeros(symbols.size, dtype=bool)
    for g in pathway_gene_upper:
        for idx in upper_to_indices.get(g, []):
            membership[idx] = True

    info = PathwayMembershipInfo(
        n_input_genes=int(symbols.size),
        n_retained_pathways=int(retained),
        n_pathway_genes=int(membership.sum()),
        n_nonpathway_genes=int((~membership).sum()),
        min_pathway_genes=int(min_pathway_genes),
        max_pathway_genes=int(max_pathway_genes),
        gene_symbol_source=gene_symbol_source,
    )
    return membership, info


def make_two_level_penalty(
    pathway_membership_model: np.ndarray,
    strength: float,
    pathway_weight: float = 0.5,
    nonpathway_weight: float = 1.0,
) -> np.ndarray:
    """
    Gene-specific L1 penalties for H.

    lambda_j = strength * pathway_weight for pathway genes
             = strength * nonpathway_weight for non-pathway genes.

    The Stage-2B-1 experiment fixes the relative weights and selects only the
    overall strength from a label-free held-out reconstruction validation split.
    """
    membership = np.asarray(pathway_membership_model, dtype=bool)
    if strength < 0 or pathway_weight < 0 or nonpathway_weight < 0:
        raise ValueError("sparsity strengths/weights must be nonnegative")
    if pathway_weight >= nonpathway_weight and strength > 0:
        raise ValueError("pathway_weight must be smaller than nonpathway_weight")
    return np.where(
        membership,
        float(strength) * float(pathway_weight),
        float(strength) * float(nonpathway_weight),
    ).astype(np.float32)
