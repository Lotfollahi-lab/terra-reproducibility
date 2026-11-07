"""
Per-gene normalization factors of a cohort of silver AnnData files (raw counts in X, Ensembl ids in
var['ensembl_id']), shared by the ablation scripts.

The factors pool cells as `stjepa-reproducibility/src/compute_norm_factors.py` does (concatenated cells): per gene, the
mean over all cells of the files that probe it and the nonzero mean over its cells with a count > 0, of the raw counts
and of the gene-corrected read depth (gcrd) normalized counts. Gcrd scales every cell to the target read depth
T(G) = a + b * G, with G the number of genes of its file. The constants default to those of the tokenizer
(`normalize_by_gene_corrected_read_depth`), so that T cancels within a dataset when the tokenizer divides its gcrd counts
by these factors. The factor scripts of stjepa-reproducibility used other constants (`GCRD_CONSTANTS`), which multiplies
the count values of each gene panel by T_tokenizer(G) / T_factors(G).

Only numpy, scipy, pandas and anndata are used, so the module runs in the TERRA and in the ST-JEPA v2 environment.

Usage:
```python
from norm_factors import check_norm_factors, compute_norm_factors

norm_factor_df = compute_norm_factors(["batch0.h5ad", "batch1.h5ad"])     # file paths or AnnData objects

# Guard (pinned code on PYTHONPATH): the gcrd columns match the tokenizer's own normalizer
from terra.preprocessors.normalizers import normalize_by_gene_corrected_read_depth
ok = check_norm_factors("norm_factors.csv", ["batch0.h5ad"], normalize_by_gene_corrected_read_depth)
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Iterator

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Gcrd constants (a, b) of T(G) = a + b * G
TOKENIZER_GCRD_CONSTANTS = (153.4768, 0.0487)
GCRD_CONSTANTS = {
    # terra/preprocessors/normalizers.py, normalize_by_gene_corrected_read_depth (the tokenizer)
    "tokenizer": TOKENIZER_GCRD_CONSTANTS,
    # stjepa-reproducibility/analysis/tokenization/compute_cohort_norm_factors.py (human cohort, hst_corpus_90m)
    "human_cohort": (154.01749299636748, 0.05644618485013678),
    # stjepa-reproducibility/src/compute_norm_factors.py (mouse corpus, default cohort mst_corpus_10m)
    "mouse_corpus": (-752.364435146444, 0.8921817095038856),
    # terra-reproducibility/analysis/data_preparation/compute_cohort_norm_factors.py (TERRA-110M)
    "terra_110m": (151.1754621380761, 0.04705053555567159),
    # Same script, commented-out alternative (hst_corpus_80m)
    "hst_corpus_80m": (150.6692, 0.0646),
}
FACTOR_COLUMNS = ["nonzero_mean",
                  "mean",
                  "gene_corrected_read_depth_nonzero_mean",
                  "gene_corrected_read_depth_mean"]
NORM_FACTOR_COLUMNS = ["gene_id", *FACTOR_COLUMNS, "n_cells", "n_nonzero_cells"]


def target_read_depth(n_genes: int,
                      basis_target_read_depth: float = TOKENIZER_GCRD_CONSTANTS[0],
                      target_read_depth_per_gene: float = TOKENIZER_GCRD_CONSTANTS[1]) -> float:
    """Target read depth T(G) = a + b * G of gene-corrected read depth normalization (as in the tokenizer)."""
    return basis_target_read_depth + n_genes * target_read_depth_per_gene


def iter_adatas(adatas: ad.AnnData | str | Path | Iterable[ad.AnnData | str | Path]) -> Iterator[tuple[str, ad.AnnData]]:
    """Yield (name, AnnData) for AnnData objects or h5ad file paths; files are read one at a time."""
    if isinstance(adatas, (ad.AnnData, str, Path)):
        adatas = [adatas]
    for idx, adata in enumerate(adatas):
        if isinstance(adata, ad.AnnData):
            yield f"AnnData {idx}", adata
        else:
            yield str(adata), ad.read_h5ad(adata)


def file_statistics(adata: ad.AnnData,
                    basis_target_read_depth: float = TOKENIZER_GCRD_CONSTANTS[0],
                    target_read_depth_per_gene: float = TOKENIZER_GCRD_CONSTANTS[1]) -> dict:
    """
    Per-gene statistics of one silver file: raw count sums, gcrd sums (cells scaled to T(G) counts), depth-normalized
    sums (cells scaled to 1 count) and the number of cells with a count > 0, plus n_cells, n_genes (G) and the number
    of cells without counts (these contribute zero to the normalized sums).
    """
    X = sp.csr_matrix(adata.X, dtype=np.float64)
    n_cells, n_genes = X.shape
    if X.nnz and X.data.min() < 0:
        raise ValueError("X has negative values; expected raw counts.")
    cell_sums = np.asarray(X.sum(axis=1)).ravel()
    n_zero_count_cells = int(np.sum(cell_sums == 0))

    # Gene-corrected read depth normalization (normalizers.normalize_by_gene_corrected_read_depth); kept as in the
    # original prepare_ovca_data.py so that the factors are bitwise unchanged
    scale = np.divide(target_read_depth(n_genes, basis_target_read_depth, target_read_depth_per_gene), cell_sums,
                      out=np.zeros(n_cells), where=cell_sums > 0)
    X_gcrd = sp.diags(scale) @ X
    gcrd_sums = np.asarray(X_gcrd.sum(axis=0)).ravel()
    del X_gcrd
    depth_normalized_sums = X.T @ np.divide(1.0, cell_sums, out=np.zeros(n_cells), where=cell_sums > 0)

    return {"gene_ids": np.asarray(adata.var["ensembl_id"]).astype(str),
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_zero_count_cells": n_zero_count_cells,
            "sums": np.asarray(X.sum(axis=0)).ravel(),
            "gcrd_sums": gcrd_sums,
            "depth_normalized_sums": np.asarray(depth_normalized_sums).ravel(),
            "n_nonzero_cells": np.asarray((X > 0).sum(axis=0)).ravel()}


def compute_norm_factors(adatas: ad.AnnData | str | Path | Iterable[ad.AnnData | str | Path],
                         basis_target_read_depth: float = TOKENIZER_GCRD_CONSTANTS[0],
                         target_read_depth_per_gene: float = TOKENIZER_GCRD_CONSTANTS[1],
                         verbose: bool = False) -> pd.DataFrame:
    """
    Per-gene normalization factors of a cohort of silver files (paths, read one at a time, or AnnData objects), pooled
    over all cells of the files that probe each gene. A gene listed twice in a file is pooled as two genes' cells, as by
    `src/compute_norm_factors.py`. Raises if a gene has no counts in the cohort.

    Returns the columns gene_id, nonzero_mean, mean, gene_corrected_read_depth_nonzero_mean,
    gene_corrected_read_depth_mean, n_cells and n_nonzero_cells, genes in order of first appearance.
    """
    gene_index = {}
    sums, gcrd_sums = np.zeros(0), np.zeros(0)
    n_cells, n_nonzero_cells = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    for name, adata in iter_adatas(adatas):
        stats = file_statistics(adata, basis_target_read_depth, target_read_depth_per_gene)
        idx = np.array([gene_index.setdefault(gene_id, len(gene_index)) for gene_id in stats["gene_ids"]], dtype=int)
        n_new = len(gene_index) - len(sums)
        sums, gcrd_sums = np.append(sums, np.zeros(n_new)), np.append(gcrd_sums, np.zeros(n_new))
        n_cells = np.append(n_cells, np.zeros(n_new, dtype=np.int64))
        n_nonzero_cells = np.append(n_nonzero_cells, np.zeros(n_new, dtype=np.int64))
        np.add.at(sums, idx, stats["sums"])
        np.add.at(gcrd_sums, idx, stats["gcrd_sums"])
        np.add.at(n_cells, idx, stats["n_cells"])
        np.add.at(n_nonzero_cells, idx, stats["n_nonzero_cells"])

        n_duplicates = len(idx) - len(np.unique(idx))
        if verbose or n_duplicates or stats["n_zero_count_cells"]:
            print(f"{name}: {stats['n_cells']} cells, {stats['n_genes']} genes, target read depth "
                  f"{target_read_depth(stats['n_genes'], basis_target_read_depth, target_read_depth_per_gene):.4f}"
                  + (f"; WARNING: {n_duplicates} duplicate Ensembl ids (pooled)" if n_duplicates else "")
                  + (f"; WARNING: {stats['n_zero_count_cells']} cells without counts" if stats["n_zero_count_cells"]
                     else "") + ".")

    if np.any(n_nonzero_cells == 0):
        raise ValueError(f"{np.sum(n_nonzero_cells == 0)} genes have no counts; their factors would be 0 or NaN.")
    norm_factor_df = pd.DataFrame({"gene_id": np.array(list(gene_index), dtype=object),
                                   "nonzero_mean": sums / n_nonzero_cells,
                                   "mean": sums / n_cells,
                                   "gene_corrected_read_depth_nonzero_mean": gcrd_sums / n_nonzero_cells,
                                   "gene_corrected_read_depth_mean": gcrd_sums / n_cells,
                                   "n_cells": n_cells,
                                   "n_nonzero_cells": n_nonzero_cells})
    factors = norm_factor_df[FACTOR_COLUMNS].values
    if not (np.isfinite(factors).all() and (factors > 0).all()):
        raise ValueError("Normalization factors must be finite and positive.")
    return norm_factor_df


def verify_gcrd_factors(norm_factor_df: pd.DataFrame | str | Path,
                        adatas: ad.AnnData | str | Path | Iterable[ad.AnnData | str | Path],
                        normalize_fn: Callable,
                        n_genes: int | None = 200,
                        seed: int = 0,
                        chunk_size: int = 10_000) -> pd.DataFrame:
    """
    Recompute the gcrd columns of a factor file for a random sample of its genes (all genes if `n_genes` is None) with
    the tokenizer's own normalizer and compare. `normalize_fn` is the pinned
    `terra.preprocessors.normalizers.normalize_by_gene_corrected_read_depth`; it is called on the counts of every
    file as the tokenizer calls it (default constants), in chunks of cells, which leaves its per-cell values unchanged.

    Returns one row per sampled gene: n_cells, n_nonzero_cells and the two gcrd columns of the factor file (`*_csv`)
    and recomputed, and the relative differences `rel_diff_mean` and `rel_diff_nonzero_mean` (NaN if no file probes
    the gene).
    """
    if not isinstance(norm_factor_df, pd.DataFrame):
        norm_factor_df = pd.read_csv(norm_factor_df)
    factor_df = norm_factor_df.drop_duplicates("gene_id").set_index("gene_id")
    gene_ids = factor_df.index.values.astype(str)
    if n_genes is not None and n_genes < len(gene_ids):
        gene_ids = np.sort(np.random.default_rng(seed).choice(gene_ids, n_genes, replace=False))
    position = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}

    gcrd_sums = np.zeros(len(gene_ids))
    n_cells = np.zeros(len(gene_ids), dtype=np.int64)
    n_nonzero_cells = np.zeros(len(gene_ids), dtype=np.int64)
    for _, adata in iter_adatas(adatas):
        file_gene_ids = np.asarray(adata.var["ensembl_id"]).astype(str)
        columns = np.array([col for col, gene_id in enumerate(file_gene_ids) if gene_id in position], dtype=int)
        rows = np.array([position[gene_id] for gene_id in file_gene_ids[columns]], dtype=int)
        for start in range(0, adata.n_obs, chunk_size):
            X_gcrd = normalize_fn(adata.X[start:start + chunk_size])
            X_gcrd = X_gcrd.tocsc()[:, columns].toarray() if sp.issparse(X_gcrd) else np.asarray(X_gcrd)[:, columns]
            X_gcrd = np.asarray(X_gcrd, dtype=np.float64)
            np.add.at(gcrd_sums, rows, X_gcrd.sum(axis=0))
            np.add.at(n_nonzero_cells, rows, (X_gcrd > 0).sum(axis=0))
        np.add.at(n_cells, rows, adata.n_obs)

    with np.errstate(divide="ignore", invalid="ignore"):
        comparison_df = pd.DataFrame({
            "gene_id": gene_ids,
            "n_cells_csv": factor_df.loc[gene_ids, "n_cells"].values if "n_cells" in factor_df else np.nan,
            "n_cells": n_cells,
            "n_nonzero_cells_csv": (factor_df.loc[gene_ids, "n_nonzero_cells"].values
                                    if "n_nonzero_cells" in factor_df else np.nan),
            "n_nonzero_cells": n_nonzero_cells,
            "gene_corrected_read_depth_mean_csv": factor_df.loc[gene_ids, "gene_corrected_read_depth_mean"].values,
            "gene_corrected_read_depth_mean": np.where(n_cells > 0, gcrd_sums / n_cells, np.nan),
            "gene_corrected_read_depth_nonzero_mean_csv":
                factor_df.loc[gene_ids, "gene_corrected_read_depth_nonzero_mean"].values,
            "gene_corrected_read_depth_nonzero_mean": np.where(n_nonzero_cells > 0, gcrd_sums / n_nonzero_cells,
                                                               np.nan)})
    for column, rel_diff_column in [("gene_corrected_read_depth_mean", "rel_diff_mean"),
                                    ("gene_corrected_read_depth_nonzero_mean", "rel_diff_nonzero_mean")]:
        comparison_df[rel_diff_column] = (np.abs(comparison_df[column] - comparison_df[f"{column}_csv"])
                                          / np.abs(comparison_df[f"{column}_csv"]))
    return comparison_df


def check_norm_factors(norm_factor_file_path: str | Path,
                       silver_file_paths: ad.AnnData | str | Path | Iterable[ad.AnnData | str | Path],
                       normalize_fn: Callable,
                       n_genes: int | None = 200,
                       rtol: float = 1e-5,
                       seed: int = 0,
                       report_file_path: str | Path | None = None) -> bool:
    """
    Guard that a factor file matches the tokenizer on its cohort (the silver files, paths or AnnData objects): it lists
    every gene of the files, its n_cells and n_nonzero_cells are those of the files, and its gcrd columns equal those
    recomputed with the tokenizer's normalizer (`normalize_fn`, see `verify_gcrd_factors`) for `n_genes` sampled genes
    within the relative tolerance `rtol`. Prints the result and returns whether the factors pass; the per-gene
    comparison is written to `report_file_path` if given.
    """
    norm_factor_df = pd.read_csv(norm_factor_file_path)
    missing_columns = [column for column in NORM_FACTOR_COLUMNS if column not in norm_factor_df]
    if missing_columns:
        print(f"Norm factors {norm_factor_file_path}: columns {missing_columns} missing: MISMATCH")
        return False

    adatas = [adata for _, adata in iter_adatas(silver_file_paths)]
    probed_gene_ids = set(np.concatenate([np.asarray(adata.var["ensembl_id"]).astype(str) for adata in adatas]))
    n_missing = len(probed_gene_ids - set(norm_factor_df["gene_id"].astype(str)))
    n_other = len(set(norm_factor_df["gene_id"].astype(str)) - probed_gene_ids)

    comparison_df = verify_gcrd_factors(norm_factor_df, adatas, normalize_fn, n_genes=n_genes, seed=seed)
    if report_file_path is not None:
        comparison_df.to_csv(report_file_path, index=False)
    max_rel_diff = comparison_df[["rel_diff_mean", "rel_diff_nonzero_mean"]].max().max()
    counts_match = ((comparison_df["n_cells"] == comparison_df["n_cells_csv"])
                    & (comparison_df["n_nonzero_cells"] == comparison_df["n_nonzero_cells_csv"])).all()
    values_match = (comparison_df[["rel_diff_mean", "rel_diff_nonzero_mean"]].notna().all().all()
                    and max_rel_diff <= rtol)
    ok = n_missing == 0 and counts_match and values_match
    print(f"Norm factors {norm_factor_file_path}: {len(norm_factor_df)} genes ({n_missing} genes of the silver data "
          f"missing, {n_other} other genes); n_cells and n_nonzero_cells of {len(comparison_df)} sampled genes "
          f"{'match' if counts_match else 'DIFFER from'} the silver data; gcrd columns vs the tokenizer's normalizer: "
          f"max relative difference {max_rel_diff:.2e} (tolerance {rtol:g}): {'OK' if ok else 'MISMATCH'}")
    return ok
