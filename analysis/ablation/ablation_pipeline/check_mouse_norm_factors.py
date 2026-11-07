"""
This script checks the per-gene normalization factors of the six mouse gene-corrected read depth (gcrd) mean arms of
the normalization ablation (Supp. Fig. 5) on the farm. The tokenizer scales every cell to T(G) = 153.4768 + 0.0487 * G
counts (G: genes of its silver file) and then divides by the gcrd factors of `norm_factors.csv`, which the factor scripts
of stjepa-reproducibility computed with other constants. With matched constants T cancels within a dataset; otherwise
the values of each gene panel are multiplied by T_tokenizer(G) / T_factors(G).

(a) Factor files: columns, md5 and rows of `norm_factors.csv` of every model folder (the copy that the mouse evaluation
    used when it re-tokenized the silver data with `harmonize_tokenize_embed_pipeline`); whether the six are identical.
(b) Constants: a least-squares fit of (a, b) to the gcrd columns, from the per-gene depth-normalized count sums of
    every silver file of the training data; residual and closest candidate (`norm_factors.GCRD_CONSTANTS`); whether
    the factor cohort is these files (n_cells and raw count sums of the CSV match) or a larger corpus; the multiplier
    T_tokenizer(G) / T_factors(G) of every file.
(c) Train/eval consistency: for a sample of gold (training) cells, both streams are recomputed from the silver counts
    with the pinned normalizers and the model-folder CSV, as `CellGraphTokenizer` computes them. The index-cell block
    (the first 64 tokens) must hold the top-64 genes by the recomputed rank score in descending order, ties at the cut
    tolerated, then padding; its values must equal the recomputed count values within the relative tolerance --rtol.
Exits with an error if a model folder is incomplete, its norm methods differ from the plan, gold cells are missing from
the silver data or a check (c) fails.

Run in the ST-JEPA v2 environment with the pinned code on PYTHONPATH (as check_setup.py).

Usage:
```bash
source datasets/ovca.env   # any dataset's parameter file: only the plan, mouse artifact and data paths are used
source "${ENV_PATH}/bin/activate" && export PYTHONPATH="${CODE_PATH}/src"
python check_mouse_norm_factors.py --plan_file_path "${PLAN_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}" \
    [--n_cells 300] [--dataset_name <silver dataset folder>]
```
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from ablation_utils import NORM_METHOD_KEYS, load_plan
from norm_factors import GCRD_CONSTANTS, TOKENIZER_GCRD_CONSTANTS, file_statistics, iter_adatas, target_read_depth

# The gcrd mean arms of the mouse normalization ablation (variant names of ablation_plan.yaml)
GCRD_VARIANTS = ["gene_corrected_read_depth_mean_no_norm_rank",
                 "gene_corrected_read_depth_nz_mean_no_norm_rank",
                 "gene_corrected_read_depth_mean",
                 "gene_corrected_read_depth_nz_mean",
                 "no_norm_gene_corrected_read_depth_mean_rank",
                 "no_norm_gene_corrected_read_depth_nz_mean_rank"]
# Factor column prefix of the cell norm method (CellGraphTokenizer._tokenize_adata)
FACTOR_COLUMN_PREFIX = {None: "", "read_depth": "read_depth_", "gene_corrected_read_depth": "gene_corrected_read_depth_"}
GCRD = "gene_corrected_read_depth"
# Raw count sums of a factor file vs the silver files (cohort test), and fit residual of a matching candidate
SUM_RTOL = 1e-6
CANDIDATE_RTOL = 1e-4


def md5sum(file_path: Path) -> str:
    return hashlib.md5(Path(file_path).read_bytes()).hexdigest()


def factor_script(columns: list) -> str:
    """The factor scripts that write these columns (a hint; (b) fits the constants)."""
    if any(column.endswith("median") for column in columns):
        return "stjepa-reproducibility src/compute_norm_factors.py with median stats (default constants: mouse_corpus)"
    if "n_nonzero_cells" not in columns:
        return "stjepa-reproducibility analysis/tokenization/compute_cohort_norm_factors.py (default: human_cohort)"
    return ("src/compute_norm_factors.py --no-compute_median_stats, terra-reproducibility "
            "compute_cohort_norm_factors.py (default: terra_110m) or norm_factors.py")


def load_silver(silver_directory_path: Path) -> list[dict]:
    """The silver files of a dataset (all *.h5ad below the folder, as the tokenizer reads them) with their statistics."""
    file_paths = sorted(silver_directory_path.glob("**/*.h5ad"))
    if not file_paths:
        raise FileNotFoundError(f"No h5ad files in {silver_directory_path}.")
    silver = []
    for name, adata in iter_adatas(file_paths):
        silver.append({"name": Path(name).name,
                       "adata": adata,
                       "stats": file_statistics(adata),
                       "cell_ids": np.asarray(adata.obs["cell_id"]).astype(str)})
    return silver


def pool_silver_statistics(silver: list[dict]) -> pd.DataFrame:
    """Per-gene statistics pooled over the silver files (S: depth-normalized sums, GS: sums of G_f * S_fj)."""
    gene_index = {}
    for silver_file in silver:
        for gene_id in silver_file["stats"]["gene_ids"]:
            gene_index.setdefault(gene_id, len(gene_index))
    pooled = {key: np.zeros(len(gene_index)) for key in ["sums", "S", "GS"]}
    pooled.update({key: np.zeros(len(gene_index), dtype=np.int64)
                   for key in ["n_cells", "n_cells_detected", "n_nonzero_cells", "n_files"]})
    for silver_file in silver:
        stats = silver_file["stats"]
        idx = np.array([gene_index[gene_id] for gene_id in stats["gene_ids"]], dtype=int)
        np.add.at(pooled["sums"], idx, stats["sums"])
        np.add.at(pooled["S"], idx, stats["depth_normalized_sums"])
        np.add.at(pooled["GS"], idx, stats["n_genes"] * stats["depth_normalized_sums"])
        np.add.at(pooled["n_cells"], idx, stats["n_cells"])
        np.add.at(pooled["n_cells_detected"], idx, stats["n_cells"] * (stats["n_nonzero_cells"] > 0))
        np.add.at(pooled["n_nonzero_cells"], idx, stats["n_nonzero_cells"])
        np.add.at(pooled["n_files"], idx, 1)
    return pd.DataFrame(pooled, index=pd.Index(list(gene_index), name="gene_id"))


def relative_residuals(y: np.ndarray, A: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return (y - A @ theta) / y


def fit_gcrd_constants(norm_factor_df: pd.DataFrame, pooled_df: pd.DataFrame, silver: list[dict]) -> dict:
    """
    (b) Fit (a, b) of a factor file. If its cohort is these silver files, its gcrd sum of gene j
    (gene_corrected_read_depth_mean * n_cells) is a * S_j + b * GS_j, with S_j = sum_f S_fj and GS_j = sum_f G_f * S_fj
    (S_fj: count sum of gene j in file f after scaling every cell to 1 count; G_f: genes of file f); least squares on
    the relative residuals over genes. For a larger cohort, the cohort's gcrd means are approximated by these files'.
    """
    factor_df = norm_factor_df.assign(gene_id=norm_factor_df["gene_id"].astype(str)).drop_duplicates("gene_id")
    factor_df = factor_df.set_index("gene_id")
    genes = pooled_df.index[pooled_df.index.isin(factor_df.index)]
    pooled, factors = pooled_df.loc[genes], factor_df.loc[genes]

    # Cohort: the factor file's n_cells (over the files that probe, or detect, a gene) and raw sums are these files'
    if "n_cells" in factors:
        n_cells = factors["n_cells"].values
        n_cells_match = (n_cells == pooled["n_cells"].values) | (n_cells == pooled["n_cells_detected"].values)
    else:
        n_cells, n_cells_match = pooled["n_cells"].values, np.ones(len(genes), dtype=bool)
    sums_match = np.abs(factors["mean"].values * n_cells - pooled["sums"].values) <= SUM_RTOL * pooled["sums"].values
    matched = n_cells_match & sums_match
    is_cohort = len(genes) == len(pooled_df) and len(genes) > 0 and bool(matched.all())

    A = pooled[["S", "GS"]].values
    if is_cohort:
        y = factors[f"{GCRD}_mean"].values * n_cells
    else:
        y, A = factors[f"{GCRD}_mean"].values, A / pooled["n_cells"].values[:, None]
    keep = np.isfinite(y) & (y > 0) & (A[:, 0] > 0)
    y, A = y[keep], A[keep]
    n_genes_per_file = np.array([silver_file["stats"]["n_genes"] for silver_file in silver])
    identifiable = len(np.unique(n_genes_per_file)) > 1
    if identifiable:
        theta = np.linalg.lstsq(A / y[:, None], np.ones(len(y)), rcond=None)[0]
    else:
        # One gene panel: only T(G) = a + b * G is identifiable (reported as a, with b = 0)
        theta = np.array([np.linalg.lstsq(A[:, :1] / y[:, None], np.ones(len(y)), rcond=None)[0][0], 0.0])
    residuals = relative_residuals(y, A, theta)
    candidates = {name: float(np.sqrt(np.mean(relative_residuals(y, A, np.array(constants)) ** 2)))
                  for name, constants in GCRD_CONSTANTS.items()}
    closest = min(candidates, key=candidates.get)

    fit = {"a": theta[0], "b": theta[1], "identifiable": identifiable, "n_genes_fit": len(y),
           "rms": float(np.sqrt(np.mean(residuals ** 2))), "max": float(np.abs(residuals).max()),
           "candidates": candidates, "closest": closest, "matches": candidates[closest] <= CANDIDATE_RTOL,
           "is_cohort": is_cohort, "n_matched": int(matched.sum()), "n_genes": len(genes),
           "n_missing": len(pooled_df) - len(genes), "n_other": len(factor_df) - len(genes),
           # Multiplier of the count values of each file
           "multipliers": [target_read_depth(n_genes) / (theta[0] + theta[1] * n_genes) for n_genes in n_genes_per_file]}
    if is_cohort and "n_nonzero_cells" in factors:
        y_nonzero = (factors[f"{GCRD}_nonzero_mean"].values * factors["n_nonzero_cells"].values)[keep]
        fit["rms_nonzero"] = float(np.sqrt(np.mean(relative_residuals(y_nonzero, A, theta) ** 2)))
    return fit


def effective_scales(norm_factor_df: pd.DataFrame, pooled_df: pd.DataFrame, silver: list[dict],
                     gene_norm_method: str) -> list[float]:
    """
    Scale of the count values of each file relative to consistent factors: median over its genes of the gcrd factor
    computed with the tokenizer constants on these files, divided by the factor file's. Equals
    T_tokenizer(G) / T_factors(G) for genes of one panel if the cohort is these files; for a larger cohort it also holds
    the expression difference between the cohort and these files.
    """
    a, b = TOKENIZER_GCRD_CONSTANTS
    factor_df = norm_factor_df.assign(gene_id=norm_factor_df["gene_id"].astype(str)).drop_duplicates("gene_id")
    factors = factor_df.set_index("gene_id")[f"{GCRD}_{gene_norm_method}"]
    denominator = pooled_df["n_cells"] if gene_norm_method == "mean" else pooled_df["n_nonzero_cells"]
    ratio = (a * pooled_df["S"] + b * pooled_df["GS"]) / denominator / factors.reindex(pooled_df.index)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    return [float(ratio.reindex(silver_file["stats"]["gene_ids"]).median()) for silver_file in silver]


def recompute_stream(X, probed_genes: pd.Series, cell_norm_method: str | None, gene_norm_method: str | None,
                     count_norm_method: str | None, norm_factor_file_path: Path, normalizers) -> np.ndarray:
    """
    One stream (rank or count) of `CellGraphTokenizer._tokenize_adata` for the cells (rows) of the raw counts X, with
    the pinned normalizers called as the tokenizer calls them. Every step acts on single cells or single genes, so a
    subset of cells gets the values of the whole file.
    """
    if cell_norm_method == "read_depth":
        X = normalizers.normalize_by_read_depth(X)
    elif cell_norm_method == GCRD:
        X = normalizers.normalize_by_gene_corrected_read_depth(X)
    elif cell_norm_method is not None:
        raise NotImplementedError(f"Cell norm method {cell_norm_method}.")
    if gene_norm_method in ("mean", "nonzero_mean"):
        X = normalizers.normalize_by_factor(X,
                                            norm_factor_file_path=norm_factor_file_path,
                                            probed_genes=probed_genes,
                                            norm_factor=f"{FACTOR_COLUMN_PREFIX[cell_norm_method]}{gene_norm_method}")
    elif gene_norm_method is not None:
        raise NotImplementedError(f"Gene norm method {gene_norm_method}.")
    if count_norm_method == "shifted_log":
        X = normalizers.normalize_by_shifted_log(X)
    elif count_norm_method is not None:
        raise NotImplementedError(f"Count norm method {count_norm_method}.")
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float64)


def compare_index_block(gold_tokens: np.ndarray, gold_values: np.ndarray, token_columns: dict,
                        rank_scores: np.ndarray, count_values: np.ndarray, rtol: float) -> tuple[str | None, bool, float]:
    """
    Compare the index-cell block of a gold row with the recomputed streams of its cell (`rank_scores` and
    `count_values` over the coding genes of its file, `token_columns`: token -> gene column). The block must list the
    genes with a positive rank score in descending order of score: the top len(block) of them, i.e. every gene above
    the cut and none below it (ties at the cut are broken by the tokenizer's sort), then padding with token and value 0.
    Zero-score genes may follow the positive ones when fewer genes are expressed than the block holds (explicit zeros
    on the tokenizer's sparse path). Returns (problem or None, whether the cut falls in a tie, max relative difference
    of the values).
    """
    block_length = len(gold_tokens)
    n_block = int(np.sum(gold_tokens != 0))
    if np.any(gold_tokens[n_block:] != 0) or np.any(gold_values[n_block:] != 0):
        return "gene tokens or values after the padding", False, np.nan
    columns = np.array([token_columns.get(int(token), -1) for token in gold_tokens[:n_block]], dtype=int)
    if np.any(columns < 0):
        return f"{np.sum(columns < 0)} tokens are not genes of the cell's silver file", False, np.nan
    if len(np.unique(columns)) < n_block:
        return "duplicate gene tokens", False, np.nan

    scores = rank_scores[columns]
    positive_scores = np.sort(rank_scores[rank_scores > 0])[::-1]
    cut = positive_scores[block_length - 1] if len(positive_scores) >= block_length else 0.0
    tie_at_cut = len(positive_scores) > block_length and positive_scores[block_length] >= cut * (1 - rtol)
    n_expected = min(block_length, len(positive_scores))
    n_above_missing = np.sum(~np.isin(np.flatnonzero(rank_scores > cut * (1 + rtol)), columns))
    if n_above_missing:
        return f"{n_above_missing} genes above the top-{block_length} cut are missing", tie_at_cut, np.nan
    if np.any(scores < cut * (1 - rtol)) or n_block < n_expected:
        return (f"{np.sum(scores < cut * (1 - rtol))} genes below the cut, {n_block} genes (expected "
                f"{n_expected})"), tie_at_cut, np.nan
    if np.any(scores[1:] > scores[:-1] * (1 + rtol)):
        return "genes not in descending order of the rank score", tie_at_cut, np.nan

    expected_values = count_values[columns]
    both_nan = np.isnan(gold_values[:n_block]) & np.isnan(expected_values)
    differences = np.where(both_nan, 0.0, np.abs(gold_values[:n_block] - expected_values))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diffs = np.where(differences == 0, 0.0, differences / np.abs(expected_values))
    max_rel_diff = float(rel_diffs.max()) if n_block else 0.0
    if not np.all(rel_diffs <= rtol):
        return (f"{np.sum(~(rel_diffs <= rtol))} values differ from the recomputed count values (max relative "
                f"difference {max_rel_diff:.2e})"), tie_at_cut, max_rel_diff
    return None, tie_at_cut, max_rel_diff


def check_consistency(arm: dict, silver: list[dict], cell_index: dict, normalizers, n_cells: int, seed: int,
                      rtol: float) -> dict:
    """(c) Compare the index-cell blocks of sampled gold cells with the streams recomputed from the silver counts."""
    from datasets import load_from_disk

    data = arm["data"]
    # Index-cell block of a cell with n_neighs neighbors (process_gene_tokens: model_input_size / n_gene_segments)
    block_length = int(data["model_input_size"] / (data["n_neighs"] + 1))
    gold_dataset = load_from_disk(str(arm["gold_path"]))
    gold_cell_ids = np.asarray(
        gold_dataset.remove_columns([c for c in gold_dataset.column_names if c != "cell_id"])[:]["cell_id"]).astype(str)
    result = {"n_gold": len(gold_cell_ids), "n_shared": int(np.sum([c in cell_index for c in gold_cell_ids])),
              "block_length": block_length}

    # The same cells for every arm with the same gold cells (sampled by rank in the sorted cell ids)
    order = np.argsort(gold_cell_ids, kind="stable")
    sample = np.sort(order[np.random.default_rng(seed).choice(len(order), min(n_cells, len(order)), replace=False)])
    rows = gold_dataset.select(sample.tolist())
    rows = rows.remove_columns([c for c in rows.column_names if c not in ("cell_id", "gene_tokens", "gene_expr")])[:]
    with open(arm["token_dictionary_file_path"], "rb") as file:
        token_dict = pickle.load(file)

    methods = arm["methods"]
    problems, n_ties, max_rel_diff, n_missing_cells = [], 0, 0.0, 0
    by_file = {}
    for k, cell_id in enumerate(rows["cell_id"]):
        if str(cell_id) not in cell_index:
            n_missing_cells += 1
            continue
        file_idx, row = cell_index[str(cell_id)]
        by_file.setdefault(file_idx, []).append((k, row))
    for file_idx, cells in by_file.items():
        silver_file = silver[file_idx]
        adata = silver_file["adata"]
        X = adata.X[[row for _, row in cells]]
        streams = {stream: recompute_stream(X, adata.var["ensembl_id"], methods[f"{stream}_cell_norm_method"],
                                            methods[f"{stream}_gene_norm_method"],
                                            methods[f"{stream}_count_norm_method"], arm["norm_factor_file_path"],
                                            normalizers)
                   for stream in ["rank", "count"]}
        # Genes of the vocabulary (protein-coding and miRNA, `coding_miRNA_idx` of the tokenizer)
        gene_ids = silver_file["stats"]["gene_ids"]
        coding = np.flatnonzero([("ENS" in gene_id) and (gene_id in token_dict) for gene_id in gene_ids])
        token_columns = {int(token_dict[gene_id]): column for column, gene_id in enumerate(gene_ids[coding])}
        for j, (k, _) in enumerate(cells):
            gold_tokens = np.asarray(rows["gene_tokens"][k], dtype=np.int64)
            gold_values = np.asarray(rows["gene_expr"][k], dtype=np.float64)
            if len(gold_tokens) != data["model_input_size"]:
                problems.append(f"cell {rows['cell_id'][k]}: {len(gold_tokens)} tokens, expected "
                                f"{data['model_input_size']}")
                continue
            problem, tie_at_cut, rel_diff = compare_index_block(gold_tokens[:block_length],
                                                                gold_values[:block_length], token_columns,
                                                                streams["rank"][j, coding], streams["count"][j, coding],
                                                                rtol)
            n_ties += tie_at_cut
            max_rel_diff = np.nanmax([max_rel_diff, rel_diff])
            if problem is not None:
                problems.append(f"cell {rows['cell_id'][k]} ({silver_file['name']}): {problem}")
    result.update({"n_checked": len(rows["cell_id"]) - n_missing_cells, "n_missing_cells": n_missing_cells,
                   "n_ties": n_ties, "max_rel_diff": max_rel_diff, "problems": problems})
    return result


def main(args):
    from terra.preprocessors import normalizers  # pinned TERRA code (PYTHONPATH)

    tokenizations, variants = load_plan(args.plan_file_path)
    arms, failures = [], {}
    for variant in args.variants:
        timestamp = variants[variant]["mouse_runs"][0]
        model_folder_path = Path(args.mouse_artifact_folder_path) / "models" / timestamp
        missing = [name for name in ["model_config.yaml", "norm_factors.csv"] if not (model_folder_path / name).exists()]
        if missing:
            failures[variant] = [f"{', '.join(missing)} missing in {model_folder_path}"]
            continue
        arm = {"variant": variant, "timestamp": timestamp, "plan": tokenizations[variants[variant]["tokenization"]],
               "norm_factor_file_path": model_folder_path / "norm_factors.csv"}
        with open(model_folder_path / "model_config.yaml") as file:
            arm["data"] = yaml.safe_load(file)["data"]
        arm["methods"] = {key: arm["data"].get(key) for key in NORM_METHOD_KEYS}
        arm["token_dictionary_file_path"] = model_folder_path / "token_dictionary.pkl"
        if not arm["token_dictionary_file_path"].exists():
            arm["token_dictionary_file_path"] = Path(arm["data"]["token_dict_folder_path"])
        # Gold dataset of the training run (re-rooted under --data_directory_path if it moved)
        arm["gold_path"] = Path(arm["data"]["tokenized_data_folder_path"])
        if not arm["gold_path"].exists():
            arm["gold_path"] = (Path(args.data_directory_path) / "gold" / arm["gold_path"].parent.parent.name
                                / arm["gold_path"].parent.name / arm["gold_path"].name)
        arm["dataset_name"] = args.dataset_name or arm["gold_path"].parent.name
        arms.append(arm)

    print("== (a) Factor files (norm_factors.csv of the model folders, as used by the mouse evaluation)")
    factor_files = {}
    for arm in arms:
        arm["md5"] = md5sum(arm["norm_factor_file_path"])
        norm_factor_df = factor_files.setdefault(arm["md5"], pd.read_csv(arm["norm_factor_file_path"]))
        print(f"{arm['variant']} | {arm['timestamp']}: md5 {arm['md5']}, {len(norm_factor_df)} rows, "
              f"columns {list(norm_factor_df.columns)}")
    for md5, norm_factor_df in factor_files.items():
        print(f"{md5}: columns as written by {factor_script(list(norm_factor_df.columns))}")
    print(f"The {len(arms)} copies are " + ("identical." if len(factor_files) == 1
                                            else f"NOT identical ({len(factor_files)} distinct files)."))
    for variant, problems in failures.items():
        print(f"{variant}: {problems[0]}")

    print("== (b) Gene-corrected read depth constants of the factor files")
    silver_data, fits = {}, {}
    for arm in arms:
        if arm["dataset_name"] not in silver_data:
            silver_directory_path = Path(args.data_directory_path) / "silver" / arm["dataset_name"]
            silver = load_silver(silver_directory_path)
            pooled_df = pool_silver_statistics(silver)
            cell_index = {cell_id: (file_idx, row) for file_idx, silver_file in enumerate(silver)
                          for row, cell_id in enumerate(silver_file["cell_ids"])}
            if len(cell_index) < sum(len(silver_file["cell_ids"]) for silver_file in silver):
                raise ValueError(f"Cell ids of {silver_directory_path} are not unique across files.")
            silver_data[arm["dataset_name"]] = (silver, pooled_df, cell_index)
            print(f"Silver data {silver_directory_path}: " + ", ".join(
                f"{silver_file['name']} (G = {silver_file['stats']['n_genes']}, {silver_file['stats']['n_cells']} "
                f"cells)" for silver_file in silver) + f"; {int(np.sum(pooled_df['n_files'] > 1))} genes in more "
                f"than one file.")
        silver, pooled_df, _ = silver_data[arm["dataset_name"]]
        key = (arm["md5"], arm["dataset_name"])
        if key in fits:
            continue
        fit = fits[key] = fit_gcrd_constants(factor_files[arm["md5"]], pooled_df, silver)
        for gene_norm_method in ["mean", "nonzero_mean"]:
            fit[f"scales_{gene_norm_method}"] = effective_scales(factor_files[arm["md5"]], pooled_df, silver,
                                                                 gene_norm_method)
        print(f"Factor file {arm['md5']} on {arm['dataset_name']}:")
        print(f"  cohort: the CSV's n_cells and raw count sums match these files for {fit['n_matched']} of the "
              f"{fit['n_genes']} genes they probe ({fit['n_missing']} of their genes missing from the CSV, "
              f"{fit['n_other']} other genes in the CSV) "
              f"-> {'the factor cohort is these files' if fit['is_cohort'] else 'a LARGER or different cohort'}")
        constants = (f"a = {fit['a']:.6f}, b = {fit['b']:.7f}" if fit["identifiable"]
                     else f"T(G) = {fit['a']:.4f} (one panel: a and b not separately identifiable)")
        print(f"  {'fit' if fit['is_cohort'] else 'approximate fit (cohort means taken as those of these files)'}: "
              f"{constants} over {fit['n_genes_fit']} genes; relative residual rms {fit['rms']:.1e}, max "
              f"{fit['max']:.1e}" + (f"; nonzero-mean column rms {fit['rms_nonzero']:.1e}" if "rms_nonzero" in fit
                                     else ""))
        print("  candidates (relative residual rms): " + " | ".join(
            f"{name} {rms:.1e}" for name, rms in sorted(fit["candidates"].items(), key=lambda item: item[1]))
              + f" -> {'matches' if fit['matches'] else 'closest (none matches)'}: {fit['closest']}")
        print(f"  T_tokenizer(G) / T_factors(G){'' if fit['is_cohort'] else ' (approximate fit)'}: " + ", ".join(
            f"{silver_file['name']} (G = {silver_file['stats']['n_genes']}) x{multiplier:.3f}"
            for silver_file, multiplier in zip(silver, fit["multipliers"]))
              + f"; platform offset x{max(fit['multipliers']) / min(fit['multipliers']):.3f}")
        if not fit["is_cohort"]:
            print("  The constants cannot be identified from these files. Still valid: check (c), which uses the CSV "
                  "as is, and the effective scale of the count values of each file (median over its genes of the "
                  "factor with the tokenizer constants on these files / the CSV factor), which mixes the constants "
                  "with the expression difference between the cohort and these files: mean "
                  + ", ".join(f"x{scale:.3f}" for scale in fit["scales_mean"]) + "; nonzero mean "
                  + ", ".join(f"x{scale:.3f}" for scale in fit["scales_nonzero_mean"]))

    print(f"== (c) Train/eval consistency ({args.n_cells} gold cells per arm; index-cell block; values and rank scores "
          f"within the relative tolerance {args.rtol:g})")
    for arm in arms:
        variant, methods = arm["variant"], arm["methods"]
        plan_methods = {key: arm["plan"][key] for key in NORM_METHOD_KEYS}
        if methods != plan_methods:
            failures.setdefault(variant, []).append(f"norm methods of the model config {methods} differ from the "
                                                    f"plan {plan_methods}")
            print(f"{variant}: {failures[variant][-1]}")
        missing = [str(path) for path in [arm["gold_path"], arm["token_dictionary_file_path"]] if not path.exists()]
        if missing:
            failures.setdefault(variant, []).append(f"{', '.join(missing)} not found")
            print(f"{variant}: {failures[variant][-1]}")
            continue
        silver, _, cell_index = silver_data[arm["dataset_name"]]
        result = arm["result"] = check_consistency(arm, silver, cell_index, normalizers, args.n_cells, args.seed,
                                                   args.rtol)
        rank_differs_from_count = not (
            methods["rank_cell_norm_method"] == methods["count_cell_norm_method"]
            and methods["rank_gene_norm_method"] == methods["count_gene_norm_method"]
            and (methods["rank_count_norm_method"] == methods["count_count_norm_method"]
                 or methods["count_count_norm_method"] == "shifted_log"))
        print(f"{variant} ({'dense' if rank_differs_from_count else 'sparse'} tokenizer path, gold "
              f"{arm['gold_path'].name}): cell ids {result['n_gold']} gold, {len(cell_index)} silver, "
              f"{result['n_shared']} shared; {result['n_checked'] - len(result['problems'])} of {result['n_checked']} "
              f"cells consistent (top-{result['block_length']} cut in a tie: {result['n_ties']}); values max relative "
              f"difference {result['max_rel_diff']:.1e}")
        for problem in result["problems"][:5]:
            print(f"  {problem}")
        if result["n_shared"] < result["n_gold"]:
            failures.setdefault(variant, []).append(f"{result['n_gold'] - result['n_shared']} gold cells missing from "
                                                    f"the silver data")
        if result["problems"]:
            failures.setdefault(variant, []).append(f"{len(result['problems'])} of {result['n_checked']} sampled "
                                                    f"cells inconsistent")

    print("== Verdict")
    for variant in args.variants:
        arm = next((arm for arm in arms if arm["variant"] == variant), None)
        if arm is None:
            print(f"{variant}: INCOMPLETE ({'; '.join(failures[variant])})")
            continue
        silver, pooled_df, _ = silver_data[arm["dataset_name"]]
        fit, methods = fits[(arm["md5"], arm["dataset_name"])], arm["methods"]
        verdict = (f"MISMATCH ({'; '.join(failures[variant])})" if variant in failures
                   else "CONSISTENT (training tokens = model-folder factors)")
        constants = f"a = {fit['a']:.4f}, b = {fit['b']:.6f}" if fit["identifiable"] else f"T(G) = {fit['a']:.3f}"
        factors = (f"{fit['closest']} constants ({constants}, residual {fit['rms']:.0e})" if fit["matches"]
                   else f"no candidate fits ({'' if fit['is_cohort'] else 'approximate '}fit {constants}, residual "
                        f"{fit['rms']:.0e}; closest {fit['closest']})")
        if methods["count_cell_norm_method"] == GCRD and methods["count_gene_norm_method"] in ("mean", "nonzero_mean"):
            multipliers = fit["multipliers"] if fit["is_cohort"] else fit[f"scales_{methods['count_gene_norm_method']}"]
            effect = ("count values " + ", ".join(
                f"x{multiplier:.3f} ({silver_file['name']}, G = {silver_file['stats']['n_genes']})"
                for silver_file, multiplier in zip(silver, multipliers))
                      + f", platform offset x{max(multipliers) / min(multipliers):.3f}"
                      + ("" if fit["is_cohort"] else " (effective scale; larger cohort)"))
        else:
            effect = (f"rank stream only, values are raw counts: a per-file factor cancels within a cell, except for "
                      f"the ranks of the {int(np.sum(pooled_df['n_files'] > 1))} genes in more than one file")
        print(f"{variant} ({arm['timestamp']}): {verdict}; factors: {factors}; {effect}")

    if any(fit["matches"] and fit["closest"] != "tokenizer" for fit in fits.values()):
        print("The factors do not use the tokenizer's constants: rerunning the four arms with gcrd count values on "
              "factors from compute_norm_factors.py (same cohort) removes the platform offset.")
    if failures:
        sys.exit("Norm factor check failed, see above.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--variants',
                        type=str,
                        nargs='+',
                        default=GCRD_VARIANTS,
                        help='Variants of the ablation plan to check (default: the six gcrd mean arms).')
    parser.add_argument('--plan_file_path',
                        type=str,
                        required=True,
                        help='Ablation plan (variant names and mouse run timestamps).')
    parser.add_argument('--mouse_artifact_folder_path',
                        type=str,
                        required=True,
                        help='Artifact folder of the mouse ablation runs (models/<timestamp>/).')
    parser.add_argument('--data_directory_path',
                        type=str,
                        required=True,
                        help='Data directory with silver/<dataset_name>/*.h5ad (and gold, if the config path moved).')
    parser.add_argument('--dataset_name',
                        type=str,
                        default=None,
                        help='Silver dataset folder (default: the folder of the gold dataset of each run).')
    parser.add_argument('--n_cells',
                        type=int,
                        default=300,
                        help='Gold cells sampled per arm.')
    parser.add_argument('--rtol',
                        type=float,
                        default=1e-5,
                        help='Relative tolerance of gold vs recomputed values and rank scores.')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Seed of the cell sample.')
    args = parser.parse_args()

    main(args)
