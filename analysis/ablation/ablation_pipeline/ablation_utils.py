"""
Generic helpers of the ablation scripts: loading the ablation plan, the names of the tokenized datasets (as written by
`create_tokenized_dataset.py`) and what each was made from, the memory class of a tokenization and the order in which
the runs are launched. All dataset-specific values (paths, dataset name, labels, schedule) are command line arguments of
the scripts, taken from the dataset's parameter file (`datasets/<dataset>.env`).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml


# Shards of the tokenized datasets; part of their names (`create_tokenized_dataset.py` naming)
NUM_SHARDS = 32

NORM_METHOD_KEYS = ["rank_cell_norm_method",
                    "rank_gene_norm_method",
                    "rank_count_norm_method",
                    "count_cell_norm_method",
                    "count_gene_norm_method",
                    "count_count_norm_method"]
# Opt-in tokenizer fixes of TERRA (off = behaviour of the mouse tokenizations); the plan sets them for all
# tokenizations under `tokenizer_options`. rank_tie_break_norm_factor (a column of the norm factor file, with
# stable_rank_ties) orders tied genes before the token id. seurat_loess_span None keeps TERRA's default (0.75, the span
# the mouse tokenizations effectively used). The last two only apply to the cell-neighborhood tokenizer: seq_len_cell
# None is the original v2 split into n_neighs + 1 blocks, model_input_size / 2 TERRA's equal split
# (split_cell_neigh_equally).
TOKENIZER_OPTION_DEFAULTS = {"stable_rank_ties": False,
                             "rank_tie_break_norm_factor": None,
                             "zero_expr_from_raw_counts": False,
                             "seurat_loess_span": None,
                             "aggregate_own_neighbors": False,
                             "seq_len_cell": None}
TOKENIZATION_DEFAULTS = {"tokenizer": "cell-graph-tokenizer",
                         "model_input_size": 704,
                         "n_neighs": 10,
                         **dict.fromkeys(NORM_METHOD_KEYS),
                         **TOKENIZER_OPTION_DEFAULTS}
# Obs column of the evaluation with the section of every cell (its label file): the batch of the integration metrics
SECTION_KEY = "section"
# Launch and plotting order (figure order of the mouse ablations)
CATEGORY_ORDER = ["base",
                  "tokenizer",
                  "sequence_length",
                  "expression_encoding",
                  "spatial_encoding",
                  "masking_ratio",
                  "metatoken",
                  "normalization_unnormalized_ranks",
                  "normalization_full",
                  "normalization_unnormalized_counts",
                  "architecture"]


def load_plan(plan_file_path: Path | str) -> tuple[dict, dict]:
    """Load the tokenizations and variants of the ablation plan with defaults filled in."""
    with open(plan_file_path) as file:
        plan = yaml.safe_load(file)

    tokenizations = {name: {**TOKENIZATION_DEFAULTS, **plan.get("tokenizer_options", {}), **spec}
                     for name, spec in plan["tokenizations"].items()}
    for name, tokenization in tokenizations.items():
        if set(tokenization) - set(TOKENIZATION_DEFAULTS):
            raise ValueError(f"Tokenization {name} has unknown keys {set(tokenization) - set(TOKENIZATION_DEFAULTS)}.")
        # TERRA's cell-neighborhood tokenizer splits a sequence equally between the index cell and its neighborhood
        # (split_cell_neigh_equally) or into n_neighs + 1 blocks
        if (tokenization["tokenizer"] == "cell-neighborhood-tokenizer"
                and tokenization["seq_len_cell"] not in (None, int(tokenization["model_input_size"] / 2))):
            raise ValueError(f"Tokenization {name}: seq_len_cell must be model_input_size / 2 (equal split) or null "
                             f"(n_neighs + 1 blocks), not {tokenization['seq_len_cell']}.")
        if tokenization["rank_tie_break_norm_factor"] is not None and not tokenization["stable_rank_ties"]:
            raise ValueError(f"Tokenization {name}: rank_tie_break_norm_factor requires stable_rank_ties.")
    variants = plan["variants"]
    for name, variant in variants.items():
        variant.setdefault("tokenization", "base")
        variant.setdefault("overrides", {})
        variant.setdefault("appended_replicates", 0)
        if not 0 <= variant["appended_replicates"] < variant["replicates"]:
            raise ValueError(f"Variant {name}: appended_replicates must be at least 0 and less than replicates "
                             f"({variant['replicates']}), not {variant['appended_replicates']}.")
        # Opt-in training fixes of TERRA, set in every run config before the variant's own overrides
        variant["training_options"] = dict(plan.get("training_options", {}))
        if variant["tokenization"] not in tokenizations:
            raise ValueError(f"Variant {name} uses unknown tokenization {variant['tokenization']}.")
        if variant["category"] not in CATEGORY_ORDER:
            raise ValueError(f"Variant {name} has unknown category {variant['category']}.")
    return tokenizations, variants


def get_tokenized_dataset_path(tokenization: dict,
                               data_directory_path: Path | str,
                               dataset_name: str) -> Path:
    """
    Path of a tokenized dataset, named as by `create_tokenized_dataset.py` (kNN graph, no zero-expression genes), with
    a suffix for the enabled tokenizer fixes (so datasets of other tokenizer settings are never reused).
    """
    norm_methods = "_".join(str(tokenization[key]) for key in NORM_METHOD_KEYS)
    options = []
    if tokenization["stable_rank_ties"]:
        tie_break_norm_factor = tokenization["rank_tie_break_norm_factor"]
        options.append("ties" if tie_break_norm_factor is None else f"tiesby{tie_break_norm_factor}")
    if tokenization["zero_expr_from_raw_counts"]:
        options.append("detected")
    if tokenization["seurat_loess_span"] is not None and "seurat_v3" in norm_methods:
        # No dots: the tokenizer names its output with Path.with_suffix
        options.append(f"span{tokenization['seurat_loess_span']}".replace(".", "p"))
    if tokenization["tokenizer"] == "cell-neighborhood-tokenizer":
        if tokenization["aggregate_own_neighbors"]:
            options.append("ownknn")
        if tokenization["seq_len_cell"] is not None:
            options.append(f"cell{tokenization['seq_len_cell']}")
    file_prefix = (f"{dataset_name}_{NUM_SHARDS}_{norm_methods}_knn_{tokenization['n_neighs']}_"
                   f"{tokenization['model_input_size']}{''.join(f'_{option}' for option in options)}")
    return Path(data_directory_path) / "gold" / tokenization["tokenizer"] / dataset_name / f"{file_prefix}.dataset"


def needs_large_memory(tokenization: dict) -> bool:
    """
    Whether a tokenization needs the large memory class (`submit_tokenization.sh`; standard: 200 GB).
    - With `stable_rank_ties` or `zero_expr_from_raw_counts`, every tokenization is standard. TERRA cuts the ranked
      gene lists at `model_input_size` on both ranking paths (lossless). Peaks measured on the OVCA section (338,695
      cells) with the equivalent cut of the former pinned code: about 83 GB for the base, dense-path and 256-gene
      (`model_input_size` 2816) tokenizations, about 105 GB for Seurat v3 and Pearson residuals. That code also dropped
      trailing pad entries of dense-path lists, which TERRA keeps, so dense-path arms of a dataset of that size may need
      up to about twice as much; the mouse sections (92,520 cells) are far below the limit.
    - Without them, the dense path (rank and count normalizations differ, see
      `CellBaseTokenizer.rank_differs_from_count`) and Seurat v3 / Pearson residuals keep every gene per cell (about
      1.3 MB per cell, 0.5-0.6 TB for the OVCA section), and long sequences (`model_input_size` > 1408) also use the
      large class.
    """
    if tokenization["stable_rank_ties"] or tokenization["zero_expr_from_raw_counts"]:
        return False
    rank_differs_from_count = not (
        tokenization["rank_cell_norm_method"] == tokenization["count_cell_norm_method"]
        and tokenization["rank_gene_norm_method"] == tokenization["count_gene_norm_method"]
        and (tokenization["rank_count_norm_method"] == tokenization["count_count_norm_method"]
             or tokenization["count_count_norm_method"] == "shifted_log"))
    keeps_all_genes = any(tokenization[key] in ("seurat_v3", "analytic_pearson_residuals") for key in NORM_METHOD_KEYS)
    return rank_differs_from_count or keeps_all_genes or tokenization["model_input_size"] > 1408


def get_seq_len_cell(tokenization: dict) -> int:
    """Index-cell tokens per sequence: cell graph blocks or the cell-neighborhood split (default: k + 1 blocks)."""
    if tokenization["tokenizer"] == "cell-neighborhood-tokenizer" and tokenization["seq_len_cell"] is not None:
        return tokenization["seq_len_cell"]
    return int(tokenization["model_input_size"] / (tokenization["n_neighs"] + 1))


def cells_without_vocabulary_genes(adata, token_dict: dict, chunk_size: int = 100_000):
    """
    Boolean array over the cells of `adata`: True for a cell without a detected gene of the token dictionary (raw count
    > 0 in `adata.X` for a gene whose `var['ensembl_id']` has a token, the genes the tokenizer keeps). With
    `zero_expr_from_raw_counts`, exactly these cells have an empty ranked gene list in every tokenization. Their own
    segment holds only padding (empty embedding), and on the sparse ranking path (most arms, including base) the
    tokenizer does not count them as neighbor segments (TERRA's CellGraphTokenizer with its default
    pad_empty_neighbors=False, as the original v2 code): a cell with k such neighbors gets longer segments that no longer
    start where the datasets read them, and loses its k farthest neighbors (TERRA's pipeline removes these cells before
    tokenization since 7f6a30c). Works on backed AnnData (row chunks).
    """
    import numpy as np
    import scipy.sparse as sp

    in_vocabulary = np.array([bool(token_dict.get(gene_id, False)) for gene_id in adata.var["ensembl_id"]])
    n_detected = np.zeros(adata.n_obs, dtype=np.int64)
    for start in range(0, adata.n_obs, chunk_size):
        X = sp.csr_matrix(adata.X[start:start + chunk_size])[:, in_vocabulary]
        n_detected[start:start + X.shape[0]] = np.asarray((X > 0).sum(axis=1)).ravel()
    return n_detected == 0


def count_stored_zeros(adata, chunk_size: int = 100_000) -> int:
    """
    Number of zeros stored explicitly in a sparse `adata.X` (0 for a dense X). On the sparse ranking path the tokenizer
    turns them into gene tokens with a count of 0; filter_silver_cells.py removes them. Works on backed AnnData.
    """
    import scipy.sparse as sp

    n_stored_zeros = 0
    for start in range(0, adata.n_obs, chunk_size):
        X = adata.X[start:start + chunk_size]
        if sp.issparse(X):
            n_stored_zeros += int((X.data == 0).sum())
    return n_stored_zeros


# What each tokenized dataset was made from, written into its folder by tokenize_dataset.py and compared by
# tokenize_dataset.py (before skipping an existing dataset) and check_setup.py (check 4): the dataset name records the
# tokenization settings, but not the TERRA code or the input files
PROVENANCE_FILE_NAME = "ablation_provenance.json"


def file_md5(file_path: Path | str) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1 << 23), b""):
            digest.update(block)
    return digest.hexdigest()


def get_code_version(code_path: Path | str | None = None) -> str:
    """CODE_VERSION of `code_path`, by default of the TERRA copy that `import terra` loads; '?' without the file."""
    if code_path is None:
        import terra
        code_path = Path(terra.__file__).resolve().parents[2]  # <code_path>/src/terra/__init__.py
    version_file_path = Path(code_path) / "CODE_VERSION"
    return version_file_path.read_text().strip() if version_file_path.exists() else "?"


def get_tokenization_provenance(tokenization: dict, code_version: str, silver_file_paths: list,
                                norm_factor_file_path: Path | str, token_dictionary_file_path: Path | str,
                                md5_cache: dict | None = None) -> dict:
    """
    The inputs of a tokenized dataset: the TERRA code version, the tokenization settings of the plan, and the md5 of
    the token dictionary, the normalization factor file and each silver file (by file name). `md5_cache` keeps the md5
    of each path across calls.
    """
    md5_cache = {} if md5_cache is None else md5_cache

    def md5(file_path):
        key = os.path.abspath(file_path)
        if key not in md5_cache:
            md5_cache[key] = file_md5(file_path)
        return md5_cache[key]

    return {"code_version": code_version,
            "tokenization": {key: tokenization[key] for key in sorted(tokenization)},
            "token_dictionary_md5": md5(token_dictionary_file_path),
            "norm_factor_md5": md5(norm_factor_file_path),
            "silver_file_md5s": {Path(file_path).name: md5(file_path) for file_path in sorted(silver_file_paths)}}


def provenance_differences(dataset_path: Path | str, expected: dict) -> list[str]:
    """The provenance keys whose value in the dataset's provenance file differs from `expected`."""
    provenance_file_path = Path(dataset_path) / PROVENANCE_FILE_NAME
    if not provenance_file_path.exists():
        return [f"no {PROVENANCE_FILE_NAME} (made before provenance was recorded)"]
    with open(provenance_file_path) as file:
        recorded = json.load(file)
    return [key for key in expected if recorded.get(key) != expected[key]]


def write_provenance(dataset_path: Path | str, provenance: dict):
    with open(Path(dataset_path) / PROVENANCE_FILE_NAME, "w") as file:
        json.dump(provenance, file, indent=1)


def ties_at_cut(adata, token_dict: dict, seq_lens, chunk_size: int = 100_000) -> dict:
    """
    How often the tie order decides which genes a raw-count ranking keeps (the base and every arm whose rank is not
    normalized): for each length k in `seq_lens`, (number of cells with more than k detected vocabulary genes, number
    of cells whose k-th and (k + 1)-th highest raw counts are equal, i.e. whose cut at k genes falls inside a group of
    tied genes). Works on backed AnnData (row chunks).
    """
    import numpy as np
    import scipy.sparse as sp

    seq_lens = sorted(set(seq_lens))
    in_vocabulary = np.array([bool(token_dict.get(gene_id, False)) for gene_id in adata.var["ensembl_id"]])
    n_cut, n_tied = dict.fromkeys(seq_lens, 0), dict.fromkeys(seq_lens, 0)
    for start in range(0, adata.n_obs, chunk_size):
        X = sp.csr_matrix(adata.X[start:start + chunk_size])[:, in_vocabulary]
        for row in range(X.shape[0]):
            counts = X.data[X.indptr[row]:X.indptr[row + 1]]
            counts = np.sort(counts[counts > 0])[::-1]
            for k in seq_lens:
                if counts.size > k:
                    n_cut[k] += 1
                    n_tied[k] += int(counts[k - 1] == counts[k])
    return {k: (n_cut[k], n_tied[k]) for k in seq_lens}


def get_section_names(label_file_paths: list) -> list[str]:
    """Section of each label file (one file = one section): its name without extension, its path if names coincide."""
    names = [Path(file_path).stem for file_path in label_file_paths]
    return names if len(set(names)) == len(names) else [str(file_path) for file_path in label_file_paths]


def get_runs(variants: dict) -> list[dict]:
    """
    Runs in launch order: the first replicate of every variant (base first, then figure order), then all second
    replicates, then all third replicates, so that a complete single-replicate picture is available early. The last
    `appended_replicates` replicates of a variant (added after the first manifest was fixed) follow all other runs, in
    the same order, so that the other runs keep their manifest rows (and wandb ids) and the campaigns of all datasets
    share one order. Replicate `r` is trained with seed `1000 * r` (1000, 2000, 3000) in every variant (replicates are
    paired by seed across variants): it seeds weight initialization, data order and, through the DataLoader worker
    seeds, masks and segment sampling. None equals the hard-coded seed 0 of the mouse runs, and the spacing keeps the
    samplers' per-epoch seeds (seed + epoch) of different replicates disjoint.
    """
    variant_order = {name: idx for idx, name in enumerate(variants)}
    runs = [{"run_name": f"{name}_rep{replicate}",
             "variant": name,
             "replicate": replicate,
             "seed": 1000 * replicate,
             "appended": replicate > variant["replicates"] - variant["appended_replicates"],
             **variant}
            for name, variant in variants.items()
            for replicate in range(1, variant["replicates"] + 1)]
    return sorted(runs, key=lambda run: (run["appended"],
                                         run["replicate"],
                                         CATEGORY_ORDER.index(run["category"]),
                                         variant_order[run["variant"]]))
