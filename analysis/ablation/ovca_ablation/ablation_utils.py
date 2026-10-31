"""
Shared helpers of the OVCA ablation scripts: default paths, loading the ablation plan, the names of the tokenized
datasets (as written by `create_tokenized_dataset.py`) and the order in which the runs are launched.
"""

from __future__ import annotations

from pathlib import Path

import yaml


DATA_DIRECTORY_PATH = "/lustre/scratch126/cellgen/lotfollahi/DATASETS"
DATASET_NAME = "xho50-1b_1p_spatialfusion"
ARTIFACT_FOLDER_PATH = "/nfs/team361/sb75/terra-reproducibility/artifacts/ovca_ablation"
BRONZE_FILE_PATH = "/nfs/team361/sb75/DATASETS/bronze/spatialfusion/benchmark_processed_data/processed_OVCA.h5ad"
SILVER_FILE_PATH = f"{DATA_DIRECTORY_PATH}/silver/{DATASET_NAME}/{DATASET_NAME}_batch0.h5ad"
TOKEN_DICTIONARY_FILE_PATH = f"{DATA_DIRECTORY_PATH}/tokenizer/token_dictionary_homo_sapiens.pkl"
GENE_MAPPING_DICT_FILE_PATH = f"{DATA_DIRECTORY_PATH}/genes/homo_sapiens_gene_name_to_ensembl_id_dict.pkl"
NORM_FACTOR_FILE_PATH = f"{DATA_DIRECTORY_PATH}/tokenizer/{DATASET_NAME}_norm_factors.csv"
PLAN_FILE_PATH = Path(__file__).resolve().parent / "ablation_plan.yaml"
NUM_SHARDS = 32

NORM_METHOD_KEYS = ["rank_cell_norm_method",
                    "rank_gene_norm_method",
                    "rank_count_norm_method",
                    "count_cell_norm_method",
                    "count_gene_norm_method",
                    "count_count_norm_method"]
# Opt-in tokenizer fixes of stjepa_ovca_ablation.patch (off = behaviour of the mouse tokenizations); the plan sets
# them for all tokenizations under `tokenizer_options`. The last two only apply to the cell-neighborhood tokenizer.
TOKENIZER_OPTION_DEFAULTS = {"stable_rank_ties": False,
                             "zero_expr_from_raw_counts": False,
                             "exact_n_nonzero_tokens": False,
                             "seurat_loess_span": None,
                             "aggregate_own_neighbors": False,
                             "seq_len_cell": None}
TOKENIZATION_DEFAULTS = {"tokenizer": "cell-graph-tokenizer",
                         "model_input_size": 704,
                         "n_neighs": 10,
                         **dict.fromkeys(NORM_METHOD_KEYS),
                         **TOKENIZER_OPTION_DEFAULTS}
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


def load_plan(plan_file_path: Path | str = PLAN_FILE_PATH) -> tuple[dict, dict]:
    """Load the tokenizations and variants of the ablation plan with defaults filled in."""
    with open(plan_file_path) as file:
        plan = yaml.safe_load(file)

    tokenizations = {name: {**TOKENIZATION_DEFAULTS, **plan.get("tokenizer_options", {}), **spec}
                     for name, spec in plan["tokenizations"].items()}
    for name, tokenization in tokenizations.items():
        if set(tokenization) - set(TOKENIZATION_DEFAULTS):
            raise ValueError(f"Tokenization {name} has unknown keys {set(tokenization) - set(TOKENIZATION_DEFAULTS)}.")
    variants = plan["variants"]
    for name, variant in variants.items():
        variant.setdefault("tokenization", "base")
        variant.setdefault("overrides", {})
        # Opt-in training fixes of the patch, set in every run config before the variant's own overrides
        variant["training_options"] = dict(plan.get("training_options", {}))
        if variant["tokenization"] not in tokenizations:
            raise ValueError(f"Variant {name} uses unknown tokenization {variant['tokenization']}.")
        if variant["category"] not in CATEGORY_ORDER:
            raise ValueError(f"Variant {name} has unknown category {variant['category']}.")
    return tokenizations, variants


def get_tokenized_dataset_path(tokenization: dict,
                               data_directory_path: Path | str = DATA_DIRECTORY_PATH,
                               dataset_name: str = DATASET_NAME,
                               num_shards: int = NUM_SHARDS) -> Path:
    """
    Path of a tokenized dataset, named as by `create_tokenized_dataset.py` (kNN graph, no zero-expression genes), with
    a suffix for the enabled tokenizer fixes (so datasets of other tokenizer settings are never reused).
    """
    norm_methods = "_".join(str(tokenization[key]) for key in NORM_METHOD_KEYS)
    options = []
    if tokenization["stable_rank_ties"]:
        options.append("ties")
    if tokenization["zero_expr_from_raw_counts"]:
        options.append("detected")
    if tokenization["exact_n_nonzero_tokens"]:
        options.append("nnz")
    if tokenization["seurat_loess_span"] is not None and "seurat_v3" in norm_methods:
        # No dots: the tokenizer names its output with Path.with_suffix
        options.append(f"span{tokenization['seurat_loess_span']}".replace(".", "p"))
    if tokenization["tokenizer"] == "cell-neighborhood-tokenizer":
        if tokenization["aggregate_own_neighbors"]:
            options.append("ownknn")
        if tokenization["seq_len_cell"] is not None:
            options.append(f"cell{tokenization['seq_len_cell']}")
    file_prefix = (f"{dataset_name}_{num_shards}_{norm_methods}_knn_{tokenization['n_neighs']}_"
                   f"{tokenization['model_input_size']}{''.join(f'_{option}' for option in options)}")
    return Path(data_directory_path) / "gold" / tokenization["tokenizer"] / dataset_name / f"{file_prefix}.dataset"


def needs_large_memory(tokenization: dict) -> bool:
    """
    Whether a tokenization needs the large memory class: without the opt-in tokenizer fixes, the dense path (rank and
    count normalizations differ, see `CellBaseTokenizer.rank_differs_from_count`) and Seurat v3 / Pearson residuals keep
    every gene per cell (about 1.3 MB per cell, 0.5-0.6 TB for the OVCA section); with `stable_rank_ties` or
    `zero_expr_from_raw_counts`, ranked gene lists are cut at `model_input_size` (lossless; about 0.09-0.21 MB per cell,
    0.09-0.10 MB with `exact_n_nonzero_tokens`, at most about 70 GB). Long sequences always use the large class.
    """
    rank_differs_from_count = not (
        tokenization["rank_cell_norm_method"] == tokenization["count_cell_norm_method"]
        and tokenization["rank_gene_norm_method"] == tokenization["count_gene_norm_method"]
        and (tokenization["rank_count_norm_method"] == tokenization["count_count_norm_method"]
             or tokenization["count_count_norm_method"] == "shifted_log"))
    keeps_all_genes = any(tokenization[key] in ("seurat_v3", "analytic_pearson_residuals") for key in NORM_METHOD_KEYS)
    cuts_ranked_lists = tokenization["stable_rank_ties"] or tokenization["zero_expr_from_raw_counts"]
    return ((rank_differs_from_count or keeps_all_genes) and not cuts_ranked_lists
            or tokenization["model_input_size"] > 1408)


def get_seq_len_cell(tokenization: dict) -> int:
    """Index-cell tokens per sequence: cell graph blocks or the cell-neighborhood split (pinned default k + 1 blocks)."""
    if tokenization["tokenizer"] == "cell-neighborhood-tokenizer" and tokenization["seq_len_cell"] is not None:
        return tokenization["seq_len_cell"]
    return int(tokenization["model_input_size"] / (tokenization["n_neighs"] + 1))


def get_runs(variants: dict) -> list[dict]:
    """
    Runs in launch order: the first replicate of every variant (base first, then figure order), then all second
    replicates, then all third replicates, so that a complete single-replicate picture is available early. Replicate
    `r` is trained with seed `1000 * r` (1000, 2000, 3000): it seeds weight initialization, data order and, through
    the DataLoader worker seeds, masks and segment sampling. None equals the hard-coded seed 0 of the mouse runs, and
    the spacing keeps the samplers' per-epoch seeds (seed + epoch) of different replicates disjoint.
    """
    variant_order = {name: idx for idx, name in enumerate(variants)}
    runs = [{"run_name": f"{name}_rep{replicate}",
             "variant": name,
             "replicate": replicate,
             "seed": 1000 * replicate,
             **variant}
            for name, variant in variants.items()
            for replicate in range(1, variant["replicates"] + 1)]
    return sorted(runs, key=lambda run: (run["replicate"],
                                         CATEGORY_ORDER.index(run["category"]),
                                         variant_order[run["variant"]]))
