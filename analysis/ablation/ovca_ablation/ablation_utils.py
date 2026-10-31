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
TOKENIZATION_DEFAULTS = {"tokenizer": "cell-graph-tokenizer",
                         "model_input_size": 704,
                         "n_neighs": 10,
                         **dict.fromkeys(NORM_METHOD_KEYS)}
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

    tokenizations = {name: {**TOKENIZATION_DEFAULTS, **spec} for name, spec in plan["tokenizations"].items()}
    variants = plan["variants"]
    for name, variant in variants.items():
        variant.setdefault("tokenization", "base")
        variant.setdefault("overrides", {})
        if variant["tokenization"] not in tokenizations:
            raise ValueError(f"Variant {name} uses unknown tokenization {variant['tokenization']}.")
        if variant["category"] not in CATEGORY_ORDER:
            raise ValueError(f"Variant {name} has unknown category {variant['category']}.")
    return tokenizations, variants


def get_tokenized_dataset_path(tokenization: dict,
                               data_directory_path: Path | str = DATA_DIRECTORY_PATH,
                               dataset_name: str = DATASET_NAME,
                               num_shards: int = NUM_SHARDS) -> Path:
    """Path of a tokenized dataset, named as by `create_tokenized_dataset.py` (kNN graph, no zero-expression genes)."""
    norm_methods = "_".join(str(tokenization[key]) for key in NORM_METHOD_KEYS)
    file_prefix = (f"{dataset_name}_{num_shards}_{norm_methods}_knn_{tokenization['n_neighs']}_"
                   f"{tokenization['model_input_size']}")
    return Path(data_directory_path) / "gold" / tokenization["tokenizer"] / dataset_name / f"{file_prefix}.dataset"


def needs_large_memory(tokenization: dict) -> bool:
    """
    Whether a tokenization takes the tokenizer's dense path (rank and count normalizations differ, see
    `CellBaseTokenizer.rank_differs_from_count`) or keeps every gene per cell (Seurat v3, Pearson residuals), or has
    long sequences. These need about 0.5-0.6 TB of memory for the OVCA section.
    """
    rank_differs_from_count = not (
        tokenization["rank_cell_norm_method"] == tokenization["count_cell_norm_method"]
        and tokenization["rank_gene_norm_method"] == tokenization["count_gene_norm_method"]
        and (tokenization["rank_count_norm_method"] == tokenization["count_count_norm_method"]
             or tokenization["count_count_norm_method"] == "shifted_log"))
    keeps_all_genes = any(tokenization[key] in ("seurat_v3", "analytic_pearson_residuals") for key in NORM_METHOD_KEYS)
    return rank_differs_from_count or keeps_all_genes or tokenization["model_input_size"] > 1408


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
