"""
This script prepares the SpatialFusion ovarian cancer section (Zenodo 20934900, `processed_OVCA.h5ad`; raw counts and
pathologist niche labels) for the OVCA ablations:

1. Harmonizes the bronze AnnData into a silver AnnData with TERRA's `harmonize_batch`, using the metadata of the same 10x
   section in the TERRA corpus (`xho50-1b_1p`: dataset id 50, batch0) so that the human token dictionary already holds
   its special value tokens. The pathologist labels (`path_region`), cell type labels and 10x cell ids (obs_names) are
   kept; only `obsm['spatial']` (microns) is kept as coordinates.
2. Computes the per-gene normalization factors of this cohort (the section itself) with `norm_factors.py`, i.e. as
   `stjepa-reproducibility/src/compute_norm_factors.py` does, with the gene-corrected read depth constants of the
   tokenizer (153.4768 + 0.0487 * n_genes), so that factors and tokenization are consistent. They are needed by the
   mean and nonzero-mean normalization arms.
3. Checks the human token dictionary: gene coverage, special value tokens, the batch special value and the token offset
   that `gt_type: rank` assumes.

Run in the TERRA environment (imports `analysis/data_preparation/harmonize_bronze_to_silver.py`).

Usage:
```bash
python prepare_ovca_data.py
```
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from ablation_utils import (BRONZE_FILE_PATH,
                            GENE_MAPPING_DICT_FILE_PATH,
                            NORM_FACTOR_FILE_PATH,
                            SILVER_FILE_PATH,
                            TOKEN_DICTIONARY_FILE_PATH)
from norm_factors import compute_norm_factors

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_preparation"))
from harmonize_bronze_to_silver import harmonize_batch  # noqa: E402

# Metadata of the same 10x section in the TERRA corpus (config/bronze_to_silver/xho50-1b_1p.yaml)
UNS_PARAMS = {"dataset_id": 50,
              "assay": "xenium",
              "species": "homo_sapiens",
              "tissue": "ovary",
              "condition": "serous_carcinoma",
              "batch": "batch0"}
LABEL_KEYS = ["path_region", "cell_labels", "minor_celltype", "major_celltype"]
# Special token offset hard-coded for `gt_type: rank` with a non-mouse vocabulary (cell_datasets.py)
RANK_SPECIAL_TOKEN_OFFSET = 21957


def check_token_dictionary(adata: ad.AnnData,
                           token_dictionary_file_path: str):
    """Print the token dictionary checks that the training configs depend on."""
    with open(token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)

    n_genes_in_dict = adata.var["ensembl_id"].isin(token_dict.keys()).sum()
    print(f"Genes in token dictionary: {n_genes_in_dict} of {adata.n_vars} (others are dropped by the tokenizer).")

    special_value_keys = [f"spv_{adata.uns['dataset_id']}_{adata.uns['batch']}", f"spv_{adata.uns['assay']}",
                          f"spv_{adata.uns['species']}", f"spv_{adata.uns['tissue']}"]
    missing_keys = [key for key in special_value_keys if key not in token_dict]
    if missing_keys:
        raise KeyError(f"Special value tokens missing from the token dictionary: {missing_keys}.")

    # Batch special value as computed by the tokenizer (cell_tokenizers.py, `batch_value`)
    spv_start_idx = min(value for key, value in token_dict.items() if key.startswith("spv_"))
    max_cls_tokens = sum(1 for key in token_dict if "cls" in key)
    batch_value = token_dict[special_value_keys[0]] - (spv_start_idx - 2 - max_cls_tokens)
    print(f"Vocabulary size: {len(token_dict)}; batch special value: {batch_value}. The special value embedding has "
          f"2 + n_special_values rows, so the configs need n_special_values >= {batch_value - 1}.")
    print(f"`gt_type: rank` maps the batch special value to token {batch_value + RANK_SPECIAL_TOKEN_OFFSET}; "
          f"the batch special value token is {token_dict[special_value_keys[0]]} "
          f"({'consistent' if batch_value + RANK_SPECIAL_TOKEN_OFFSET == token_dict[special_value_keys[0]] else 'INCONSISTENT'}).")


def main(args):
    # Keep the counts, labels and micron coordinates only
    adata_bronze = sc.read_h5ad(args.bronze_file_path)
    print(f"Read {args.bronze_file_path}: {adata_bronze.shape}.")
    adata = ad.AnnData(X=adata_bronze.X,
                       obs=adata_bronze.obs[[key for key in LABEL_KEYS if key in adata_bronze.obs]].copy(),
                       var=pd.DataFrame(index=adata_bronze.var_names.copy()),
                       obsm={"spatial": np.asarray(adata_bronze.obsm["spatial"])})
    del adata_bronze

    # Harmonize exactly as the TERRA corpus
    with open(args.gene_mapping_dict_file_path, "rb") as file:
        gene_ensembl_map_dict = pickle.load(file)
    adata = harmonize_batch(adata,
                            uns_params=UNS_PARAMS,
                            misc_params={"min_genes": 10},
                            gene_ensembl_map_dict=gene_ensembl_map_dict)
    adata.X.eliminate_zeros()
    assert adata.obs_names.is_unique, "Cell ids (obs_names) must be unique to join the labels after embedding."

    os.makedirs(Path(args.silver_file_path).parent, exist_ok=True)
    adata.write_h5ad(args.silver_file_path)
    print(f"Saved silver AnnData {adata.shape} to {args.silver_file_path}.")
    print(adata.obs[LABEL_KEYS[0]].value_counts())

    # Normalization factors of this cohort, with the tokenizer's gene-corrected read depth constants
    norm_factor_df = compute_norm_factors([adata])
    norm_factor_df.to_csv(args.norm_factor_file_path, index=False)
    print(f"Saved normalization factors of {len(norm_factor_df)} genes to {args.norm_factor_file_path}.")

    check_token_dictionary(adata, args.token_dictionary_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--bronze_file_path',
                        type=str,
                        default=BRONZE_FILE_PATH,
                        help='SpatialFusion OVCA AnnData (raw counts, gene symbols, pathologist labels).')
    parser.add_argument('--silver_file_path',
                        type=str,
                        default=SILVER_FILE_PATH,
                        help='Output silver AnnData (the only file in its folder).')
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        default=NORM_FACTOR_FILE_PATH,
                        help='Output per-gene normalization factors.')
    parser.add_argument('--gene_mapping_dict_file_path',
                        type=str,
                        default=GENE_MAPPING_DICT_FILE_PATH,
                        help='Gene symbol to Ensembl id dictionary of the TERRA corpus.')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        default=TOKEN_DICTIONARY_FILE_PATH,
                        help='Human token dictionary.')
    args = parser.parse_args()

    main(args)
