"""
Usage: python reproducibility/analysis/data_preparation/create_nested_stratified_group_splits.py --config reproducibility/config/create-nested-split/create-split_xhs_bk51_niche_transfer.yaml

This script creates nested stratified group k-fold cross-validation splits for within-donor or across-donor datasets.

The config file is expected to be in the following format:
```
data:
  adata_path: /lustre/scratch126/cellgen/lotfollahi/am84/NEMO/data/xhs_harmonized_nemo_heldout.h5ad
  dataset_path: /lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_110m/hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_10.dataset
split:
  in_patient: true
  stratify_group: batch
  patient_id: BK51
  label_column: niche19
  k_outer: 4
  k_inner: 2
  shuffle: true
  seed: 42
  root_save_dir: /lustre/scratch126/cellgen/lotfollahi/am84/NEMO/data/xhs_bk51_niche-transfer
```

The `in_patient` flag indicates whether to split within a single donor or across donors.
The `stratify_group` flag indicates the column in the adata.obs to use as groups for stratification.
The `patient_id` flag indicates the patient ID to split within.
The `label_column` flag indicates the column in the adata.obs to use as labels for stratification.
The `k_outer` flag indicates the number of outer folds.
The `k_inner` flag indicates the number of inner folds.
The `shuffle` flag indicates whether to shuffle the data before splitting.
The `seed` flag indicates the random seed to use for reproducibility.
The `root_save_dir` flag indicates the root directory to save the splits.

For details of the splits and the nested stratified group k-fold cross-validation, see the documentation of the `NestedStratifiedGroupKFold` class.
"""

import argparse
import yaml
import scanpy as sc
from datasets import load_from_disk
from nichejepa.utils.nested_stratified_group_split import NestedStratifiedGroupKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config",
    )
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    adata = sc.read_h5ad(cfg["data"]["adata_path"])
    dataset = load_from_disk(cfg["data"]["dataset_path"])
    
    if cfg["split"]["in_patient"]:
        assert cfg["split"]["patient_id"] is not None
        print(f"Splitting tissue sections within Donor {cfg['split']['patient_id']}")
        adata = adata[adata.obs["Patient ID"] == cfg["split"]["patient_id"]]
    else:
        print(f"Splitting donors.")

    splitter = NestedStratifiedGroupKFold(
        stratify_group=cfg["split"]["stratify_group"],
        label_column=cfg["split"]["label_column"],
        K_outer=cfg["split"]["k_outer"],
        K_inner=cfg["split"]["k_inner"],
        shuffle=cfg["split"].get("shuffle", True),
        seed=cfg["split"].get("seed", 42),
        require_all_labels=cfg["split"].get("require_all_labels", False),
        group_balance_tolerance=cfg["split"].get("group_balance_tolerance", 1),
        max_retries=cfg["split"].get("max_retries", 10),
    )

    split_info = splitter.design_splits(adata)

    splitter.build_split_data(
        adata=adata,
        dataset=dataset,
        split_info=split_info,
        root_save_dir=cfg["split"]["root_save_dir"],
    )

if __name__ == "__main__":
    main()