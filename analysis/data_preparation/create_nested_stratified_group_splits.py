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

With --lightweight: only cell_ids and labels are saved per fold (no h5ad/dataset copies).
Output structure:
  root_save_dir/
    split_metadata.pkl
    outer-fold_1/
      train_cell_ids.npy
      test_cell_ids.npy
      train_labels.npy
      test_labels.npy
    ...

For details of the splits and the nested stratified group k-fold cross-validation, see the documentation of the `NestedStratifiedGroupKFold` class.
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import pickle
import numpy as np
import yaml
import scanpy as sc
from pathlib import Path
from datasets import load_from_disk
from terra.utils.nested_stratified_group_split import NestedStratifiedGroupKFold


def save_lightweight_splits(adata, cfg, split_info):
    label_column = cfg["split"]["label_column"]
    root_save_dir = Path(cfg["split"]["root_save_dir"])
    root_save_dir.mkdir(parents=True, exist_ok=True)

    vd = split_info["_validation_data"]
    global_indices = vd["global_indices"]
    outer_fold_validations = vd["outer_fold_validations"]
    inner_fold_validations = vd["inner_fold_validations"]

    K_inner = split_info["K_inner"]

    for outer_val in outer_fold_validations:
        fold_num = outer_val["outer_fold"] + 1  # 0-based → 1-based
        trainval_obs = global_indices[outer_val["trainval_pos"]]
        test_obs = global_indices[outer_val["test_pos"]]

        trainval_adata = adata[trainval_obs]
        test_adata = adata[test_obs]

        fold_dir = root_save_dir / f"outer-fold_{fold_num}"
        fold_dir.mkdir(exist_ok=True)

        np.save(fold_dir / "train_cell_ids.npy", trainval_adata.obs["cell_id"].values)
        np.save(fold_dir / "test_cell_ids.npy", test_adata.obs["cell_id"].values)
        np.save(fold_dir / "train_labels.npy", trainval_adata.obs[label_column].values)
        np.save(fold_dir / "test_labels.npy", test_adata.obs[label_column].values)

        print(f"  outer-fold_{fold_num}: {len(trainval_obs)} train, {len(test_obs)} test cells")

        # Inner folds — positions are relative to the outer trainval subset
        inner_folds = inner_fold_validations[outer_val["outer_fold"]]
        if K_inner == 1:
            inner_folds = inner_folds[:1]

        for inner_val in inner_folds:
            inner_num = inner_val["inner_fold"] + 1  # 0-based → 1-based
            inner_train_obs = trainval_obs[inner_val["train_pos"]]
            inner_val_obs = trainval_obs[inner_val["val_pos"]]

            inner_dir = fold_dir / f"inner-fold_{inner_num}"
            inner_dir.mkdir(exist_ok=True)

            np.save(inner_dir / "train_cell_ids.npy", adata[inner_train_obs].obs["cell_id"].values)
            np.save(inner_dir / "val_cell_ids.npy", adata[inner_val_obs].obs["cell_id"].values)

            print(f"    inner-fold_{inner_num}: {len(inner_train_obs)} train, {len(inner_val_obs)} val cells")

    # Save clean metadata (strip _validation_data as build_split_data does)
    split_info_clean = {k: v for k, v in split_info.items() if k != "_validation_data"}
    with open(root_save_dir / "split_metadata.pkl", "wb") as f:
        pickle.dump(split_info_clean, f)

    print(f"Lightweight splits saved to {root_save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Save only cell_ids and labels per fold; skip h5ad/dataset file creation",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    adata = sc.read_h5ad(cfg["data"]["adata_path"])

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

    if args.lightweight:
        save_lightweight_splits(adata, cfg, split_info)
    else:
        dataset = load_from_disk(cfg["data"]["dataset_path"])
        splitter.build_split_data(
            adata=adata,
            dataset=dataset,
            split_info=split_info,
            root_save_dir=cfg["split"]["root_save_dir"],
        )

if __name__ == "__main__":
    main()
