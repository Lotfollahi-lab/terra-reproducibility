"""
This script checks the OVCA ablation setup on the farm before launching the job array:
1. Config parity: every variant's generated config is compared with the params of its mouse runs (model_config.yaml
   or params.yaml under --mouse_artifact_folder_path). Expected differences (OVCA data paths, special value
   rows, run folder and resumption, seed, step-matched epochs) are ignored; any other difference means the plan does not reproduce the
   mouse experiment and should be resolved first. Inferred overrides (cell-neighborhood layout, dimension ablations)
   are verified here.
2. Special values: the human batch special value fits into the special value embedding of the configs, and the
   metatoken offset hard-coded for `gt_type: rank` matches the human token dictionary.
3. Tokenized datasets: each exists and has one row per silver cell.
Exits with an error if check 2 or 3 fails; check 1 differences have to be reviewed by eye.

Usage:
```bash
python check_setup.py [--skip_datasets]
```
"""

from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

from ablation_utils import (ARTIFACT_FOLDER_PATH,
                            PLAN_FILE_PATH,
                            SILVER_FILE_PATH,
                            TOKEN_DICTIONARY_FILE_PATH,
                            get_tokenized_dataset_path,
                            load_plan)

# Keys that differ between the mouse and OVCA runs by design
IGNORED_KEYS = {"data.dataset_name",
                "data.raw_data_folder_path",
                "data.token_dict_folder_path",
                "data.tokenized_data_folder_path",
                "data.n_special_values",
                "data.vocab_size",
                "seed",
                "state.folder_path",
                "state.load_checkpoint",
                "state.read_checkpoint",
                "state.read_timestamp",
                "state.checkpoint_freq",
                # Matched to the mouse runs in optimizer steps instead of epochs
                "optimization.epochs",
                "optimization.warmup"}
# RoPE keys (added on 21 Apr, after the plotted runs) are ignored by the pinned v2 code
IGNORED_KEY_PREFIXES = ("meta.rope_",)
# Special token offset hard-coded for `gt_type: rank` with a non-mouse vocabulary (cell_datasets.py)
RANK_SPECIAL_TOKEN_OFFSET = 21957


def flatten(config: dict, prefix: str = "") -> dict:
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat_config.update(flatten(value, f"{prefix}{key}."))
        else:
            flat_config[f"{prefix}{key}"] = value
    return flat_config


def find_mouse_params(mouse_artifact_folder_path: str, timestamp: str) -> Path | None:
    for pattern in [f"{mouse_artifact_folder_path}/models/{timestamp}/model_config.yaml",
                    f"{mouse_artifact_folder_path}/*/{timestamp}/params.yaml"]:
        file_paths = sorted(glob.glob(pattern))
        if file_paths:
            return Path(file_paths[0])
    return None


def check_config_parity(variants: dict, config_file_paths: dict, mouse_artifact_folder_path: str):
    print("== 1. Config parity with the mouse runs (differences other than OVCA data, run folder and seed)")
    for variant, spec in variants.items():
        with open(config_file_paths[variant]) as file:
            ovca_config = flatten(yaml.safe_load(file))
        for timestamp in spec["mouse_runs"]:
            mouse_params_file_path = find_mouse_params(mouse_artifact_folder_path, timestamp)
            if mouse_params_file_path is None:
                print(f"{variant} | mouse run {timestamp}: params not found")
                continue
            with open(mouse_params_file_path) as file:
                mouse_config = flatten(yaml.safe_load(file))
            differences = {key: (mouse_config.get(key, "<missing>"), ovca_config.get(key, "<missing>"))
                           for key in sorted(set(mouse_config) | set(ovca_config))
                           if key not in IGNORED_KEYS and not key.startswith(IGNORED_KEY_PREFIXES)
                           and mouse_config.get(key) != ovca_config.get(key)}
            print(f"{variant} | mouse run {timestamp}: "
                  f"{'identical' if not differences else f'mouse vs OVCA {differences}'}")


def check_special_values(config_file_path: Path, token_dictionary_file_path: str, silver_file_path: str) -> bool:
    print("== 2. Batch special value")
    import scanpy as sc

    uns = sc.read_h5ad(silver_file_path, backed="r").uns
    with open(token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)
    spv_start_idx = min(value for key, value in token_dict.items() if key.startswith("spv_"))
    max_cls_tokens = sum(1 for key in token_dict if "cls" in key)
    batch_value_token = token_dict[f"spv_{uns['dataset_id']}_{uns['batch']}"]
    batch_value = batch_value_token - (spv_start_idx - 2 - max_cls_tokens)
    with open(config_file_path) as file:
        n_special_values = yaml.safe_load(file)["data"]["n_special_values"]
    fits = batch_value <= n_special_values + 1
    rank_offset_consistent = batch_value + RANK_SPECIAL_TOKEN_OFFSET == batch_value_token
    print(f"batch_value {batch_value}, special value embedding rows {n_special_values + 2}: "
          f"{'OK' if fits else 'TOO SMALL'}")
    print(f"`gt_type: rank` metatoken {batch_value + RANK_SPECIAL_TOKEN_OFFSET} vs token {batch_value_token}: "
          f"{'OK' if rank_offset_consistent else 'INCONSISTENT (rank arms would train with a wrong metatoken)'}")
    return fits and rank_offset_consistent


def check_tokenized_datasets(tokenizations: dict, silver_file_path: str) -> bool:
    print("== 3. Tokenized datasets")
    import scanpy as sc
    from datasets import load_from_disk

    n_cells = sc.read_h5ad(silver_file_path, backed="r").n_obs
    complete = True
    for name, tokenization in tokenizations.items():
        dataset_path = get_tokenized_dataset_path(tokenization)
        if not dataset_path.exists():
            print(f"{name}: missing ({dataset_path})")
            complete = False
            continue
        n_rows = load_from_disk(str(dataset_path)).num_rows
        print(f"{name}: {n_rows} rows ({'OK' if n_rows == n_cells else f'expected {n_cells}'})")
        complete &= n_rows == n_cells
    return complete


def main(args):
    tokenizations, variants = load_plan(args.plan_file_path)
    manifest_df = pd.read_csv(Path(args.artifact_folder_path) / "manifest.tsv", sep="\t")
    config_file_paths = (manifest_df[manifest_df["replicate"] == 1]
                         .set_index("variant")["config_file_path"].to_dict())
    check_config_parity(variants, config_file_paths, args.mouse_artifact_folder_path)
    ok = check_special_values(config_file_paths["base"], args.token_dictionary_file_path, args.silver_file_path)
    if not args.skip_datasets:
        ok &= check_tokenized_datasets(tokenizations, args.silver_file_path)
    if not ok:
        sys.exit("Setup check failed, see above.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--plan_file_path',
                        type=str,
                        default=PLAN_FILE_PATH,
                        help='Ablation plan.')
    parser.add_argument('--artifact_folder_path',
                        type=str,
                        default=ARTIFACT_FOLDER_PATH,
                        help='Folder with the generated configs.')
    parser.add_argument('--silver_file_path',
                        type=str,
                        default=SILVER_FILE_PATH,
                        help='OVCA silver AnnData.')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        default=TOKEN_DICTIONARY_FILE_PATH,
                        help='Human token dictionary.')
    parser.add_argument('--mouse_artifact_folder_path',
                        type=str,
                        default="/nfs/team361/sb75/stjepa-reproducibility/artifacts",
                        help='Artifact folder of the mouse ablation runs (models/<timestamp>/model_config.yaml).')
    parser.add_argument('--skip_datasets',
                        action='store_true',
                        help='Skip the tokenized dataset checks (before tokenization).')
    args = parser.parse_args()

    main(args)
