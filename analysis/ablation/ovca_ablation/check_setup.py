"""
This script checks the OVCA ablation setup on the farm before launching the job array:
1. Config parity: every variant's generated config is compared with the params of its mouse runs (model_config.yaml
   or params.yaml under --mouse_artifact_folder_path). Expected differences (OVCA data paths, special value
   rows, run folder and resumption, seed, step-matched epochs) are ignored; any other difference means the plan does not reproduce the
   mouse experiment and should be resolved first. Inferred overrides (cell-neighborhood layout, dimension ablations)
   are verified here.
2. Special values: the human batch special value fits into the special value embedding of the configs, and the
   `gt_type: rank` metatoken gets the right token id. The patched dataset derives the token offset from the token
   dictionary; the offset hard-coded in the unpatched code is reported for the human and the mouse dictionary (the
   latter tells whether the mouse rank arms trained with the right metatoken).
3. Tokenized datasets: each exists and has one row per silver cell.
4. Normalization factors: the gene-corrected read depth columns of the OVCA factor file match the tokenizer's own
   normalizer (same constants), and every silver gene has factors (`norm_factors.check_norm_factors`).
0. Pinned code: the patch is applied (predictor fix, metatoken inference option).
Exits with an error if check 0, 2, 3 or 4 fails; check 1 differences have to be reviewed by eye.

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
                            NORM_FACTOR_FILE_PATH,
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
                # Opt-in fixes (`training_options` and `tokenizer_options` of the plan; off in the mouse runs)
                "mask.pad_masks_to_batch_max",
                "data.stable_rank_ties",
                "data.zero_expr_from_raw_counts",
                "data.aggregate_own_neighbors",
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
# Special token offsets hard-coded for `gt_type: rank` in the unpatched code (cell_datasets.py), by vocabulary size
MOUSE_VOCAB_SIZE, MOUSE_RANK_OFFSET, HUMAN_RANK_OFFSET = 2785, 1674, 21957
MOUSE_TOKEN_DICTIONARY_FILE_PATH = "/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/token_dictionary_mus_musculus.pkl"


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


def check_pinned_code(code_path: str) -> bool:
    print("== 0. Pinned code")
    import inspect

    sys.path.insert(0, f"{code_path}/src")
    from app.inference.embed import embed_dataset
    from stjepa.models.gene_transformers import GeneTransformerCombinedPredictor
    import stjepa

    version = (Path(code_path) / "CODE_VERSION").read_text().strip() if (Path(code_path) / "CODE_VERSION").exists() else "?"
    patched = ("pad_special_tokens" in inspect.signature(GeneTransformerCombinedPredictor.forward).parameters
               and "keep_special_tokens" in inspect.signature(embed_dataset).parameters)
    print(f"{stjepa.__file__}: {version}; {'patched' if patched else 'NOT PATCHED (rerun setup_code.sh)'}")
    return patched


def hard_coded_rank_offset(token_dict: dict) -> int:
    return MOUSE_RANK_OFFSET if len(token_dict) == MOUSE_VOCAB_SIZE else HUMAN_RANK_OFFSET


def check_special_values(config_file_path: Path, token_dictionary_file_path: str, silver_file_path: str,
                         mouse_token_dictionary_file_path: str) -> bool:
    print("== 2. Batch special value")
    import scanpy as sc
    from stjepa.datasets.cell_datasets import get_spv_token_offset

    uns = sc.read_h5ad(silver_file_path, backed="r").uns
    with open(token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)
    spv_offset = get_spv_token_offset(token_dict)
    batch_value_token = token_dict[f"spv_{uns['dataset_id']}_{uns['batch']}"]
    batch_value = batch_value_token - spv_offset
    with open(config_file_path) as file:
        n_special_values = yaml.safe_load(file)["data"]["n_special_values"]
    fits = batch_value <= n_special_values + 1
    print(f"batch_value {batch_value}, special value embedding rows {n_special_values + 2}: "
          f"{'OK' if fits else 'TOO SMALL'}")
    print(f"`gt_type: rank` metatoken: token {batch_value + spv_offset} (offset from the token dictionary) for "
          f"spv token {batch_value_token}; the unpatched code used offset {hard_coded_rank_offset(token_dict)} "
          f"({'same' if hard_coded_rank_offset(token_dict) == spv_offset else 'DIFFERENT'})")

    # Mouse dictionary: did the mouse rank arms (unpatched code) get the right metatoken?
    if Path(mouse_token_dictionary_file_path).exists():
        with open(mouse_token_dictionary_file_path, "rb") as file:
            mouse_token_dict = pickle.load(file)
        mouse_offset = get_spv_token_offset(mouse_token_dict)
        print(f"Mouse token dictionary ({len(mouse_token_dict)} tokens): offset {mouse_offset}, unpatched code used "
              f"{hard_coded_rank_offset(mouse_token_dict)} -> mouse rank arms trained with "
              f"{'the right' if mouse_offset == hard_coded_rank_offset(mouse_token_dict) else 'a WRONG'} metatoken")
    else:
        print(f"Mouse token dictionary not found ({mouse_token_dictionary_file_path}).")
    return fits


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
    ok = check_pinned_code(args.code_path)
    check_config_parity(variants, config_file_paths, args.mouse_artifact_folder_path)
    ok &= check_special_values(config_file_paths["base"], args.token_dictionary_file_path, args.silver_file_path,
                               args.mouse_token_dictionary_file_path)
    if not args.skip_datasets:
        ok &= check_tokenized_datasets(tokenizations, args.silver_file_path)
    print("== 4. Normalization factors")
    from norm_factors import check_norm_factors
    from stjepa.preprocessors.normalizers import normalize_by_gene_corrected_read_depth
    ok &= check_norm_factors(args.norm_factor_file_path, [args.silver_file_path],
                             normalize_by_gene_corrected_read_depth)
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
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        default=NORM_FACTOR_FILE_PATH,
                        help='Per-gene normalization factors of the OVCA section.')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        default=TOKEN_DICTIONARY_FILE_PATH,
                        help='Human token dictionary.')
    parser.add_argument('--code_path',
                        type=str,
                        default="/nfs/team361/sb75/stjepa_ovca_ablation",
                        help='Pinned code (setup_code.sh).')
    parser.add_argument('--mouse_token_dictionary_file_path',
                        type=str,
                        default=MOUSE_TOKEN_DICTIONARY_FILE_PATH,
                        help='Mouse token dictionary (rank metatoken check of the mouse runs).')
    parser.add_argument('--mouse_artifact_folder_path',
                        type=str,
                        default="/nfs/team361/sb75/stjepa-reproducibility/artifacts",
                        help='Artifact folder of the mouse ablation runs (models/<timestamp>/model_config.yaml).')
    parser.add_argument('--skip_datasets',
                        action='store_true',
                        help='Skip the tokenized dataset checks (before tokenization).')
    args = parser.parse_args()

    main(args)
