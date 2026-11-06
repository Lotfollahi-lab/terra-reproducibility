"""
This script checks the setup of an ablation campaign on the farm before launching the job array:
0. Pinned code and campaign: the patch is applied (predictor fix, metatoken inference option), its CODE_VERSION is the
   `code_version` of the configs, and `dataset.env` of the artifact folder (what the job array reads) holds the values
   of the arguments, i.e. the configs were regenerated after the last change of the env file.
1. Config parity: every variant's generated config is compared with the params of its mouse runs (model_config.yaml
   or params.yaml under --mouse_artifact_folder_path). Expected differences (data paths, special value rows, run
   folder and resumption, seed, step-matched epochs, code version) are ignored; any other difference means the plan
   does not reproduce the mouse experiment and should be resolved first. Inferred overrides (cell-neighborhood layout,
   dimension ablations) are verified here.
2. Special values and vocabulary genes: the batch special value of every silver file fits into the special value
   embedding of the configs, and the `gt_type: rank` metatoken gets the right token id. The patched dataset derives the
   token offset from the token dictionary; the offset hard-coded in the unpatched code is reported for the dataset's
   and the mouse dictionary (the latter tells whether the mouse rank arms trained with the right metatoken). Fails if a
   silver cell has no detected gene of the token dictionary: its ranked gene lists are empty, which shifts the segments
   of the cells around it on the sparse ranking path (remove such cells with filter_silver_cells.py).
3. Labels and sections: every label file (LABEL_FILE_PATHS, default: the silver files) is one section, the batch of the
   integration metrics. Prints its cells, uns dataset id and batch, label keys and their most frequent values (to spot
   the unassigned value) and how many cells carry the unassigned label. Fails if a file is missing or has no
   obs['cell_id'], a label key is in no file, a cell id occurs twice, or, with two or more sections, scib_metrics
   does not import.
4. Tokenized datasets: each exists, has one row per silver cell and all its cells are in the label files (with the
   cells per section).
5. Normalization factors: the gene-corrected read depth columns of the dataset's factor file match the tokenizer's own
   normalizer (same constants), and every silver gene has factors (`norm_factors.check_norm_factors`).
6. Schedule end values (report): the LR, weight decay and EMA schedules span `ipe_scale` x epochs (1.25, as in
   V-JEPA), but training stops after `epochs`, so they end at 80% of their path; the values to state in Methods.
Exits with an error if check 0, 2, 3, 4 or 5 fails; check 1 differences have to be reviewed by eye. --skip_datasets
skips check 4 (before tokenization).

Usage (`source datasets/<dataset>.env` first, see README.md; add --skip_datasets before tokenization):
```bash
python check_setup.py --plan_file_path "${PLAN_FILE_PATH}" --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" \
    --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}" \
    --silver_file_paths ${SILVER_FILE_PATHS} --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" \
    --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --code_path "${CODE_PATH}" \
    --mouse_token_dictionary_file_path "${MOUSE_TOKEN_DICTIONARY_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}"
```
"""

from __future__ import annotations

import argparse
import glob
import math
import pickle
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ablation_utils import cells_without_vocabulary_genes, get_section_names, get_tokenized_dataset_path, load_plan

# Keys that differ between the mouse runs and the reruns by design
IGNORED_KEYS = {"code_version",
                "data.dataset_name",
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
                "data.exact_n_nonzero_tokens",
                "data.seurat_loess_span",
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
# Label values listed per key (check 3), and values that may mark unassigned cells (listed in addition)
TOP_VALUES = 10
UNASSIGNED_PATTERN = re.compile(r"unassign|unknown|undefined|unlabel|^(none|nan|na|n/a|other|-|)$", re.IGNORECASE)


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
    print("== 1. Config parity with the mouse runs (differences other than data, run folder, seed and code version)")
    for variant, spec in variants.items():
        with open(config_file_paths[variant]) as file:
            config = flatten(yaml.safe_load(file))
        for timestamp in spec["mouse_runs"]:
            mouse_params_file_path = find_mouse_params(mouse_artifact_folder_path, timestamp)
            if mouse_params_file_path is None:
                print(f"{variant} | mouse run {timestamp}: params not found")
                continue
            with open(mouse_params_file_path) as file:
                mouse_config = flatten(yaml.safe_load(file))
            differences = {key: (mouse_config.get(key, "<missing>"), config.get(key, "<missing>"))
                           for key in sorted(set(mouse_config) | set(config))
                           if key not in IGNORED_KEYS and not key.startswith(IGNORED_KEY_PREFIXES)
                           and mouse_config.get(key) != config.get(key)}
            print(f"{variant} | mouse run {timestamp}: "
                  f"{'identical' if not differences else f'mouse vs rerun {differences}'}")


def check_pinned_code(code_path: str, config_file_path: str) -> bool:
    print("== 0. Pinned code and campaign")
    import inspect

    sys.path.insert(0, f"{code_path}/src")
    from app.inference.embed import embed_dataset
    from stjepa.models.gene_transformers import GeneTransformerCombinedPredictor
    import stjepa

    version_file_path = Path(code_path) / "CODE_VERSION"
    version = version_file_path.read_text().strip() if version_file_path.exists() else "?"
    patched = ("pad_special_tokens" in inspect.signature(GeneTransformerCombinedPredictor.forward).parameters
               and "keep_special_tokens" in inspect.signature(embed_dataset).parameters)
    print(f"{stjepa.__file__}: {version}; {'patched' if patched else 'NOT PATCHED (rerun setup_code.sh)'}")
    with open(config_file_path) as file:
        config_version = yaml.safe_load(file).get("code_version")
    same_version = config_version == version
    print(f"Code version of the configs: {config_version} "
          f"({'same' if same_version else 'DIFFERENT: regenerate the configs or set up that code version'})")
    return patched and same_version


def check_dataset_env(artifact_folder_path: str, expected: dict) -> bool:
    """dataset.env (generate_ablation_runs.py; read by the job array) holds the values of the arguments."""
    dataset_env_file_path = Path(artifact_folder_path) / "dataset.env"
    if not dataset_env_file_path.is_file():
        print(f"{dataset_env_file_path} is MISSING: run generate_ablation_runs.py")
        return False
    values = {}
    for line in dataset_env_file_path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key] = " ".join(shlex.split(value))
    differences = {key: (values.get(key, "<missing>"), value) for key, value in expected.items()
                   if values.get(key) != value}
    if differences:
        print(f"{dataset_env_file_path} (read by the job array) is DIFFERENT (dataset.env vs arguments): "
              f"{differences}; regenerate the configs with generate_ablation_runs.py")
    else:
        print(f"{dataset_env_file_path} (read by the job array): same values as the arguments")
    return not differences


def hard_coded_rank_offset(token_dict: dict) -> int:
    return MOUSE_RANK_OFFSET if len(token_dict) == MOUSE_VOCAB_SIZE else HUMAN_RANK_OFFSET


def check_special_values(config_file_path: Path, token_dictionary_file_path: str, silver_file_paths: list,
                         mouse_token_dictionary_file_path: str) -> bool:
    print("== 2. Batch special values and vocabulary genes")
    import scanpy as sc
    from stjepa.datasets.cell_datasets import get_spv_token_offset

    with open(token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)
    spv_offset = get_spv_token_offset(token_dict)
    with open(config_file_path) as file:
        n_special_values = yaml.safe_load(file)["data"]["n_special_values"]
    fits = True
    n_empty_cells = 0
    for silver_file_path in silver_file_paths:
        adata = sc.read_h5ad(silver_file_path, backed="r")
        uns = adata.uns
        # Cells without a detected vocabulary gene, among those the tokenizer keeps (obs['filter_pass'], filters.py)
        empty = cells_without_vocabulary_genes(adata, token_dict)
        if "filter_pass" in adata.obs.columns:
            empty &= np.array([filter_pass == 1 for filter_pass in adata.obs["filter_pass"]])
        n_empty_cells += int(empty.sum())
        print(f"{silver_file_path}: {int(empty.sum())} of {adata.n_obs} cells without a detected gene of the token "
              f"dictionary{' (OK)' if not empty.any() else ', e.g. ' + str(list(adata.obs_names[empty][:3]))}")
        batch_value_token = token_dict[f"spv_{uns['dataset_id']}_{uns['batch']}"]
        batch_value = batch_value_token - spv_offset
        fits &= batch_value <= n_special_values + 1
        print(f"{silver_file_path}: batch_value {batch_value}, special value embedding rows {n_special_values + 2}: "
              f"{'OK' if batch_value <= n_special_values + 1 else 'TOO SMALL'}")
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
    if n_empty_cells > 0:
        print(f"ERROR: {n_empty_cells} cells have no detected gene of the token dictionary, so their ranked gene lists "
              f"are empty; on the sparse ranking path the cells around them get shifted segments and lose their "
              f"farthest neighbors. Remove them with filter_silver_cells.py (README.md) and use the filtered files.")
    return fits and n_empty_cells == 0


def check_labels(label_file_paths: list, label_keys: list, unassigned_label: str) -> tuple[bool, pd.Series]:
    """Check 3; returns whether it passed and the section of every cell id of the label files."""
    print("== 3. Labels and sections (one label file per section; the sections are the batches of the integration "
          "metrics)")
    import scanpy as sc

    ok = True
    cell_sections = []
    files_with_key = {key: [] for key in label_keys}
    n_unassigned = 0
    for file_path, section_name in zip(label_file_paths, get_section_names(label_file_paths)):
        if not Path(file_path).is_file():
            print(f"{file_path}: MISSING")
            ok = False
            continue
        adata = sc.read_h5ad(file_path, backed="r")
        obs, uns = adata.obs, adata.uns
        keys = [key for key in label_keys if key in obs.columns]
        print(f"{file_path}: section {section_name}; {adata.n_obs} cells; uns dataset_id "
              f"{uns.get('dataset_id', '<missing>')}, batch {uns.get('batch', '<missing>')}; label keys {keys}")
        if "cell_id" in obs.columns:
            cell_sections.append(pd.Series(section_name, index=obs["cell_id"].astype(str).to_numpy()))
        else:
            print("  ERROR: no obs['cell_id'] (the tokenizer cell ids)")
            ok = False
        for key in keys:
            files_with_key[key].append(file_path)
            # Values as the evaluation compares them (strings; missing values are unlabelled)
            labelled = obs[key].notna().to_numpy()
            counts = pd.Series(obs[key].astype(str).to_numpy()[labelled]).value_counts()
            n_key_unassigned = int(counts.get(unassigned_label, 0))
            n_unassigned += n_key_unassigned
            print(f"  {key}: {len(counts)} values, {int((~labelled).sum())} cells without a value; top "
                  f"{min(TOP_VALUES, len(counts))}: "
                  f"{', '.join(f'{value} ({count})' for value, count in counts.head(TOP_VALUES).items())}")
            candidates = [f"{value} ({count})" for value, count in counts.items() if UNASSIGNED_PATTERN.search(value)]
            if candidates:
                print(f"    values that may mark unassigned cells: {', '.join(candidates)}")
            print(f"    UNASSIGNED_LABEL {unassigned_label!r}: {n_key_unassigned} cells")
    for key, file_paths in files_with_key.items():
        if file_paths:
            print(f"Label key {key}: in {len(file_paths)} of {len(label_file_paths)} label files")
        else:
            print(f"ERROR: label key {key} is in none of the label files")
            ok = False
    if n_unassigned == 0:
        print(f"WARNING: UNASSIGNED_LABEL {unassigned_label!r} occurs in no label file, so every labelled cell is "
              f"scored (as in the mouse evaluation); set it to the unassigned value above if there is one.")
    cell_section = pd.concat(cell_sections) if cell_sections else pd.Series(dtype=object)
    duplicated = cell_section.index[cell_section.index.duplicated()].unique()
    if len(duplicated) > 0:
        print(f"ERROR: {len(duplicated)} cell ids occur more than once in the label files, e.g. "
              f"{list(duplicated[:3])}")
        ok = False
        cell_section = cell_section[~cell_section.index.duplicated()]
    else:
        print(f"Cell ids: {len(cell_section)}, unique over the label files")
    if len(label_file_paths) >= 2:
        try:
            import scib_metrics
            print(f"{len(label_file_paths)} sections: the evaluation computes the batch-integration metrics (iLISI, "
                  f"ASW, MMD) between them (scib_metrics {scib_metrics.__version__})")
        except ImportError as error:
            print(f"ERROR: {len(label_file_paths)} sections, but scib_metrics (batch-integration metrics) does not "
                  f"import: {error}")
            ok = False
    else:
        print("1 section: no batch-integration metrics")
    return ok, cell_section


def report_schedule_end_values(config_file_paths: dict):
    print("== 6. Schedule end values (schedules span ipe_scale x epochs; training stops after epochs)")
    schedules = {}
    for variant, config_file_path in config_file_paths.items():
        with open(config_file_path) as file:
            optimization = yaml.safe_load(file)["optimization"]
        schedules.setdefault(yaml.safe_dump(optimization, sort_keys=True), []).append(variant)
    for optimization, variants in schedules.items():
        opt = yaml.safe_load(optimization)
        epochs, warmup, scale = opt["epochs"], opt["warmup"], opt["ipe_scale"]
        # Progress of WarmupCosineSchedule (after warmup), CosineWDSchedule and the EMA schedule at the last step
        lr_end = opt["final_lr"] + (opt["lr"] - opt["final_lr"]) * 0.5 * (
            1 + math.cos(math.pi * (epochs - warmup) / (scale * epochs - warmup)))
        wd_end = opt["final_weight_decay"] + (opt["weight_decay"] - opt["final_weight_decay"]) * 0.5 * (
            1 + math.cos(math.pi / scale))
        ema_end = opt["ema"][0] + (opt["ema"][1] - opt["ema"][0]) / scale
        print(f"{len(variants)} variants (e.g. {variants[0]}), {epochs} epochs, ipe_scale {scale}: "
              f"LR {opt['lr']:g} -> {lr_end:.3g} (configured {opt['final_lr']:g}); "
              f"WD {opt['weight_decay']:g} -> {wd_end:.3g} (configured {opt['final_weight_decay']:g}); "
              f"EMA {opt['ema'][0]:g} -> {ema_end:.4g} (configured {opt['ema'][1]:g})")


def check_tokenized_datasets(tokenizations: dict, silver_file_paths: list, data_directory_path: str,
                             dataset_name: str, cell_section: pd.Series) -> bool:
    print("== 4. Tokenized datasets (one row per silver cell; every cell in the label files)")
    import scanpy as sc
    from datasets import load_from_disk

    n_cells = sum(sc.read_h5ad(silver_file_path, backed="r").n_obs for silver_file_path in silver_file_paths)
    complete = True
    for name, tokenization in tokenizations.items():
        dataset_path = get_tokenized_dataset_path(tokenization, data_directory_path, dataset_name)
        if not dataset_path.exists():
            print(f"{name}: missing ({dataset_path})")
            complete = False
            continue
        dataset = load_from_disk(str(dataset_path))
        # Cell ids as the evaluation joins them to the labels
        cell_ids = pd.Index([str(cell_id) for cell_id in dataset["cell_id"]])
        sections = cell_section.reindex(cell_ids)
        unlabelled = sections.isna().to_numpy()
        coverage = ("all in the label files" if not unlabelled.any() else
                    f"{unlabelled.sum()} NOT in the label files, e.g. {list(cell_ids[unlabelled][:3])}")
        print(f"{name}: {dataset.num_rows} rows ({'OK' if dataset.num_rows == n_cells else f'expected {n_cells}'}); "
              f"cells {coverage}; per section {sections.value_counts().to_dict()}")
        complete &= dataset.num_rows == n_cells and not unlabelled.any()
    return complete


def main(args):
    label_file_paths = args.label_file_paths or args.silver_file_paths
    tokenizations, variants = load_plan(args.plan_file_path)
    manifest_df = pd.read_csv(Path(args.artifact_folder_path) / "manifest.tsv", sep="\t")
    config_file_paths = (manifest_df[manifest_df["replicate"] == 1]
                         .set_index("variant")["config_file_path"].to_dict())
    ok = check_pinned_code(args.code_path, config_file_paths["base"])
    ok &= check_dataset_env(args.artifact_folder_path,
                            {"DATASET_NAME": args.dataset_name,
                             "SILVER_FILE_PATHS": " ".join(args.silver_file_paths),
                             "LABEL_FILE_PATHS": " ".join(label_file_paths),
                             "LABEL_KEYS": " ".join(args.label_keys),
                             "UNASSIGNED_LABEL": args.unassigned_label,
                             "NORM_FACTOR_FILE_PATH": args.norm_factor_file_path,
                             "CODE_PATH": args.code_path})
    check_config_parity(variants, config_file_paths, args.mouse_artifact_folder_path)
    ok &= check_special_values(config_file_paths["base"], args.token_dictionary_file_path, args.silver_file_paths,
                               args.mouse_token_dictionary_file_path)
    labels_ok, cell_section = check_labels(label_file_paths, args.label_keys, args.unassigned_label)
    ok &= labels_ok
    if not args.skip_datasets:
        ok &= check_tokenized_datasets(tokenizations, args.silver_file_paths, args.data_directory_path,
                                       args.dataset_name, cell_section)
    print("== 5. Normalization factors")
    from norm_factors import check_norm_factors
    from stjepa.preprocessors.normalizers import normalize_by_gene_corrected_read_depth
    ok &= check_norm_factors(args.norm_factor_file_path, args.silver_file_paths,
                             normalize_by_gene_corrected_read_depth)
    report_schedule_end_values(config_file_paths)
    if not ok:
        sys.exit("Setup check failed, see above.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--plan_file_path',
                        type=str,
                        required=True,
                        help='Ablation plan.')
    parser.add_argument('--artifact_folder_path',
                        type=str,
                        required=True,
                        help='Folder with the generated configs and manifest.tsv.')
    parser.add_argument('--data_directory_path',
                        type=str,
                        required=True,
                        help='Data directory with the gold subdirectory (tokenized datasets).')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help='Name of the dataset (tokenized dataset names).')
    parser.add_argument('--silver_file_paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Silver AnnData files of the dataset.')
    parser.add_argument('--label_file_paths',
                        type=str,
                        nargs='+',
                        help='Label files of the evaluation, one per section (LABEL_FILE_PATHS; default: the silver '
                             'files).')
    parser.add_argument('--label_keys',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Keys of the niche labels in the obs of the label files.')
    parser.add_argument('--unassigned_label',
                        type=str,
                        required=True,
                        help='Label of cells outside annotated regions (excluded from the metrics).')
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        required=True,
                        help='Per-gene normalization factors of the dataset.')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        required=True,
                        help='Token dictionary of the dataset\'s species.')
    parser.add_argument('--code_path',
                        type=str,
                        required=True,
                        help='Pinned code (setup_code.sh).')
    parser.add_argument('--mouse_token_dictionary_file_path',
                        type=str,
                        required=True,
                        help='Mouse token dictionary (rank metatoken check of the mouse runs).')
    parser.add_argument('--mouse_artifact_folder_path',
                        type=str,
                        required=True,
                        help='Artifact folder of the mouse ablation runs (models/<timestamp>/model_config.yaml).')
    parser.add_argument('--skip_datasets',
                        action='store_true',
                        help='Skip the tokenized dataset checks (check 4; before tokenization).')
    args = parser.parse_args()

    main(args)
