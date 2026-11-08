"""
This script generates the training configs and the run manifest of an ablation campaign from the ablation plan
(`ablation_plan.yaml`) and the base config (`base_config.yaml`: the mouse ablation base in TERRA's config format), for
the dataset given by the arguments (values
of `datasets/<dataset>.env`). Every run config is the base config with
- the dataset (silver folder, token dictionary, the variant's tokenized dataset and its tokenization keys; the keys
  are recorded so that a re-tokenization from the model folder, e.g. with `harmonize_tokenize_embed_pipeline` on other
  data, matches training; the ablation evaluation itself embeds the run's own tokenized training dataset and does not
  re-tokenize),
- `n_special_values` large enough for the dataset's batch special values (the mouse value 200 is too small for the
  human token dictionary), and `vocab_size`, the size of the token dictionary (TERRA builds the models from the token
  dictionary and does not read the key; recorded for the packaged model config),
- the schedule (`--epochs`, `--warmup`; for OVCA, 82 epochs with 1 warmup epoch match the 300 / 3 epochs of the mouse
  runs in optimizer steps),
- a fixed run folder `runs/<run name>_<config hash>` that is resumed from `terra-latest.pth.tar` (TERRA raises on a
  missing checkpoint, so the job array's launch config sets `state.load_checkpoint` only once the file exists), and
  only the latest checkpoint kept; a changed config gets a new folder instead of resuming or reusing an earlier one,
- `seed: 1000 * replicate` (TERRA's top-level `seed`; without it all replicates use seed 0 like the mouse runs),
- `code_version`: the CODE_VERSION of the pinned code (`setup_code.sh`), so that the config hash changes with the code
  (the training code ignores the key),
- the variant overrides (dotted keys that must exist in the base config).
Every config must set `meta.zero_fully_masked_attention_rows: true` (base config): an attention query whose keys are
all padding (e.g. a context of padding only) then gives zeros instead of NaN.
The manifest lists the runs in launch order for the job array (`get_runs`: replicates added later to the plan,
`appended_replicates`, come last, so the other runs keep their rows). `dataset.env` in the artifact folder holds the
dataset-level values of the job array (`run_ablation_array.lsf`), so that the artifact folder defines the campaign,
including the label files of the evaluation (`--label_file_paths`, one per section; default: the silver files).

Usage (`source datasets/<dataset>.env` first, see README.md; LABEL_FILE_PATHS is optional):
```bash
python generate_ablation_runs.py --plan_file_path "${PLAN_FILE_PATH}" --base_config_file_path "${BASE_CONFIG_FILE_PATH}" \
    --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}" \
    --dataset_name "${DATASET_NAME}" --silver_file_paths ${SILVER_FILE_PATHS} \
    --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" --experiment_tag "${EXPERIMENT_TAG}" \
    --epochs "${EPOCHS}" --warmup "${WARMUP}" --n_special_values "${N_SPECIAL_VALUES}" --code_path "${CODE_PATH}"
```
"""

import argparse
import copy
import hashlib
import os
import pickle
import re
import shlex
from pathlib import Path

import yaml

from ablation_utils import (NORM_METHOD_KEYS,
                            SECTION_KEY,
                            TOKENIZER_OPTION_DEFAULTS,
                            get_runs,
                            get_seq_len_cell,
                            get_tokenized_dataset_path,
                            load_plan)

# run_ablation_array.lsf reads the rows with these columns (in this order)
MANIFEST_COLUMNS = ["index", "run_id", "run_name", "variant", "category", "label", "replicate", "seed", "code_version",
                    "config_file_path", "run_folder_path", "eval_folder_path"]
TOKENIZER_TYPES = {"cell_graph": "cell-graph-tokenizer", "cell_neigh": "cell-neighborhood-tokenizer"}


def set_key(config: dict, dotted_key: str, value, allow_new: bool = False):
    """
    Set a dotted key, which must already exist in the config (catches typos in the plan) unless `allow_new` (opt-in
    keys that the base config does not have).
    """
    *parents, key = dotted_key.split(".")
    section = config
    for parent in parents:
        section = section[parent]
    if key not in section and not allow_new:
        raise KeyError(f"Override {dotted_key} does not exist in the base config.")
    section[key] = value


def validate_config(config: dict, tokenization: dict, run_name: str):
    """Check that sequence lengths match the token layout of the tokenized dataset and the model dimensions."""
    data, meta = config["data"], config["meta"]
    if TOKENIZER_TYPES[data["tokenizer_type"]] != tokenization["tokenizer"]:
        raise ValueError(f"{run_name}: tokenizer_type {data['tokenizer_type']} does not match {tokenization}.")
    # Cell graph: n_neighs + 1 blocks of seq_len_cell tokens; cell neighborhood: the tokenizer's split (`seq_len_cell`
    # of the plan). The datasets expect exactly seq_len_cell + seq_len_neighborhood = model_input_size tokens.
    model_input_size = tokenization["model_input_size"]
    seq_len_cell = get_seq_len_cell(tokenization)
    if (data["seq_len_cell"], data["seq_len_neighborhood"]) != (seq_len_cell, model_input_size - seq_len_cell):
        raise ValueError(f"{run_name}: seq_len_cell / seq_len_neighborhood {data['seq_len_cell']} / "
                         f"{data['seq_len_neighborhood']} do not match the tokenized data "
                         f"({seq_len_cell} / {model_input_size - seq_len_cell}).")
    for key in ["enc_emb_dim", "pred_emb_dim"]:
        if meta[key] % meta["num_heads"]:
            raise ValueError(f"{run_name}: {key} {meta[key]} is not divisible by num_heads {meta['num_heads']}.")
    # Training and TERRA's inference read restrict_special_attention from `mask` (the v2 inference read `meta`): keep
    # them equal
    mask = config["mask"]
    if meta.get("restrict_special_attention", False) != mask.get("restrict_special_attention", False):
        raise ValueError(f"{run_name}: meta.restrict_special_attention and mask.restrict_special_attention differ.")
    if not 0.0 <= mask.get("special_token_pad_ratio", 0.0) <= 1.0:
        raise ValueError(f"{run_name}: mask.special_token_pad_ratio must be in [0, 1].")
    # Without it, a sample whose context is padding only (e.g. an index cell with at most 2 detected genes and a padded
    # or no metatoken) gives NaN attention outputs on some GPU kernels, which spread to the loss and the weights
    if meta.get("zero_fully_masked_attention_rows") is not True:
        raise ValueError(f"{run_name}: meta.zero_fully_masked_attention_rows must be true (base config).")


def make_run_config(base_config: dict,
                    tokenization: dict,
                    run: dict,
                    run_folder_path: Path,
                    args) -> dict:
    config = copy.deepcopy(base_config)

    # Dataset and the tokenization of this variant
    data = config["data"]
    data["dataset_name"] = args.dataset_name
    data["raw_data_folder_path"] = str(get_silver_folder_path(args))
    data["token_dict_folder_path"] = args.token_dictionary_file_path
    data["tokenized_data_folder_path"] = str(get_tokenized_dataset_path(tokenization,
                                                                        args.data_directory_path,
                                                                        args.dataset_name))
    data["model_input_size"] = tokenization["model_input_size"]
    data["n_neighs"] = tokenization["n_neighs"]
    for key in NORM_METHOD_KEYS:
        data[key] = tokenization[key]
    # Opt-in tokenizer fixes, recorded so that a re-tokenization from the model folder (`tokenize_adata`) matches; the
    # cell-neighborhood split is the model's own data.seq_len_cell (validated below)
    for key in TOKENIZER_OPTION_DEFAULTS:
        if key != "seq_len_cell":
            data[key] = tokenization[key]
    data["n_special_values"] = args.n_special_values
    data["vocab_size"] = args.vocab_size

    # Schedule, run folder with resumption, seed
    config["optimization"]["epochs"] = args.epochs
    config["optimization"]["warmup"] = args.warmup
    config["state"]["folder_path"] = str(run_folder_path)
    config["state"]["load_checkpoint"] = True
    config["state"]["read_checkpoint"] = f"{config['state']['write_tag']}-latest.pth.tar"
    config["state"]["checkpoint_freq"] = args.epochs + 1  # never write epoch snapshots, only the latest checkpoint
    config["seed"] = run["seed"]
    config["code_version"] = args.code_version

    for key, value in run["training_options"].items():
        set_key(config, key, value, allow_new=True)
    for key, value in run["overrides"].items():
        set_key(config, key, value)
    validate_config(config, tokenization, run["run_name"])
    return config


def get_silver_folder_path(args) -> Path:
    """Folder of the silver data, which the tokenizer reads (`tokenize_dataset.py`)."""
    return Path(args.data_directory_path) / "silver" / args.dataset_name


def read_code_version(code_path: str) -> str:
    """CODE_VERSION of the pinned code, written by setup_code.sh."""
    code_version_file_path = Path(code_path) / "CODE_VERSION"
    if not code_version_file_path.is_file():
        raise FileNotFoundError(f"{code_version_file_path} does not exist: set up the pinned code with setup_code.sh "
                                f"first.")
    code_version = code_version_file_path.read_text().strip()
    if not code_version or "\n" in code_version or "\t" in code_version:
        raise ValueError(f"{code_version_file_path} must hold one non-empty line, not {code_version!r}.")
    return code_version


def validate_args(args):
    """Values that end up in dataset.env and the manifest: lists without blanks, ids that wandb accepts."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment_tag):
        raise ValueError(f"--experiment_tag {args.experiment_tag!r} may only contain letters, digits, '_' and '-' "
                         f"(it starts the wandb run ids).")
    for name in ["silver_file_paths", "label_file_paths", "label_keys"]:
        for value in getattr(args, name):
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"--{name}: {value!r} is empty or contains whitespace (the lists are passed on "
                                 f"space-separated).")
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            raise ValueError(f"--{name} {getattr(args, name)} has duplicates.")
    if SECTION_KEY in args.label_keys:
        raise ValueError(f"--label_keys: '{SECTION_KEY}' holds the section of each cell in the evaluation; rename the "
                         f"key.")
    # The tokenizer reads every h5ad file below the silver folder, so the silver files must be there (the label files
    # may be anywhere)
    silver_folder_path = os.path.abspath(get_silver_folder_path(args))
    outside = [file_path for file_path in args.silver_file_paths
               if os.path.commonpath([silver_folder_path, os.path.abspath(file_path)]) != silver_folder_path]
    if outside:
        raise ValueError(f"Silver files {outside} are not in {silver_folder_path}, the folder that the tokenizer reads.")


def write_dataset_env(args, dataset_env_file_path: Path):
    """Dataset-level values of the job array (sourced by run_ablation_array.lsf)."""
    values = {"DATASET_NAME": args.dataset_name,
              "SILVER_FILE_PATHS": " ".join(args.silver_file_paths),
              "LABEL_FILE_PATHS": " ".join(args.label_file_paths),
              "LABEL_KEYS": " ".join(args.label_keys),
              "UNASSIGNED_LABEL": args.unassigned_label,
              "NORM_FACTOR_FILE_PATH": args.norm_factor_file_path,
              "EXPERIMENT_TAG": args.experiment_tag,
              "CODE_PATH": args.code_path,
              "CODE_VERSION": args.code_version}
    with open(dataset_env_file_path, "w") as file:
        file.write("# Written by generate_ablation_runs.py: dataset-level values of this campaign, sourced by every\n"
                   "# element of run_ablation_array.lsf (regenerate instead of editing).\n")
        for key, value in values.items():
            file.write(f"{key}={shlex.quote(str(value))}\n")


def main(args):
    args.label_file_paths = args.label_file_paths or args.silver_file_paths
    validate_args(args)
    args.code_version = read_code_version(args.code_path)
    with open(args.token_dictionary_file_path, "rb") as file:
        args.vocab_size = len(pickle.load(file))
    tokenizations, variants = load_plan(args.plan_file_path)
    with open(args.base_config_file_path) as file:
        base_config = yaml.safe_load(file)

    artifact_folder_path = Path(args.artifact_folder_path)
    config_folder_path = artifact_folder_path / "configs"
    os.makedirs(config_folder_path, exist_ok=True)

    runs = get_runs(variants)
    manifest_rows = []
    for index, run in enumerate(runs, start=1):
        # Run ids carry a hash of the config, so that a changed config (e.g. other epochs or code) never resumes or
        # reuses the checkpoints and metrics of an earlier config; the hash excludes the run folder, which contains it
        config = make_run_config(base_config, tokenizations[run["tokenization"]], run, Path(), args)
        config["state"]["folder_path"] = None
        config_hash = hashlib.sha1(yaml.dump(config, sort_keys=True).encode()).hexdigest()[:8]
        run_id = f"{run['run_name']}_{config_hash}"
        run_folder_path = artifact_folder_path / "runs" / run_id
        config["state"]["folder_path"] = str(run_folder_path)
        config_file_path = config_folder_path / f"{run_id}.yaml"
        with open(config_file_path, "w") as file:
            yaml.dump(config, file, sort_keys=False)
        manifest_rows.append({"index": index,
                              "run_id": run_id,
                              **{key: run[key] for key in ["run_name", "variant", "category", "label",
                                                           "replicate", "seed"]},
                              "code_version": args.code_version,
                              "config_file_path": config_file_path,
                              "run_folder_path": run_folder_path,
                              "eval_folder_path": artifact_folder_path / "eval" / run["variant"] / run_id})

    # The job array splits the rows at tabs (consecutive tabs would merge): every field must be non-empty
    for row in manifest_rows:
        for column in MANIFEST_COLUMNS:
            value = str(row[column])
            if not value.strip() or any(character in value for character in "\t\n\r"):
                raise ValueError(f"Manifest row {row['index']} ({row['run_id']}): {column} {value!r} is empty or "
                                 f"contains a tab or line break.")
    manifest_file_path = artifact_folder_path / "manifest.tsv"
    with open(manifest_file_path, "w") as file:
        file.write("\t".join(MANIFEST_COLUMNS) + "\n")
        for row in manifest_rows:
            file.write("\t".join(str(row[column]) for column in MANIFEST_COLUMNS) + "\n")
    dataset_env_file_path = artifact_folder_path / "dataset.env"
    write_dataset_env(args, dataset_env_file_path)
    print(f"Wrote {len(manifest_rows)} run configs to {config_folder_path}, the manifest to {manifest_file_path} and "
          f"the dataset values to {dataset_env_file_path} (code version: {args.code_version}).")
    label_files = ("the silver files" if args.label_file_paths == args.silver_file_paths else
                   f"{len(args.label_file_paths)} files that are not the silver files")
    print(f"Evaluation labels: {args.label_keys} (unassigned label {args.unassigned_label!r}) from {label_files}: "
          f"{args.label_file_paths}.")
    n_appended = sum(run["appended"] for run in runs)
    if n_appended:
        print(f"Rows {len(manifest_rows) - n_appended + 1}-{len(manifest_rows)}: the {n_appended} appended replicates "
              f"(appended_replicates of the plan).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--plan_file_path',
                        type=str,
                        required=True,
                        help='Ablation plan.')
    parser.add_argument('--base_config_file_path',
                        type=str,
                        required=True,
                        help='Base model config (the mouse base config).')
    parser.add_argument('--artifact_folder_path',
                        type=str,
                        required=True,
                        help='Folder for configs, manifest, dataset.env, runs, evaluations and logs.')
    parser.add_argument('--data_directory_path',
                        type=str,
                        required=True,
                        help='Data directory with silver and gold subdirectories.')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help='Name of the dataset (silver folder and tokenized dataset names).')
    parser.add_argument('--silver_file_paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Silver AnnData files of the dataset (in the silver folder).')
    parser.add_argument('--label_file_paths',
                        type=str,
                        nargs='+',
                        help='AnnData files with obs["cell_id"] and the label keys, one per section, from which the '
                             'evaluation reads the labels (LABEL_FILE_PATHS; default: the silver files).')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        required=True,
                        help='Token dictionary of the species.')
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        required=True,
                        help='Per-gene normalization factors of the dataset (packaged with every run).')
    parser.add_argument('--label_keys',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Keys of the niche labels in the obs of the label files (evaluation).')
    parser.add_argument('--unassigned_label',
                        type=str,
                        required=True,
                        help='Label of cells outside annotated regions; excluded from the metrics.')
    parser.add_argument('--experiment_tag',
                        type=str,
                        required=True,
                        help='Short tag of the campaign; wandb run ids are <experiment_tag>_<manifest index>_<config '
                             'hash> and must be new.')
    parser.add_argument('--epochs',
                        type=int,
                        required=True,
                        help='Training epochs (OVCA: 82, i.e. 216,972 steps, matching the 300 mouse epochs in steps).')
    parser.add_argument('--warmup',
                        type=int,
                        required=True,
                        help='Warmup epochs (OVCA: 1, i.e. 2,646 steps; mouse: 3, i.e. 2,166 steps).')
    parser.add_argument('--n_special_values',
                        type=int,
                        required=True,
                        help='Rows of the special value embedding; must cover the batch special values.')
    parser.add_argument('--code_path',
                        type=str,
                        required=True,
                        help='Pinned code (setup_code.sh); its CODE_VERSION is written into every config.')
    args = parser.parse_args()

    main(args)
