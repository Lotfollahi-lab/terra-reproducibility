"""
This script generates the training configs and the run manifest of the OVCA ablations from `ablation_plan.yaml` and the
mouse base config (`base_config.yaml`). Every run config is the base config with
- the OVCA data (silver path, human token dictionary, the variant's tokenized dataset and its tokenization keys, which
  the evaluation reads to re-tokenize the section exactly as for training),
- `n_special_values` large enough for the human batch special value (the mouse value 200 is too small),
- a fixed run folder `runs/<run name>_<config hash>` that is resumed from `stjepa-latest.pth.tar` (on the first
  launch the file is missing and training starts from scratch), and only the latest checkpoint kept; a changed config
  gets a new folder instead of resuming or reusing an earlier one,
- `seed: 1000 * (replicate - 1)` (stjepa_ovca_ablation.patch; without it all replicates use seed 0 like the mouse runs),
- the variant overrides (dotted keys that must exist in the base config).
The manifest lists the runs in launch order for the job array.

Usage:
```bash
python generate_ablation_runs.py [--epochs 300 --warmup 3]
```
"""

import argparse
import copy
import hashlib
import os
from pathlib import Path

import yaml

from ablation_utils import (ARTIFACT_FOLDER_PATH,
                            DATA_DIRECTORY_PATH,
                            DATASET_NAME,
                            NORM_METHOD_KEYS,
                            PLAN_FILE_PATH,
                            SILVER_FILE_PATH,
                            TOKEN_DICTIONARY_FILE_PATH,
                            get_runs,
                            get_tokenized_dataset_path,
                            load_plan)

MANIFEST_COLUMNS = ["index", "run_id", "run_name", "variant", "category", "label", "replicate", "seed", "experiment_name",
                    "config_file_path", "run_folder_path", "eval_folder_path"]
TOKENIZER_TYPES = {"cell_graph": "cell-graph-tokenizer", "cell_neigh": "cell-neighborhood-tokenizer"}


def set_key(config: dict, dotted_key: str, value):
    """Set a dotted key that must already exist in the config (catches typos in the plan)."""
    *parents, key = dotted_key.split(".")
    section = config
    for parent in parents:
        section = section[parent]
    if key not in section:
        raise KeyError(f"Override {dotted_key} does not exist in the base config.")
    section[key] = value


def validate_config(config: dict, tokenization: dict, run_name: str):
    """Check that sequence lengths match the token layout of the tokenized dataset and the model dimensions."""
    data, meta = config["data"], config["meta"]
    if TOKENIZER_TYPES[data["tokenizer_type"]] != tokenization["tokenizer"]:
        raise ValueError(f"{run_name}: tokenizer_type {data['tokenizer_type']} does not match {tokenization}.")
    # Cell graph: n_neighs + 1 blocks of seq_len_cell tokens; cell neighborhood (tokenize_ovca.py): equal split. The
    # datasets expect exactly seq_len_cell + seq_len_neighborhood = model_input_size tokens.
    model_input_size = tokenization["model_input_size"]
    n_blocks = tokenization["n_neighs"] + 1 if data["tokenizer_type"] == "cell_graph" else 2
    seq_len_cell = model_input_size // n_blocks
    if (data["seq_len_cell"], data["seq_len_neighborhood"]) != (seq_len_cell, model_input_size - seq_len_cell):
        raise ValueError(f"{run_name}: seq_len_cell / seq_len_neighborhood {data['seq_len_cell']} / "
                         f"{data['seq_len_neighborhood']} do not match the tokenized data "
                         f"({seq_len_cell} / {model_input_size - seq_len_cell}).")
    for key in ["enc_emb_dim", "pred_emb_dim"]:
        if meta[key] % meta["num_heads"]:
            raise ValueError(f"{run_name}: {key} {meta[key]} is not divisible by num_heads {meta['num_heads']}.")


def make_run_config(base_config: dict,
                    tokenization: dict,
                    run: dict,
                    run_folder_path: Path,
                    args) -> dict:
    config = copy.deepcopy(base_config)

    # OVCA data and the tokenization of this variant
    data = config["data"]
    data["dataset_name"] = args.dataset_name
    data["raw_data_folder_path"] = str(Path(args.silver_file_path).parent)
    data["token_dict_folder_path"] = args.token_dictionary_file_path
    data["tokenized_data_folder_path"] = str(get_tokenized_dataset_path(tokenization,
                                                                        args.data_directory_path,
                                                                        args.dataset_name))
    data["model_input_size"] = tokenization["model_input_size"]
    data["n_neighs"] = tokenization["n_neighs"]
    for key in NORM_METHOD_KEYS:
        data[key] = tokenization[key]
    data["n_special_values"] = args.n_special_values

    # Schedule, run folder with resumption, seed
    config["optimization"]["epochs"] = args.epochs
    config["optimization"]["warmup"] = args.warmup
    config["state"]["folder_path"] = str(run_folder_path)
    config["state"]["load_checkpoint"] = True
    config["state"]["read_checkpoint"] = "stjepa-latest.pth.tar"
    config["state"]["checkpoint_freq"] = args.epochs + 1  # never write epoch snapshots, only stjepa-latest
    config["seed"] = run["seed"]

    for key, value in run["overrides"].items():
        set_key(config, key, value)
    validate_config(config, tokenization, run["run_name"])
    return config


def main(args):
    tokenizations, variants = load_plan(args.plan_file_path)
    with open(args.base_config_file_path) as file:
        base_config = yaml.safe_load(file)

    artifact_folder_path = Path(args.artifact_folder_path)
    config_folder_path = artifact_folder_path / "configs"
    os.makedirs(config_folder_path, exist_ok=True)

    manifest_rows = []
    for index, run in enumerate(get_runs(variants), start=1):
        # Run ids carry a hash of the config, so that a changed config (e.g. other epochs) never resumes or reuses
        # the checkpoints and metrics of an earlier config; the hash excludes the run folder, which contains it
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
                              "experiment_name": args.experiment_name,
                              "config_file_path": config_file_path,
                              "run_folder_path": run_folder_path,
                              "eval_folder_path": artifact_folder_path / "eval" / run["variant"] / run_id})

    manifest_file_path = artifact_folder_path / "manifest.tsv"
    with open(manifest_file_path, "w") as file:
        file.write("\t".join(MANIFEST_COLUMNS) + "\n")
        for row in manifest_rows:
            file.write("\t".join(str(row[column]) for column in MANIFEST_COLUMNS) + "\n")
    print(f"Wrote {len(manifest_rows)} run configs to {config_folder_path} and the manifest to {manifest_file_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--epochs',
                        type=int,
                        default=300,
                        help='Training epochs (mouse: 300; OVCA has 3.66x the iterations per epoch).')
    parser.add_argument('--warmup',
                        type=int,
                        default=3,
                        help='Warmup epochs (mouse: 3).')
    parser.add_argument('--n_special_values',
                        type=int,
                        default=1448,
                        help='Rows of the special value embedding; must cover the human batch special value.')
    parser.add_argument('--experiment_name',
                        type=str,
                        default="stjepa_v2_ovca_ablation",
                        help='wandb experiment name; wandb run ids are <experiment_name>_<run_name> and must be new.')
    parser.add_argument('--plan_file_path',
                        type=str,
                        default=PLAN_FILE_PATH,
                        help='Ablation plan.')
    parser.add_argument('--base_config_file_path',
                        type=str,
                        default=Path(__file__).resolve().parent / "base_config.yaml",
                        help='Mouse base model config.')
    parser.add_argument('--artifact_folder_path',
                        type=str,
                        default=ARTIFACT_FOLDER_PATH,
                        help='Folder for configs, manifest, runs, evaluations and logs.')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default=DATA_DIRECTORY_PATH,
                        help='Data directory with silver and gold subdirectories.')
    parser.add_argument('--dataset_name',
                        type=str,
                        default=DATASET_NAME,
                        help='Name of the OVCA dataset.')
    parser.add_argument('--silver_file_path',
                        type=str,
                        default=SILVER_FILE_PATH,
                        help='OVCA silver AnnData.')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        default=TOKEN_DICTIONARY_FILE_PATH,
                        help='Human token dictionary.')
    args = parser.parse_args()

    main(args)
