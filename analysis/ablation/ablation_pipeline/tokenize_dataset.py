"""
This script creates one tokenized dataset of the ablation plan from the silver data of a dataset, like
`stjepa-reproducibility/analysis/tokenization/create_tokenized_dataset.py` created the mouse ablation datasets (same
tokenizer arguments and output naming), but with the pinned TERRA tokenizer (`terra.tokenizers`), sequential
processing of the silver files (same output, lower peak memory), the opt-in tokenizer fixes set in the plan
(`tokenizer_options`: deterministic top-k ties ordered by corpus mean, detected genes only, the Seurat v3 LOESS span,
own-neighbor aggregation and the equal index-cell/neighborhood split of the cell-neighborhood tokenizer; their names are
appended to the dataset name; TERRA counts the real gene tokens for the length-grouped sampler without an option) and
without wandb logging. The tokenizer reads every h5ad file
below `<data_directory_path>/silver/<dataset_name>`; they must be exactly `--silver_file_paths`, so that the
evaluation's labels cover the tokenized cells. The dataset is written under a temporary name and renamed when
complete, so that interrupted jobs are never mistaken for finished datasets. It records its inputs (TERRA code version,
tokenization settings, md5 of the token dictionary, the factor file and the silver files) in `ablation_provenance.json`;
an existing dataset is skipped only if it was made from the same inputs, otherwise the job stops (the dataset name does
not encode the code or the input files, so a changed pin or recomputed factors would otherwise reuse stale datasets).

Usage (`submit_tokenization.sh` runs the first two; `tokenize_dataset.lsf` the last):
```bash
python tokenize_dataset.py --list [DATASET ARGS]                                  # index, name, memory class, path
python tokenize_dataset.py --list_indices standard [DATASET ARGS]                 # indices of one memory class
python tokenize_dataset.py --tokenization_index ${LSB_JOBINDEX} [DATASET ARGS] \
    --silver_file_paths ... --norm_factor_file_path ... --token_dictionary_file_path ...   # create one dataset
```
DATASET ARGS: --plan_file_path ... --data_directory_path ... --dataset_name ...
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from ablation_utils import (NORM_METHOD_KEYS,
                            NUM_SHARDS,
                            get_code_version,
                            get_tokenization_provenance,
                            get_tokenized_dataset_path,
                            load_plan,
                            needs_large_memory,
                            provenance_differences,
                            write_provenance)


def check_silver_files(silver_folder_path: Path, silver_file_paths: list[str]):
    """The tokenizer's input files (all h5ad files below the silver folder) must be exactly the given silver files."""
    missing = [file_path for file_path in silver_file_paths if not Path(file_path).is_file()]
    if missing:
        raise FileNotFoundError(f"Silver files {missing} do not exist: the data preparation must finish first.")
    tokenizer_inputs = {os.path.abspath(file_path) for file_path in silver_folder_path.glob("**/*.h5ad")}
    given = {os.path.abspath(file_path) for file_path in silver_file_paths}
    if tokenizer_inputs != given:
        raise ValueError(f"The tokenizer would read {sorted(tokenizer_inputs)} from {silver_folder_path}, but the "
                         f"silver files are {sorted(given)}: move other files out of the folder or list them.")


def main(args):
    tokenizations, _ = load_plan(args.plan_file_path)
    tokenization_names = list(tokenizations)

    if args.list or args.list_indices:
        for idx, name in enumerate(tokenization_names, start=1):
            memory_class = "large" if needs_large_memory(tokenizations[name]) else "standard"
            if args.list:
                print(f"{idx}\t{name}\t{memory_class}\t"
                      f"{get_tokenized_dataset_path(tokenizations[name], args.data_directory_path, args.dataset_name)}")
        if args.list_indices:
            print(",".join(str(idx) for idx, name in enumerate(tokenization_names, start=1)
                           if ("large" if needs_large_memory(tokenizations[name]) else "standard") == args.list_indices))
        return

    if not 1 <= args.tokenization_index <= len(tokenization_names):
        raise ValueError(f"--tokenization_index {args.tokenization_index} is not in 1-{len(tokenization_names)}.")
    name = tokenization_names[args.tokenization_index - 1]
    tokenization = tokenizations[name]
    output_path = get_tokenized_dataset_path(tokenization, args.data_directory_path, args.dataset_name)
    silver_folder_path = Path(args.data_directory_path) / "silver" / args.dataset_name
    check_silver_files(silver_folder_path, args.silver_file_paths)
    # The pinned TERRA copy (first on PYTHONPATH in tokenize_dataset.lsf)
    provenance = get_tokenization_provenance(tokenization, get_code_version(), args.silver_file_paths,
                                             args.norm_factor_file_path, args.token_dictionary_file_path)
    if output_path.exists():
        differences = provenance_differences(output_path, provenance)
        if differences:
            raise ValueError(f"{output_path} exists, but was not made from the current inputs (differs in: "
                             f"{', '.join(differences)}). Move or delete it and resubmit.")
        print(f"{output_path} already exists, made from the same inputs. Skipping tokenization...")
        return
    print(f"Tokenization {args.tokenization_index} ({name}): {tokenization}; code {provenance['code_version']}.")

    from terra.tokenizers.cell_tokenizers import CellGraphTokenizer, CellNeighborhoodTokenizer

    Tokenizer = {"cell-graph-tokenizer": CellGraphTokenizer,
                 "cell-neighborhood-tokenizer": CellNeighborhoodTokenizer}[tokenization["tokenizer"]]
    tokenizer_kwargs = {"stable_rank_ties": tokenization["stable_rank_ties"],
                        "zero_expr_from_raw_counts": tokenization["zero_expr_from_raw_counts"]}
    if tokenization["rank_tie_break_norm_factor"] is not None:
        tokenizer_kwargs["rank_tie_break_norm_factor"] = tokenization["rank_tie_break_norm_factor"]
    if tokenization["seurat_loess_span"] is not None:
        tokenizer_kwargs["seurat_loess_span"] = tokenization["seurat_loess_span"]
    if tokenization["tokenizer"] == "cell-neighborhood-tokenizer":
        # seq_len_cell is model_input_size / 2 (equal split) or None (n_neighs + 1 blocks; checked by load_plan)
        tokenizer_kwargs["split_cell_neigh_equally"] = tokenization["seq_len_cell"] is not None
        tokenizer_kwargs["aggregate_own_neighbors"] = tokenization["aggregate_own_neighbors"]
    # Sequential processing (one silver file at a time): parallel mode pickles every tokenized file back from a worker
    # process, which doubles the peak memory (without the ranked-list cut: ~1.4 TB for the dense-path OVCA tokenizations)
    tk = Tokenizer(
        nproc=int(os.environ.get("LSB_DJOB_NUMPROC", 1)),
        processing_mode="sequential",
        model_input_size=tokenization["model_input_size"],
        include_zero_expr_genes=False,
        n_neighs=tokenization["n_neighs"],
        radius=None,
        delaunay=False,
        **{key: tokenization[key] for key in NORM_METHOD_KEYS},
        norm_factor_file_path=args.norm_factor_file_path,
        token_dictionary_file_path=args.token_dictionary_file_path,
        **tokenizer_kwargs)

    # Write under a temporary name, then rename
    incomplete_file_prefix = f"{output_path.stem}_incomplete"
    incomplete_output_path = output_path.with_name(f"{incomplete_file_prefix}.dataset")
    shutil.rmtree(incomplete_output_path, ignore_errors=True)
    cache_directory_path = Path(args.data_directory_path) / "cache" / args.dataset_name / name
    tk.tokenize_data(input_directory=silver_folder_path,
                     output_directory=output_path.parent,
                     output_file_prefix=incomplete_file_prefix,
                     use_generator=True,
                     cache_directory_path=cache_directory_path,
                     keep_in_memory=False,
                     num_shards=NUM_SHARDS)
    write_provenance(incomplete_output_path, provenance)
    os.rename(incomplete_output_path, output_path)
    shutil.rmtree(cache_directory_path, ignore_errors=True)
    print(f"Saved tokenized dataset to {output_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--tokenization_index',
                      type=int,
                      help='1-based index of the tokenization in the ablation plan (job array index).')
    mode.add_argument('--list',
                      action='store_true',
                      help='List all tokenizations and exit.')
    mode.add_argument('--list_indices',
                      type=str,
                      choices=['standard', 'large'],
                      help='Print the comma-separated indices of one memory class and exit.')
    parser.add_argument('--plan_file_path',
                        type=str,
                        required=True,
                        help='Ablation plan.')
    parser.add_argument('--data_directory_path',
                        type=str,
                        required=True,
                        help='Data directory with silver, gold and cache subdirectories.')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help='Name of the silver dataset folder.')
    parser.add_argument('--silver_file_paths',
                        type=str,
                        nargs='+',
                        help='Silver AnnData files of the dataset: all h5ad files of the silver folder (tokenization).')
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        help='Per-gene normalization factors of the dataset (tokenization).')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        help='Token dictionary of the species (tokenization).')
    args = parser.parse_args()
    if args.tokenization_index is not None:
        for name in ["silver_file_paths", "norm_factor_file_path", "token_dictionary_file_path"]:
            if getattr(args, name) is None:
                parser.error(f"--{name} is required with --tokenization_index.")

    main(args)
