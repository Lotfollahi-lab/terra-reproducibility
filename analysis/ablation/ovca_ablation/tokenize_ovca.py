"""
This script creates one tokenized dataset of the OVCA ablation plan from the OVCA silver data like
`stjepa-reproducibility/analysis/tokenization/create_tokenized_dataset.py` created the mouse ablation datasets (same
tokenizer arguments and output naming), but with the pinned ST-JEPA v2 tokenizer (`stjepa.tokenizers`), sequential
processing of the single silver file (same output, lower peak memory), the opt-in tokenizer fixes set in the plan
(`tokenizer_options`: deterministic top-k ties, detected genes only, the real gene token count for the sampler, the
Seurat v3 LOESS span, own-neighbor aggregation and an explicit index-cell/neighborhood split for the cell-neighborhood tokenizer; their names
are appended to the dataset name) and without wandb logging. The dataset is written under a temporary name and renamed when
complete, so that interrupted jobs are never mistaken for finished datasets.

Usage:
```bash
python tokenize_ovca.py --list                                  # index, name, memory class and path of every dataset
python tokenize_ovca.py --list_indices standard                 # indices of the datasets of one memory class
python tokenize_ovca.py --tokenization_index ${LSB_JOBINDEX}    # create one dataset (job array element)
```
"""

import argparse
import os
import shutil
from pathlib import Path

from ablation_utils import (DATA_DIRECTORY_PATH,
                            DATASET_NAME,
                            NORM_FACTOR_FILE_PATH,
                            NORM_METHOD_KEYS,
                            NUM_SHARDS,
                            PLAN_FILE_PATH,
                            TOKEN_DICTIONARY_FILE_PATH,
                            get_tokenized_dataset_path,
                            load_plan,
                            needs_large_memory)


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

    name = tokenization_names[args.tokenization_index - 1]
    tokenization = tokenizations[name]
    output_path = get_tokenized_dataset_path(tokenization, args.data_directory_path, args.dataset_name)
    if output_path.exists():
        print(f"{output_path} already exists. Skipping tokenization...")
        return
    print(f"Tokenization {args.tokenization_index} ({name}): {tokenization}.")

    from stjepa.tokenizers.cell_tokenizers import CellGraphTokenizer, CellNeighborhoodTokenizer

    Tokenizer = {"cell-graph-tokenizer": CellGraphTokenizer,
                 "cell-neighborhood-tokenizer": CellNeighborhoodTokenizer}[tokenization["tokenizer"]]
    tokenizer_kwargs = {"stable_rank_ties": tokenization["stable_rank_ties"],
                        "zero_expr_from_raw_counts": tokenization["zero_expr_from_raw_counts"],
                        "exact_n_nonzero_tokens": tokenization["exact_n_nonzero_tokens"],
                        "seurat_loess_span": tokenization["seurat_loess_span"]}
    if tokenization["tokenizer"] == "cell-neighborhood-tokenizer":
        tokenizer_kwargs["seq_len_cell"] = tokenization["seq_len_cell"]
        tokenizer_kwargs["aggregate_own_neighbors"] = tokenization["aggregate_own_neighbors"]
    # Sequential processing: the silver data is one file, and parallel mode pickles the whole tokenized file back from a
    # worker process, which doubles the peak memory of the dense-path tokenizations (~1.4 TB for OVCA)
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
    tk.tokenize_data(input_directory=Path(args.data_directory_path) / "silver" / args.dataset_name,
                     output_directory=output_path.parent,
                     output_file_prefix=incomplete_file_prefix,
                     use_generator=True,
                     cache_directory_path=cache_directory_path,
                     keep_in_memory=False,
                     num_shards=NUM_SHARDS)
    os.rename(incomplete_output_path, output_path)
    shutil.rmtree(cache_directory_path, ignore_errors=True)
    print(f"Saved tokenized dataset to {output_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--tokenization_index',
                        type=int,
                        default=None,
                        help='1-based index of the tokenization in the ablation plan (job array index).')
    parser.add_argument('--list',
                        action='store_true',
                        help='List all tokenizations and exit.')
    parser.add_argument('--list_indices',
                        type=str,
                        choices=['standard', 'large'],
                        default=None,
                        help='Print the comma-separated indices of one memory class and exit.')
    parser.add_argument('--plan_file_path',
                        type=str,
                        default=PLAN_FILE_PATH,
                        help='Ablation plan.')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default=DATA_DIRECTORY_PATH,
                        help='Data directory with silver, gold and cache subdirectories.')
    parser.add_argument('--dataset_name',
                        type=str,
                        default=DATASET_NAME,
                        help='Name of the silver dataset folder.')
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        default=NORM_FACTOR_FILE_PATH,
                        help='Per-gene normalization factors of the OVCA section.')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        default=TOKEN_DICTIONARY_FILE_PATH,
                        help='Human token dictionary.')
    args = parser.parse_args()

    main(args)
