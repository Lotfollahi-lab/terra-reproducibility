import argparse
import os
import tracemalloc
from pathlib import Path

import datasets
import pandas as pd


def main(args):
    ds_list = []
    ds_path_list = [
        '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_110m/hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_10.dataset',
        '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/xhs1022-1-55b_1p/xhs1022-1-55b_1p_32_None_None_None_None_None_shifted_log_knn_10_2816.dataset',
        '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/xhs1022-2-55b_1p/xhs1022-2-55b_1p_32_None_None_None_None_None_shifted_log_knn_10_2816.dataset',
        '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/xhs1022-3-55b_1p/xhs1022-3-55b_1p_32_None_None_None_None_None_shifted_log_knn_10_2816.dataset'
    ]

    for ds_path in ds_path_list:
        ds = datasets.load_from_disk(ds_path)
        ds_list.append(ds)

    print("Concatenating dataset...")
    concatenated_dataset = datasets.concatenate_datasets(ds_list)
    print(concatenated_dataset)

    def replace_batch_values(batch):
        batch["batch_value"] = [
            [872] if val[0] == 450 else val
            for val in batch["batch_value"]
        ]
        return batch

    print("Replacing wrong batch value...")
    concatenated_dataset = concatenated_dataset.map(
        replace_batch_values,
        batched=True,
        batch_size=10000,
        num_proc=args.num_workers,
        desc="Updating batch_value"
    )

    print("Filtering dataset...")
    filtered_dataset = concatenated_dataset.filter(
        lambda ids: [i.split('_')[0] + '_' + i.split('_')[1] != args.filtered_batch_id for i in ids],
        input_columns="cell_id",
        batched=True,
        batch_size=10000,
        num_proc=args.num_workers,
        desc="Filtering by dataset_id"
    )

    print("Flattening indices...")
    # Materialize the new physical order
    filtered_dataset = filtered_dataset.flatten_indices(
        num_proc=args.num_workers,
        writer_batch_size=1000)

    print("Writing to disk...")
    filtered_dataset.save_to_disk(
        '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_110m/hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_10_new.dataset'
        #num_shards=args.num_shards
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS', # '/nfs/team361/sb75/DATASETS'
                        help='Data directory path.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=64,
                        help='Number of workers.')
    parser.add_argument('--filtered_batch_id',
                        type=str,
                        default='1010_batch26',
                        help='Batch ID to be filtered.')
    args = parser.parse_args()
    
    main(args)
