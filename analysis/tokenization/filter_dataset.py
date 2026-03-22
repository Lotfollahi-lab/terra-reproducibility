import argparse
import os
import tracemalloc
from pathlib import Path

import datasets
import pandas as pd


def main(args):
    if args.concatenate_datasets:
        print("Concatenating dataset...")
        ds_list = []
        for ds_path in args.dataset_load_paths:
            ds = datasets.load_from_disk(ds_path)

            if args.drop_columns:
                print(f"Droping unnnecessary columns for {ds_path}...")
                drop_cols = [
                    'seg_tokens',
                    'batch_token',
                    'gene_panel_token',
                    'assay_token',
                    'species_token',
                    'tissue_token']
                for col in drop_cols:
                    if col in ds.column_names:
                        ds = ds.remove_columns(col)

            ds_list.append(ds)
        concatenated_dataset = datasets.concatenate_datasets(ds_list)
    else:
        print("Loading dataset...")
        concatenated_dataset = load_from_disk(args.dataset_load_paths[0])
    print(concatenated_dataset)

    if args.deduplicate_coords:
        print("Removing duplicate coordinates...")
        def keep_every_256(batch, n_segments=11):
            return {
                "rel_x_coord": [
                    x[::256] if len(x) > n_segments else x
                    for x in batch["rel_x_coord"]
                ],
                "rel_y_coord": [
                    y[::256] if len(y) > n_segments else y
                    for y in batch["rel_y_coord"]
                ],
            }

        concatenated_dataset = concatenated_dataset.map(
            keep_every_256,
            batched=True,
            batch_size=10000,
            num_proc=args.num_workers,
            desc="Removing duplicate coordinates"
        )

    if args.fix_coord_order:
        print("Reordering coordinates...")
        def _reorder_coord_pair(xs, ys):
            # Stable sort by squared radius without NumPy overhead
            order = sorted(range(len(xs)), key=lambda i: xs[i]*xs[i] + ys[i]*ys[i])
            return [xs[i] for i in order], [ys[i] for i in order]

        def reorder_coords(batch):
            rx, ry = batch["rel_x_coord"], batch["rel_y_coord"]
            new_x, new_y = [], []
            append_x, append_y = new_x.append, new_y.append
            for xs, ys in zip(rx, ry):
                xs_sorted, ys_sorted = _reorder_coord_pair(xs, ys)
                append_x(xs_sorted); append_y(ys_sorted)
            batch["rel_x_coord"] = new_x
            batch["rel_y_coord"] = new_y
            return batch

        concatenated_dataset = concatenated_dataset.map(
            reorder_coords,
            batched=True,
            batch_size=10000,
            num_proc=args.num_workers,
            desc="Reordering rel_x_coord / rel_y_coord by radius"
        )

    if args.replace_batch_value:
        print("Replacing wrong batch value...")

        def replace_batch_values(batch):
            batch["batch_value"] = [
                [872] if val[0] == 450 else val
                for val in batch["batch_value"]
            ]
            return batch

        concatenated_dataset = concatenated_dataset.map(
            replace_batch_values,
            batched=True,
            batch_size=10000,
            num_proc=args.num_workers,
            desc="Updating batch_value"
        )

    if args.filtering_dataset:
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
        args.dataset_save_path
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
    parser.add_argument('--concatenate_datasets',
                        type=bool,
                        default=False,
                        help='Whether to concatenate the corrected skin datasets.')
    parser.add_argument('--drop_columns',
                        type=bool,
                        default=False,
                        help='Whether to drop unnecessary columns.')
    parser.add_argument('--deduplicate_coords',
                        type=bool,
                        default=False,
                        help='Whether to deduplicate coordinates.')
    parser.add_argument('--fix_coord_order',
                        type=bool,
                        default=False,
                        help='Whether to fix coordinate order.')
    parser.add_argument('--replace_batch_value',
                        type=bool,
                        default=False,
                        help='Whether to replace the wrong batch value.')
    parser.add_argument('--filtering_dataset',
                        type=bool,
                        default=False,
                        help='Whether to filter out some dataset based on batch_id.')
    parser.add_argument('--filtered_batch_id',
                        type=str,
                        default='1010_batch26',
                        help='Batch ID to be filtered.')
    parser.add_argument(
        '--dataset_load_paths',
        type=str,
        nargs='+',
        default=[
            '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_110m/hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_10.dataset',
            '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/xhs1022-1-55b_1p/xhs1022-1-55b_1p_32_None_None_None_None_None_shifted_log_knn_10_2816.dataset',
            '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/xhs1022-2-55b_1p/xhs1022-2-55b_1p_32_None_None_None_None_None_shifted_log_knn_10_2816.dataset',
            '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/xhs1022-3-55b_1p/xhs1022-3-55b_1p_32_None_None_None_None_None_shifted_log_knn_10_2816.dataset'
        ],
        help='List of dataset paths to load.'
    )
    parser.add_argument('--dataset_save_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_110m/hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_10_new.dataset',
                        help='Dataset save path.')

    args = parser.parse_args()
    
    main(args)
