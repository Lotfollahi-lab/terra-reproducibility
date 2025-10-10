import argparse
import pickle

import numpy as np
from datasets import load_from_disk


def main(args):
    print("Start script.")

    # Load Huggingface dataset
    ds = load_from_disk(args.dataset_path)

    # Retrieve indices for reordering
    with open(args.idx_path, "rb") as f: 
        indices = pickle.load(f)

    ds_all_indices = list(range(len(ds)))

    # Add test and val indices at end of training indices
    seen = set(indices)
    to_add = []
    for idx in ds_all_indices:
        if idx not in seen:
            to_add.append(idx)
            seen.add(idx)

    indices.extend(to_add)

    ds_ordered = ds.select(indices,
                           indices_cache_file_name=args.cache_path)

    print("Start flattening.")

    # Materialize the new physical order
    ds_ordered = ds_ordered.flatten_indices(num_proc=128, writer_batch_size=1000)

    print("Flattening done.")

    new_path = args.dataset_path.replace('.dataset', '_ordered.dataset')
    ds_ordered.save_to_disk(new_path, num_shards=4000)
    print("Saved to:", new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_90m/hst_corpus_90m_0.01_sample_32_None_None_None_None_None_shifted_log_knn_10.dataset',
                        help='Path where the Huggingface dataset will be loaded from.')
    parser.add_argument('--idx_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/hst_corpus_90m_0.01_sample_32_None_None_None_None_None_shifted_log_knn_10_precomputed_loader_idx.pkl',
                        help='Path where the index will be loaded from.')
    parser.add_argument('--cache_path',
                        type=str,
                        default='"/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/hst_corpus_90m_0.01_sample_32_None_None_None_None_None_shifted_log_knn_10_loader_reidx.arrow"',
                        help='Path where the cache will be saved.')
    args = parser.parse_args()

    main(args)

