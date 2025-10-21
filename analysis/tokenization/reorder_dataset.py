import argparse
import pickle

import numpy as np
from datasets import load_from_disk


def main(args):
    print("Start script.")

    # Load Huggingface dataset
    ds = load_from_disk(args.dataset_path)

    # Retrieve training indices for filtering
    # (required for full data as reordering idx was computed on training set;
    # sample datasets contain only training set anyhow)
    if args.train_idx_path == "None":
            args.train_idx_path = None

    if args.train_idx_path:
        with open(args.train_idx_path, "rb") as f: 
            train_indices = pickle.load(f)
        
        ds = ds.select(train_indices)        

    # Retrieve indices for reordering
    with open(args.order_idx_path, "rb") as f: 
        order_indices = pickle.load(f)

    ds = ds.select(order_indices)

    print("Start flattening.")

    # Materialize the new physical order
    ds = ds.flatten_indices(
        num_proc=32,
        writer_batch_size=1000)

    print("Flattening done.")

    new_path = args.dataset_path.replace('.dataset', '_ordered.dataset')
    ds.save_to_disk(new_path, num_shards=4000)
    print("Saved to:", new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_90m/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10.dataset',
                        help='Path where the Huggingface dataset will be loaded from.')
    parser.add_argument('--train_idx_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10_precomputed_split_train.pkl',
                        help='Path where the train index will be loaded from.')
    parser.add_argument('--order_idx_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10_precomputed_loader_idx.pkl',
                        help='Path where the order index will be loaded from.')
    args = parser.parse_args()

    main(args)

