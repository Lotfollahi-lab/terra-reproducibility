import argparse
import pickle
import time
import timeit
import random

import torch
from datasets import load_from_disk

from nichejepa.masks.block_masking  import BlockMaskCollator
from nichejepa.datasets.cell_datasets import init_cell_dataset
from nichejepa.datasets.dataloaders import init_dataloader_and_sampler, CustomDistributedLengthGroupedSampler


def main(args):
    # Load Huggingface dataset
    dataset = load_from_disk(args.dataset_path)

    # Retrieve training indices for filtering
    # (required for full data as reordering idx is computed on training set;
    # sample datasets contain only training set anyhow)
    if args.train_idx_path == "None":
        args.train_idx_path = None

    if args.train_idx_path:
        with open(args.train_idx_path, "rb") as f: 
            train_indices = pickle.load(f)
        
        dataset = dataset.select(train_indices)

    print(dataset)

    # Load precomputed nonzero tokens
    if args.nz_tokens_path == "None":
        args.nz_tokens_path = None

    if args.nz_tokens_path:
        with open(args.nz_tokens_path, "rb") as f: 
            n_nonzero_tokens = pickle.load(f)
    else:
        n_nonzero_tokens = None

    # Set params
    token_dict_folder_path = '/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/token_dictionary_homo_sapiens.pkl'

    with open(token_dict_folder_path, 'rb') as file:
        token_dict = pickle.load(file)
    vocab_size = len(token_dict)

    special_tokens = []
    sampling_strategy = None
    sep_gene_tokens_neb = False

    ### Initialize dataloader and sampler ###
    # Initialize cell dataset
    cell_dataset = init_cell_dataset(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=args.seq_len_cell,
        seq_len_neighborhood=args.seq_len_neighborhood,
        tokenizer_type='cell_graph',
        gt_type=args.gt_type,
        cell_pos_enc=args.cell_pos_enc,
        special_tokens=special_tokens,
        sampling_strategy=sampling_strategy,
        n_nonzero_tokens_list=n_nonzero_tokens,
        include_cell_id=False,
        sep_gene_tokens_neb=sep_gene_tokens_neb)

    # Initialize mask collator
    mask_collator = BlockMaskCollator(
        n_targets=args.n_targets,
        n_contexts=args.n_contexts,
        n_segments=args.n_segments,
        seq_len_cell=args.seq_len_cell,
        seq_len_neighborhood=args.seq_len_neighborhood,
        n_special_tokens=args.n_special_tokens,
        per_block_mask_ratio=args.per_block_mask_ratio,
        sample_segments=True,
        sample_gene_masks=True)

    # Initialize loader and sampler
    distributed = True # True, False
    world_size = 1
    rank = 0
    pin_memory = True
    num_workers = 0

    if not distributed:
        loader, sampler = init_dataloader_and_sampler(
            cell_dataset=cell_dataset,
            batch_size=args.batch_size,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            collate_fn=mask_collator,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=True,
            #prefetch_factor=4,
            persistent_workers=False)
    else:
        loader, sampler = init_dataloader_and_sampler(
            cell_dataset=cell_dataset,
            batch_size=args.batch_size,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            collate_fn=mask_collator,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=True,
            #prefetch_factor=4,
            persistent_workers=False)
    ###############################

    ### Sample length grouped indices ###
    g = torch.Generator()
    g.manual_seed(0)
    indices = sampler._get_length_grouped_indices(generator=g)

    with open(args.idx_save_path, "wb") as f: 
        pickle.dump(indices, f)
    #####################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_90m/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10.dataset',
                        help='Path where the Huggingface dataset will be loaded from.')
    parser.add_argument('--nz_tokens_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10_n_nonzero_tokens_train.pkl',
                        help='Path where the precomputed nonzero tokens will be loaded from.')
    parser.add_argument('--train_idx_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10_precomputed_split_train.pkl',
                        help='Path where the train index will be loaded from.')
    parser.add_argument('--idx_save_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10_precomputed_loader_idx.pkl',
                        help='Path where the index will be saved.')
    parser.add_argument('--seq_len_cell',
                        type=int,
                        default=256,
                        help='Cell sequence length.')
    parser.add_argument('--seq_len_neighborhood',
                        type=int,
                        default=2560,
                        help='Neighborhood sequence length.')
    parser.add_argument('--gt_type',
                        type=str,
                        default='combined',
                        help='Gene transformer type.')
    parser.add_argument('--cell_pos_enc',
                        type=str,
                        default='coord',
                        help='Cell positional encoding.')
    parser.add_argument('--n_targets',
                        type=int,
                        default=1,
                        help='Number of targets.')
    parser.add_argument('--n_contexts',
                        type=int,
                        default=1,
                        help='Number of contexts.')
    parser.add_argument('--n_segments',
                        type=int,
                        default=11,
                        help='Number of segments.')
    parser.add_argument('--n_special_tokens',
                        type=int,
                        default=0,
                        help='Number of special tokens.')
    parser.add_argument('--per_block_mask_ratio',
                        type=float,
                        default=0.6,
                        help='Per block mask ratio.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=124,
                        help='Batch size.')
    args = parser.parse_args()

    main(args)