import argparse
import ast
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from datasets import concatenate_datasets, load_from_disk


def main(args):
    ### Prepare stratified sampling ###
    # Load Huggingface dataset and keep training indices only
    dataset = load_from_disk(args.dataset_path)
    with open(args.train_idx_path, 'rb') as f:
        train_idx = pickle.load(f)
    dataset = dataset.select(train_idx)
    print(dataset)

    # Retrieve dataset IDs
    dataset_ids = [cell_id.split('_')[0] for cell_id in list(dataset['cell_id'])]

    # Get validation and test batches
    with open(args.dataset_split_path, "r") as f:
        dataset_split = f.read()
    dataset_split_data = {}
    exec(dataset_split, {}, dataset_split_data)
    val_batch_ids = dataset_split_data['__annotations__']['val_batch_ids']
    test_batch_ids = dataset_split_data['__annotations__']['test_batch_ids']

    # Read corpus statistics
    data_artifacts_folder_path = '/nfs/team361/sb75/nichejepa-reproducibility/artifacts/data'
    dataset_stats_df = pd.read_csv(f'{data_artifacts_folder_path}/{args.cohort_name}/{args.cohort_name}_statistics.csv')
    dataset_stats_df['dataset_id_batch'] = dataset_stats_df['dataset_id'].astype(str) + '_' + dataset_stats_df['batch'].astype(str)
    dataset_stats_df = dataset_stats_df[~dataset_stats_df['dataset_id_batch'].isin(val_batch_ids + test_batch_ids)] # remove test and val batches
    print(dataset_stats_df.head())
    ###################################


    ### Filter cells stratified by tissue ###
    random.seed(0)
    dataset_ids_set = sorted(set(dataset_ids))
    sampled_dataset_ids_set = random.sample(dataset_ids_set, len(dataset_ids_set))
    print(sampled_dataset_ids_set)

    sampled_dataset_ids = []
    datasets_list = []

    for sample_ratio in args.sample_ratios:
        print(f"Sample ratio: {sample_ratio}.")
        tissue_cell_count_dict = {}
        tissue_cell_count_diff_dict = {}
        max_tissue_cell_count = (dataset_stats_df.groupby('tissue')['n_cells'].sum() * sample_ratio).astype(int)
        for dataset_id in sampled_dataset_ids_set:
            dataset_tissue = dataset_stats_df[dataset_stats_df['dataset_id'] == int(dataset_id)]['tissue'].values[0]
            dataset_cell_count = dataset_stats_df[dataset_stats_df['dataset_id'] == int(dataset_id)]['n_cells'].values.sum()
            if dataset_tissue not in tissue_cell_count_dict:
                tissue_cell_count_dict[dataset_tissue] = 0
                tissue_cell_count_diff_dict[dataset_tissue] = max_tissue_cell_count[dataset_tissue]
            if tissue_cell_count_diff_dict[dataset_tissue] >= 0:
                current_dataset = dataset.filter(
                    lambda ids: [i.split('_')[0] == dataset_id for i in ids],
                    input_columns="cell_id",
                    batched=True,
                    batch_size=10000,
                    num_proc=args.num_workers,
                    desc="Filtering by dataset_id"
                )
                print(f"Complete length of current dataset: {len(current_dataset)}.")
                tissue_cell_count_dict[dataset_tissue] += dataset_cell_count
                tissue_cell_count_diff_dict[dataset_tissue] = max_tissue_cell_count[dataset_tissue] - tissue_cell_count_dict[dataset_tissue]
                if tissue_cell_count_diff_dict[dataset_tissue] < 0:
                    current_dataset = current_dataset.select(range(0, len(current_dataset) + tissue_cell_count_diff_dict[dataset_tissue]))
                    print(f"Tissue cell count diff: {tissue_cell_count_diff_dict[dataset_tissue]}.")    
                    print(f"Truncated length of current dataset: {len(current_dataset)}.")
                datasets_list.append(current_dataset)
            else:
                print(f"Tissue cell count diff: {tissue_cell_count_diff_dict[dataset_tissue]}. Continuing...") 
                continue
                
        combined_dataset = concatenate_datasets(datasets_list)
        print(f"Combined dataset length: {len(combined_dataset)}.")                                            
        print('Saving combined dataset...')
        combined_dataset.save_to_disk(args.dataset_path.replace(f'{args.cohort_name}_', f'{args.cohort_name}_{sample_ratio}_sample_'))
    #########################################


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
                        help='Path where the training index is stored.')
    parser.add_argument('--dataset_split_path',
                        type=str,
                        default='/nfs/team361/sb75/nichejepa-reproducibility/analysis/data_preparation/hst_corpus_90m/hst_corpus_90m_dataset_split.txt',
                        help='Path where the dataset split is stored.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default='hst_corpus_90m',
                        help='Name of data cohort.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of workers.')
    parser.add_argument('--sample_ratios',
                        type=float,
                        nargs='+',
                        default=[0.01, 0.1, 0.25, 0.5],
                        help='Sample ratios for cell filtering.')
    args = parser.parse_args()

    main(args)