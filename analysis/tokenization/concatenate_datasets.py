import argparse
import os
import tracemalloc
from pathlib import Path

import pandas as pd
import datasets


def main(args):
    gold_data_path = Path(args.data_directory_path) / 'gold' / args.tokenizer

    hparam_prefix = '_'.join([str(s) for s in [
        args.num_shards,
        args.rank_cell_norm_method,
        args.rank_gene_norm_method,
        args.rank_count_norm_method,
        args.count_cell_norm_method,
        args.count_gene_norm_method,
        args.count_count_norm_method,
        "knn",
        args.n_neighs]])
    print(hparam_prefix)

    ds_list = []

    # Read datasets
    datasets_df = pd.read_csv(
        args.cohort_dataset_names_file_path,
        index_col=0,
        header=None)
    print(datasets_df)

    for dataset_name, _ in datasets_df.iterrows():
        print('Processing: ', dataset_name)
        
        gold_dataset_path = gold_data_path / dataset_name
        dir_substring = f"{dataset_name}_{hparam_prefix}"
        
        gold_dataset_dir = [p for p in gold_data_path.rglob("*") if p.is_dir() and dir_substring in p.name][0]
        
        print(gold_dataset_dir)

        ds = datasets.load_from_disk(str(gold_dataset_dir))
        ds_list.append(ds)
        
    cohort_dir = gold_data_path / args.cohort_name
    cohort_dataset_dir = cohort_dir / f"{args.cohort_name}_{hparam_prefix}"
    output_path = str(
        (Path(cohort_dir) / f"{args.cohort_name}_{hparam_prefix}").with_suffix(
            '.dataset'))

    os.makedirs(cohort_dir, exist_ok=True)

    # Save the dataset
    print("Concatenating datasets...")
    tracemalloc.start()
    concatenated_dataset = datasets.concatenate_datasets(ds_list)
    concatenated_dataset.save_to_disk(output_path, num_shards=args.num_shards)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/DATASETS",
                        help='Data directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="human_cohort1_60m",
                        help='Name of the cohort.')
    parser.add_argument('--cohort_dataset_names_file_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/analysis/tokenization/human_cohort1_60m_dataset_names.txt",
                        help='Path to the file containing all datasets in the cohort.')
    parser.add_argument('--tokenizer',
                        type=str,
                        default="cell-graph-tokenizer",
                        help='Tokenizer.')
    parser.add_argument('--num_shards',
                        type=int,
                        default="32",
                        help='Number of shards.')
    parser.add_argument('--rank_cell_norm_method',
                        type=str,
                        default=None,
                        help='Rank cell norm method.')
    parser.add_argument('--rank_gene_norm_method',
                        type=str,
                        default=None,
                        help='Rank gene norm method.')
    parser.add_argument('--rank_count_norm_method',
                        type=str,
                        default=None,
                        help='Rank count norm method.')
    parser.add_argument('--count_cell_norm_method',
                        type=str,
                        default="gene_corrected_read_depth",
                        help='Count cell norm method.')
    parser.add_argument('--count_gene_norm_method',
                        type=str,
                        default=None,
                        help='Count gene norm method.')
    parser.add_argument('--count_count_norm_method',
                        type=str,
                        default="shifted_log",
                        help='Count count norm method.')
    parser.add_argument('--n_neighs',
                        type=int,
                        default="10",
                        help='Number of neighbors.')
    args = parser.parse_args()
    
    main(args)
