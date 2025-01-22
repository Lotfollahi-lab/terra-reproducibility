import argparse

import pandas as pd
import datasets
from pathlib import Path
import os
import tracemalloc


tokenizer='cell-graph-tokenizer'

num_shards = 32
norm_factor = "None"
norm_method = "shifted_log"
n_neighs = 10
hparam_prefix = '_'.join([str(s) for s in [num_shards, norm_factor, norm_method, "knn", n_neighs]])
print(hparam_prefix)


def main(args):
    gold_data_path = Path(args.data_directory_path) / 'gold' / tokenizer

    cohort_dir = gold_data_path / args.cohort_name
    input_path = str(
        (Path(cohort_dir) / f"{args.cohort_name}_{hparam_prefix}").with_suffix(
            '.dataset'))
    output_path = str(
        (Path(cohort_dir) / f"{args.cohort_name}_{hparam_prefix}_appended").with_suffix(
            '.dataset'))

    ds_list = []
    ds = datasets.load_from_disk(input_path)
    ds_list.append(ds)

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
        
    cohort_dataset_dir = cohort_dir / f"{args.cohort_name}_{hparam_prefix}"

    os.makedirs(cohort_dir, exist_ok=True)

    # Save the dataset
    print("Concatenating datasets...")
    tracemalloc.start()
    concatenated_dataset = datasets.concatenate_datasets(ds_list)
    concatenated_dataset.save_to_disk(output_path, num_shards=num_shards)
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
                        default="/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/analysis/tokenization/human_cohort1_60m_add_dataset_names.txt",
                        help='Path to the file containing datasets to be added to the cohort.')
    args = parser.parse_args()
    
    main(args)