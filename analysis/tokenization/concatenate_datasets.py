import argparse
import json
import os
import tracemalloc
from pathlib import Path

import datasets
import pyarrow.compute as pc
import pandas as pd


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
        args.n_neighs,
        args.seq_len]])
    print(hparam_prefix)

    ds_list = []
    tokenizer_params = []

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

        gold_dataset_dir = [p for p in gold_data_path.rglob("*") if p.is_dir() and (dir_substring + '.dataset' in p.name) or (dir_substring + f'_{args.seq_len}.dataset' in p.name)][0]
        
        print(gold_dataset_dir)

        ds = datasets.load_from_disk(str(gold_dataset_dir))
        params_path = gold_dataset_dir / 'terra_tokenizer_params.json'
        tokenizer_params.append(
            json.loads(params_path.read_text()) if params_path.exists()
            else None)

        ds_list.append(ds)

    cohort_dir = gold_data_path / args.cohort_name
    cohort_dataset_dir = cohort_dir / f"{args.cohort_name}_{hparam_prefix}"
    output_path = str(
        (Path(cohort_dir) / f"{args.cohort_name}_{hparam_prefix}").with_suffix(
            '.dataset'))

    os.makedirs(cohort_dir, exist_ok=True)

    # Save dataset
    print('Concatenating datasets...')
    tracemalloc.start()
    concatenated_dataset = datasets.concatenate_datasets(ds_list)
    concatenated_dataset.save_to_disk(
        output_path
        #num_shards=args.num_shards
        )
    # The concatenated dataset keeps the tokenizer settings record (which
    # train() checks the model config against) only if all parts share it
    if any(p is None for p in tokenizer_params):
        print('WARNING: some datasets have no terra_tokenizer_params.json; '
              'the concatenated dataset gets none.')
    else:
        if any(p != tokenizer_params[0] for p in tokenizer_params[1:]):
            raise ValueError(
                'The datasets were tokenized with different settings '
                '(terra_tokenizer_params.json differs); re-tokenize them '
                'with one configuration before concatenating.')
        with open(Path(output_path) / 'terra_tokenizer_params.json', 'w') as f:
            json.dump(tokenizer_params[0], f, indent=2, sort_keys=True)
    # Cell ids must be unique: duplicated rows were once removed by a
    # hard-coded row range (fix_dataset_duplicates.py)
    n_unique = pc.count_distinct(
        concatenated_dataset.data.column('cell_id')).as_py()
    n_duplicates = len(concatenated_dataset) - n_unique
    if n_duplicates:
        raise ValueError(f'{n_duplicates} duplicated cell ids in {output_path}.')
    current, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS', # '/nfs/team361/sb75/DATASETS'
                        help='Data directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default='hst_corpus_110m',
                        help='Name of the cohort.')
    parser.add_argument('--cohort_dataset_names_file_path',
                        type=str,
                        default='/nfs/team361/sb75/terra-reproducibility/analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt', #'/nfs/team361/sb75/terra-reproducibility/analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names_additional.txt'
                        help='Path to the file containing all datasets in the cohort.')
    parser.add_argument('--tokenizer',
                        type=str,
                        default='cell-graph-tokenizer',
                        help='Tokenizer.')
    parser.add_argument('--num_shards',
                        type=int,
                        default='32',
                        help='Number of shards.')
    parser.add_argument('--rank_cell_norm_method',
                        type=str,
                        default=None, # None, 'gene_corrected_read_depth'
                        help='Rank cell norm method.')
    parser.add_argument('--rank_gene_norm_method',
                        type=str,
                        default=None, # None, 'nonzero_mean'
                        help='Rank gene norm method.')
    parser.add_argument('--rank_count_norm_method',
                        type=str,
                        default=None, # None
                        help='Rank count norm method.')
    parser.add_argument('--count_cell_norm_method',
                        type=str,
                        default=None, # None, 'gene_corrected_read_depth'
                        help='Count cell norm method.')
    parser.add_argument('--count_gene_norm_method',
                        type=str,
                        default=None, # None, 'nonzero_mean'
                        help='Count gene norm method.')
    parser.add_argument('--count_count_norm_method',
                        type=str,
                        default='shifted_log', # 'shifted_log', None
                        help='Count count norm method.')
    parser.add_argument('--n_neighs',
                        type=int,
                        default=6, # 10, 6
                        help='Number of neighbors.')
    parser.add_argument('--seq_len',
                        type=int,
                        default=7168, # 2816, 7168
                        help='Sequence length.')
    args = parser.parse_args()
    
    main(args)
