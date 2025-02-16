import argparse
import os
from pathlib import Path

import pandas as pd
import yaml


def main(args):
    config_path = Path('/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/config') / args.tokenizer

    # Read datasets
    datasets_df = pd.read_csv(
        args.cohort_dataset_names_file_path,
        index_col=0,
        header=None)
    print(datasets_df)

    for dataset_name, _ in datasets_df.iterrows():
        print(f'Setting config of dataset: {dataset_name}.')
        
        config_dataset_path = config_path / f'{args.tokenizer}_{dataset_name}.yaml'

        with open(config_dataset_path, 'r') as file:
            dataset_config = yaml.safe_load(file)
            print(dataset_config)

            dataset_config['tokenization']['rank_cell_norm_method'] = args.rank_cell_norm_method
            dataset_config['tokenization']['rank_gene_norm_method'] = args.rank_gene_norm_method
            dataset_config['tokenization']['rank_count_norm_method'] = args.rank_count_norm_method
            dataset_config['tokenization']['count_cell_norm_method'] = args.count_cell_norm_method
            dataset_config['tokenization']['count_gene_norm_method'] = args.count_gene_norm_method
            dataset_config['tokenization']['count_count_norm_method'] = args.count_count_norm_method
            dataset_config['tokenization']['n_neighs'] = args.n_neighs

        with open(config_dataset_path, 'w') as file:
           yaml.dump(dataset_config, file, default_flow_style=False)
    
    print('Finished setting all configs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--cohort_dataset_names_file_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/analysis/data_preparation/hst_corpus_70m/hst_corpus_70m_dataset_names.txt",
                        help='Path to the file containing all datasets in the cohort.')
    parser.add_argument('--tokenizer',
                        type=str,
                        default="cell-graph-tokenizer",
                        help='Tokenizer.')
    parser.add_argument('--rank_cell_norm_method',
                        type=str,
                        default='gene_corrected_read_depth', # None, 'gene_corrected_read_depth'
                        help='Rank cell norm method.')
    parser.add_argument('--rank_gene_norm_method',
                        type=str,
                        default='nonzero_mean', # None, 'nonzero_mean'
                        help='Rank gene norm method.')
    parser.add_argument('--rank_count_norm_method',
                        type=str,
                        default=None, # None
                        help='Rank count norm method.')
    parser.add_argument('--count_cell_norm_method',
                        type=str,
                        default=None, # None
                        help='Count cell norm method.')
    parser.add_argument('--count_gene_norm_method',
                        type=str,
                        default=None, # None
                        help='Count gene norm method.')
    parser.add_argument('--count_count_norm_method',
                        type=str,
                        default="shifted_log", # None
                        help='Count count norm method.')
    parser.add_argument('--n_neighs',
                        type=int,
                        default="10",
                        help='Number of neighbors.')
    args = parser.parse_args()
    
    main(args)
