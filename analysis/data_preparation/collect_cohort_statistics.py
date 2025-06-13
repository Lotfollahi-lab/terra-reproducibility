"""
This script collects the basic statistics of all datasets in a cohort. A cohort is defined as a collection of one or more datasets which themselves are a collection of one or more samples/batches. The statistics collected include the number of cells, number of genes, counts per cell, genes per cell, dataset ID, batch ID, assay, species, tissue, and the path to the dataset. The statistics are saved in a CSV file and counts and genes per cell are saved in pickle files.

Usage:
```bash
python collect_dataset_statistics.py --cohort_dataset_names_file_path [COHORT_NAME]
```

The `cohort_dataset_names_file_path` is the path to a file containing all dataset names for which the statistics are to be collected.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def compute_batch_statistics(cohort_name: str,
                             dataset_name: str,
                             adata_batch_path: Path,
                            ) -> dict:
    print(f"Processing {adata_batch_path}...")
    adata_batch = sc.read_h5ad(adata_batch_path)

    # Compute QC metrics if not available
    if ("total_counts" not in adata_batch.obs) or ("n_genes_by_counts" not in adata_batch.obs):
        sc.pp.calculate_qc_metrics(adata_batch, inplace=True, percent_top=None)

    batch_stats_dict = {'cohort_name': cohort_name,
                        'dataset_name': dataset_name,
                        'dataset_id': adata_batch.uns['dataset_id'],
                        'batch': adata_batch.uns['batch'],
                        'n_cells': adata_batch.shape[0],
                        'n_genes': adata_batch.shape[1],
                        'min_counts_per_cell': min(adata_batch.obs['total_counts']),
                        'max_counts_per_cell': max(adata_batch.obs['total_counts']),
                        'median_counts_per_cell': np.median(adata_batch.obs['total_counts']),
                        'min_genes_per_cell': min(adata_batch.obs['n_genes_by_counts']),
                        'max_genes_per_cell': max(adata_batch.obs['n_genes_by_counts']),
                        'median_genes_per_cell': np.median(adata_batch.obs['n_genes_by_counts']),
                        'assay': adata_batch.uns['assay'],
                        'species': adata_batch.uns['species'],
                        'tissue': adata_batch.uns['tissue'],
                        'adata_batch_path': str(adata_batch_path)}

    batch_counts_per_cell = adata_batch.obs['total_counts'].values.tolist()
    batch_genes_per_cell = adata_batch.obs['n_genes_by_counts'].values.tolist()

    return batch_stats_dict, batch_counts_per_cell, batch_genes_per_cell


def compute_cohort_statistics(
        input_directory: Path,
        cohort_name: str,
        dataset_names: list[str]):
    print(f"Cohort name: {cohort_name}")
    
    batch_statistics = []
    counts_per_cell = []
    genes_per_cell = []

    for dataset_name in dataset_names:
        for adata_batch_path in input_directory.glob(f"{dataset_name}/*.h5ad"):
            batch_stats_dict, batch_counts_per_cell, batch_genes_per_cell = compute_batch_statistics(
                cohort_name,
                dataset_name,
                adata_batch_path)
            batch_statistics.append(batch_stats_dict)
            counts_per_cell.append(batch_counts_per_cell)
            genes_per_cell.append(batch_genes_per_cell)
    
    cohort_stats_df = pd.DataFrame(batch_statistics)
    cohort_stats_df = cohort_stats_df.sort_values(
        by=['cohort_name', 'dataset_id', 'batch'])
    
    return cohort_stats_df, counts_per_cell, genes_per_cell


def main(args):
    silver_directory_path = Path(args.data_directory_path) / "silver"
    datasets_df = pd.read_csv(
        args.cohort_dataset_names_file_path,
        index_col=0,
        header=None)
    dataset_names = datasets_df.index.tolist()
    
    cohort_stats_df, counts_per_cell, genes_per_cell = compute_cohort_statistics(
        input_directory=silver_directory_path,
        cohort_name=args.cohort_name,
        dataset_names=dataset_names)

    # Save statistics to disk
    os.makedirs(f"{args.results_directory_path}/{args.cohort_name}", exist_ok=True)
    statistics_file = f"{args.results_directory_path}/{args.cohort_name}/{args.cohort_name}_statistics.csv"
    counts_file = f"{args.results_directory_path}/{args.cohort_name}/{args.cohort_name}_counts.pkl"
    genes_file = f"{args.results_directory_path}/{args.cohort_name}/{args.cohort_name}_genes.pkl"
    cohort_stats_df.to_csv(
        statistics_file,
        index=False,
        sep=',')
    with open(counts_file, 'wb') as file:
        pickle.dump(counts_per_cell, file)
    with open(genes_file, 'wb') as file:
        pickle.dump(genes_per_cell, file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default="/nfs/team361/sb75/DATASETS",
                        help='Data directory path.')
    parser.add_argument('--results_directory_path',
                        type=str,
                        default="/nfs/team361/sb75/nichejepa-reproducibility/artifacts/data",
                        help='Results directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="hst_corpus_80m", # "hst_corpus_70m"
                        help='Path to the file containing all datasets in the cohort.')
    parser.add_argument('--cohort_dataset_names_file_path',
                        type=str,
                        default="/nfs/team361/sb75/nichejepa-reproducibility/analysis/data_preparation/hst_corpus_80m/hst_corpus_80m_dataset_names.txt", # "/nfs/team361/sb75/nichejepa-reproducibility/analysis/data_preparation/hst_corpus_70m/hst_corpus_70m_dataset_names.txt",
                        help='Path to the file containing all datasets in the cohort.')
    args = parser.parse_args()
    
    main(args)