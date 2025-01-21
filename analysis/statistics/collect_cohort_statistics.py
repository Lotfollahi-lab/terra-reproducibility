"""
This script collects the basic statistics of the datasets in a cohort. A cohort is defined as a collection of one or more datasets which themselves are a collection of one or more batches. The statistics collected include the number of cells, number of genes, dataset ID, batch ID, assay, species, tissue, and the path to the dataset. The statistics are saved in a CSV file.

Usage:
```bash
python collect_dataset_statistics.py --data_directory_path /lustre/scratch126/cellgen/team361/DATASETS --cohort_dataset_names_file_path [COHORT_NAME]
```

The `cohort_name` is the name of a directory representing the datasets for which the statistics are to be collected. The `data_directory_path` is the path to the directory where the datasets are stored. The script will look for the datasets in the `data_directory_path/silver/[cohort_name]` directory. The statistics are saved in the `analysis/statistics/results/[cohort_name]_statistics.csv` file.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def compute_batch_statistics(
        cohort_name: str,
        dataset_name: str,
        adata_batch_path: Path,
    ) -> dict:
    print(f"Processing {adata_batch_path}...")
    adata_batch = sc.read_h5ad(adata_batch_path)

    # Compute QC metrics if not available
    if ("total_counts" not in adata_batch.obs) or ("n_genes_by_counts" not in adata_batch.obs):
        sc.pp.calculate_qc_metrics(adata_batch, inplace=True, percent_top=None)

    batch_stats_dict = {
                        'cohort_name': cohort_name,
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
                        'adata_batch_path': str(adata_batch_path),
                        }

    return batch_stats_dict


def compute_cohort_statistics(
        input_directory: Path,
        cohort_name: str,
        datasets: list[str]):
    print(f"Cohort Name: {cohort_name}")
    
    batch_statistics = []

    for dataset in datasets:
        for adata_batch_path in input_directory.glob(f"{dataset}/*.h5ad"):
            batch_stats_dict = compute_batch_statistics(
                cohort_name,
                dataset,
                adata_batch_path)
            batch_statistics.append(batch_stats_dict)
    
    cohort_stats_df = pd.DataFrame(batch_statistics)
    cohort_stats_df = cohort_stats_df.sort_values(
        by=['cohort_name', 'dataset_id', 'batch'])
    
    return cohort_stats_df


def main(args):
    silver_directory_path = Path(args.data_directory_path) / "silver"
    datasets_df = pd.read_csv(
        args.cohort_dataset_names_file_path,
        index_col=0,
        header=None)
    datasets = datasets_df.index.tolist()
    
    cohort_stats_df = compute_cohort_statistics(
        input_directory=silver_directory_path,
        cohort_name=args.cohort_name,
        datasets=datasets)

    statistics_file = f"{args.results_directory_path}/{args.cohort_name}_statistics.csv"
    cohort_stats_df.to_csv(
        statistics_file,
        index=False,
        sep=',')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/DATASETS",
                        help='Data directory path.')
    parser.add_argument('--results_directory_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/artifacts/data",
                        help='Results directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="human_cohort1_60m",
                        help='Path to the file containing all datasets in the cohort.')
    parser.add_argument('--cohort_dataset_names_file_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/analysis/tokenization/human_cohort1_60m_dataset_names.txt",
                        help='Path to the file containing all datasets in the cohort.')
    args = parser.parse_args()
    
    main(args)