"""
This script collects the statistics of the basic statistics of the datasets in a cohort. A cohort is defined as a collection of one or more datasets which themselves are a collection of one or more batches. The statistics collected include the number of cells, number of genes, dataset ID, batch ID, assay, species, tissue, and the path to the dataset. The statistics are saved in a CSV file.

Usage:
```bash
python collect_dataset_statistics.py --data_directory_path /lustre/scratch126/cellgen/team361/DATASETS --cohort_name [COHORT_NAME]
```

The `cohort_name` is the name of a directory representing the datasets for which the statistics are to be collected. The `data_directory_path` is the path to the directory where the datasets are stored. The script will look for the datasets in the `data_directory_path/silver/[cohort_name]` directory. The statistics are saved in the `analysis/statistics/results/[cohort_name]_statistics.csv` file.
"""

import argparse
import pandas as pd
import scanpy as sc
from pathlib import Path


def compute_batch_statistics(
        cohort_name: str,
        dataset_name: str,
        adata_batch_path: Path,
    ):
    adata_batch = sc.read(adata_batch_path)

    batch_stats_dict = {
                        'cohort_name': cohort_name,
                        'dataset_name': dataset_name,
                        'dataset_id': adata_batch.uns['dataset_id'],
                        'batch': adata_batch.uns['batch'],
                        'n_cells': adata_batch.shape[0],
                        'n_genes': adata_batch.shape[1],
                        'assay': adata_batch.uns['assay'],
                        'species': adata_batch.uns['species'],
                        'tissue': adata_batch.uns['tissue'],
                        'adata_batch_path': str(adata_batch_path),
                        }

    return batch_stats_dict


def compute_cohort_statistics(
        cohort_name: str,
        input_directory: Path,
    ):
    print(f"Cohort Name: {cohort_name}")
    
    batch_statistics = []
    
    for adata_batch_path in input_directory.glob("**/*.h5ad"):
        dataset_name = adata_batch_path.parent.name if "cohort" in cohort_name else cohort_name
        batch_stats_dict = compute_batch_statistics(
                                cohort_name,
                                dataset_name,
                                adata_batch_path,
                                )
        batch_statistics.append(batch_stats_dict)
    
    cohort_stats_df = pd.DataFrame(batch_statistics)
    cohort_stats_df = cohort_stats_df.sort_values(by=['cohort_name', 'dataset_id', 'batch'])
    
    return cohort_stats_df


def main(args):

    silver_directory_path = Path(args.data_directory_path) / "silver"
    input_directory = silver_directory_path / args.cohort_name
    
    cohort_stats_df = compute_cohort_statistics(
                            cohort_name=args.cohort_name,
                            input_directory=input_directory,
                            )

    statistics_file = f"analysis/statistics/results/{args.cohort_name}_statistics.csv"
    cohort_stats_df.to_csv(
        statistics_file,
        index=False,sep=','
    )
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    parser.add_argument('--data_directory_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/DATASETS",
                        help='Data directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="xhs1000-39b_1p",
                        help='Name of the dataset.')
    
    args = parser.parse_args()
    
    main(args)