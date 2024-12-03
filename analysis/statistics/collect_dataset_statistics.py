import re
import argparse
import pandas as pd
import scanpy as sc
from pathlib import Path


def compute_batch_statistics(
        dataset_id,
        dataset_name,
        adata_batch_path,
    ):
    adata_batch = sc.read(adata_batch_path)

    batch_stats_dict = {
                        'dataset_name': dataset_name,
                        'dataset_id': dataset_id,
                        'batch': adata_batch.uns['batch'],
                        'n_cells': adata_batch.shape[0],
                        'n_genes': adata_batch.shape[1],
                        'assay': adata_batch.uns['assay'],
                        'species': adata_batch.uns['species'],
                        'tissue': adata_batch.uns['tissue'],
                        'adata_batch_path': str(adata_batch_path),
                        }

    return batch_stats_dict


def compute_dataset_statistics(
        dataset_name,
        input_directory,
    ):
    print(f"Dataset Name: {dataset_name}")
    dataset_prefix = dataset_name.split("-")[0]
    match = re.search(r'(\d+)$', dataset_prefix)
    dataset_id = int(match.group(1))
    
    batch_statistics = []
    
    for adata_batch_path in input_directory.glob("**/*.h5ad"):
        batch_stats_dict = compute_batch_statistics(
                                dataset_id,
                                dataset_name,
                                adata_batch_path,
                                )
        batch_statistics.append(batch_stats_dict)
    
    dataset_stats_df = pd.DataFrame(batch_statistics)
    
    return dataset_stats_df


def main():

    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    parser.add_argument('--data_directory_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/DATASETS",
                        help='Data directory path.')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="xhs1000-39b_1p",
                        help='Name of the dataset.')
    
    args = parser.parse_args()
    
    silver_directory_path = Path(args.data_directory_path) / "silver"
    input_directory = silver_directory_path / args.dataset_name
    
    statistics_file = "/lustre/scratch126/cellgen/team361/am84/NicheJEPA/results/silver_dataset_statistics.csv"
    
    dataset_stats_df = compute_dataset_statistics(
                            dataset_name=args.dataset_name,
                            input_directory=input_directory,
                            )

    if Path(statistics_file).exists():
        statistics_df = pd.read_csv(statistics_file)

        # if there are any previous statistics for this dataset, remove them
        statistics_df = statistics_df[statistics_df['dataset_name'] != args.dataset_name]

    else:
        statistics_df = pd.DataFrame()


    statistics_df = pd.concat(
                        [statistics_df, dataset_stats_df],
                        axis=0,
                        join='outer',
                        ignore_index=True,
                        )

    statistics_df = statistics_df.sort_values(by=['dataset_id', 'batch'])
    statistics_df.to_csv(statistics_file, index=False, sep=',')
    


if __name__ == '__main__':

    main()