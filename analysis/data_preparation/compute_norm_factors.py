import argparse
import math
import pickle
import tracemalloc
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import scanpy as sc


def main(args):
    silver_data_path = Path(args.data_directory_path) / 'silver'

    # Read dataset names
    datasets_df = pd.read_csv(
        args.cohort_dataset_names_file_path,
        index_col=0,
        header=None)
    print(datasets_df)

    batchwise_norm_factor_dict = {}

    print("Computing norm factors...")
    tracemalloc.start()

    for dataset_name, _ in datasets_df.iterrows():
        print('Processing: ', dataset_name)
        
        silver_dataset_path = silver_data_path / dataset_name

        for file in silver_dataset_path.glob('*.h5ad'):
            adata = sc.read_h5ad(file)

            # Compute batchwise nonzero means and means per gene
            X_dense = adata.X.toarray()
            nonzero_mask = X_dense > 0
            nonzero_counts = np.sum(nonzero_mask, axis=0)
            nonzero_sums = np.sum(X_dense * nonzero_mask, axis=0)
            nonzero_means = np.true_divide(nonzero_sums, nonzero_counts)
            means = X_dense.mean(0)

            # Perform gene corrected read depth normalization
            X = adata.X / adata.X.sum(axis=1).reshape(-1, 1) * (
                args.basis_target_read_depth + adata.X.shape[1] *
                args.target_read_depth_per_gene)

            # Compute batchwise gene corrected read depth normalized nonzero
            # means and means per gene
            X_dense_gcrd = X.toarray()
            nonzero_mask_gcrd = X_dense_gcrd > 0
            nonzero_counts_gcrd = np.sum(nonzero_mask_gcrd, axis=0)
            nonzero_sums_gcrd = np.sum(X_dense_gcrd * nonzero_mask_gcrd, axis=0)
            nonzero_means_gcrd = np.true_divide(
                nonzero_sums_gcrd, nonzero_counts_gcrd)
            means_gcrd = X_dense_gcrd.mean(0)

            # Store batchwise norm factors
            for idx, gene_id in enumerate(adata.var['ensembl_id'].values):
                if math.isnan(nonzero_means_gcrd[idx]): # math.isnan(nonzero_means[idx]) is equivalent
                    nonzero_means[idx] = 0.0
                    nonzero_means_gcrd[idx] = 0.0
                    n_cells = 0
                else:
                    n_cells = len(adata)
                if f"{gene_id}_n_cells" in batchwise_norm_factor_dict.keys():
                    batchwise_norm_factor_dict[f"{gene_id}_nonzero_means"].append(nonzero_means[idx])
                    batchwise_norm_factor_dict[f"{gene_id}_means"].append(means[idx])
                    batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_nonzero_means"].append(nonzero_means_gcrd[idx])
                    batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_means"].append(means_gcrd[idx])
                    batchwise_norm_factor_dict[f"{gene_id}_n_cells"].append(n_cells)
                else:
                    batchwise_norm_factor_dict[f"{gene_id}_nonzero_means"] = [nonzero_means[idx]]
                    batchwise_norm_factor_dict[f"{gene_id}_means"] = [means[idx]]
                    batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_nonzero_means"] = [nonzero_means_gcrd[idx]]
                    batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_means"] = [means_gcrd[idx]]
                    batchwise_norm_factor_dict[f"{gene_id}_n_cells"] = [n_cells]

            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB.")

            # Write to disk
            with open(f'/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/artifacts/data/{args.cohort_name}_batchwise_norm_factors.pkl', 'wb') as file:
                pickle.dump(batchwise_norm_factor_dict, file)

    # Compute overall nonzero means and means per gene and store cell numbers
    norm_factor_dict = {}
    gene_ids = list(set([key.split('_')[0] for key in norm_factor_dict.keys()]))
    
    for gene_id in gene_ids:
        norm_factor_dict[gene_id] = {}
        norm_factor_dict[gene_id]["nonzero_mean"] = np.dot(
            np.array(batchwise_norm_factor_dict[f"{gene_id}_nonzero_means"]),
            np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"])
        ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
        norm_factor_dict[gene_id]["mean"] = np.dot(
            np.array(batchwise_norm_factor_dict[f"{gene_id}_means"]),
            np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"])
        ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
        norm_factor_dict[gene_id]["gene_corrected_read_depth_nonzero_mean"] = np.dot(
            np.array(batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_nonzero_means"]),
            np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"])
        ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
        norm_factor_dict[gene_id]["gene_corrected_read_depth_mean"] = np.dot(
            np.array(batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_means"]),
            np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"])
        ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
        norm_factor_dict[gene_id]["n_cells"] = np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()

    norm_factor_df = pd.DataFrame.from_dict(norm_factor_dict, orient='index').reset_index()
    norm_factor_df = norm_factor_df.rename(columns={'index': 'gene_id'})

    # Write to disk
    norm_factor_df.to_csv(f'/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/artifacts/data/{args.cohort_name}_norm_factors.csv',
                        index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/DATASETS",
                        help='Data directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="hst_corpus_70m",
                        help='Name of the cohort.')
    parser.add_argument('--cohort_dataset_names_file_path',
                        type=str,
                        default="/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/analysis/data_preparation/hst_corpus_70m/hst_corpus_70m_dataset_names.txt",
                        help='Path to the file containing datasets of the cohort.')                        
    parser.add_argument('--basis_target_read_depth',
                        type=float,
                        default=137.7060,
                        help='Read depth independent of number of probed genes.')
    parser.add_argument('--target_read_depth_per_gene',
                        type=float,
                        default=0.0901,
                        help='Additional read depth increment per probed gene.')
    args = parser.parse_args()
    
    main(args)