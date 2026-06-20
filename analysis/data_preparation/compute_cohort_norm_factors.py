import argparse
import math
import pickle
import tracemalloc
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def main(args):
    silver_data_path = Path(args.data_directory_path) / 'silver'

    # Output artifacts go into a per-cohort subdirectory:
    #   {artifacts_base}/{cohort_name}/{cohort_name}_norm_factors.csv
    #   {artifacts_base}/{cohort_name}/{cohort_name}_pf_targets.csv
    #   {artifacts_base}/{cohort_name}/{cohort_name}_batchwise_norm_factors.pkl
    output_dir = Path(args.artifacts_base_path) / args.cohort_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read dataset names
    datasets_df = pd.read_csv(
        args.cohort_dataset_names_file_path,
        index_col=0,
        header=None)
    print(datasets_df)

    running_n_cells = 0
    batchwise_norm_factor_dict = {}

    # ------------------------------------------------------------------
    # PFlog1pPF corpus targets -- pass 1 of 2.
    # PFlog1pPF is a per-cell transform whose only cross-cell dependence
    # is two global scalars: the first-PF target s1 (corpus mean cell
    # depth) and the second-PF target s2 (corpus mean per-cell log-sum).
    # Computing them once over the whole cohort lets the tokenizer FREEZE
    # them instead of recomputing per file, which keeps the value scale
    # consistent across shards and -- crucially -- between training and
    # inference (mirrors how gene_corrected_read_depth carries fixed
    # corpus depth targets). s2 depends on s1, so s1 is computed first in
    # a lightweight sparse pass here; s2 is accumulated in the main loop.
    # ------------------------------------------------------------------
    pf_depth_target = None
    pf_logsum_sum, pf_logsum_n_cells = 0.0, 0
    if not args.skip_pf_targets:
        print("Computing PFlog1pPF first-PF target (corpus mean depth)...")
        pf_count_sum, pf_n_cells = 0.0, 0
        for dataset_name, _ in datasets_df.iterrows():
            silver_dataset_path = silver_data_path / dataset_name
            for file in silver_dataset_path.glob('*.h5ad'):
                adata = sc.read_h5ad(file)
                cell_sums = np.asarray(adata.X.sum(axis=1)).ravel()
                pf_count_sum += float(cell_sums.sum())
                pf_n_cells += int(adata.n_obs)
        pf_depth_target = pf_count_sum / max(pf_n_cells, 1)
        print(f"  PF depth target s1 = {pf_depth_target:.6f} "
              f"(over {pf_n_cells} cells)")

    print("Computing norm factors...")
    tracemalloc.start()

    for dataset_name, _ in datasets_df.iterrows():
        print('Processing: ', dataset_name)
        
        silver_dataset_path = silver_data_path / dataset_name

        for file in silver_dataset_path.glob('*.h5ad'):
            adata = sc.read_h5ad(file)
            running_n_cells += len(adata)

            # PFlog1pPF second-PF target (s2) -- pass 2 of 2. Accumulate
            # the sum of per-cell log-sums, where each cell is first
            # depth-equalized to the GLOBAL s1 (pf_depth_target) and
            # log1p-transformed. Done on the sparse matrix to avoid an
            # extra dense copy. All cells (incl. zero-count) are counted
            # so the resulting mean matches what normalize_by_pflog1ppf
            # computes when applied to the pooled corpus at once.
            if not args.skip_pf_targets:
                cell_sums_pf = np.asarray(adata.X.sum(axis=1)).ravel()
                safe_pf = np.where(cell_sums_pf > 0, cell_sums_pf, 1.0)
                scale_pf = (pf_depth_target / safe_pf).reshape(-1, 1)
                if sp.issparse(adata.X):
                    x_pf = adata.X.multiply(scale_pf).tocsr()
                    x_pf.data = np.log1p(x_pf.data)
                    log_sums_pf = np.asarray(x_pf.sum(axis=1)).ravel()
                else:
                    x_pf = np.asarray(adata.X, dtype=np.float64) * scale_pf
                    log_sums_pf = np.log1p(x_pf).sum(axis=1)
                pf_logsum_sum += float(np.asarray(log_sums_pf).sum())
                pf_logsum_n_cells += int(adata.n_obs)

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
                    n_nonzero_cells = nonzero_counts[idx]
                if f"{gene_id}_n_cells" in batchwise_norm_factor_dict.keys():
                    batchwise_norm_factor_dict[f"{gene_id}_nonzero_means"].append(nonzero_means[idx])
                    batchwise_norm_factor_dict[f"{gene_id}_means"].append(means[idx])
                    batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_nonzero_means"].append(nonzero_means_gcrd[idx])
                    batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_means"].append(means_gcrd[idx])
                    batchwise_norm_factor_dict[f"{gene_id}_n_cells"].append(n_cells)
                    batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"].append(n_nonzero_cells)
                else:
                    batchwise_norm_factor_dict[f"{gene_id}_nonzero_means"] = [nonzero_means[idx]]
                    batchwise_norm_factor_dict[f"{gene_id}_means"] = [means[idx]]
                    batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_nonzero_means"] = [nonzero_means_gcrd[idx]]
                    batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_means"] = [means_gcrd[idx]]
                    batchwise_norm_factor_dict[f"{gene_id}_n_cells"] = [n_cells]
                    batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"] = [n_nonzero_cells]

            print(f"Running number of cells: {running_n_cells}.")

            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB.")

            # Write to disk
            with open(output_dir / f'{args.cohort_name}_batchwise_norm_factors.pkl', 'wb') as file:
                pickle.dump(batchwise_norm_factor_dict, file)

    # Compute overall nonzero means and means per gene and store cell numbers
    norm_factor_dict = {}
    gene_ids = list(set([key.split('_')[0] for key in batchwise_norm_factor_dict.keys()]))
    
    for gene_id in gene_ids:
        norm_factor_dict[gene_id] = {}
        norm_factor_dict[gene_id]["nonzero_mean"] = np.dot(
            np.array(batchwise_norm_factor_dict[f"{gene_id}_nonzero_means"]),
            np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"])
        ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"]).sum()
        norm_factor_dict[gene_id]["mean"] = np.dot(
            np.array(batchwise_norm_factor_dict[f"{gene_id}_means"]),
            np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"])
        ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
        norm_factor_dict[gene_id]["gene_corrected_read_depth_nonzero_mean"] = np.dot(
            np.array(batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_nonzero_means"]),
            np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"])
        ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"]).sum()
        norm_factor_dict[gene_id]["gene_corrected_read_depth_mean"] = np.dot(
            np.array(batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_means"]),
            np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"])
        ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
        norm_factor_dict[gene_id]["n_cells"] = np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
        norm_factor_dict[gene_id]["n_nonzero_cells"] = np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"]).sum()

    norm_factor_df = pd.DataFrame.from_dict(norm_factor_dict, orient='index').reset_index()
    norm_factor_df = norm_factor_df.rename(columns={'index': 'gene_id'})

    # Write to disk
    norm_factor_df.to_csv(output_dir / f'{args.cohort_name}_norm_factors.csv',
                        index=False)

    # Write the corpus-wide PFlog1pPF targets (s1, s2) for the tokenizer
    # to load (via target_size / logsum_target on normalize_by_pflog1ppf),
    # so the pflog1ppf arm uses frozen corpus scales instead of per-file
    # statistics.
    if not args.skip_pf_targets:
        pf_logsum_target = pf_logsum_sum / max(pf_logsum_n_cells, 1)
        pf_targets_df = pd.DataFrame([{
            'pf_depth_target': pf_depth_target,
            'pf_logsum_target': pf_logsum_target,
            'n_cells': pf_logsum_n_cells,
        }])
        pf_targets_path = output_dir / f'{args.cohort_name}_pf_targets.csv'
        pf_targets_df.to_csv(pf_targets_path, index=False)
        print(f"PFlog1pPF targets written to {pf_targets_path}:")
        print(f"  pf_depth_target  (s1) = {pf_depth_target:.6f}")
        print(f"  pf_logsum_target (s2) = {pf_logsum_target:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default="/nfs/team361/sb75/DATASETS",
                        help='Data directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="hst_corpus_110m", # "hst_corpus_90m", "hst_corpus_80m", "hst_corpus_70m"
                        help='Name of the cohort.')
    parser.add_argument('--artifacts_base_path',
                        type=str,
                        default="/nfs/team361/sb75/terra-reproducibility/artifacts/data",
                        help='Base directory for output artifacts. Outputs '
                             'are written to '
                             '{artifacts_base_path}/{cohort_name}/ as '
                             '{cohort_name}_norm_factors.csv, '
                             '{cohort_name}_pf_targets.csv and '
                             '{cohort_name}_batchwise_norm_factors.pkl '
                             '(the directory is created if missing).')
    parser.add_argument('--cohort_dataset_names_file_path',
                        type=str,
                        default="/nfs/team361/sb75/terra-reproducibility/analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt", # "/nfs/team361/sb75/terra-reproducibility/analysis/data_preparation/hst_corpus_90m/hst_corpus_90m_dataset_names.txt", "/nfs/team361/sb75/terra-reproducibility/analysis/data_preparation/hst_corpus_80m/hst_corpus_80m_dataset_names.txt", "/nfs/team361/terra-reproducibility/analysis/data_preparation/hst_corpus_70m/hst_corpus_70m_dataset_names.txt",
                        help='Path to the file containing datasets of the cohort.')                        
    parser.add_argument('--basis_target_read_depth',
                        type=float,
                        default=151.1754621380761, # 154.01749299636748 (hst_corpus_90m); 150.6692 (hst_corpus_80m)
                        help='Read depth independent of number of probed genes.')
    parser.add_argument('--target_read_depth_per_gene',
                        type=float,
                        default=0.04705053555567159, # 0.05644618485013678 (hst_corpus_90m); 0.0646 (hst_corpus_80m)
                        help='Additional read depth increment per probed gene.')
    parser.add_argument('--skip_pf_targets',
                        action='store_true',
                        help='Skip computing the corpus-wide PFlog1pPF '
                             'targets (first-PF mean depth s1 and second-PF '
                             'mean log-sum s2). Computing them adds one '
                             'extra lightweight (sparse) pass over the '
                             'corpus and writes {cohort}_pf_targets.csv.')
    args = parser.parse_args()
    
    main(args)