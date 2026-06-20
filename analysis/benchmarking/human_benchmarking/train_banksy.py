import os
import yaml
import time
import wandb
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Literal, Union

import scib
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import squidpy as sq
import scipy.sparse as sp
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('/lustre/scratch126/cellgen/team361/am84/TERRA/third-party-software/banksy')
from banksy.main import median_dist_to_nearest_neighbour
from banksy_utils.load_data import load_adata, display_adata
from banksy_utils.filter_utils import normalize_total, filter_hvg, print_max_min
from banksy_utils.filter_utils import filter_cells
from banksy.main import median_dist_to_nearest_neighbour
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy_utils.umap_pca import pca_umap


def train_banksy_models(dataset,
                        model_name,
                        reference_batches,
                        st_data_gold_folder_path,
                        cell_type_key,
                        spatial_key,
                        mapping_entity_key,
                        condition_key,
                        adj_key,
                        latent_key,
                        benchmarking_folder_path,
                        adata_new=None,
                        n_start_run=1,
                        n_end_run=1,
                        n_neighbor_list=[4],
                        filter_genes: bool=False,
                        n_svg: int=3000):
    # Configure figure folder path
    # dataset_figure_folder_path = f"{figure_folder_path}/{dataset}/single_sample_method_benchmarking/" \
                                #  f"{model_name}/{current_timestamp}"
    # os.makedirs(dataset_figure_folder_path, exist_ok=True)

    # Create new adata to store results from training runs in storage-efficient way
    if adata_new is None:  
        adata_batch_list = []
        if reference_batches is not None:
            for batch in reference_batches:
                adata_batch = ad.read_h5ad(
                    f"{st_data_gold_folder_path}/{dataset}_{batch}.h5ad")
                adata_batch.obs[mapping_entity_key] = "reference"
                adata_batch_list.append(adata_batch)
            adata_original = ad.concat(adata_batch_list, join="inner")
        else:
            adata_original = ad.read_h5ad(f"{st_data_gold_folder_path}/{dataset}.h5ad")
            adata_original.obs[mapping_entity_key] = "reference"

        adata_new = sc.AnnData(sp.csr_matrix(
            (adata_original.shape[0], adata_original.shape[1]),
            dtype=np.float32))
        adata_new.var_names = adata_original.var_names
        adata_new.obs_names = adata_original.obs_names
        adata_new.obs["cell_types"] = adata_original.obs[cell_type_key].values
        adata_new.obsm["spatial"] = adata_original.obsm["spatial"]
        # adata_new.obs[condition_key] = adata_original.obs[condition_key]
        adata_new.obs[mapping_entity_key] = adata_original.obs[mapping_entity_key] 
        del(adata_original)

    model_seeds = list(range(1))
    for run_number, n_neighbors in zip(np.arange(n_start_run, n_end_run+1), n_neighbor_list):
        if reference_batches is not None:
            adata_batch_list = []
            for batch in reference_batches:
                print(f"Processing batch {batch}...")
                print("Loading data...")
                adata = ad.read_h5ad(
                    f"{st_data_gold_folder_path}/{dataset}_{batch}.h5ad")
                adata.obs[mapping_entity_key] = "reference"

                if filter_genes:

                    # Compute (separate) spatial neighborhood graphs
                    sq.gr.spatial_neighbors(adata,
                                            coord_type="generic",
                                            spatial_key=spatial_key,
                                            n_neighs=n_neighbors)
                    # Make adjacency matrix symmetric
                    adata.obsp[adj_key] = (
                        adata.obsp[adj_key].maximum(
                            adata.obsp[adj_key].T))

                    sc.pp.filter_genes(adata,
                                       min_cells=0)
                    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
                    sv_genes = adata.uns["moranI"].index[:n_svg].tolist()
                    adata.var["spatially_variable"] = adata.var_names.isin(sv_genes)
                    adata = adata[:, adata.var["spatially_variable"] == True].copy()
                    print(f"Keeping {len(adata.var_names)} spatially variable genes.")

                start_time = time.time()

                # Set default model hyperparams
                max_m = 1 # use both mean and AFT
                nbr_weight_decay = "scaled_gaussian" # can also choose "reciprocal", "uniform" or "ranked"
                lambda_list = [0.8]
                pca_dims = [20]

                # Define spatial coordinates
                adata.obs["spatial_x"] = adata.obsm['spatial'][:, 0]
                adata.obs["spatial_y"] = adata.obsm['spatial'][:, 1]

                banksy_dict = initialize_banksy(
                    adata,
                    ("spatial_x", "spatial_y", "spatial"),
                    n_neighbors,
                    nbr_weight_decay=nbr_weight_decay,
                    max_m=max_m,
                    plt_edge_hist=False,
                    plt_nbr_weights=False,
                    plt_agf_angles=False, # takes long time to plot
                    plt_theta=False)

                banksy_dict, banksy_matrix = generate_banksy_matrix(
                    adata,
                    banksy_dict,
                    lambda_list,
                    max_m)

                pca_umap(
                    banksy_dict,
                    pca_dims = pca_dims,
                    add_umap = True,
                    plt_remaining_var = False)

                adata.obsm[latent_key] = banksy_dict[nbr_weight_decay][lambda_list[0]]["adata"].obsm["reduced_pc_20"]
                adata_batch_list.append(adata)
            adata_original = ad.concat(adata_batch_list, join="outer")

            # Integrate with harmony
            sc.external.pp.harmony_integrate(adata_original, "batch", basis=latent_key)

            # Measure time for model training
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Duration of model training in run {run_number}: "
                f"{int(hours)} hours, {int(minutes)} minutes and {int(seconds)} seconds.")
            adata_original.uns[f"{model_name}_model_training_duration_run{run_number}"] = (
                elapsed_time)
            adata_new.uns[f"{model_name}_model_training_duration_run{run_number}"] = adata_original.uns[f"{model_name}_model_training_duration_run{run_number}"]

            adata_new.obsm[latent_key + f"_run{run_number}"] = adata_original.obsm['X_pca_harmony']

            adata_new.write(f"{benchmarking_folder_path}/{dataset}_{model_name}.h5ad")  
        else:
            adata = ad.read_h5ad(f"{st_data_gold_folder_path}/{dataset}.h5ad")

            if filter_genes:
                # Compute (separate) spatial neighborhood graphs
                sq.gr.spatial_neighbors(adata,
                                        coord_type="generic",
                                        spatial_key=spatial_key,
                                        radius=27.5)

                # Make adjacency matrix symmetric
                adata.obsp[adj_key] = (
                    adata.obsp[adj_key].maximum(
                        adata.obsp[adj_key].T))

                sc.pp.filter_genes(adata,
                                   min_cells=0)
                sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
                sv_genes = adata.uns["moranI"].index[:n_svg].tolist()
                adata.var["spatially_variable"] = adata.var_names.isin(sv_genes)
                adata = adata[:, adata.var["spatially_variable"] == True].copy()
                print(f"Keeping {len(adata.var_names)} spatially variable genes.")

            start_time = time.time()

            # Set default model hyperparams
            max_m = 1 # use both mean and AFT
            nbr_weight_decay = "scaled_gaussian" # can also choose "reciprocal", "uniform" or "ranked"
            lambda_list = [0.8]
            pca_dims = [20]

            # Define spatial coordinates
            adata.obs["spatial_x"] = adata.obsm['spatial'][:, 0]
            adata.obs["spatial_y"] = adata.obsm['spatial'][:, 1]

            banksy_dict = initialize_banksy(
                adata,
                ("spatial_x", "spatial_y", "spatial"),
                n_neighbors,
                nbr_weight_decay=nbr_weight_decay,
                max_m=max_m,
                plt_edge_hist=False,
                plt_nbr_weights=False,
                plt_agf_angles=False, # takes long time to plot
                plt_theta=False)

            banksy_dict, banksy_matrix = generate_banksy_matrix(
                adata,
                banksy_dict,
                lambda_list,
                max_m)

            pca_umap(
                banksy_dict,
                pca_dims = pca_dims,
                add_umap = True,
                plt_remaining_var = False)

            adata.obsm[latent_key] = banksy_dict[nbr_weight_decay][lambda_list[0]]["adata"].obsm["reduced_pc_20"]

            # Measure time for model training
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Duration of model training in run {run_number}: "
              f"{int(hours)} hours, {int(minutes)} minutes and {int(seconds)} seconds.")
            adata_new.uns[f"{model_name}_model_training_duration_run{run_number}"] = (
                elapsed_time)

            # Store latent representation
            adata_new.obsm[latent_key + f"_run{run_number}"] = adata.obsm[latent_key]

            # Store intermediate adata to disk
            adata_new.write(f"{benchmarking_folder_path}/{dataset}_{model_name}.h5ad") 

            # Free memory
            del(adata)
            # gc.collect()

    # Store final adata to disk
    adata_new.write(f"{benchmarking_folder_path}/{dataset}_{model_name}.h5ad")


def main():
    model_name = "banksy"
    latent_key = f"{model_name}_latent"
    lambda_list = [0.2]  # list of lambda parameters
    mapping_entity_key = "reference"
    condition_key = "batch"
    spatial_key = "spatial"
    adj_key = "spatial_connectivities"    

    st_data_silver_folder_path = "/lustre/scratch126/cellgen/team361/am84/TERRA/datasets/am84/st_data/silver/sim1_1105fts_10000locs"
    st_data_results_folder_path = "./temp/datasets/st_data/results" 
    benchmarking_folder_path = "./temp/artifacts/sample_integration_method_benchmarking"

    # Create required directories
    os.makedirs(st_data_results_folder_path, exist_ok=True)
    os.makedirs(benchmarking_folder_path, exist_ok=True)

    train_banksy_models(st_data_gold_folder_path=st_data_silver_folder_path,
                    dataset="sim1_1105fts_10000locs",
                    model_name=model_name,
                    spatial_key=spatial_key,
                    adj_key=adj_key,
                    latent_key=latent_key,
                    mapping_entity_key=mapping_entity_key,
                    condition_key=condition_key,
                    benchmarking_folder_path=benchmarking_folder_path,
                    reference_batches=None,
                    cell_type_key="cell_types",
                    adata_new=None,
                    n_start_run=1,
                    n_end_run=8,
                    n_neighbor_list=[4])


if __name__ == "__main__":
    main()