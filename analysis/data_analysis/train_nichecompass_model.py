#!/usr/bin/env python
# coding: utf-8

###############################################################################
# NicheCompass Model Training #
###############################################################################

###############################################################################
## 1. Setup ##
###############################################################################

###############################################################################
### 1.1 Import Libraries ###
###############################################################################

import argparse
import os
import sys
from datetime import datetime

import anndata as ad
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq

from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                add_multimodal_mask_to_adata,
                                extract_gp_dict_from_collectri_tf_network,
                                extract_gp_dict_from_mebocost_ms_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps,
                                filter_and_combine_gp_dict_gps_v2,
                                generate_multimodal_mapping_dict,
                                get_gene_annotations,
                                get_unique_genes_from_gp_dict)


def main(args):
    print(f"Dataset: {args.dataset}")

    if args.cat_covariates_embeds_injection == [None]:
        args.cat_covariates_embeds_injection = []
    if args.cat_covariates_keys == [None]:
        args.cat_covariates_keys = None
    if args.cat_covariates_no_edges == [None]:
        args.cat_covariates_no_edges = None
    if args.cat_covariates_embeds_nums == [None]:
        args.cat_covariates_embeds_nums = None
        
    if args.include_atac_modality:
        save_adata_atac = True
    else:
        save_adata_atac = False

    # Get time of script execution for timestamping saved artifacts
    now = datetime.now()
    current_timestamp = now.strftime("%d%m%Y_%H%M%S")

    print(f"Run timestamp: {current_timestamp + args.timestamp_suffix}.")
    print("Script arguments:")
    print(sys.argv)

    # Set mlflow experiment
    if args.mlflow_tracking:
        experiment = mlflow.set_experiment(f"{args.dataset}_{args.model_label}")
        mlflow_experiment_id = experiment.experiment_id
        
        # Track params that are not part of model
        mlflow.log_param("timestamp", current_timestamp + args.timestamp_suffix)
        mlflow.log_param("nichenet_keep_target_genes_ratio",
                        args.nichenet_keep_target_genes_ratio)
        mlflow.log_param("nichenet_max_n_target_genes_per_gp",
                        args.nichenet_max_n_target_genes_per_gp)
        mlflow.log_param("include_nichenet_gps",
                        args.include_nichenet_gps)
        mlflow.log_param("include_mebocost_gps",
                        args.include_mebocost_gps)
        mlflow.log_param("include_collectri_gps",
                        args.include_collectri_gps)
        mlflow.log_param("include_brain_marker_gps",
                        args.include_brain_marker_gps)
        mlflow.log_param("species",
                        args.species)
        mlflow.log_param("gp_filter_mode",
                        args.gp_filter_mode)
        mlflow.log_param("combine_overlap_gps",
                        args.combine_overlap_gps)
        mlflow.log_param("overlap_thresh_source_genes",
                        args.overlap_thresh_source_genes)
        mlflow.log_param("overlap_thresh_target_genes",
                        args.overlap_thresh_target_genes)
        mlflow.log_param("overlap_thresh_genes",
                        args.overlap_thresh_genes)
        mlflow.log_param("add_fc_gps_instead_of_gp_dict_gps",
                        args.add_fc_gps_instead_of_gp_dict_gps)
        mlflow.log_param("n_neighbors",
                        args.n_neighbors)
        mlflow.log_param("filter_genes",
                        args.filter_genes)
        mlflow.log_param("n_hvg",
                        args.n_hvg)
        mlflow.log_param("n_svg",
                        args.n_svg)
        mlflow.log_param("n_svp",
                        args.n_svp)
        mlflow.log_param("include_atac_modality",
                        args.include_atac_modality)
        if args.include_atac_modality:
            mlflow.log_param("filter_peaks", args.filter_peaks)
            mlflow.log_param("min_cell_peak_thresh_ratio",
                            args.min_cell_peak_thresh_ratio)
    else:
        mlflow_experiment_id = None

    ###############################################################################
    ### 1.3 Configure Paths and Create Directories ###
    ###############################################################################

    gp_data_folder_path = f"{args.data_folder_path}/gene_programs"
    ga_data_folder_path = f"{args.data_folder_path}/gene_annotations"
    nichenet_lr_network_file_path = gp_data_folder_path + \
                                    "/nichenet_lr_network_v2_" \
                                    f"{args.species}.csv"
    nichenet_ligand_target_matrix_file_path = gp_data_folder_path + \
                                            "/nichenet_ligand_target_matrix_" \
                                            f"v2_{args.species}.csv"
    omnipath_lr_network_file_path = gp_data_folder_path + \
                                    "/omnipath_lr_network.csv"
    collectri_tf_network_file_path = gp_data_folder_path + \
                                    f"/collectri_tf_network_{args.species}.csv"
    gene_orthologs_mapping_file_path = ga_data_folder_path + \
                                    "/human_mouse_gene_orthologs.csv"
    gtf_file_path = ga_data_folder_path + \
                    "/gencode.vM32.chr_patch_hapl_scaff.annotation.gtf.gz"
    os.makedirs(args.model_folder_path, exist_ok=True)
    os.makedirs(args.result_folder_path, exist_ok=True)

    ###############################################################################
    ## 2. Gene Program Mask ##
    ###############################################################################

    print("\nPreparing the gene program mask...")
    if args.include_omnipath_gps:
        if args.use_new_gp_mask:
            min_curation_effort = 2
        else:
            min_curation_effort = 0

        # OmniPath gene programs
        omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
            species=args.species,
            min_curation_effort=min_curation_effort,
            load_from_disk=False,
            save_to_disk=True,
            lr_network_file_path=omnipath_lr_network_file_path,
            gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
            plot_gp_gene_count_distributions=False)

        omnipath_genes = get_unique_genes_from_gp_dict(
            gp_dict=omnipath_gp_dict,
            retrieved_gene_entities=["sources", "targets"])

        # Combine gene programs into one dictionary
        if args.use_new_gp_mask:
            gp_dicts = [omnipath_gp_dict]
        combined_gp_dict = dict(omnipath_gp_dict)
    else:
        if args.use_new_gp_mask:
            gp_dicts = []
        combined_gp_dict = dict()    

    if args.include_nichenet_gps:
        # NicheNet gene programs
        nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
            species=args.species,
            version="v2",
            keep_target_genes_ratio=args.nichenet_keep_target_genes_ratio,
            max_n_target_genes_per_gp=args.nichenet_max_n_target_genes_per_gp,
            load_from_disk=False,
            save_to_disk=True,
            lr_network_file_path=nichenet_lr_network_file_path,
            ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
            gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
            plot_gp_gene_count_distributions=False)

        nichenet_lr_genes = get_unique_genes_from_gp_dict(
            gp_dict=nichenet_gp_dict,
            retrieved_gene_categories=["ligand", "receptor"])
        
        if args.use_new_gp_mask:
            gp_dicts.append(nichenet_gp_dict)
        combined_gp_dict.update(nichenet_gp_dict)

    #if args.filter_genes:
    #    # Get gene program relevant genes
    #    gp_relevant_genes = [gene.upper() for gene in list(set(
    #        omnipath_genes + nichenet_lr_genes))]

    # Mebocost gene programs
    if args.include_mebocost_gps:
        mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
        dir_path=f"{gp_data_folder_path}/metabolite_enzyme_sensor_gps",
        species=args.species,
        plot_gp_gene_count_distributions=False)
        
        mebocost_genes = get_unique_genes_from_gp_dict(
            gp_dict=mebocost_gp_dict,
            retrieved_gene_entities=["sources", "targets"])

        if args.use_new_gp_mask:
            gp_dicts.append(mebocost_gp_dict)
        combined_gp_dict.update(mebocost_gp_dict)
        
        #if args.filter_genes:
            # Update gene program relevant genes
            #gp_relevant_genes = list(set(gp_relevant_genes + mebocost_genes))
            
    if args.include_collectri_gps:
        collectri_gp_dict = extract_gp_dict_from_collectri_tf_network(
            species=args.species,
            tf_network_file_path=collectri_tf_network_file_path,
            load_from_disk=True,
            save_to_disk=False,
            plot_gp_gene_count_distributions=False)
        
        collectri_genes = get_unique_genes_from_gp_dict(
            gp_dict=collectri_gp_dict,
            retrieved_gene_entities=["sources", "targets"],
            retrieved_gene_categories=["tf"])
        
        if args.use_new_gp_mask:
            gp_dicts.append(collectri_gp_dict)
        combined_gp_dict.update(collectri_gp_dict)
        
        #if args.filter_genes:
            # Update gene program relevant genes (with only tf genes)
            #gp_relevant_genes = list(set(gp_relevant_genes + collectri_genes))

    if args.use_new_gp_mask:
        combined_new_gp_dict = filter_and_combine_gp_dict_gps_v2(
            gp_dicts,
            verbose=True)
    else:
        # Filter and combine gene programs
        combined_new_gp_dict = filter_and_combine_gp_dict_gps(
            gp_dict=combined_gp_dict,
            gp_filter_mode=args.gp_filter_mode,
            combine_overlap_gps=args.combine_overlap_gps,
            overlap_thresh_source_genes=args.overlap_thresh_source_genes,
            overlap_thresh_target_genes=args.overlap_thresh_target_genes,
            overlap_thresh_genes=args.overlap_thresh_genes,
            verbose=False)

    print("Number of gene programs before filtering and combining: "
        f"{len(combined_gp_dict)}.")
    print(f"Number of gene programs after filtering and combining: "
        f"{len(combined_new_gp_dict)}.")

    ###############################################################################
    ## 3. Data ##
    ###############################################################################

    ###############################################################################
    ### 3.1 Load Data & Compute Spatial Neighbor Graph ###
    ###############################################################################

    # RNA-seq data
    adata_batch_list = []
    if len(args.batch_file_paths) > 1:
        for batch_file_path in args.batch_file_paths:
            print(f"\nProcessing batch with path {batch_file_path}...")
            print("Loading data...")
            adata_batch = ad.read_h5ad(
                f"{args.dataset_folder_path}/{batch_file_path}.h5ad")
            print("Computing spatial neighborhood graph...")
            # Compute spatial neighborhood graphs
            if args.graph_type == "radius":
                sq.gr.spatial_neighbors(adata_batch,
                                        coord_type="generic",
                                        spatial_key=args.spatial_key,
                                        radius=args.radius)
            elif args.graph_type == "knn":
                sq.gr.spatial_neighbors(adata_batch,
                                        coord_type="generic",
                                        spatial_key=args.spatial_key,
                                        n_neighs=args.n_neighbors)
            # Make adjacency matrix symmetric
            adata_batch.obsp[args.adj_key] = (
                adata_batch.obsp[args.adj_key].maximum(
                    adata_batch.obsp[args.adj_key].T))
            adata_batch_list.append(adata_batch)
        adata = ad.concat(adata_batch_list, join="inner")

        # Combine spatial neighborhood graphs as disconnected components
        batch_connectivities = []
        len_before_batch = 0
        for i in range(len(adata_batch_list)):
            if i == 0: # first batch
                after_batch_connectivities_extension = sp.csr_matrix(
                    (adata_batch_list[0].shape[0],
                    (adata.shape[0] -
                    adata_batch_list[0].shape[0])))
                batch_connectivities.append(sp.hstack(
                    (adata_batch_list[0].obsp[args.adj_key],
                    after_batch_connectivities_extension)))
            elif i == (len(adata_batch_list) - 1): # last batch
                before_batch_connectivities_extension = sp.csr_matrix(
                    (adata_batch_list[i].shape[0],
                    (adata.shape[0] -
                    adata_batch_list[i].shape[0])))
                batch_connectivities.append(sp.hstack(
                    (before_batch_connectivities_extension,
                    adata_batch_list[i].obsp[args.adj_key])))
            else: # middle batches
                before_batch_connectivities_extension = sp.csr_matrix(
                    (adata_batch_list[i].shape[0], len_before_batch))
                after_batch_connectivities_extension = sp.csr_matrix(
                    (adata_batch_list[i].shape[0],
                    (adata.shape[0] -
                    adata_batch_list[i].shape[0] -
                    len_before_batch)))
                batch_connectivities.append(sp.hstack(
                    (before_batch_connectivities_extension,
                    adata_batch_list[i].obsp[args.adj_key],
                    after_batch_connectivities_extension)))
            len_before_batch += adata_batch_list[i].shape[0]
        connectivities = sp.vstack(batch_connectivities)
        adata.obsp[args.adj_key] = connectivities
    else:
        adata = ad.read_h5ad(
                f"{args.dataset_folder_path}/{args.batch_file_paths[0]}.h5ad")
        # Compute (separate) spatial neighborhood graphs
        if args.graph_type == "radius":
            sq.gr.spatial_neighbors(adata,
                                    coord_type="generic",
                                    spatial_key=args.spatial_key,
                                    radius=args.radius)
        elif args.graph_type == "knn":
            sq.gr.spatial_neighbors(adata,
                                    coord_type="generic",
                                    spatial_key=args.spatial_key,
                                    n_neighs=args.n_neighbors)
        # Make adjacency matrix symmetric
        adata.obsp[args.adj_key] = (
            adata.obsp[args.adj_key].maximum(
                adata.obsp[args.adj_key].T))
    adata.obs[args.mapping_entity_key] = "reference"
    adata.obs_names_make_unique()

    # ATAC data (if included)
    if args.include_atac_modality:
        adata_atac_batch_list = []
        if len(args.batch_file_paths) > 1:
            for batch_file_path in args.batch_file_paths:
                print(f"\nProcessing ATAC batch with path {batch_file_path}...")
                print("Loading data...")
                adata_atac_batch = ad.read_h5ad(
                    f"{args.dataset_folder_path}/{batch_file_path}_atac.h5ad")
            adata_atac = ad.concat(adata_atac_batch_list, join="inner")
        else:
            adata_atac = ad.read_h5ad(
                f"{args.dataset_folder_path}/{args.batch_file_paths[0]}_atac.h5ad")
        adata_atac.obs_names_make_unique()
    else:
        adata_atac = None
        
    ###############################################################################
    ### 3.2 Filter Omics Features ###
    ###############################################################################

    # RNA-seq data
    if args.filter_genes:
        print("\nFiltering genes...")
        # Filter genes and only keep gp relevant, the 'n_svg' spatially variable,
        # and 'n_hvg' highly variable genes
        gp_dict_genes = get_unique_genes_from_gp_dict(
            gp_dict=combined_new_gp_dict,
            retrieved_gene_entities=["sources", "targets"])
        print(f"Starting with {len(adata.var_names)} genes.")
        min_cells = int(adata.shape[0] * args.min_cell_gene_thresh_ratio)
        sc.pp.filter_genes(adata,
                        min_cells=min_cells)
        print(f"Keeping {len(adata.var_names)} genes after filtering genes with "
            "expression in {} cells.")

        if args.counts_key is not None:
            hvg_layer = args.counts_key
            if (adata.layers[args.counts_key].astype(int).astype(np.float32).sum() == 
            adata.layers[args.counts_key].sum()): # raw counts
                hvg_flavor = "seurat_v3"
            else: # log normalized counts
                hvg_flavor = "seurat"
        else:
            hvg_layer = None
            if adata.X.astype(int).astype(np.float32).sum() == adata.X.sum():
            # raw counts
                hvg_flavor = "seurat_v3"
            else: # log normalized counts
                hvg_flavor = "seurat"

        if args.n_hvg != 0:
            sc.pp.highly_variable_genes(
                adata,
                layer=hvg_layer,
                n_top_genes=args.n_hvg,
                flavor=hvg_flavor,
                batch_key=args.condition_key,
                subset=False)
        else:
            adata.var["highly_variable"] = False
        
        # Filter spatially variable genes
        if args.n_svg != 0:
            sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
            sv_genes = adata.uns["moranI"].index[:args.n_svg].tolist()
            adata.var["spatially_variable"] = adata.var_names.isin(sv_genes)
        else:
            adata.var["spatially_variable"] = False

        gp_relevant_genes = [] # remove gp relevant genes
        adata.var["gp_relevant"] = (
            adata.var.index.str.upper().isin(gp_relevant_genes))
        adata.var["keep_gene"] = (adata.var["gp_relevant"] | 
                                adata.var["highly_variable"] |
                                adata.var["spatially_variable"])
        adata = adata[:, adata.var["keep_gene"] == True]
        print(f"Keeping {len(adata.var_names)} spatially variable, highly "
            "variable or gene program relevant genes.")
        #adata = (adata[:, adata.var_names[adata.var_names.str.upper().isin(
        #            [gene.upper() for gene in gp_dict_genes])].sort_values()])
        #print(f"Keeping {len(adata.var_names)} genes after filtering genes not in "
        #      "gp dict.")

    # ATAC data (if included)
    if args.include_atac_modality:
        if args.filter_peaks:
            print("\nFiltering peaks...")
            print(f"Starting with {len(adata_atac.var_names)} peaks.")
            # Filter out peaks that are rarely detected
            min_cells = int(adata_atac.shape[0] * args.min_cell_peak_thresh_ratio)
            sc.pp.filter_genes(adata_atac, min_cells=min_cells)
            print(f"Keeping {len(adata_atac.var_names)} peaks after filtering "
                " peaks with counts in less than "
                f"{int(adata_atac.shape[0] * args.min_cell_peak_thresh_ratio)} "
                "cells.")
            
        # Filter spatially variable peaks
        if args.n_svp > 0:
            adata_atac.obsp["spatial_connectivities"] = (
                adata.obsp["spatial_connectivities"])
            adata_atac.obsp["spatial_distances"] = adata.obsp["spatial_distances"]

            sq.gr.spatial_autocorr(adata_atac,
                                mode="moran",
                                genes=adata_atac.var_names)
            sv_peaks = adata_atac.uns["moranI"].index[:args.n_svp].tolist()
            adata_atac.var["spatially_variable"] = (
                adata_atac.var_names.isin(sv_peaks))
            adata_atac = adata_atac[:,
                                    adata_atac.var["spatially_variable"] == True]
            print(f"Keeping {len(adata_atac.var_names)} peaks after filtering "
                "spatially variable peaks.")

    ###############################################################################
    ### 3.3 Annotate Genes (If ATAC Modality Incl.) ###
    ###############################################################################

    if args.include_atac_modality:
        adata, adata_atac = get_gene_annotations(
            adata=adata,
            adata_atac=adata_atac,
            gtf_file_path=gtf_file_path)

    ###############################################################################
    ### 3.4 Add Gene Program Mask to Data ###
    ###############################################################################

    # Add the gene program dictionary as binary masks to the adata for model 
    # training
    add_gps_from_gp_dict_to_adata(
        gp_dict=combined_new_gp_dict,
        adata=adata,
        genes_uppercase=True,
        gp_targets_mask_key=args.gp_targets_mask_key,
        gp_sources_mask_key=args.gp_sources_mask_key,
        gp_names_key=args.gp_names_key,
        min_genes_per_gp=1,
        min_source_genes_per_gp=0,
        min_target_genes_per_gp=0,
        max_genes_per_gp=None,
        max_source_genes_per_gp=None,
        max_target_genes_per_gp=None,
        filter_genes_not_in_masks=False,
        add_fc_gps_instead_of_gp_dict_gps=args.add_fc_gps_instead_of_gp_dict_gps)

    ###############################################################################
    ### 3.5 Add Chromatin Accessibility Mask to Data (If ATAC Modality Incl.) ###
    ###############################################################################

    if args.include_atac_modality:
        gene_peak_dict = generate_multimodal_mapping_dict(
            adata,
            adata_atac)

        adata, adata_atac = add_multimodal_mask_to_adata(
            adata=adata,
            adata_atac=adata_atac,
            gene_peak_mapping_dict=gene_peak_dict)

        print(f"Keeping {len(adata_atac.var_names)} peaks after filtering peaks "
            "with no matching genes in gp mask.")

    ###############################################################################
    ### 4. Model ###
    ###############################################################################

    ###############################################################################
    ## 4.1 Initialize, Train & Save Model ##
    ###############################################################################

    print("\nTraining model...")
    # Initialize model
    model = NicheCompass(adata,
                        adata_atac,
                        counts_key=args.counts_key,
                        adj_key=args.adj_key,
                        cat_covariates_embeds_injection=args.cat_covariates_embeds_injection,
                        cat_covariates_keys=args.cat_covariates_keys,
                        cat_covariates_no_edges=args.cat_covariates_no_edges,
                        cat_covariates_embeds_nums=args.cat_covariates_embeds_nums,
                        gp_names_key=args.gp_names_key,
                        active_gp_names_key=args.active_gp_names_key,
                        gp_targets_mask_key=args.gp_targets_mask_key,
                        gp_sources_mask_key=args.gp_sources_mask_key,
                        latent_key=args.latent_key,
                        n_addon_gp=args.n_addon_gp,
                        active_gp_thresh_ratio=args.active_gp_thresh_ratio,
                        gene_expr_recon_dist=args.gene_expr_recon_dist,
                        n_fc_layers_encoder=args.n_fc_layers_encoder,
                        n_layers_encoder=args.n_layers_encoder,
                        conv_layer_encoder=args.conv_layer_encoder,
                        n_hidden_encoder=args.n_hidden_encoder,
                        log_variational=args.log_variational,
                        node_label_method=args.node_label_method,
                        seed=args.seed)

    # Train model
    model.train(n_epochs=args.n_epochs,
                n_epochs_all_gps=args.n_epochs_all_gps,
                n_epochs_no_cat_covariates_contrastive=args.n_epochs_no_cat_covariates_contrastive,
                lr=args.lr,
                lambda_edge_recon=args.lambda_edge_recon,
                lambda_gene_expr_recon=args.lambda_gene_expr_recon,
                lambda_chrom_access_recon=args.lambda_chrom_access_recon,
                lambda_cat_covariates_contrastive=args.lambda_cat_covariates_contrastive,
                contrastive_logits_pos_ratio=args.contrastive_logits_pos_ratio,
                contrastive_logits_neg_ratio=args.contrastive_logits_neg_ratio,
                lambda_group_lasso=args.lambda_group_lasso,
                lambda_l1_masked=args.lambda_l1_masked,
                lambda_l1_addon=args.lambda_l1_addon,
                l1_targets_categories=args.l1_targets_categories,
                l1_sources_categories=args.l1_sources_categories,
                edge_batch_size=args.edge_batch_size,
                node_batch_size=args.node_batch_size,
                edge_val_ratio=args.edge_val_ratio,
                node_val_ratio=args.node_val_ratio,
                n_sampled_neighbors=args.n_sampled_neighbors,
                mlflow_experiment_id=mlflow_experiment_id,
                seed=args.seed,
                verbose=True)

    print("\nFinished model training...")

    if args.compute_knn_graph:
        print("\nComputing neighbor graph...")
        # Use latent representation for UMAP generation
        sc.pp.neighbors(model.adata,
                        use_rep=args.latent_key,
                        key_added=args.latent_key)

        print("\nComputing UMAP embedding...")
        sc.tl.umap(model.adata,
                   neighbors_key=args.latent_key)

    # Store adata to disk
    model.adata.write(
        f"{args.result_folder_path}/{args.dataset}_{args.model_label}.h5ad")

    print("\nSaving model...")
    # Save trained model
    os.makedirs(f"{args.model_folder_path}/{args.dataset}", exist_ok=True)
    model.save(dir_path=f"{args.model_folder_path}/{args.dataset}",
               overwrite=True,
               save_adata=True,
               adata_file_name=f"{args.dataset}_{args.model_label}.h5ad",
               save_adata_atac=save_adata_atac,
               adata_atac_file_name=f"{args.dataset}_{args.model_label}_atac.h5ad")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')

    def none_or_value(value):
        if value == "None":
            return None
        return value

    def none_or_int(value):
        if value == "None":
            return None
        return int(value)

    def none_or_bool(value):
        if value == "None":
            return None
        return value == "True"

    ###############################################################################
    ### 1.2 Define Parameters ###
    ###############################################################################
    # Gene program mask
    parser.add_argument(
        "--species",
        type=str,
        default="human",
        help="Species that is used for the retrieval of gene programs.")
    parser.add_argument(
        "--nichenet_keep_target_genes_ratio",
        type=float,
        default=0.01,
        help="Ratio how many of the overall top scored target genes are kept.")
    parser.add_argument(
        "--nichenet_max_n_target_genes_per_gp",
        type=int,
        default=25344,
        help="After this number of genes the genes are clipped from the gp.")
    parser.add_argument(
        "--include_omnipath_gps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Indicator whether to include Omnipath gene programs.")
    parser.add_argument(
        "--include_nichenet_gps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Indicator whether to include NicheNet gene programs.")
    parser.add_argument(
        "--include_mebocost_gps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Indicator whether to include MEBOCOST gene programs.")
    parser.add_argument(
        "--include_collectri_gps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Indicator whether to include CollecTRI gene programs.")
    parser.add_argument(
        "--gp_filter_mode",
        type=str,
        default="subset",
        help="Which kind of gene programs are filtered.")
    parser.add_argument(
        "--combine_overlap_gps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Indicator whether to combine overlapping gene programs.")
    parser.add_argument(
        "--overlap_thresh_source_genes",
        type=float,
        default=0.9,
        help="Threshold for source genes above which gene programs are combined.")
    parser.add_argument(
        "--overlap_thresh_target_genes",
        type=float,
        default=0.9,
        help="Threshold for target genes above which gene programs are combined.")
    parser.add_argument(
        "--overlap_thresh_genes",
        type=float,
        default=0.9,
        help="Threshold for overall genes above which gene programs are combined.")
    parser.add_argument(
        "--add_fc_gps_instead_of_gp_dict_gps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Indicator whether to add a fc gp mask instead of a gp dict gp mask."
            "This is used for ablation studies.")
    parser.add_argument(
        "--use_new_gp_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Indicator whether new gene program mask should be used.")

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        help="Input dataset name.")
    parser.add_argument(
        "--counts_key",
        type=none_or_value,
        default=None,
        help="s. NicheCompass class signature.")
    parser.add_argument(
        "--condition_key",
        type=str,
        default="batch",
        help="s. NicheCompass class signature.")
    parser.add_argument(
        "--cat_covariates_keys",
        nargs='+',
        type=none_or_value,
        default=None,
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--cat_covariates_no_edges",
        nargs='+',
        type=none_or_bool,
        default=None,
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--graph_type",
        type=str,
        default="knn",
        help="Determines how the spatial neighborhood graphs are computed."
            "Either 'radius' or 'knn'.")
    parser.add_argument(
        "--radius",
        type=float,
        default=0.,
        help="Radius used to compute the spatial neighborhood graphs.")
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=10,
        help="Number of neighbors used to compute the spatial neighborhood graphs.")
    parser.add_argument(
        "--spatial_key",
        type=str,
        default="spatial",
        help="Key in `adata.obsm` where the spatial coordinates are stored.")
    parser.add_argument(
        "--adj_key",
        type=str,
        default="spatial_connectivities",
        help="s. NicheCompass class signature.")
    parser.add_argument(
        "--mapping_entity_key",
        type=str,
        default="mapping_entity",
        help="Key in `adata.obsm` where the mapping entities will be stored.")
    parser.add_argument(
        "--filter_genes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Indicator whether genes should be filtered.")
    parser.add_argument(
        "--n_hvg",
        type=int,
        default=0,
        help="Number of highly variable genes that are kept if `filter_genes` is "
            "`True`.")
    parser.add_argument(
        "--n_svg",
        type=int,
        default=3000,
        help="Number of spatially variable genes that are kept if `filter_genes` "
            "is `True`.")
    parser.add_argument(
        "--n_svp",
        type=int,
        default=3000,
        help="Number of spatially variable peaks that are kept if `filter_genes` "
            "is `True`.")
    parser.add_argument(
        "--gp_targets_mask_key",
        type=str,
        default="nichecompass_gp_targets",
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--gp_sources_mask_key",
        type=str,
        default="nichecompass_gp_sources",
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--gp_names_key",
        type=str,
        default="nichecompass_gp_names",
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--include_atac_modality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Indicator whether to include atac modality.")
    parser.add_argument(
        "--filter_peaks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Indicator whether peaks should be filtered.")
    parser.add_argument(
        "--min_cell_peak_thresh_ratio",
        type=float,
        default=0.0005,
        help="Ratio of cells in which peaks need to be detected to not be "
            "discarded.")
    parser.add_argument(
        "--min_cell_gene_thresh_ratio",
        type=float,
        default=0.0005,
        help="Ratio of cells in which peaks need to be detected to not be "
            "discarded.")

    # Model
    parser.add_argument(
        "--model_label",
        type=str,
        default="reference",
        help="Label of the model under which it will be saved.")
    parser.add_argument(
        "--active_gp_names_key",
        type=str,
        default="nichecompass_active_gp_names",
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--latent_key",
        type=str,
        default="nichecompass_latent",
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--n_addon_gp",
        type=int,
        default=100,
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--active_gp_thresh_ratio",
        type=float,
        default=0.05,
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--gene_expr_recon_dist",
        type=str,
        default="nb",
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--cat_covariates_embeds_injection",
        nargs='+',
        default=["gene_expr_decoder"],
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--cat_covariates_embeds_nums",
        nargs='+',
        type=none_or_int,
        default=None,
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--log_variational",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="s. NicheCompass class signature") # counts as input
    parser.add_argument(
        "--node_label_method",
        type=str,
        default="one-hop-norm",
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--n_layers_encoder",
        type=int,
        default=1,
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--n_fc_layers_encoder",
        type=int,
        default=1,
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--conv_layer_encoder",
        type=str,
        default="gatv2conv",
        help="s. NicheCompass class signature")
    parser.add_argument(
        "--n_hidden_encoder",
        type=none_or_int,
        default=None,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--n_epochs_all_gps",
        type=int,
        default=25,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--n_epochs_no_cat_covariates_contrastive",
        type=int,
        default=5,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--lambda_edge_recon",
        type=float,
        default=500000.,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--lambda_gene_expr_recon",
        type=float,
        default=300.,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--lambda_chrom_access_recon",
        type=float,
        default=100.,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--lambda_cat_covariates_contrastive",
        type=float,
        default=0.,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--contrastive_logits_pos_ratio",
        type=float,
        default=0.,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--contrastive_logits_neg_ratio",
        type=float,
        default=0.,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--lambda_group_lasso",
        type=float,
        default=0.,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--lambda_l1_masked",
        type=float,
        default=0.,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--lambda_l1_addon",
        type=float,
        default=30.,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--l1_targets_categories",
        nargs='+',
        type=none_or_value,
        default="target_gene",
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--l1_sources_categories",
        nargs='+',
        type=none_or_value,
        default=None,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--edge_batch_size",
        type=int,
        default=1024,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--node_batch_size",
        type=none_or_int,
        default=None,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--edge_val_ratio",
        type=float,
        default=0.1,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--node_val_ratio",
        type=float,
        default=0.1,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--n_sampled_neighbors",
        type=int,
        default=4,
        help="s. NicheCompass train method signature")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="s. NicheCompass class or train method signature")

    # Other
    parser.add_argument(
        "--mlflow_tracking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Indicator whether to use mlflow tracking.")
    parser.add_argument(
        "--compute_knn_graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Indicator whether to compute the knn graph.")
    parser.add_argument(
        "--timestamp_suffix",
        type=str,
        default="",
        help="Timestamp suffix for saving artifacts if timestamp overlaps")
    parser.add_argument(
        "--dataset_folder_path",
        type=str,
        default="/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver/xhp2000-1b_1p",
        help="Dataset folder path.")
    parser.add_argument('--batch_file_paths',
                        type=str,
                        nargs='+',
                        default=["Xenium_Pancreatic_Cancer_with_Xenium_Human_Multi-Tissue_and_Cancer_Panel.h5ad"],
                        help='Batch file paths.')
    parser.add_argument("--data_folder_path",
                        type=str,
                        default="/nfs/team361/sb75/nichejepa-reproducibility/artifacts/nichecompass/data",
                        help="Data folder path.")
    parser.add_argument("--model_folder_path",
                        type=str,
                        default="/nfs/team361/sb75/nichejepa-reproducibility/artifacts/nichecompass/models",
                        help="Model folder path.")
    parser.add_argument("--result_folder_path",
                        type=str,
                        default="/nfs/team361/sb75/nichejepa-reproducibility/artifacts/nichecompass/results",
                        help="Result folder path.")

    args = parser.parse_args()

    main(args)
