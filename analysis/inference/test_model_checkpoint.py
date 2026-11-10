import argparse
import sys
sys.path.append("../../utils")

import logging
import math
import os
import re
import warnings
import pickle
import random
from collections import defaultdict
from datetime import datetime
from itertools import chain, compress
from pathlib import Path

import anndata as ad
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scib_metrics
import seaborn as sns
import squidpy as sq
from datasets import concatenate_datasets, load_from_disk
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from terra.inference import get_average_gene_embed, get_gene_embed, infer
from terra.datasets import prepare_dataset
from terra.utils.config import create_params_from_YAML_wandb_config
from terra.utils.distributed import init_distributed
from terra.utils.embedding import collect_adata_from_folder
from terra.utils.evaluation import clustering_metrics, classification_metrics

from analysis_utils import *


def main(args):
    ### Define paths ###
    global artifacts_folder_path
    global folder_path
    artifacts_folder_path = '/nfs/team361/sb75/terra-reproducibility/artifacts'
    datasets_folder_path = '/nfs/team361/sb75/DATASETS'
    token_dict_path = f'{datasets_folder_path}/tokenizer/token_dictionary_homo_sapiens.pkl'
    folder_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}'
    ####################

    ### Set up figure params ###
    os.makedirs(f'{folder_path}/figures', exist_ok=True)
    plt.rcParams['font.size'] = 5
    plt.rcParams['text.usetex'] = False
    plt.rcParams['svg.fonttype'] = 'none'

    # Improve reproducibility of UMAP and Leiden
    os.environ['NUMBA_CPU_NAME'] = 'generic'

    sc.settings.figdir = f'{folder_path}/figures'

    sc.settings.set_figure_params(
        dpi=300,
        dpi_save=300,
        figsize=(6, 4),
        facecolor='white',
        fontsize=10
    )
    ############################


    ### Set up logger ###
    rank=0
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    #####################


    ### Load config ###
    params = create_params_from_YAML_wandb_config(
        f'{folder_path}/params.yaml', logger)
    trained_tokenized_data_folder_path = params['data'].get(
        'tokenized_data_folder_path')

    params['data']['batch_size'] = args.batch_size

    if args.read_checkpoint:
        params['state']['read_checkpoint'] = args.read_checkpoint
    if args.load_timestamp:
        params['state']['load_checkpoint'] = True
    if args.aws_run:
        if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
            params['data']['precomputed_epoch_n_nonzero_tokens'] = params['data']['precomputed_epoch_n_nonzero_tokens'].replace(
                f'/fsx-shared/s3/{args.cohort_name}', f'{args.datasets_path}/tokenizer'
            )
        elif 'precomputed_n_nonzero_tokens' in params['data'].keys():
            params['data']['precomputed_n_nonzero_tokens'] = params['data']['precomputed_n_nonzero_tokens'].replace(
                f'/fsx-shared/s3/{args.cohort_name}', f'{args.datasets_path}/tokenizer'
            )
        if 'precomputed_epoch_splits' in params['data'].keys():
            params['data']['precomputed_epoch_splits'] = params['data']['precomputed_epoch_splits'].replace(
                f'/fsx-shared/s3/{args.cohort_name}', f'{args.datasets_path}/tokenizer'
            )
        elif 'precomputed_split' in params['data'].keys():
            params['data']['precomputed_split'] = params['data']['precomputed_split'].replace(
                f'/fsx-shared/s3/{args.cohort_name}', f'{args.datasets_path}/tokenizer'
            )
        params['data']['token_dict_folder_path'] = params['data']['token_dict_folder_path'].replace(
            '/fsx-shared/s3', f'{args.datasets_path}/tokenizer'
        )
        if args.overwrite_tokenized_data_folder_path:
            params['data']['tokenized_data_folder_path'] = args.overwrite_tokenized_data_folder_path
        else:
            params['data']['tokenized_data_folder_path'] = params['data']['tokenized_data_folder_path'].replace(
                '/fsx-shared/s3', f'{args.datasets_path}/gold/cell-graph-tokenizer'
            ).replace(
                'hst_corpus_80m', 'hst_corpus_110m'
            ).replace(
                'hst_corpus_90m', 'hst_corpus_110m'
            )
        if 'test_batch_ids' in params['data']:
            del(params['data']['test_batch_ids'])
        if 'val_batch_ids' in params['data']:
            del(params['data']['val_batch_ids'])

    #params['data']['precomputed_n_nonzero_tokens'] = f'{args.datasets_path}/tokenizer/{params['data']['tokenized_data_folder_path'].split('/')[-1].split('.dataset')[0]}_n_nonzero_tokens_train.pkl"
    # The model must be evaluated on data tokenized exactly like its training
    # data (normalization, kNN, ...). With --aws_run the tokenized dataset is
    # replaced by --overwrite_tokenized_data_folder_path, whose default is one
    # specific tokenization, so make any mismatch visible.
    used_tokenized_data_folder_path = params['data']['tokenized_data_folder_path']
    logger.info(f'Model trained on tokenized data: {trained_tokenized_data_folder_path}')
    logger.info(f'Evaluating on tokenized data: {used_tokenized_data_folder_path}')
    if (trained_tokenized_data_folder_path is not None
            and Path(str(trained_tokenized_data_folder_path).rstrip('/')).name
            != Path(str(used_tokenized_data_folder_path).rstrip('/')).name):
        logger.warning(
            'Evaluation dataset differs from the training dataset; results are '
            'only meaningful if both use the same tokenization and number of '
            'neighbours. Pass an empty --overwrite_tokenized_data_folder_path to '
            'evaluate on the training dataset unless intended.')

    global layer
    layer = params['meta']['enc_depth']
    ###################


    ### Load token dict ###
    with open(token_dict_path, 'rb') as f:
        token_dict = pickle.load(f)

    token_dict_reverse =  {value: key for key, value in token_dict.items()}

    spv_dict = {
        k: v for k, v in token_dict.items() if k.startswith('spv_')}
    spv_start_idx = min(spv_dict.values())
    max_cls_tokens = sum(1 for key in token_dict if "cls" in key)
    spv_idx_subtract = spv_start_idx - 2 - max_cls_tokens
    #######################


    ### Define functions ###
    def run_inference(dataset_name,
                      params,
                      read_checkpoint,
                      obs_cols,
                      rerun_inference=False,
                      write_to_disk=True):
        # Load inference results if they exist
        if os.path.exists(f'{folder_path}/adata_{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}.h5ad') and not rerun_inference:
            adata = sc.read_h5ad(f'{folder_path}/adata_{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}.h5ad')
        else:
            # Load data
            _, _, test_dataset = prepare_dataset(params)
            test_dataset
            
            # Run model inference
            adata = infer(params,
                          test_dataset,
                          load_folder_path=folder_path,
                          dataset_ids=list(set([batch_id.split('_')[0] for batch_id in params['data']['test_batch_ids']])),
                          obs_cols=obs_cols,
                          agg_type='avg',
                          masked_tokens=None,
                          agg_excluded_genes=None,
                          cell_gene_ids=[],
                          neighborhood_gene_ids=[])

        # Extract batch from cell ID
        adata.obs["batch"] = [cell_id.split('_')[1].replace('batch', '') for cell_id in adata.obs.index.tolist()]
        
        # Remove missing ground truth
        for obs_col in obs_cols:
            adata = adata[~adata.obs[obs_col].isna()]

        if write_to_disk:
            # Save adata
            adata.write(f'{folder_path}/adata_{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}.h5ad')

        return adata

    def compute_embedding_similarity(adata,
                                    dataset_name,
                                    read_checkpoint,
                                    emb_keys=['cell', 'neighborhood'],
                                    label='niche_type',
                                    write_to_disk=True):
        for emb_key in emb_keys:
            similarity_stats = compute_similarity_stats(
                adata,
                n_samples=None,
                type_key=label,
                embedding_key=f'{emb_key}_emb_layer_{layer}')

            avg_cos_sim = np.mean(similarity_stats["Mean Similarity"])
        
            cosine_df = pd.DataFrame({
                'metric': [f'avg_{label}_cos_sim'],
                'value': [avg_cos_sim]
            })
            print(f'Average {label} cosine similarity is {avg_cos_sim}.')
            cosine_df.to_csv(f'{folder_path}/{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}_avg_{label}_cos_sim.csv',
                            index=False)
            
            plot_similarity_heatmaps(
                similarity_stats,
                save_file_path=f'{folder_path}/figures/{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}_{emb_key}_emb_{label}_cos_sim.svg')

    def compute_neighborhood_umap(adata,
                                  dataset_name,
                                  read_checkpoint,
                                  write_to_disk=True):
        if 'neighborhood_connectivities' not in adata.obsp.keys():
            # Compute UMAP representation for (aggregated) neighborhood gene embeddings
            sc.pp.neighbors(adata,
                            n_neighbors=15,
                            use_rep=f'neighborhood_emb_layer_{layer}',
                            key_added='neighborhood')
        if 'neighborhood_umap' not in adata.obsm.keys():
            sc.tl.umap(adata,
                    neighbors_key='neighborhood',
                    key_added='neighborhood_umap')
        
            if write_to_disk:
                # Save adata
                adata.write(f'{folder_path}/adata_{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}.h5ad')
        
        if 'cell_connectivities' not in adata.obsp.keys():
            # Compute UMAP representation for (aggregated) cell gene embeddings
            sc.pp.neighbors(adata,
                            n_neighbors=15,
                            use_rep=f'cell_emb_layer_{layer}',
                            key_added='cell')
        if 'cell_umap' not in adata.obsm.keys():
            sc.tl.umap(adata,
                    neighbors_key='cell',
                    key_added='cell_umap')

            if write_to_disk:
                # Save adata
                adata.write(f'{folder_path}/adata_{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}.h5ad')

        return adata

    def compute_latent_clusters(adata,
                                dataset_name,
                                read_checkpoint,
                                emb_keys=['cell', 'neighborhood'],
                                obs_col='niche_type',
                                latent_leiden_resolution_cell=1.0,
                                latent_leiden_resolution_neighborhood=0.4,
                                write_to_disk=True):
        for emb_key in emb_keys:
            latent_leiden_resolutions = []
            if emb_key == 'cell':
                latent_leiden_resolution = latent_leiden_resolution_cell
            elif emb_key == 'neighborhood':
                latent_leiden_resolution = latent_leiden_resolution_neighborhood
            adata.obsm['X_umap'] = adata.obsm[f'{emb_key}_umap']
            # Perform clustering until number of ground truth niches is matched
            i = 0
            n_unique_vals = len(adata.obs[obs_col].value_counts() < 100)
            print(f'Target number of unique values: {n_unique_vals}.')
            n_latent_clusters = 0
            while (n_latent_clusters != n_unique_vals):
                if latent_leiden_resolution in latent_leiden_resolutions:
                    latent_leiden_resolution += random.uniform(0.0001, 0.001)
                latent_leiden_resolutions.append(latent_leiden_resolution)
                print(f'Leiden resolution: {latent_leiden_resolution}.')
                latent_cluster_key = f'{emb_key}_emb_leiden_res{str(latent_leiden_resolution).replace(".", "_")}'
                i += 1
                sc.tl.leiden(adata,
                            neighbors_key=emb_key,
                            key_added=latent_cluster_key,
                            resolution=latent_leiden_resolution)
            
                n_latent_clusters = len(adata.obs[latent_cluster_key].value_counts() < 100)
                print(f"Number of latent clusters {n_latent_clusters}.")
                if n_latent_clusters == n_unique_vals:
                    break
                elif n_latent_clusters < (n_unique_vals - 8):
                    latent_leiden_resolution += 0.1
                elif n_latent_clusters < (n_unique_vals - 5):
                    latent_leiden_resolution += 0.05
                elif n_latent_clusters < (n_unique_vals - 3):
                    latent_leiden_resolution += 0.02
                elif n_latent_clusters < n_unique_vals:
                    latent_leiden_resolution += 0.001
                elif n_latent_clusters > (n_unique_vals + 8):
                    latent_leiden_resolution -= 0.1
                elif n_latent_clusters > (n_unique_vals + 5):
                    latent_leiden_resolution -= 0.05
                elif n_latent_clusters > (n_unique_vals + 3):
                    latent_leiden_resolution -= 0.02
                else:
                    latent_leiden_resolution -= 0.001
            
            # Compute metrics and store to disk
            labels = adata.obs[obs_col]
            labels_pred = adata.obs[latent_cluster_key]
            nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
            ari = adjusted_rand_score(labels, labels_pred)
            print(f'NMI is {nmi}. ARI is {ari}.')
            metrics_df = pd.DataFrame({
                'metric': ['NMI', 'ARI'],
                'value': [nmi, ari]
            })
            metrics_df.to_csv(f'{folder_path}/{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}_{latent_cluster_key}_metrics.csv',
                            index=False)
            
            # Plot UMAPs
            os.makedirs(f'{folder_path}/figures', exist_ok=True)
            labels = ['batch', obs_col, latent_cluster_key]
            # Plot (aggregated) neighborhood gene embeddings, colored by labels
            for label in labels:
                if len(np.unique(adata.obs[label])) > 1:
                    sc.pl.umap(adata,
                            neighbors_key=emb_key,
                            title=(emb_key + "_emb_" + label if emb_key not in label else label),
                            color=label,
                            palette=(custom_palette if label == latent_cluster_key else None),
                            show=False)
                    plt.savefig(f'{folder_path}/figures/{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}{"_" + emb_key + "_emb" if emb_key not in label else ""}_{label}_umap.svg',
                                dpi=300,
                                bbox_inches='tight')
            
            # Plot niches in tissue
            batch_ids = adata.obs['batch'].unique().tolist()
            for i, batch_id in enumerate(batch_ids):
                adata_batch = adata[adata.obs['batch'] == batch_id]
                sc.pl.spatial(adata_batch,
                            color=latent_cluster_key,
                            palette=custom_palette,
                            spot_size=spot_sizes[i],
                            save=f'{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}_{latent_cluster_key}_batch_{batch_id}.svg')
                os.rename(f'{folder_path}/figures/show{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}_{latent_cluster_key}_batch_{batch_id}.svg',
                        f'{folder_path}/figures/{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}_{latent_cluster_key}_batch_{batch_id}.svg'.replace("show", ""))
        
            if write_to_disk:
                # Save adata
                adata.write(f'{folder_path}/adata_{read_checkpoint.split("-")[1].split(".")[0]}_{dataset_name}.h5ad')
        
        return adata


    print(f'Load timestamp: {args.load_timestamp}.')
    print(f'Read checkpoint: {args.read_checkpoint}.')


    ### Run inference for xhs49-1b_1p ###
    if args.include_xhs49:
        # Set test batch IDs
        params['data']['precomputed_split'] = False
        params['data']['test_batch_ids'] = [
            '49_batch0']
        params['data']['val_batch_ids'] = None
        if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
            del(params['data']['precomputed_epoch_n_nonzero_tokens'])
        if 'precomputed_epoch_splits' in params['data'].keys():
            del(params['data']['precomputed_epoch_splits'])

        # Set dataset params
        dataset_name = 'xhs49-1b_1p'
        niche_key = 'niche_name'
        latent_leiden_resolution_cell = 1.0
        latent_leiden_resolution_neighborhood = 0.4
        spot_sizes = [20] * len(params['data']['test_batch_ids'])

        # Run model inference
        adata = run_inference(dataset_name,
                              params,
                              args.read_checkpoint,
                              obs_cols=[niche_key],
                              write_to_disk=True)

        # Plot embedding similiarity stats
        compute_embedding_similarity(adata,
                                     dataset_name,
                                     args.read_checkpoint,
                                     emb_keys=['neighborhood'],
                                     label=niche_key,
                                     write_to_disk=True)

        # Identify niches & cell types
        adata = compute_neighborhood_umap(adata,
                                          dataset_name,
                                          args.read_checkpoint,
                                          write_to_disk=True)

        adata = compute_latent_clusters(adata,
                                        dataset_name,
                                        args.read_checkpoint,
                                        emb_keys=['neighborhood'],
                                        obs_col=niche_key,
                                        latent_leiden_resolution_cell=latent_leiden_resolution_cell,
                                        latent_leiden_resolution_neighborhood=latent_leiden_resolution_neighborhood,
                                        write_to_disk=True)
    #####################################


    ### Run inference for xhs1000-39b_1p ###
    if args.include_xhs1000:
        # Set test batch IDs
        params['data']['precomputed_split'] = False
        params['data']['test_batch_ids'] = [
            '1000_batch11',
            '1000_batch19',
            '1000_batch32']
        params['data']['val_batch_ids'] = None
        if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
            del(params['data']['precomputed_epoch_n_nonzero_tokens'])
        if 'precomputed_epoch_splits' in params['data'].keys():
            del(params['data']['precomputed_epoch_splits'])

        # Set dataset params
        dataset_name = 'xhs1000-39b_1p'
        niche_key = 'niche_type'
        batch_key = 'sample'
        latent_leiden_resolution_cell = 1.0
        latent_leiden_resolution_neighborhood = 0.4
        spot_sizes = [20] * len(params['data']['test_batch_ids'])

        # Run model inference
        adata = run_inference(dataset_name,
                              params,
                              args.read_checkpoint,
                              obs_cols=[niche_key],
                              write_to_disk=True)
                            
        # Plot embedding similarity stats
        compute_embedding_similarity(adata,
                                     dataset_name,
                                     args.read_checkpoint,    
                                     emb_keys=['neighborhood'],
                                     label=niche_key,
                                     write_to_disk=True)

        # Identify niches & cell types
        adata = compute_neighborhood_umap(adata,
                                          dataset_name,
                                          args.read_checkpoint,
                                          write_to_disk=True)

        adata = compute_latent_clusters(adata,
                                        dataset_name,
                                        args.read_checkpoint,
                                        emb_keys=['neighborhood'],
                                        obs_col=niche_key,
                                        latent_leiden_resolution_cell=latent_leiden_resolution_cell,
                                        latent_leiden_resolution_neighborhood=latent_leiden_resolution_neighborhood,
                                        write_to_disk=True)
    ########################################


    ### Run inference for xhk1007-41b_1p ###
    if args.include_xhk1007:
        # Set test batch IDs
        params['data']['precomputed_split'] = False
        params['data']['test_batch_ids'] = [
            '1007_batch1',
            '1007_batch14'
        ]
        params['data']['val_batch_ids'] = None
        if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
            del(params['data']['precomputed_epoch_n_nonzero_tokens'])
        if 'precomputed_epoch_splits' in params['data'].keys():
            del(params['data']['precomputed_epoch_splits'])

        # Set dataset params
        dataset_name = 'xhk1007-41b_1p'
        cell_type_key = 'level_1_cell_type'
        batch_key = 'batch'
        latent_leiden_resolution_cell = 0.75
        latent_leiden_resolution_neighborhood = 0.2
        spot_sizes = [30] * len(params['data']['test_batch_ids'])

        # Run model inference
        adata = run_inference(dataset_name,
                              params,
                              args.read_checkpoint,
                              obs_cols=[cell_type_key],
                              write_to_disk=True)

        # Plot embedding similarity stats
        compute_embedding_similarity(adata,
                                     dataset_name,
                                     args.read_checkpoint,
                                     emb_keys=['cell'],
                                     label=cell_type_key,
                                     write_to_disk=True)

        # Identify niches & cell types
        adata = compute_neighborhood_umap(adata,
                                          dataset_name,
                                          args.read_checkpoint,
                                          write_to_disk=True)

        adata = compute_latent_clusters(adata,
                                        dataset_name,
                                        args.read_checkpoint,
                                        emb_keys=['cell'],
                                        obs_col=cell_type_key,
                                        latent_leiden_resolution_cell=latent_leiden_resolution_cell,
                                        latent_leiden_resolution_neighborhood=latent_leiden_resolution_neighborhood,
                                        write_to_disk=True)  
    ########################################


    ### Run inference for xhb1002-75b_3p ###
    if args.include_xhb1002:
        # Set test batch IDs
        params['data']['precomputed_split'] = False
        params['data']['test_batch_ids'] = [
            '1002_batch3',
            '1002_batch15',
            '1002_batch27']
        params['data']['val_batch_ids'] = None
        if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
            del(params['data']['precomputed_epoch_n_nonzero_tokens'])
        if 'precomputed_epoch_splits' in params['data'].keys():
            del(params['data']['precomputed_epoch_splits'])

        # Set dataset params
        dataset_name = 'xhb1002-75b_3p'
        cell_type_key = 'leiden_log1pnorm_res-1'
        batch_key = 'batch'
        latent_leiden_resolution_cell = 1.0
        latent_leiden_resolution_neighborhood = 0.4
        spot_sizes = [30] * len(params['data']['test_batch_ids'])

        # Run model inference
        adata = run_inference(dataset_name,
                              params,
                              args.read_checkpoint,
                              obs_cols=[cell_type_key],
                              write_to_disk=True)

        # Plot embedding similarity stats
        compute_embedding_similarity(adata,
                                     dataset_name,
                                     args.read_checkpoint,
                                     emb_keys=['cell'],
                                     label=cell_type_key,
                                     write_to_disk=True)

        # Identify niches & cell types
        adata = compute_neighborhood_umap(adata,
                                          dataset_name,
                                          args.read_checkpoint,
                                          write_to_disk=True)

        adata = compute_latent_clusters(adata,
                                        dataset_name,
                                        args.read_checkpoint,
                                        emb_keys=['cell'],
                                        obs_col=cell_type_key,
                                        latent_leiden_resolution_cell=latent_leiden_resolution_cell,
                                        latent_leiden_resolution_neighborhood=latent_leiden_resolution_neighborhood,
                                        write_to_disk=True)

    ########################################


    ### Run inference for chl59-8b_1p ###
    if args.include_chl59:
        # Set test batch IDs
        params['data']['precomputed_split'] = False
        params['data']['test_batch_ids'] = [
            '59_batch0',
            #'59_batch3', # unannotated
            '59_batch6',
            '59_batch7',
            #'59_batch2', # unannotated
            '59_batch4',
            '59_batch1',
            '59_batch5']
        params['data']['val_batch_ids'] = None
        if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
            del(params['data']['precomputed_epoch_n_nonzero_tokens'])
        if 'precomputed_epoch_splits' in params['data'].keys():
            del(params['data']['precomputed_epoch_splits'])

        # Set dataset params
        dataset_name = 'chl59-8b_1p'
        niche_key = 'niche'
        batch_key = 'batch'
        latent_leiden_resolution_cell = 1.0
        latent_leiden_resolution_neighborhood = 0.6
        spot_sizes = [80] * len(params['data']['test_batch_ids'])

        # Run model inference
        adata = run_inference(dataset_name,
                              params,
                              args.read_checkpoint,
                              obs_cols=[niche_key],
                              write_to_disk=True)

        # Plot embedding similarity stats
        compute_embedding_similarity(adata,
                                     dataset_name,
                                     args.read_checkpoint,
                                     emb_keys=['neighborhood'],
                                     label=niche_key,
                                     write_to_disk=True)

        # Identify niches & cell types
        adata = compute_neighborhood_umap(adata,
                                          dataset_name,
                                          args.read_checkpoint,
                                          write_to_disk=True)

        adata = compute_latent_clusters(adata,
                                        dataset_name,
                                        args.read_checkpoint,
                                        emb_keys=['neighborhood'],
                                        obs_col=niche_key,
                                        latent_leiden_resolution_cell=latent_leiden_resolution_cell,
                                        latent_leiden_resolution_neighborhood=latent_leiden_resolution_neighborhood,
                                        write_to_disk=True)
    #####################################


    ### Run inference for chr78-11b_1p ###
    if args.include_chr78:
        # Set test batch IDs
        params['data']['precomputed_split'] = False
        params['data']['test_batch_ids'] = [
            '78_batch3',
            '78_batch10',
            '78_batch1',
            '78_batch0',
            '78_batch2',
            '78_batch6',
            '78_batch7',
            '78_batch9',
            '78_batch8',
            '78_batch4',
            '78_batch5']
        params['data']['val_batch_ids'] = None
        if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
            del(params['data']['precomputed_epoch_n_nonzero_tokens'])
        if 'precomputed_epoch_splits' in params['data'].keys():
            del(params['data']['precomputed_epoch_splits'])

        # Set dataset params
        dataset_name = 'chr78-11b_1p'
        niche_key = 'broad_anatomy'
        batch_key = 'batch'
        latent_leiden_resolution_cell = 1.0
        latent_leiden_resolution_neighborhood = 0.4
        spot_sizes = [1920 * 0.093] * len(params['data']['test_batch_ids'])

        # Run model inference
        adata = run_inference(dataset_name,
                              params,
                              args.read_checkpoint,
                              obs_cols=[niche_key],
                              write_to_disk=True)

        # Plot embedding similarity stats
        compute_embedding_similarity(adata,
                                     dataset_name,
                                     args.read_checkpoint,
                                     emb_keys=['neighborhood'],
                                     label=niche_key,
                                     write_to_disk=True)

        # Identify niches & cell types
        adata = compute_neighborhood_umap(adata,
                                          dataset_name,
                                          args.read_checkpoint,
                                          write_to_disk=True)

        adata = compute_latent_clusters(adata,
                                        dataset_name,
                                        args.read_checkpoint,
                                        emb_keys=['neighborhood'],
                                        obs_col=niche_key,
                                        latent_leiden_resolution_cell=latent_leiden_resolution_cell,
                                        latent_leiden_resolution_neighborhood=latent_leiden_resolution_neighborhood,
                                        write_to_disk=True)
    ######################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--load_timestamp',
                        type=str,
                        default='28092025_173911_891',
                        help='Model load timestamp.')
    parser.add_argument('--read_checkpoint',
                        type=str,
                        default='terra-ep3.pth.tar',
                        help='Model read checkpoint.')
    parser.add_argument("--aws_run",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Flag whether the model was trained on AWS.')    
    parser.add_argument('--cohort_name',
                        type=str,
                        default='hst_corpus_110m',
                        help='Name of data cohort.')
    parser.add_argument('--datasets_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS',
                        help='Path to the DATASETS folder.')
    parser.add_argument("--include_xhs49",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Flag whether to run inference on xhs49.')
    parser.add_argument("--include_xhs1000",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Flag whether to run inference on xhs1000.')
    parser.add_argument("--include_xhk1007",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Flag whether to run inference on xhk1007.')
    parser.add_argument("--include_xhb1002",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Flag whether to run inference on xhb1002.') 
    parser.add_argument("--include_chl59",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Flag whether to run inference on chl59.') 
    parser.add_argument("--include_chr78",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Flag whether to run inference on chr78.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size.')
    parser.add_argument('--overwrite_tokenized_data_folder_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_110m/hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_10.dataset',
                        help='Path to the tokenized data folder if different than model config.')
    args = parser.parse_args()
    
    main(args)