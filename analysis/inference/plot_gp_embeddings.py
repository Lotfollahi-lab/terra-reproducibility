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
import celltypist
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

from app.infer import get_gene_embed, get_average_gene_embed, infer
from app.train import train
from terra.datasets import prepare_dataset
from terra.utils.config import create_params_from_YAML_wandb_config
from terra.utils.distributed import init_distributed
from terra.utils.embedding import collect_adata_from_folder
from terra.utils.evaluation import clustering_metrics, classification_metrics

from analysis_utils import *


def main(args):
    ### Define paths ###
    artifacts_folder_path = '/nfs/team361/sb75/terra-reproducibility/artifacts'
    datasets_folder_path = '/nfs/team361/sb75/DATASETS'
    folder_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}'
    model_folder_path = f'/nfs/team361/sb75/terra-reproducibility/artifacts/models/{args.load_timestamp}'
    ####################


    ### Set up logger ###
    rank=0
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    #####################


    ### Load config ###
    params = create_params_from_YAML_wandb_config(
        f'{folder_path}/params.yaml', logger)

    if args.read_checkpoint:
        params['state']['read_checkpoint'] = args.read_checkpoint
    if args.load_timestamp:
        params['state']['load_checkpoint'] = True
    if args.aws_run:
        if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
            params['data']['precomputed_epoch_n_nonzero_tokens'] = params['data']['precomputed_epoch_n_nonzero_tokens'].replace(
                f'/fsx-shared/s3/{args.cohort_name}', '/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer'
            )
        elif 'precomputed_n_nonzero_tokens' in params['data'].keys():
            params['data']['precomputed_n_nonzero_tokens'] = params['data']['precomputed_n_nonzero_tokens'].replace(
                f'/fsx-shared/s3/{args.cohort_name}', '/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer'
            )
        if 'precomputed_epoch_splits' in params['data'].keys():
            params['data']['precomputed_epoch_splits'] = params['data']['precomputed_epoch_splits'].replace(
                f'/fsx-shared/s3/{args.cohort_name}', '/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer'
            )
        elif 'precomputed_split' in params['data'].keys():
            params['data']['precomputed_split'] = params['data']['precomputed_split'].replace(
                f'/fsx-shared/s3/{args.cohort_name}', '/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer'
            )
        params['data']['token_dict_folder_path'] = params['data']['token_dict_folder_path'].replace(
            '/fsx-shared/s3', '/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer'
        )
        params['data']['tokenized_data_folder_path'] = params['data']['tokenized_data_folder_path'].replace(
            '/fsx-shared/s3', '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer'
        ).replace(
            'hst_corpus_80m', 'hst_corpus_90m'
        )
        if 'test_batch_ids' in params['data']:
            del(params['data']['test_batch_ids'])
        if 'val_batch_ids' in params['data']:
            del(params['data']['val_batch_ids'])

    #params['data']['precomputed_n_nonzero_tokens'] = f"/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/{params['data']['tokenized_data_folder_path'].split('/')[-1].split('.dataset')[0]}_n_nonzero_tokens_train.pkl"
    ###################


    ### Load gene embed dict ###
    with open(f'{folder_path}/output_gene_embed_{args.read_checkpoint.split("-")[1].split(".")[0]}_{args.load_suffix}.pkl', 'rb') as f:
        output_gene_embed = pickle.load(f)
    ############################


    ### Get avg nonezero gene embeddings for each emb_key and plot ###
    dataset = load_from_disk(args.dataset_path)    

    for emb_key in args.emb_keys:

        gene_emb = [v for k, v in output_gene_embed.items() if emb_key in k.split('_')[0]]
        
        # Stack them along a new axis: shape becomes (num_keys, N, D)
        #gene_emb_stacked = np.stack(gene_emb, axis=0)
        # Compute the mean across the keys (axis=0), keeping shape (N, D)
        #mean_gene_emb = gene_emb_stacked.mean(axis=0)
        # Stack into shape: (12, 700000, 384)
        stacked = np.stack(gene_emb, axis=0)
        # Mask of nonzero rows: shape (12, 700000)
        nonzero_mask = ~(stacked == 0).all(axis=2)
        # Count of valid arrays per row: shape (700000,)
        valid_counts = nonzero_mask.sum(axis=0)
        # To avoid divide-by-zero later
        valid_counts = np.where(valid_counts == 0, 1, valid_counts)
        # Zero out fully-zero rows before summing
        masked = stacked * nonzero_mask[:, :, np.newaxis] # shape: (12, 700000, 384)
        # Sum across arrays and normalize
        sum_embeddings = masked.sum(axis=0) # shape: (700000, 384)
        mean_gene_emb = sum_embeddings / valid_counts[:, np.newaxis] # shape: (700000, 384)

        nonzero_row_mask = ~(mean_gene_emb == 0).all(axis=1)
        filtered_mean_gene_emb = mean_gene_emb[nonzero_row_mask]
        
        # Create AnnData
        adata = ad.AnnData(filtered_mean_gene_emb)
        adata.obs['cell_id'] = list(np.array(dataset['cell_id'])[nonzero_row_mask])
        adata.obs['tissue'] = list(np.array([
            token_dict_reverse[tissue_value[0] + spv_idx_subtract].split('spv_')[1]
            for tissue_value in dataset['tissue_value']])[nonzero_row_mask])
        adata.obs = adata.obs.set_index('cell_id')

        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata)
        sc.tl.pca(adata, use_highly_variable=False, svd_solver='arpack')
        
        for label in ['tissue']:
            # Neighbors + UMAP
            sc.pl.umap(adata,
                       color=label,
                       show=False)
            plt.title(f'{emb_key}_{args.load_suffix} by {label} (UMAP)', fontsize=14)
            plt.savefig(
                f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/figures/{emb_key}_{args.load_suffix}_umap_{label}.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()
    ##################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--load_timestamp',
                        type=str,
                        default='18062025_082526_412',
                        help='Model load timestamp.')
    parser.add_argument('--read_checkpoint',
                        type=str,
                        default='terra-ep16_0.pth.tar',
                        help='Model read checkpoint.')
    parser.add_argument("--aws_run",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Flag whether the model was trained on AWS.')    
    parser.add_argument('--cohort_name',
                        type=str,
                        default='hst_corpus_80m',
                        help='Name of data cohort.')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/nfs/team361/sb75/terra-reproducibility/artifacts/hst_corpus_80m/18062025_082526_412/ep16_0_5k_myeloid.dataset',
                        help='Path to the Huggingface dataset for which to retrieve the gene program embeddings.')    
    parser.add_argument('--annotated_gp_path',
                        type=str,
                        default='/nfs/team361/dj17/RCC_TERRA/GPannotation/myeloid_v1/myeloidGPs_v1_annotated.h5ad',
                        help='Path to the AnnData file with the annotated gene programs.')
    parser.add_argument('--gp_name',
                        type=str,
                        default='Apelin signalling GP', # 'VEGF signalling GP',
                        help='Gene program name.')
    parser.add_argument('--load_suffix',
                        type=str,
                        default='5k_myeloid_apelin_gp',
                        help='Suffix for saving the gene embedding pickle file.')
    parser.add_argument('--emb_keys',
                        type=str,
                        nargs='+',
                        default=['cell', 'neighborhood', 'spatial_cell'],
                        help='Embedding keys.')
    args = parser.parse_args()
    
    main(args)