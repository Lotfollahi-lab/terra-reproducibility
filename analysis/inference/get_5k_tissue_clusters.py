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
from nichejepa.datasets import prepare_dataset
from nichejepa.utils.config import create_params_from_YAML_wandb_config
from nichejepa.utils.distributed import init_distributed
from nichejepa.utils.embedding import collect_adata_from_folder
from nichejepa.utils.evaluation import clustering_metrics, classification_metrics

from analysis_utils import *


def main(args):
    ### Define paths ###
    artifacts_folder_path = '/nfs/team361/sb75/nichejepa-reproducibility/artifacts'
    datasets_folder_path = '/nfs/team361/sb75/DATASETS'
    folder_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}'
    model_folder_path = f'/nfs/team361/sb75/nichejepa-reproducibility/artifacts/models/{args.load_timestamp}'
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


    ### Load AnnData ###
    adata_processed_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_complete_processed.h5ad"
    adata_complete_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_complete.h5ad"
    if os.path.exists(adata_processed_path):
        print("Load existing processedAnnData.")
        adata_emb = sc.read_h5ad(adata_processed_path)
    elif os.path.exists(adata_complete_path):
        print("Load existing complete AnnData.")
        adata_emb = sc.read_h5ad(adata_complete_path)
    else:
        raise ValueError("No processed or complete AnnData found.")


    ### Run Leiden clustering ###
    for tissue in args.tissues:
        if os.path.exists(adata_processed_path):
            adata_tissue_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('.')[0]}_5k_{tissue}_processed.h5ad"
        else:
            adata_tissue_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('.')[0]}_5k_{tissue}_complete.h5ad"
        adata_emb_tissue = adata_emb[adata_emb.obs['tissue'] == tissue]

        # Compute tissue-specific neighborhood graphs and UMAP coordinates
        for emb_key in args.emb_keys:
            if f'{emb_key}_{tissue}_connectivities' not in adata_emb_tissue.obsp.keys():
                sc.pp.neighbors(adata_emb_tissue,
                                n_neighbors=args.n_neighbors,
                                use_rep=f'{emb_key}_emb_layer_{args.n_layer}',
                                key_added=f'{emb_key}_{tissue}')
            if f'{emb_key}_{tissue}_umap' not in adata_emb_tissue.obsm.keys():
                sc.tl.umap(adata_emb_tissue,
                           neighbors_key=f'{emb_key}_{tissue}',
                           key_added=f'{emb_key}_{tissue}_umap')

            # Write to disk
            adata_emb_tissue.write(adata_tissue_path)
    ##############################


    ### Combine per tissue clusters ###
    for emb_key in args.emb_keys:
        cols_to_combine = [tissue + f"_{emb_key}_clusters_res_{latent_leiden_resolution}" for tissue in all_5k_tissues]
        adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"] = (
            adata_emb.obs[cols_to_combine]
            .bfill(axis=1) # fill NaN with next non-NaN to the right
            .iloc[:, 0] # take the first filled column
        )

        # Remove NAN values
        adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"] = adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"].astype(str)
        adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"] = adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"].astype("category")
        adata_emb = adata_emb[adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"] != 'nan']

    print(adata_emb.obs.head(1))
    ###################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--load_timestamp',
                        type=str,
                        default='27092025_012758_922',
                        help='Model load timestamp.')
    parser.add_argument('--read_checkpoint',
                        type=str,
                        default='nichejepa-ep2.pth.tar',
                        help='Model read checkpoint.')
    parser.add_argument("--aws_run",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Flag whether the model was trained on AWS.')    
    parser.add_argument('--cohort_name',
                        type=str,
                        default='hst_corpus_90m',
                        help='Name of data cohort.') 
    parser.add_argument('--tissues',
                        type=str,
                        nargs='+',
                        default=['brain', 'embryo', 'kidney', 'liver', 'lung', 'ovary', 'skin'],
                        help='List of tissues to get average gene embeddings for.')
    parser.add_argument('--emb_keys',
                        type=str,
                        nargs='+',
                        default=['cell', 'neighborhood', 'spatial_cell'],
                        help='Embedding keys.')
    parser.add_argument('--n_layer',
                        type=int,
                        default=12,
                        help='Number of layers.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of workers.')
    parser.add_argument('--n_neighbors',
                        type=int,
                        default=15,
                        help='Number of neighbors.')
    args = parser.parse_args()
    
    main(args)