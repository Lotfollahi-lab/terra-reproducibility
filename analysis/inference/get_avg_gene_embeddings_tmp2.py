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
    token_dict_path = f'{datasets_folder_path}/tokenizer/token_dictionary_homo_sapiens.pkl'
    folder_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}'
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

    params['data']['batch_size'] = args.batch_size

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
            'hst_corpus_80m', args.cohort_name
        )    
        del(params['data']['test_batch_ids'])
        del(params['data']['val_batch_ids'])

    #params['data']['precomputed_n_nonzero_tokens'] = f"/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/{params['data']['tokenized_data_folder_path'].split('/')[-1].split('.dataset')[0]}_n_nonzero_tokens_train.pkl"
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


    ### Retrieve Huggingface dataset and keep only 5k gene panel cells ###
    # Load full cohort
    dataset, _, _ = prepare_dataset(params)
    dataset # 88,626,282 cells


    ###############################################

    #### Create AnnData with tissue-specific gene embeddings ###
    # Define params
    all_5k_tissues = ['brain', 'embryo', 'kidney', 'liver', 'lung', 'ovary', 'skin']
    all_emb_keys = ['cell', 'spatial_cell', 'neighborhood']
    gene_count_thresh = 20

    # Load one example AnnData just to get gene IDs/names
    adata_tmp = sc.read_h5ad('/nfs/team361/sb75/DATASETS/silver/xhe1008-1-20b_1p/section020_SOB26.h5ad')

    # Build mapping: ensembl_id -> gene_name
    gene_mapping_dict = {
        ensembl_id: gene_name
        for ensembl_id, gene_name in zip(
            adata_tmp.var['ensembl_id'].tolist(),
            adata_tmp.var.index.tolist()
        )
    }
    gene_names = [gene_mapping_dict[gene] for gene in adata_tmp.var['ensembl_id'].tolist()]

    # Retrieve and save gene embeddings
    for emb_key in all_emb_keys:
        tissue_avg_gene_embed_list = []
        filtered_genes_list = []
        tissue_list = []
        
        for selected_tissue in all_5k_tissues:
            tissue_avg_gene_embed_path = f'{artifacts_folder_path}/hst_corpus_80m/{load_timestamp}/{read_checkpoint.split("-")[1].split(".")[0]}_5k_{selected_tissue}_avg_gene_embed.pkl'
            with open(tissue_avg_gene_embed_path, 'rb') as f:
                tissue_avg_gene_embed = pickle.load(f)
        
            gene_mask = (
                tissue_avg_gene_embed[f'{emb_key}_gene_emb_counts_per_data'] > gene_count_thresh).ravel()
            tissue_avg_gene_embed = tissue_avg_gene_embed[f'{emb_key}_gene_emb_average_per_data'][gene_mask]
            filtered_genes = [f'{selected_tissue}_{gene}' for gene in list(compress(gene_names, gene_mask))]
            filtered_tissues = [selected_tissue] * len(filtered_genes)
        
            tissue_avg_gene_embed_list.append(tissue_avg_gene_embed)
            filtered_genes_list.extend(filtered_genes)
            tissue_list.extend(filtered_tissues)
        
        X = np.concatenate(tissue_avg_gene_embed_list, axis=0)

        adata_tissue_avg_gene_embed = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=filtered_genes_list),
            var=pd.DataFrame(index=[f"dim_{i}" for i in range(X.shape[1])])
        )
        adata_tissue_avg_gene_embed.obs['tissue'] = tissue_list

        adata_tissue_avg_gene_embed_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/{args.read_checkpoint.split("-")[1].split(".")[0]}_5k_{emb_key}_tissue_avg_gene_embed.h5ad'
        adata_tissue_avg_gene_embed.write(adata_tissue_avg_gene_embed_path)
    ############################################################


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
                        default='hst_corpus_80m',
                        help='Name of data cohort.')
    parser.add_argument('--inference_step_size',
                        type=int,
                        default=200000,
                        help='Inference step size.')
    parser.add_argument('--start_step',
                        type=int,
                        default=20,
                        help='Inference start step.')
    parser.add_argument('--end_step',
                        type=int,
                        default=40,
                        help='Inference start step.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of workers.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size.')
    args = parser.parse_args()
    
    main(args)