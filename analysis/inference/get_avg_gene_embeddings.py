import argparse
import logging
import math
import os
import pickle
import random
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import chain, compress
from pathlib import Path

import anndata as ad
import celltypist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scib_metrics
import seaborn as sns
import squidpy as sq
import torch
from datasets import concatenate_datasets, load_from_disk
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from app.infer import get_average_gene_embed, get_gene_embed, infer
from app.train import train
from nichejepa.datasets_new import prepare_dataset
from nichejepa.utils_new.config import create_params_from_YAML_wandb_config
from nichejepa.utils_new.distributed import init_distributed
from nichejepa.utils_new.embedding import collect_adata_from_folder
from nichejepa.utils_new.evaluation import classification_metrics, clustering_metrics

sys.path.append("../../utils")

from analysis_utils import *


def main(args):
    ### Define paths ###
    artifacts_folder_path = '/nfs/team361/sb75/nichejepa-reproducibility/artifacts'
    datasets_folder_path = '/nfs/team361/sb75/DATASETS'
    token_dict_path = f'{datasets_folder_path}/tokenizer/token_dictionary_homo_sapiens.pkl'
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
        )

    # Change config from hst_corpus_80m to hst_corpus_90m
    for key in params['data'].keys():
        if type(params['data'][key]) == str:
            params['data'][key] = params['data'][key].replace('hst_corpus_80m', 'hst_corpus_90m')

    # Load full corpus dataset (train, val and test)
    params['data']['precomputed_epoch_splits'] = False
    params['data']['precomputed_split'] = False
    params['data']['test_batch_ids'] = False
    params['data']['val_batch_ids'] = False
    if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
        del(params['data']['precomputed_epoch_n_nonzero_tokens'])
    if 'precomputed_n_nonzero_tokens' in params['data'].keys():
        del(params['data']['precomputed_n_nonzero_tokens'])

    params['data']['batch_size'] = args.batch_size
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


    #### Create AnnData with tissue-specific gene embeddings ###
    # Load Huggingface dataset
    dataset = load_from_disk(args.dataset_path)
    print(dataset)

    # Get tissues for each cell
    tissue_values = [
        token_dict_reverse[tissue_value[0] + spv_idx_subtract].split('spv_')[1]
        for tissue_value in dataset['tissue_value']]

    # Convert format of dataset
    columns = list(dataset.features.keys())
    dataset.set_format(
        type="torch",
        columns=columns,
        output_all_columns=True)

    for tissue in args.tissues:
        # Get indices where the tissue is in the allowed list
        keep_indices = [i for i, t in enumerate(tissue_values) if t in tissue]
        
        # Filter the HuggingFace dataset
        tissue_dataset = dataset.select(keep_indices)

        # Get all gene ensembl IDs
        adata_tmp = sc.read_h5ad('/nfs/team361/sb75/DATASETS/silver/xhe1008-1-20b_1p/section020_SOB26.h5ad')        
        cell_gene_ensembl_id = adata_tmp.var['ensembl_id'].tolist()
        neighborhood_gene_ensembl_id = adata_tmp.var['ensembl_id'].tolist()
        
        # Get average gene embeddings
        tissue_avg_gene_embed = get_average_gene_embed(
            dataset=tissue_dataset,
            model_folder_path=model_folder_path,
            emb_layer=args.emb_layer,
            batch_size=args.batch_size,
            pin_memory=False,
            num_workers=args.num_workers,
            cell_gene_ensembl_id=cell_gene_ensembl_id,
            neighborhood_gene_ensembl_id=neighborhood_gene_ensembl_id,
            include_spatial_cell_emb=False,
        )
        
        tissue_avg_gene_embed_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/{args.read_checkpoint.split("-")[1].split(".")[0]}_{args.save_prefix}_{tissue}_avg_gene_embed.pkl'
        
        # Write to disk
        with open(tissue_avg_gene_embed_path, "wb") as f:
            pickle.dump(tissue_avg_gene_embed, f)
    ############################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--load_timestamp',
                        type=str,
                        default='07112025_054432_709',
                        help='Model load timestamp.')
    parser.add_argument('--read_checkpoint',
                        type=str,
                        default='nichejepa-ep3.pth.tar',
                        help='Model read checkpoint.')
    parser.add_argument("--aws_run",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Flag whether the model was trained on AWS.')    
    parser.add_argument('--cohort_name',
                        type=str,
                        default='hst_corpus_110m',
                        help='Name of data cohort.')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/nfs/team361/sb75/nichejepa-reproducibility/artifacts/hst_corpus_110m/07112025_054432_709/5k.dataset',
                        help='Path where the Huggingface dataset will be loaded from.')
    parser.add_argument('--save_prefix',
                        type=str,
                        default='5k',
                        help='Save prefix.')   
    parser.add_argument('--tissues',
                        type=str,
                        nargs='+',
                        default=['embryo', 'kidney', 'liver', 'lung', 'ovary', 'skin'],
                        help='List of tissues to get average gene embeddings for.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of workers.')
    parser.add_argument('--emb_layer',
                        type=int,
                        default=None,
                        help='Embedding layer.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size.')
    args = parser.parse_args()
    
    main(args)