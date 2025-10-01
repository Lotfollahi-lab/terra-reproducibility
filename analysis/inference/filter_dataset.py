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

    # Get gene panel values for 5k gene panel
    min_gene_panel_size = 4948
    max_gene_panel_size = 4949

    # Perform index subtraction as in tokenizer
    max_cls_tokens = 100
    spv_dict = {
        k: v for k, v in token_dict.items() if k.startswith('spv_')}
    spv_start_idx = min(spv_dict.values())
    spv_idx_subtract = spv_start_idx - 2 - max_cls_tokens

    keep_gene_panel_values = torch.tensor([
        v - spv_idx_subtract for k,v in token_dict.items() if ('spv_gene_panel' in k) and (int(k.split('panel')[1]) >= min_gene_panel_size) and (int(k.split('panel')[1]) <= max_gene_panel_size)])

    # Filter dataset
    dataset.set_format(type=None) # for extreme speed-up

    def keep_row(x):
        return torch.isin(torch.tensor(x["gene_panel_value"]), keep_gene_panel_values)

    filtered_dataset = dataset.filter(
        keep_row,
        batched=True,
        batch_size=10000, 
        num_proc=args.num_workers)
    filtered_dataset # 26,621,953 cells

    columns = list(filtered_dataset.features.keys())
    columns.remove("cell_id")
    filtered_dataset.set_format(
        type="torch",
        columns=columns,
        output_all_columns=True)
    filtered_dataset # 18,168,008 cells
    filtered_dataset.save_to_disk(args.dataset_path)
    ###############################



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
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of workers.')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/nfs/team361/sb75/nichejepa-reproducibility/artifacts/hst_corpus_80m/18062025_082526_412/ep16_0_5k.dataset',
                        help='Path where the filtered Huggingface dataset will be stored.')    
    args = parser.parse_args()
    
    main(args)