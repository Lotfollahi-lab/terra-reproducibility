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
    global artifacts_folder_path
    global folder_path
    artifacts_folder_path = '/nfs/team361/sb75/nichejepa-reproducibility/artifacts'
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

    params['data']['batch_size'] = args.batch_size
    params['data']['raw_data_folder_path'] = '/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver'

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
            dataset = load_from_disk(f'/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/{dataset_name}/{dataset_name}_32_None_None_None_None_None_shifted_log_knn_10_2816.dataset')

            columns = list(dataset.features.keys())
            columns.remove("cell_id")
            dataset.set_format(
                type="torch",
                columns=columns,
                output_all_columns=True)  
            
            # Run model inference
            adata = infer(params,
                          dataset,
                          load_folder_path=folder_path,
                          dataset_ids=list(set([batch_id.split('_')[0] for batch_id in params['data']['test_batch_ids']])),
                          obs_cols=obs_cols,
                          agg_type='avg',
                          masked_tokens=None,
                          agg_excluded_tokens=None,
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


    ### Run inference ###
    # Set test batch ID
    params['data']['precomputed_split'] = False
    params['data']['test_batch_ids'] = [args.batch_id]
    params['data']['val_batch_ids'] = None
    if 'precomputed_epoch_n_nonzero_tokens' in params['data'].keys():
        del(params['data']['precomputed_epoch_n_nonzero_tokens'])
    if 'precomputed_epoch_splits' in params['data'].keys():
        del(params['data']['precomputed_epoch_splits'])

    # Run model inference
    adata = run_inference(args.dataset_name,
                          params,
                          args.read_checkpoint,
                          obs_cols=[], #[niche_key],
                          write_to_disk=True)
    #####################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--load_timestamp',
                        type=str,
                        default='28092025_173911_891',
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
    parser.add_argument('--datasets_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS',
                        help='Path to the DATASETS folder.')
    parser.add_argument('--dataset_name',
                        type=str,
                        default='xhb2000-1b_1p',
                        help='Dataset name.')
    parser.add_argument('--batch_id',
                        type=str,
                        default='2000_batch0',
                        help='Batch ID of the batch to run inference on.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size.')
    args = parser.parse_args()
    
    main(args)