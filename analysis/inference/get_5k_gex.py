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
    raw_data_folder_path = f'/nfs/team361/sb75/DATASETS/silver/{args.cohort_name}'
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


    ### Define columns for inference ###
    annotation_cols = [
        'level_1_cell_type',
        'cell_type',
        'annotation_new7']
    obs_cols = [
        'leiden_log1pnorm_res-1',
        'level_1_cell_type',
        'level_2_cell_type',
        'level_3_cell_type',
        'annotation_new7',
        'cell_type']
    uns_cols = [
        'assay',
        'batch',
        'donor_id',
        'tissue']
    ####################################


    ### Load Huggingface dataset ###
    dataset = load_from_disk(args.dataset_path)
    print(dataset)
    ################################


    ### Get gene expression for all 5k panel cells ###
    adata_expr_path = f'{folder_path}/adata_{args.read_checkpoint.split("-")[1].split(".")[0]}_5k_expr.h5ad'
    if os.path.exists(adata_expr_path):
        print("Load existing AnnData.")
        adata_expr_filt = sc.read_h5ad(adata_expr_path)
    else:
        print("Gather new AnnData.")
        cell_ids = dataset['cell_id']
        dataset_ids = list(set([cell_id.split('_')[0] for cell_id in cell_ids]))
        
        adata_expr = collect_adata_from_folder(
            load_folder_path=raw_data_folder_path,
            cell_ids=None,
            dataset_ids=dataset_ids,
            obs_cols=obs_cols,
            uns_cols=uns_cols)

        adata_expr.obs_names = adata_expr.obs['cell_id']

        # Filter rows and columns to keep only 5k panel cells and genes
        adata_expr_filt = adata_expr[adata_expr.obs['cell_id'].isin(cell_ids)]
        del adata_expr
        adata_tmp = sc.read_h5ad(f'{raw_data_folder_path}/xhl1015-R1-28b_1p/output-XETG00272__0045132__DF31-LIV-0-FFPE-1-S1-i__20250626__122204.h5ad')
        adata_expr_filt = adata_expr_filt[:, adata_expr_filt.var_names.isin(adata_tmp.var_names.tolist())]
        adata_expr_filt.write(adata_expr_path)

        # adata_expr_filt.write(adata_expr_path)

        # Write counts to csv files for CellTypist annotation
        X = adata_expr_filt.X.tocsr()
        output_prefix = adata_expr_path.replace('.h5ad', '') # Base name for chunked files
        
        for i in range(0, X.shape[0], args.chunk_size):
            end = min(i + args.chunk_size, X.shape[0])
            chunk = X[i:end].toarray().astype(np.float32)
            
            count_df = pd.DataFrame(
                chunk,
                index=adata_expr_filt.obs_names[i:end],
                columns=adata_expr_filt.var_names)
        
            output_path = f"{output_prefix}_chunk_{i}_{end}.csv"
            count_df.to_csv(output_path, header=True, index=True)
            print(f"Saved {output_path}")

        # Iteratively get annotations from CellTypist
        output_prefix = adata_expr_path.replace('.h5ad', '') # Base name for chunked files
        
        all_predictions = []
        
        for i in range(0, math.ceil(len(adata_expr_filt) / args.chunk_size)):
            start = i * args.chunk_size
            if i < (math.ceil(len(adata_expr_filt) / args.chunk_size) - 1):
                end = (i + 1) * args.chunk_size
            else:
                end = len(adata_expr_filt)
            file_path = f"{output_prefix}_chunk_{start}_{end}.csv"
            
            preds = celltypist.annotate(file_path, majority_voting=True)
            all_predictions.append(preds.predicted_labels)
        
        final_predictions = pd.concat(all_predictions)

        adata_expr_filt.obs = adata_expr_filt.obs.join(final_predictions['majority_voting'])
        
        adata_expr_filt.write(adata_expr_path)

    # Change order to be aligned with embedding AnnData
    adata_expr_aligned = adata_expr_filt[adata_emb.obs_names].copy()
    del adata_expr_filt
    ##################################################


    kidney_level_1_cell_type = adata_expr_aligned.obs['level_1_cell_type']
    kidney_level_2_cell_type = adata_expr_aligned.obs['level_2_cell_type']
    skin_cell_type = adata_expr_aligned.obs['annotation_new7']
    adata_expr_aligned.obs = adata_emb.obs
    adata_expr_aligned.obs['kidney_cell_type'] = kidney_cell_type
    adata_expr_aligned.obs['skin_cell_type'] = skin_cell_type
    adata_expr_aligned.uns = adata_emb.uns
    adata_expr_aligned.obsm = adata_emb.obsm
    adata_expr_aligned.obsp = adata_emb.obsp
    adata_expr_aligned.layers = adata_emb.layers
    adata_emb = adata_expr_aligned



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--load_timestamp',
                        type=str,
                        default='18062025_082526_412',
                        help='Model load timestamp.')
    parser.add_argument('--read_checkpoint',
                        type=str,
                        default='nichejepa-ep16_0.pth.tar',
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
                        default='/nfs/team361/sb75/nichejepa-reproducibility/artifacts/hst_corpus_80m/18062025_082526_412/ep16_0_5k_myeloid.dataset',
                        help='Path to the Huggingface dataset for which to retrieve the gene program embeddings.')    
    parser.add_argument('--emb_layer',
                        type=int,
                        default=None,
                        help='Embedding layer.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of workers.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size.')
    parser.add_argument('--chunk_size',
                        type=int,
                        default=500_000,
                        help='Chunk size.')
    args = parser.parse_args()
    
    main(args)