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
        if 'test_batch_ids' in params['data']:
            del(params['data']['test_batch_ids'])
        if 'val_batch_ids' in params['data']:
            del(params['data']['val_batch_ids'])

    #params['data']['precomputed_n_nonzero_tokens'] = f"/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/{params['data']['tokenized_data_folder_path'].split('/')[-1].split('.dataset')[0]}_n_nonzero_tokens_train.pkl"
    ###################


    ### Load dataset ###
    dataset = load_from_disk(args.dataset_path)

    columns = list(dataset.features.keys())
    columns.remove("cell_id")
    dataset.set_format(
        type="torch",
        columns=columns,
        output_all_columns=True)
    print(dataset)


    ### Get gene program genes ###
    adata_gp = sc.read_h5ad(args.annotated_gp_path)

    gp_genes = adata_gp[
        adata_gp.obs['gene_clusters_annotated'] == args.gp_name].obs.index.tolist()
    ##############################


    ### Retrieve gene embeddings ###
    # Get ensembl IDs
    gene_name_to_ensembl_id_dict_path = '/nfs/team361/sb75/DATASETS/genes/homo_sapiens_gene_name_to_ensembl_id_dict.pkl'
    with open(gene_name_to_ensembl_id_dict_path, 'rb') as f:
        gene_name_to_ensembl_id_dict = pickle.load(f)

    ensembl_ids = [gene_name_to_ensembl_id_dict.get(gene, '') for gene in gp_genes]
    cell_gene_ensembl_id = [gene for gene in ensembl_ids if gene != '']
    neighborhood_gene_ensembl_id = [gene for gene in ensembl_ids if gene != '']

    # Get gene embeddings
    output_gene_embed = get_gene_embed(
        dataset=dataset,
        model_folder_path=model_folder_path,
        emb_layer=args.emb_layer,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=args.num_workers,
        cell_gene_ensembl_id=cell_gene_ensembl_id,
        neighborhood_gene_ensembl_id=neighborhood_gene_ensembl_id
    )

    # Save output_gene_embed dict
    with open(f'{folder_path}/output_gene_embed_{args.read_checkpoint.split("-")[1].split(".")[0]}_{args.save_suffix}.pkl', 'wb') as f:
        pickle.dump(output_gene_embed, f)
    ################################


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
    parser.add_argument('--annotated_gp_path',
                        type=str,
                        default='/nfs/team361/dj17/RCC_nicheJEPA/GPannotation/myeloid_v1/myeloidGPs_v1_annotated.h5ad',
                        help='Path to the AnnData file with the annotated gene programs.')
    parser.add_argument('--gp_name',
                        type=str,
                        default='VEGF signalling GP',
                        help='Gene program name.')
    parser.add_argument('--save_suffix',
                        type=str,
                        default='myeloid_vegf_gp',
                        help='Suffix for saving the gene embedding pickle file.')
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
    args = parser.parse_args()
    
    main(args)