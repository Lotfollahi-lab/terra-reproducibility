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


### Combine inferenced AnnData objects as "adata_5k_complete.h5ad" ###
######################################################################
adata_complete_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_complete.h5ad"
if os.path.exists(adata_complete_path):
    print("Load existing AnnData.")
    adata_emb = sc.read_h5ad(adata_complete_path)
else:
    print("Combine new AnnData.")
    adata_list = []
    
    for i in range(args.inference_max_step):
        adata_tmp = sc.read_h5ad(f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('-')[1].split('.')[0]}_{args.inference_step_size}_{i}.h5ad")
        adata_list.append(adata_tmp)
        del adata_tmp
    
    adata_emb = sc.concat(adata_list, join='outer', index_unique=None)
    
    adata_emb.obs[['dataset', 'batch', 'cell']] = [s.split('_') for s in adata_emb.obs.index]
    adata_emb.obs['dataset_batch'] = adata_emb.obs['dataset'].astype(str) + adata_emb.obs['batch'].astype(str)
    adata_emb.obs['cell'] = adata_emb.obs['cell'].astype(int)

    adata_emb.write(adata_complete_path)
adata_emb
######################################################################


### Get neighborhood graphs, UMAP coordinates, Leiden clusters, and store as "adata_5k_complete_processed.h5ad" ###
adata_processed_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_complete_processed.h5ad"
if os.path.exists(adata_processed_path):
    print("Load existing AnnData.")
    adata_emb = sc.read_h5ad(adata_processed_path)
else:
    print("Process new AnnData.")
    for emb_key in args.emb_keys:
        sc.pp.neighbors(adata_emb,
                        n_neighbors=args.n_neighbors,
                        use_rep=f'{emb_key}_emb_layer_{args.n_layer}',
                        n_pcs=(args.n_pcs if args.n_pcs > 0 else None),
                        key_added=emb_key)
     
        sc.tl.umap(adata_emb,
                   neighbors_key=emb_key,
                   key_added=f'{emb_key}_umap')
    
        adata_emb.write(adata_processed_path)

    for emb_key in args.emb_keys:

        latent_cluster_key = f"{emb_key}_clusters_res_{args.latent_leiden_resolution}"
        
        sc.tl.leiden(adata_emb,
                     neighbors_key=emb_key,
                     key_added=latent_cluster_key,
                     resolution=args.latent_leiden_resolution)
    
        adata_emb.write(adata_processed_path)
adata_emb
###################################################################################################################


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
    parser.add_argument('--emb_keys',
                        type=str,
                        nargs='+',
                        default=['cell', 'neighborhood', 'spatial_cell'],
                        help='Embedding keys.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of workers.')
    parser.add_argument('--n_neighbors',
                        type=int,
                        default=15,
                        help='Number of neighbors.')
    parser.add_argument('--latent_leiden_resolution',
                        type=float,
                        default=1.0,
                        help='Latent leiden resolution.')
    parser.add_argument('--n_layer',
                        type=int,
                        default=12,
                        help='Number of layers.')
    parser.add_argument('--n_pcs',
                        type=int,
                        default=0,
                        help='Number of principal components.')
    args = parser.parse_args()
    
    main(args)