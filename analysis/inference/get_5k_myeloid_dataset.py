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



    adata_myeloid_annotated_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_myeloid_annotated.h5ad"
    if os.path.exists(adata_myeloid_annotated_path):
        print("Load existing AnnData.")
        adata_emb_myeloid = sc.read_h5ad(adata_myeloid_annotated_path)
    else:
        adata_annotated_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_annotated.h5ad"
        adata_emb = sc.read_h5ad(adata_annotated_path)
        brain_TAM_names = [
            'Pro-inflammatory TAMs',
            'Interferon TAMs',
            'Astrocyte-like TAMs',
            'Angiogenic TAMs',
            'Anti-inflammatory TAMs',
            'Resident BAM TAMs',
            'Resident-TAMs ',
            'RTN1+ TAMs',
            'Monocytes',
            'Proliferatve TAMs',
            'Stress-response TAMs'
        ]
        brain_DC_names = [
            'Plasma cells',
            'Dendritic cells'
        ]
        skin_myeloid_cell_names = [
            'LC',
            'MigDC',
            'cDC1',
            'cDC2',
            'Mac',
            'pDC',
            'LC/MigDC',
            'Mast cell'
        ]
        celltypist_myeloid_cell_names = [
            'Alveolar macrophages',
            'Classical monocytes',
            'DC',
            'DC1',
            'DC2',
            'Hofbauer cells',
            'Intestinal macrophages',
            'Kupffer cells',
            'Macrophages',
            'Mast cells',
            'Migratory DCs',
            'Mono-mac',
            'Monocytes',
            'pDC'
        ]
        
        adata_emb_myeloid = adata_emb[
            (adata_emb.obs['brain_cell_type'] == 'Lymphocytes') & (adata_emb.obs['brain_sub_cell_type'].isin(brain_DC_names)) |
            (adata_emb.obs['brain_cell_type'] == 'Macrophages') & (adata_emb.obs['brain_sub_cell_type'].isin(brain_TAM_names)) |
            (adata_emb.obs['kidney_cell_type'] == 'Myeloid') |
            (adata_emb.obs['skin_cell_type'].isin(skin_myeloid_cell_names)) |
            (adata_emb.obs['brain_cell_type'].isna()) & (
                adata_emb.obs['kidney_cell_type'].isna()) & (
                adata_emb.obs['skin_cell_type'].isna()) & (
                adata_emb.obs['majority_voting'].isin(celltypist_myeloid_cell_names))
            ]

        # Write AnnData to disk
        adata_emb_myeloid.write(adata_myeloid_annotated_path)
    adata_emb_myeloid

    # Store myeloid cell IDs to disk
    myeloid_cell_ids = adata_emb_myeloid.obs.index.tolist()

    myeloid_cell_ids_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/{args.read_checkpoint.split("-")[1].split(".")[0]}_5k_myeloid_idx.pkl'
    with open(myeloid_cell_ids_path, "wb") as f:
        pickle.dump(myeloid_cell_ids, f)


myeloid_cell_ids_path = f'{artifacts_folder_path}/hst_corpus_80m/{load_timestamp}/{read_checkpoint.split("-")[1].split(".")[0]}_5k_myeloid_idx.pkl'
with open(myeloid_cell_ids_path, 'rb') as f:
    myeloid_cell_ids = pickle.load(f)

myeloid_cell_ids = np.array(list(myeloid_cell_ids), dtype=object)

myeloid_dataset = filtered_dataset.filter(
    lambda ids: np.isin(ids, myeloid_cell_ids).tolist(),
    input_columns="cell_id",
    batched=True,
    batch_size=10000,
    num_proc=args.num_workers,
    desc="Filtering by cell_id"
)


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
    parser.add_argument('--inference_step_size',
                        type=int,
                        default=200000,
                        help='Inference step size.')
    parser.add_argument('--inference_max_step',
                        type=int,
                        default=134,
                        help='Inference maximum step.')  
    args = parser.parse_args()
    
    main(args)