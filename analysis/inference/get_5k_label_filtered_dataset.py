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
from terra.datasets import prepare_dataset
from terra.utils.config import create_params_from_YAML_wandb_config
from terra.utils.distributed import init_distributed
from terra.utils.embedding import collect_adata_from_folder
from terra.utils.evaluation import classification_metrics, clustering_metrics

sys.path.append("../../utils")

from analysis_utils import *


def main(args):
    ### Define paths ###
    artifacts_folder_path = '/nfs/team361/sb75/terra-reproducibility/artifacts'
    datasets_folder_path = '/nfs/team361/sb75/DATASETS'
    token_dict_path = f'{datasets_folder_path}/tokenizer/token_dictionary_homo_sapiens.pkl'
    folder_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}'
    ####################


    ### Get label filtered cell IDs ###
    cell_ids_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/{args.read_checkpoint.split("-")[1].split(".")[0]}_5k_{args.label_value}_idx.pkl'
    if os.path.exists(cell_ids_path):
        with open(cell_ids_path, 'rb') as f:
            cell_ids = pickle.load(f)
    else:
        adata_label_annotated_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_{args.label_value}_annotated.h5ad"
        if os.path.exists(adata_label_annotated_path):
            print("Load existing AnnData.")
            adata_emb_label = sc.read_h5ad(adata_label_annotated_path)
        else:
            adata_annotated_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_annotated.h5ad"
            adata_emb = sc.read_h5ad(adata_annotated_path)
            if args.label_value == 'myeloid':
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
                
                adata_emb_label = adata_emb[
                    (adata_emb.obs['brain_cell_type'] == 'Lymphocytes') & (adata_emb.obs['brain_sub_cell_type'].isin(brain_DC_names)) |
                    (adata_emb.obs['brain_cell_type'] == 'Macrophages') & (adata_emb.obs['brain_sub_cell_type'].isin(brain_TAM_names)) |
                    (adata_emb.obs['kidney_cell_type'] == 'Myeloid') |
                    (adata_emb.obs['skin_cell_type'].isin(skin_myeloid_cell_names)) |
                    (adata_emb.obs['brain_cell_type'].isna()) & (
                        adata_emb.obs['kidney_cell_type'].isna()) & (
                        adata_emb.obs['skin_cell_type'].isna()) & (
                        adata_emb.obs['majority_voting'].isin(celltypist_myeloid_cell_names))
                    ]

            elif args.label_value == 'macrophage':
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
                skin_macrophage_cell_names = [
                    'LC',
                    'Mac',
                ]
                celltypist_macrophage_cell_names = [
                    'Alveolar macrophages',
                    'Classical monocytes',
                    'Intestinal macrophages',
                    'Kupffer cells',
                    'Macrophages',
                    'Mono-mac',
                    'Monocytes',
                ]
                
                adata_emb_label = adata_emb[
                    (adata_emb.obs['brain_cell_type'] == 'Macrophages') & (adata_emb.obs['brain_sub_cell_type'].isin(brain_TAM_names)) |
                    (adata_emb.obs['kidney_cell_type_level_2'] == 'Macrophage') |
                    (adata_emb.obs['skin_cell_type'].isin(skin_macrophage_cell_names)) |
                    (adata_emb.obs['brain_cell_type'].isna()) & (
                        adata_emb.obs['kidney_cell_type'].isna()) & (
                        adata_emb.obs['skin_cell_type'].isna()) & (
                        adata_emb.obs['majority_voting'].isin(celltypist_macrophage_cell_names))
                    ]

            elif args.label_value == 'fibroblasts':
                skin_fibroblast_cell_names = [
                    "F1: Superficial",
                    "F2: Universal",
                    "F2/3: Bridge",
                    "F3: TRC-like",
                    "F4: Mixed",
                    "F4: TNN+COCH+",
                    "F5: Schwann-like"
                ]
            
                adata_emb_label = adata_emb[
                    (adata_emb.obs['kidney_cell_type_level_2'] == 'Fibroblast') |
                    (adata_emb.obs['skin_cell_type'].isin(skin_fibroblast_cell_names)) |
                    (adata_emb.obs['brain_cell_type'].isna()) & (
                        adata_emb.obs['kidney_cell_type'].isna()) & (
                        adata_emb.obs['skin_cell_type'].isna()) & (
                        adata_emb.obs['majority_voting'] == 'Fibroblasts')
                    ]

            # Write AnnData to disk
            adata_emb_label.write(adata_label_annotated_path)
        
        print(adata_emb_label)

        # Store label filtered cell IDs to disk
        cell_ids = adata_emb_label.obs.index.tolist()

        # Save to disk
        with open(cell_ids_path, "wb") as f:
            pickle.dump(cell_ids, f)
        ############################


    ### Get label filtered dataset ###
    filtered_dataset = load_from_disk(args.filtered_dataset_path)    

    cell_ids = set(cell_ids)
    label_filtered_dataset = filtered_dataset.filter(
        lambda ids: [i in cell_ids for i in ids],
        input_columns="cell_id",
        batched=True,
        batch_size=10000,
        num_proc=args.num_workers,
        desc="Filtering by cell_id"
    )

    print('Saving label filtered dataset...')
    label_filtered_dataset.save_to_disk(args.label_filtered_dataset_path)
    ###########################


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
    parser.add_argument('--cohort_name',
                        type=str,
                        default='hst_corpus_80m',
                        help='Name of data cohort.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of workers.')
    parser.add_argument('--label_value',
                        type=str,
                        default='myeloid',
                        help='Label value used for filtering.')
    parser.add_argument('--filtered_dataset_path',
                        type=str,
                        default='/nfs/team361/sb75/terra-reproducibility/artifacts/hst_corpus_80m/18062025_082526_412/ep16_0_5k.dataset',
                        help='Path where the filtered Huggingface dataset will be loaded from.')
    parser.add_argument('--label_filtered_dataset_path',
                        type=str,
                        default='/nfs/team361/sb75/terra-reproducibility/artifacts/hst_corpus_80m/18062025_082526_412/ep16_0_5k_myeloid.dataset',
                        help='Path where the label filtered Huggingface dataset will be stored.')   
    args = parser.parse_args()
    
    main(args)