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
from terra.datasets import prepare_dataset
from terra.utils.config import create_params_from_YAML_wandb_config
from terra.utils.distributed import init_distributed
from terra.utils.embedding import collect_adata_from_folder
from terra.utils.evaluation import clustering_metrics, classification_metrics

from analysis_utils import *


def main(args):
    ### Define paths ###
    artifacts_folder_path = '/nfs/team361/sb75/terra-reproducibility/artifacts'
    datasets_folder_path = '/nfs/team361/sb75/DATASETS'
    token_dict_path = f'{datasets_folder_path}/tokenizer/token_dictionary_homo_sapiens.pkl'
    folder_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}'
    ####################


    ##### Subsample dataset and only keep 100k genes per tissue #####
    rng_seed = 42

    # Step 1: Shuffle the full dataset once
    shuffled_dataset = filtered_dataset.shuffle(seed=rng_seed)

    # Step 2: Extract the tissue_value column (this is fast and avoids full row loading)
    tissue_values = shuffled_dataset["tissue_value"]

    # Step 3: Collect sample indices efficiently
    tissue_counts = defaultdict(int)
    tissue_indices = defaultdict(list)

    for idx, tissue in enumerate(tissue_values):
        tissue = tissue[0]

        if tissue_counts[tissue] < args.n_cells_per_tissue:
            tissue_indices[tissue].append(idx)
            tissue_counts[tissue] += 1

    # Step 4: Build final dataset with select
    final_samples = [
        shuffled_dataset.select(indices) for indices in tissue_indices.values()
    ]
    final_dataset = concatenate_datasets(final_samples)
    columns = list(final_dataset.features.keys())
    columns.remove("cell_id")
    final_dataset.set_format(
        type="torch",
        columns=columns,
        output_all_columns=True)
    final_dataset # 700,000 cells

    # Step 5: Write to disk
    #final_dataset.save_to_disk(
    #    "/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_80m/hst_corpus_80m_32_None_None_None_None_None_shifted_log_knn_10_5k_subsample.dataset")
    ######################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--load_timestamp',
                        type=str,
                        default='27092025_012758_922',
                        help='Model load timestamp.')
    parser.add_argument('--read_checkpoint',
                        type=str,
                        default='terra-ep2.pth.tar',
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
    parser.add_argument('--n_cells_per_tissue',
                        type=int,
                        default=100_000,
                        help='Number of cells per tissue.')
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