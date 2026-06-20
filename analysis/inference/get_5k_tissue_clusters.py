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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

sys.path.append("../../utils")

from analysis_utils import *


def main(args):
    ### Define paths ###
    artifacts_folder_path = '/nfs/team361/sb75/terra-reproducibility/artifacts'
    datasets_folder_path = '/nfs/team361/sb75/DATASETS'
    folder_path = f'{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}'
    model_folder_path = f'/nfs/team361/sb75/terra-reproducibility/artifacts/models/{args.load_timestamp}'
    ####################


    ### Get tissue-specific clusters ###
    ####################################
    ### Load AnnData ###
    adata_processed_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_complete_processed.h5ad"
    adata_complete_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('-')[1].split('.')[0]}_5k_complete.h5ad"
    if os.path.exists(adata_processed_path):
        print("Load existing processed AnnData.")
        adata_emb = sc.read_h5ad(adata_processed_path)
    elif os.path.exists(adata_complete_path):
        print("Load existing complete AnnData.")
        adata_emb = sc.read_h5ad(adata_complete_path)
    else:
        raise ValueError("No processed or complete AnnData found.")


    ### Run Leiden clustering ###
    for tissue in args.tissues:
        if os.path.exists(adata_processed_path):
            adata_tissue_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('.')[0]}_5k_{tissue}_processed.h5ad"
        else:
            adata_tissue_path = f"{artifacts_folder_path}/{args.cohort_name}/{args.load_timestamp}/adata_{args.read_checkpoint.split('.')[0]}_5k_{tissue}_complete.h5ad"
        adata_emb_tissue = adata_emb[adata_emb.obs['tissue'] == tissue]

        # Compute tissue-specific neighborhood graphs and UMAP coordinates
        for emb_key in args.emb_keys:
            if f'{emb_key}_{tissue}_connectivities' not in adata_emb_tissue.obsp.keys():
                if args.use_rapids:
                    rsc.pp.neighbors(adata_emb_tissue,
                                     n_neighbors=args.n_neighbors,
                                     use_rep=f'{emb_key}_emb_layer_{args.n_layer}',
                                     n_pcs=(args.n_pcs if args.n_pcs > 0 else None),
                                     key_added=f'{emb_key}_{tissue}')
                else:
                    sc.pp.neighbors(adata_emb_tissue,
                                    n_neighbors=args.n_neighbors,
                                    use_rep=f'{emb_key}_emb_layer_{args.n_layer}',
                                    n_pcs=(args.n_pcs if args.n_pcs > 0 else None),
                                    key_added=f'{emb_key}_{tissue}')
            if f'{emb_key}_{tissue}_umap' not in adata_emb_tissue.obsm.keys():
                if args.use_rapids:
                    rsc.tl.umap(adata_emb_tissue,
                                neighbors_key=f'{emb_key}_{tissue}',
                                key_added=f'{emb_key}_{tissue}_umap')
                else:
                    sc.tl.umap(adata_emb_tissue,
                               neighbors_key=f'{emb_key}_{tissue}',
                               key_added=f'{emb_key}_{tissue}_umap')

            # Write to disk
            adata_emb_tissue.write(adata_tissue_path)

        # Perform tissue-specific clustering and add to main object
        for emb_key in args.emb_keys:
            latent_cluster_key = tissue + f"_{emb_key}_clusters_res_{latent_leiden_resolution}"
            if args.use_rapids:
                rsc.tl.leiden(adata_emb_tissue,
                              neighbors_key=f'{emb_key}_{tissue}',
                              key_added=latent_cluster_key,
                              resolution=args.latent_leiden_resolution,
                              random_state=0)
            else:
                sc.tl.leiden(adata_emb_tissue,
                             neighbors_key=f'{emb_key}_{tissue}',
                             key_added=latent_cluster_key,
                             resolution=args.latent_leiden_resolution,
                             random_state=0)
            adata_emb.obs[latent_cluster_key] = (
                adata_emb_tissue.obs[latent_cluster_key].reindex(adata_emb.obs_names)
            )
    adata_emb.write(adata_processed_path)
    ####################################


    ### Combine per tissue clusters ###
    for emb_key in args.emb_keys:
        cols_to_combine = [tissue + f"_{emb_key}_clusters_res_{latent_leiden_resolution}" for tissue in args.tissues]
        adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"] = (
            adata_emb.obs[cols_to_combine]
            .bfill(axis=1) # fill NaN with next non-NaN to the right
            .iloc[:, 0] # take the first filled column
        )

        # Remove NAN values
        adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"] = adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"].astype(str)
        adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"] = adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"].astype("category")
        adata_emb = adata_emb[adata_emb.obs[f"tissue_specific_{emb_key}_clusters_res_{latent_leiden_resolution}"] != 'nan']

    print(adata_emb.obs.head(1))
    ###################################


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
    parser.add_argument("--use_rapids",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Flag whether to use rapids_singlecell.')    
    parser.add_argument('--cohort_name',
                        type=str,
                        default='hst_corpus_90m',
                        help='Name of data cohort.') 
    parser.add_argument('--tissues',
                        type=str,
                        nargs='+',
                        default=['brain', 'breast', 'cervix', 'embryo', 'kidney', 'liver', 'lung', 'ovary', 'pancreas', 'prostate', 'skin'],
                        help='List of tissues to get average gene embeddings for.')
    parser.add_argument('--emb_keys',
                        type=str,
                        nargs='+',
                        default=['cell', 'neighborhood', 'spatial_cell'],
                        help='Embedding keys.')
    parser.add_argument('--n_layer',
                        type=int,
                        default=12,
                        help='Number of layers.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of workers.')
    parser.add_argument('--n_neighbors',
                        type=int,
                        default=15,
                        help='Number of neighbors.')
    parser.add_argument('--n_pcs',
                        type=int,
                        default=0,
                        help='Number of principal components.')
    parser.add_argument('--latent_leiden_resolution',
                        type=float,
                        default=1.0,
                        help='Latent leiden resolution.')
    args = parser.parse_args()
    
    main(args)