"""
Usage:
    python reproducibility/analysis/testing/measure_batch_integration.py --config_file reproducibility/config/measure_batch_integration_merfish_mouse_brain_2-batch_read-depth_shifted-log.yaml
"""

import argparse
import yaml
from typing import Dict
from pathlib import Path
from datasets import load_from_disk
import scanpy as sc
import os
from nichejepa.aggregators.aggregate_neighbors import aggregate_neighbors
from nichejepa.normalizers.normalize_by_read_depth import normalize_by_read_depth
from nichejepa.preprocessors.filter_poor_quality_cells import filter_poor_quality_cells
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Iterable
import matplotlib.pyplot as plt
import seaborn as sns


def compute_lisi(X: np.array,
    metadata: pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float=30):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.
    
    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    
    Reference:
        https://github.com/slowkow/harmonypy/blob/master/harmonypy/lisi.py
    """
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn = NearestNeighbors(n_neighbors = perplexity * 3, algorithm = 'kd_tree').fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    indices = indices[:,1:]
    distances = distances[:,1:]
    # Save the result
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
        lisi_df[:,i] = 1 / simpson
    return lisi_df


def compute_simpson(distances: np.ndarray,
                        indices: np.ndarray,
                        labels: pd.Categorical,
                        n_categories: int,
                        perplexity: float,
                        tol: float=1e-5):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:,i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:,i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:,i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
    ----------
    - argparse.Namespace: The command line arguments.
    
    Usage:
    ----------
    .. terminal
        >>> python create_cell_neighborhood_tokenized_dataset.py --config_file config/cell-neighborhood-tokenization_sim1_1105fts_10000locs_read-depth_shifted-log.yaml --override nproc=1
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    parser.add_argument('--config_file', 
                        type=str, 
                        help='Path to the config file')
    # parser.add_argument('--data_type',
    #                     type=str,
    #                     default='anndata',
    #                     help='Type of data to process')
    # parser.add_argument('--overwrite',
    #                     type=str,
    #                     default=False,
    #                     help='Overwrite existing files')
    parser.add_argument('--override', 
                        nargs='+', 
                        help='Override parameters in the config file')
    
    return parser.parse_args()


def collect_configs(args: argparse.Namespace) -> Dict:
    """
    Collect configurations from the config file and command line arguments.
    
    Parameters:
    ----------
    - args (argparse.Namespace): The command line arguments.
    
    Returns:
    ----------
    - Dict: The configuration dictionary.
    """
    # Get the config file name from command line argument
    config_fname = args.config_file

    # Read parameters from config file
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)
    
    # Override parameters from command line
    if args.override:
        for arg in args.override:
            key, value = arg.split("=")
            if key == 'data_file_format':
                config['batch_integration'][key] = str(value)
            if key == 'overwrite':
                if value == 'True':
                    config['experiment'][key] = True
                elif value == 'False':
                    config['experiment'][key] = False
                else:
                    raise ValueError(f"Invalid value for 'overwrite': {value}")
            
    return config


def load_concat_batches(data_directory: Path):
    adata_batch_list = []
    for file_path in data_directory.glob(f"*.h5ad"):
        print(f"Loading {file_path}")
        adata_batch = sc.read_h5ad(file_path)
        print("Filtering cells.")
        adata_batch = filter_poor_quality_cells(adata_batch)
        adata_batch_list.append(adata_batch)
    if len(adata_batch_list) == 1:
        adata = adata_batch_list[0]
    else:
        print("Concatenating batches.")
        adata = ad.concat(adata_batch_list, join="inner", axis=0)
    
    print(adata)
    
    print("Normalizing by read depth.")
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("Log-transforming.")
    sc.pp.log1p(adata)
    print("Scaling.")
    sc.tl.pca(adata)
    print("Computing neighbors.")
    sc.pp.neighbors(adata)
    print("Computing UMAP.")
    sc.tl.umap(adata)        

    print(adata)
    
    return adata
        

def compute_umap_from_anndata(umap_path: Path,
                              st_data_silver_folder_path: Path, 
                              species: str, 
                              dataset_name: str, 
                              n_batches: int,
                              overwrite: bool):
    umap_file_path = umap_path / f"{dataset_name}_umap.h5ad"
    print(f"UMAP file path: {umap_file_path}")
    print(f"Umap File Exists: {umap_file_path.exists()} | Overwrite: {overwrite}")
    if umap_file_path.exists() and not overwrite:
        print(f"Loading UMAP from {umap_file_path}")
        adata = sc.read(umap_file_path)
    else:
        print(f"Computing UMAP for {dataset_name}.")
        exit()
        adata = load_concat_batches(st_data_silver_folder_path / dataset_name)
        sc.write(umap_file_path, adata)
        
    return adata


def plot_umap(adata, 
              umap_path, 
              dataset_name, 
              tokenizer):
    umap_plot_path = str(umap_path / f"{dataset_name}_{tokenizer}_anndata_umap_plot.pdf")
    print(f"Saving UMAP plot to {umap_plot_path}")
    fig, ax = plt.subplots(figsize=(10, 10))
    sc.pl.umap(adata, color=["batch"], ax=ax)
    ax.set_title(f"{dataset_name}_{tokenizer} UMAP")
    plt.tight_layout()
    plt.savefig(umap_plot_path, bbox_inches="tight")
    

def main(config):

    # Configure experiment arguments
    tokenizer = config["experiment"]["tokenizer"]
    dataset_name = config["experiment"]["dataset_name"]
    species = config["experiment"]["species"]
    n_batches = config["experiment"]["n_batches"]
    overwrite = config["experiment"]["overwrite"]
    data_file_format = config["batch_integration"]["data_file_format"]
    print(f"Overwrite: {overwrite}")
    print(f"Tokenizer: {tokenizer}")
    print(f"Dataset: {dataset_name}")
    print(f"Data file format: {data_file_format}")
    
    # Configure data paths
    data_directory_path = Path(config["paths"]["data_directory_path"])
    
    if data_file_format == "anndata":
        st_data_silver_folder_path = data_directory_path / "silver"
        print(f"Silver data folder: {st_data_silver_folder_path}")
        umap_path = data_directory_path / "umap" / dataset_name / "anndata"
        os.makedirs(umap_path, exist_ok=True)
        print(f"UMAP folder: {umap_path}")
        
        adata = compute_umap_from_anndata(umap_path,
                                         st_data_silver_folder_path, 
                                         species, 
                                         dataset_name, 
                                         n_batches,
                                         overwrite)
        
        if config['batch_integration']['plot_umap']:
            print("Plotting UMAP.")            
            plot_umap(adata, umap_path, dataset_name, tokenizer)
            
        if config['batch_integration']['compute_blisi']:
            print("Computing LISI.")
            blisi_df = compute_lisi(adata.obsm["X_umap"], adata.obs, ["batch"])
            blisi_npy_path = umap_path / f"{dataset_name}_{tokenizer}_anndata_blisi.npy"
            print(f"Saving LISI to {blisi_npy_path}")
            np.save(blisi_npy_path, blisi_df)
            avg_blisi = np.mean(blisi_df)
            stddev_blisi = np.std(blisi_df)
            print(f"Average bLISI: {avg_blisi} | Standard Deviation: {stddev_blisi}")
        
    elif data_file_format == "hugging_face":
        st_data_gold_folder_path = data_directory_path / "gold"
        print(f"Gold data folder: {st_data_gold_folder_path}")
        umap_path = data_directory_path / "umap" / dataset_name / "hugging_face"
        os.makedirs(umap_path, exist_ok=True)
        print(f"UMAP folder: {umap_path}")
    
    elif data_file_format == "pytorch":
        st_data_gold_folder_path = data_directory_path / "gold"
        print(f"Gold data folder: {st_data_gold_folder_path}")
        umap_path = data_directory_path / "umap" / dataset_name / "pytorch"
        os.makedirs(umap_path, exist_ok=True)
        print(f"UMAP folder: {umap_path}")
        
    else:
        raise ValueError("Invalid data type. Choose either 'anndata' or 'hugging_face' or 'pytorch'.")
            

if __name__ == '__main__':
    # Parse the arguments
    args = parse_arguments()

    # Collect configs 
    config = collect_configs(args)
    
    main(config)