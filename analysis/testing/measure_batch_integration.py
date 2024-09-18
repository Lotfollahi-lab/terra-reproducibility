"""
Usage:
    python reproducibility/analysis/testing/measure_batch_integration.py --config_file reproducibility/config/measure_batch_integration_merfish_mouse_brain_2-batch_read-depth_shifted-log.yaml
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Iterable

from datasets import load_from_disk
from sklearn.neighbors import NearestNeighbors


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
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    parser.add_argument('--config_file', 
                        type=str, 
                        help='Path to the config file')
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


def plot_umap(adata: ad.AnnData, 
              umap_plot_path: str, 
              dataset_name: str, 
              tokenizer: str,
              avg_blisi: float,
              stddev_blisi: float, 
              data_file_format: str) -> None:
    '''
    Plot UMAP for the AnnData object.
    
    Parameters:
    ----------
    - adata (ad.AnnData): The AnnData object containing the UMAP coordinates.
    - umap_plot_path (str): The path to save the UMAP plot.
    - dataset_name (str): The name of the dataset.
    - tokenizer (str): The name of the tokenizer.
    - avg_blisi (float): The average bLISI value.
    - stddev_blisi (float): The standard deviation of bLISI values.
    - data_file_format (str): The format of the data file.
    
    Returns:
    ----------
    - None: The UMAP plot is saved to the specified path.
    '''
    fig, ax = plt.subplots(figsize=(5,5))
    sc.pl.umap(adata, color=["batch"], ax=ax)
    ax.set_title(f"UMAP: {dataset_name} | {tokenizer} | {data_file_format} \n mean bLISI: {avg_blisi:.2f} ± {stddev_blisi:.2f}")
    plt.tight_layout()
    plt.savefig(umap_plot_path, bbox_inches="tight")
    plt.close()


def compute_umap(adata: ad.AnnData) -> ad.AnnData:
    '''
    Compute UMAP for the AnnData object.
    
    Parameters:
    ----------
    - adata (ad.AnnData): The AnnData object containing cell-gene count matrix if `data_file_format` is AnnData and the token sequence for all cells if the `data_file_format` is Hugging Face tokenized dataset.
    
    Returns:
    ----------
    - ad.AnnData: The AnnData object containing the UMAP coordinates.
    '''
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

    return adata
        

def build_anndata_from_batches_of_anndata(data_directory: Path) -> ad.AnnData:
    '''
    Build a concatenated AnnData object from batches of AnnData objects.
    
    Parameters:
    ----------
    - data_directory (Path): The path to the directory containing the batches of AnnData objects.
    
    Returns:
    ----------
    - ad.AnnData: The AnnData object containing the concatenated batches.
    '''
    adata_batch_list = []
    for file_path in data_directory.glob(f"*.h5ad"):
        print(f"Loading {file_path}")
        adata_batch = sc.read_h5ad(file_path)
        adata_batch_list.append(adata_batch)
    if len(adata_batch_list) == 1:
        adata = adata_batch_list[0]
    else:
        print("Concatenating batches.")
        adata = ad.concat(adata_batch_list, join="inner", axis=0)
    return adata


def build_anndata_from_hugging_face_data(data_directory: Path) -> ad.AnnData:
    '''
    Build an AnnData object from the Hugging Face tokenized dataset.
    
    Parameters:
    ----------
    - data_directory (Path): The path to the directory containing the Hugging Face tokenized dataset.
    
    Returns:
    ----------
    - ad.AnnData: The AnnData object containing the token sequence for all cells.
    '''
    file_found = 0
    for file_path in data_directory.glob(f"*.dataset"):
        file_found = 1
        print(f"Loading {file_path}")
        dataset = load_from_disk(str(file_path))
        print("Collecting token sequence for all cells...")
        token_sequence_per_cell = np.array(dataset['input_ids']) # this is be time-consuming
        batches = np.array(dataset['batch_id'])
        obs = pd.DataFrame(data={"batch": batches}, index=np.arange(batches.shape[0]))
        print("Creating AnnData object.")
        adata = ad.AnnData(X=token_sequence_per_cell, obs=obs)
        if file_found:
            break
    if not file_found:
        raise ValueError(f"No dataset found in {data_directory}")
    return adata
   

def compute_batch_lisi(adata: ad.AnnData, 
                        blisi_npy_path: Path, 
                        overwrite: bool) -> Tuple[float, float]:
    '''
    Compute batch Local Inverse Simpson Index (bLISI) for each cell in the dataset.
    
    Parameters:
    ----------
    - adata (ad.AnnData): The AnnData object containing the UMAP coordinates.
    - blisi_npy_path (Path): The path to save the bLISI numpy file.
    - overwrite (bool): Whether to overwrite the bLISI file if it already exists.
    
    Returns:
    ----------
    - float: The average bLISI value.
    - float: The standard deviation of bLISI values.
    '''
    # If bLISI file exists and overwrite is set to False, load it. Otherwise, compute bLISI.
    if blisi_npy_path.exists() and not overwrite:
        print(f"Loading bLISI from {blisi_npy_path}")
        blisi_df = np.load(blisi_npy_path)
    else:
        print("Computing bLISI.")
        blisi_df = compute_lisi(X=adata.obsm["X_umap"], 
                                metadata=adata.obs,
                                label_colnames=["batch"])
        
        print(f"Saving bLISI to {blisi_npy_path}")
        np.save(blisi_npy_path, blisi_df)

    # Print the average and standard deviation of bLISI
    avg_blisi = np.mean(blisi_df)
    stddev_blisi = np.std(blisi_df)
    return avg_blisi, stddev_blisi
     

def main(config):

    # Configure experiment arguments
    tokenizer = config["experiment"]["tokenizer"]
    dataset_name = config["experiment"]["dataset_name"]
    overwrite = config["experiment"]["overwrite"]
    data_file_format = config["batch_integration"]["data_file_format"]
    num_shards = config['tokenization']['num_shards']
    print(f"Overwrite: {overwrite}")
    print(f"Tokenizer: {tokenizer}")
    print(f"Dataset: {dataset_name}")
    print(f"Data file format: {data_file_format}")
    
    # Configure data directory paths
    data_directory_path = Path(config["paths"]["data_directory_path"])
    umap_path = data_directory_path / "umap" / dataset_name / tokenizer / data_file_format
    os.makedirs(umap_path, exist_ok=True)
    print(f"UMAP folder: {umap_path}")
    
    # Configure file path for Anndata with integrated UMAP
    umap_file_path = umap_path / f"{dataset_name}_{tokenizer}_{data_file_format}_umap.h5ad"
    print(f"UMAP file path: {umap_file_path}")
    print(f"Umap File Exists: {umap_file_path.exists()} | Overwrite: {overwrite}")
    
    # If UMAP file exists and overwrite is set to False, load it. Otherwise, compute UMAP.
    if umap_file_path.exists() and not overwrite:
        print(f"Loading UMAP from {umap_file_path}")
        adata = sc.read(umap_file_path)
    else:
        if data_file_format == "anndata":
            st_data_silver_folder_path = data_directory_path / "silver"
            batched_data_directory = st_data_silver_folder_path / dataset_name
            print(f"Data Directory containing AnnData Batches: {batched_data_directory}")
            
            adata = build_anndata_from_batches_of_anndata(batched_data_directory)
            
        elif data_file_format == "hugging_face":
            st_data_gold_folder_path = data_directory_path / "gold"
            hf_data_directory = st_data_gold_folder_path / tokenizer / dataset_name / f"{num_shards}-shards"
            print(f"Data Directory containing Hugging Face Dataset: {hf_data_directory}")
                
            adata = build_anndata_from_hugging_face_data(hf_data_directory)
                                                            
        elif data_file_format == "pytorch":
            raise NotImplementedError("PyTorch data format is not implemented yet.")
            # st_data_gold_folder_path = data_directory_path / "gold"
            # print(f"Gold data folder: {st_data_gold_folder_path}")
            
        else:
            raise ValueError("Invalid data type. Choose either 'anndata' or 'hugging_face' or 'pytorch'.")
        
        # Compute UMAP
        print("Computing UMAP.")
        adata = compute_umap(adata)
        
        # Save Anndata with integrated UMAP to disk
        print(f"Saving UMAP to {umap_file_path}")
        sc.write(umap_file_path, adata)

    # Compute bLISI
    blisi_npy_path = umap_path / f"{dataset_name}_{tokenizer}_{data_file_format}_blisi.npy"
    avg_blisi, stddev_blisi = compute_batch_lisi(adata, blisi_npy_path, overwrite)
    print(f"Average bLISI: {avg_blisi} | Standard Deviation: {stddev_blisi}")
        
    # Plot UMAP if specified in the config file
    if config['batch_integration']['plot_umap']:
        print("Plotting UMAP.")
        umap_plot_path = str(umap_path / f"{dataset_name}_{tokenizer}_{data_file_format}_umap_plot.pdf")
        plot_umap(adata, umap_plot_path, dataset_name, tokenizer, avg_blisi, stddev_blisi, data_file_format)
        print(f"Saved UMAP plot to {umap_plot_path}")
        

if __name__ == '__main__':
    # Parse the arguments
    args = parse_arguments()

    # Collect configs 
    config = collect_configs(args)
    
    main(config)