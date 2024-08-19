"""
Adapted from cell_neighborhood_rank_tokenizer.ipynb.

Usage
----------
.. terminal
    >>> python create_cell_neighborhood_tokenized_dataset.py config/cell-neighborhood-tokenization_sim1_shifted-log.yaml

Description
----------
This script tokenizes the gene expression data from the preprocessed AnnData (silver) dataset using the Cell Neighborhood Rank Tokenizer. It logs tokenization time and CPU statistics to wandb. It also optionally creates a wandb artifact containing the tokenized dataset and the token dictionary.
"""

import os
import sys
import yaml
import time
import wandb
import pickle
import numpy as np
import scanpy as sc
from pathlib import Path
import scipy.sparse as sp
from typing import Dict, Union, Literal

from nichejepa import tokenizers
from nichejepa.aggregators.aggregate_neighbors import aggregate_neighbors
from nichejepa.normalizers.normalize_by_read_depth import normalize_by_read_depth
from nichejepa.preprocessors.filter_poor_quality_cells import filter_poor_quality_cells


def load_data(st_data_silver_folder_path: Union[Path, str], 
                            dataset_name: str, 
                            species: str,
                            n_batches: int) -> sc.AnnData:
    """
    Load preprocessed data stored in an AnnData object.

    Parameters
    ----------
    - st_data_silver_folder_path (Union[Path, str]): The path to the ST data silver folder.
    - dataset_name (str): The name of the dataset.
    - species (str): The species of the dataset.
    - n_batches (int): The number of batches.

    Returns:
    ----------
    - sc.AnnData: The AnnData object.

    Raises:
    ----------
    - NotImplementedError: If simulated data with multiple batches is not implemented.
    - NotImplementedError: If species other than 'sim' is not implemented.
    """
    if species == 'sim':
        # Load data
        if n_batches == 1:
            adata = sc.read_h5ad(f"{st_data_silver_folder_path}/{dataset_name}/{dataset_name}.h5ad")
            if type(adata.X) == sp._csr.csr_matrix:
                adata.X = adata.X.toarray()
        else:
            raise NotImplementedError("Simulated data with multiple batches not implemented.")
    else:
        raise NotImplementedError(f"Species {species} not implemented.")

    return adata


def create_base_token_dictionary(tokenizer_directory_path: str,
                            token_dictionary_file: str, 
                            adata: Dict,
                            species: str) -> Dict:
    """
    Create a base token dictionary for gene IDs and save it to a file.

    Parameters:
    ----------
    - tokenizer_directory_path (str): The path to the tokenizer directory. (Used for loading geneformer token dictionary)
    - token_dictionary_file (str): The file path to save the token dictionary.
    - adata (Dict): The dictionary containing the gene IDs.
    - species (str): The species of the dataset.
    
    Returns:
    ----------
    - Dict: The token dictionary with gene IDs as keys and indices as values.
    
    Raises:
    ----------
    - NotImplementedError: If species other than 'sim' is not implemented.
    """
    if species == 'sim':
        gene_ids = adata.var["ensembl_id"]
        base_token_dict = {'<pad>': 0, '<mask>': 1}
        base_token_dict.update({gene_id: idx + 2 for idx, gene_id in enumerate(gene_ids)})
    else:
        raise NotImplementedError(f"Species {species} not implemented.")

    with open(token_dictionary_file, "wb") as f:
        pickle.dump(base_token_dict, f)
        
    return base_token_dict


def create_nichejepa_token_dictionary(nichejepa_token_dictionary_file: str,
                                      adata: sc.AnnData, 
                                      token_dict: Dict, 
                                      add_cls_tokens: Literal[True, False],
                                      use_separate_cell_and_neighborhood_tokens: Literal[True, False]) -> Dict:
    """
    Create a NicheJEPA token dictionary.

    Parameters:
    ----------
    - nichejepa_token_dictionary_file (str): The file path to save the token dictionary.
    - adata (sc.AnnData): The AnnData object containing the dataset.
    - token_dict (Dict): The initial token dictionary.
    - add_cls_tokens (bool): Whether to add cls tokens.
    - use_separate_cell_and_neighborhood_tokens (bool): Whether to separate cell and neighborhood tokens.

    Returns:
    ----------
    - Dict: The updated NicheJEPA token dictionary.
    - Int: The number of gene tokens.
    """

    # Add cls token
    if add_cls_tokens:
        def insert_at_index(d, index, key, value):
            items = list(d.items())
            items.insert(index, (key, value))
            new_d = dict(items)
            return new_d
        token_dict = insert_at_index(token_dict, 2, "<cls_cell>", 2)
        if use_separate_cell_and_neighborhood_tokens:
            token_dict = insert_at_index(token_dict, 3, "<cls_neighborhood>", 3)

    # Remove genes not in dataset
    gene_list = adata.var["ensembl_id"].tolist()
    token_dict_copy = token_dict.copy()
    for token in token_dict_copy.keys():
        if "ENS" in token and token not in gene_list:
            del(token_dict[token])
    del(token_dict_copy)

    # Separate gene tokens into cell and neighborhood tokens
    if use_separate_cell_and_neighborhood_tokens:
        nichejepa_token_dict = {
            (key + "_cell" if "ENS" in key else key): value
            for value, key in enumerate(token_dict.keys())}
        nichejepa_token_dict_copy = nichejepa_token_dict.copy()
        i = 0
        for key in nichejepa_token_dict_copy.keys():
            if "ENS" in key:
                nichejepa_token_dict[key.replace("cell", "neighborhood")] = len(nichejepa_token_dict_copy) + i
                i += 1
        del(nichejepa_token_dict_copy)
    else:
        nichejepa_token_dict = {
            key: value
            for value, key in enumerate(token_dict.keys())}

    n_gene_tokens = len([token for token in nichejepa_token_dict.keys() if "ENS" in token])

    # Write to disk
    with open(nichejepa_token_dictionary_file, "wb") as file:
        pickle.dump(nichejepa_token_dict, file)
    
    return nichejepa_token_dict, n_gene_tokens


def normalize_gene_counts(adata: sc.AnnData, 
                          n_batches: int, 
                          norm_factor: str) -> sc.AnnData:
    """
    Normalize gene counts within cells (row-wise) in the given AnnData object.

    Parameters:
    ----------
    - adata (sc.AnnData): Annotated data matrix with gene expression information.
    - n_batches (int): Number of batches in the data.
    - norm_factor (str): List of normalization factors.

    Returns:
    ----------
    - sc.AnnData: The cell-wise gene-count normalized AnnData object.

    Raises:
    ----------
    - NotImplementedError: If `norm_factor` other than `read_depth` is not implemented.
    - NotImplementedError: If simulated data with multiple batches is not implemented.
    """
    if n_batches == 1:
        # Filter to remove poor quality cells
        print("Filtering cells.")
        adata = filter_poor_quality_cells(adata)

        # Aggregate neighborhood cell gene expression
        print("Computing spatial neighborhood graph and aggregating counts.")
        adata = aggregate_neighbors(adata,
                                    radius=27.5)

        # Normalize gene counts by norm_factor
        if norm_factor == "read_depth":
            # Normalize gene counts within each cell (row)
            adata.X = normalize_by_read_depth(adata.X, target_size=10000)
        
            # Normalize gene counts within each neighborhood (row)
            adata.layers["X_neighborhood"] = normalize_by_read_depth(adata.layers["X_neighborhood"], target_size=10000)            
        else:
            raise NotImplementedError(f"Normalization factor {norm_factor} not implemented.")
    else:
        raise NotImplementedError("Simulated data with multiple batches not implemented.")

    return adata


def compute_cell_and_neighborhood_gene_means(adata: sc.AnnData) -> np.ndarray:
    """
    Compute means of gene counts across cells and neighborhoods (column-wise).

    Parameters:
    ----------
    - adata (sc.AnnData): Annotated data matrix with gene expression information.

    Returns:
    ----------
    - np.ndarray: 1D array containing the means of gene counts across cells for each gene
    - np.ndarray: 1D array containing the means of gene counts across neighborhoods for each gene
    """
    cell_gene_means = np.mean(adata.X, axis=0)
    cell_gene_means = np.array(cell_gene_means).reshape(-1)

    neighborhood_gene_means = np.mean(adata.layers["X_neighborhood"], axis=0)
    neighborhood_gene_means = np.array(neighborhood_gene_means).reshape(-1)

    return cell_gene_means, neighborhood_gene_means


def compute_cell_and_neighborhood_gene_nzmeans(adata: sc.AnnData) -> np.ndarray:
    """
    Compute means of non-zero gene counts across cells and neighborhoods (column-wise).

    Parameters:
    ----------
    - adata (sc.AnnData): Annotated data matrix with gene expression information.

    Returns:
    ----------
    - np.ndarray: 1D array containing the means of non-zero gene counts across cells for each gene
    - np.ndarray: 1D array containing the means of non-zero gene counts across neighborhoods for each gene

    """
    masked_data_cells = np.ma.masked_equal(adata.X, 0)
    cell_gene_nzmeans = np.ma.mean(masked_data_cells, axis=0)
    cell_gene_nzmeans = np.array(cell_gene_nzmeans).reshape(-1)

    masked_data_neighborhood = np.ma.masked_equal(adata.layers["X_neighborhood"], 0)
    neighborhood_gene_nzmeans = np.ma.mean(masked_data_neighborhood, axis=0)
    neighborhood_gene_nzmeans = np.array(neighborhood_gene_nzmeans).reshape(-1)

    return cell_gene_nzmeans, neighborhood_gene_nzmeans


def compute_cell_and_neighborhood_gene_logmeans(adata: sc.AnnData) -> np.ndarray:
    """
    Compute log-means of gene counts across cells and neighborhoods (column-wise).

    Parameters:
    ----------
    - adata (sc.AnnData): Annotated data matrix with gene expression information.

    Returns:
    ----------
    - np.ndarray: 1D array containing the log-means of gene counts across cells for each gene
    - np.ndarray: 1D array containing the log-means of gene counts across neighborhoods for each gene
    """
    sc.pp.log1p(adata)
    cell_gene_logmeans = np.mean(adata.X, axis=0)
    cell_gene_logmeans = np.array(cell_gene_logmeans).reshape(-1)

    sc.pp.log1p(adata, layer="X_neighborhood")
    neighborhood_gene_logmeans = np.mean(adata.layers["X_neighborhood"], axis=0)
    neighborhood_gene_logmeans = np.array(neighborhood_gene_logmeans).reshape(-1)

    return cell_gene_logmeans, neighborhood_gene_logmeans


def main():
    
    # Get the config file name from command line argument
    config_fname = sys.argv[1]

    # Read arguments from config file
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb run
    run = wandb.init(project=config["wandb"]["project"],
                        job_type=config["wandb"]["job_type"],
                        tags=config["wandb"]["tags"],
                        config=config)
        
    # Configure experiment arguments
    tokenizer = config["experiment"]["tokenizer"]
    dataset_name = config["experiment"]["dataset_name"]
    n_batches = config["experiment"]["n_batches"]
    species = config["experiment"]["species"]
    print(f"Tokenizer: {tokenizer}")
    print(f"Dataset: {dataset_name}")
    
    # Configure data paths
    data_directory_path = config["paths"]["data_directory_path"]
    st_data_silver_folder_path = f"{data_directory_path}/silver"
    st_data_gold_folder_path = f"{data_directory_path}/gold"

    # Configure tokenization parameters
    add_cls_tokens = config["tokenization"]["add_cls_tokens"]
    use_separate_cell_and_neighborhood_tokens = config["tokenization"]["use_separate_cell_and_neighborhood_tokens"]
    norm_method = config["tokenization"]["norm_method"]
    norm_factor = config["tokenization"]["norm_factor"]
    custom_attr_name_dict = config["tokenization"]["custom_attr_name_dict"]
    label_cats = config["tokenization"]["label_cats"]

    # Configure tokenizer paths
    tokenizer_directory_path = config["paths"]["tokenizer_directory_path"]
    tokenizer_folder_path = f"{tokenizer_directory_path}/{tokenizer}/{dataset_name}"
    token_dictionary_file = f"{tokenizer_folder_path}/token_dictionary.pkl"
    nichejepa_token_dictionary_file = f"{tokenizer_folder_path}/token_dictionary" \
                                    f"{'_cls' if add_cls_tokens else '_no_cls'}" \
                                    f"{'_sep' if use_separate_cell_and_neighborhood_tokens else '_no_sep'}.pkl"
    os.makedirs(f"{tokenizer_folder_path}/{dataset_name}", exist_ok=True)

    # Set input and output directories
    input_directory = f"{st_data_silver_folder_path}/{dataset_name}"
    output_directory = f"{st_data_gold_folder_path}/{tokenizer}/{dataset_name}"

    # Configure tokenized dictionary specific paths
    cell_gene_means_dictionary_file = f"{tokenizer_folder_path}/cell_gene_{norm_factor}_means_dictionary.pkl"
    cell_gene_nzmeans_dictionary_file = f"{tokenizer_folder_path}/cell_gene_{norm_factor}_nzmeans_dictionary.pkl"
    cell_gene_logmeans_dictionary_file = f"{tokenizer_folder_path}/cell_gene_{norm_factor}_logmeans_dictionary.pkl"
    neighborhood_gene_means_dictionary_file = f"{tokenizer_folder_path}/neighborhood_gene_{norm_factor}_means_dictionary.pkl"
    neighborhood_gene_nzmeans_dictionary_file = f"{tokenizer_folder_path}/neighborhood_gene_{norm_factor}_nzmeans_dictionary.pkl"
    neighborhood_gene_logmeans_dictionary_file = f"{tokenizer_folder_path}/neighborhood_gene_{norm_factor}_logmeans_dictionary.pkl"
        
    # Load data
    adata = load_data(st_data_silver_folder_path, dataset_name, species, n_batches)

    # Create token dictionary
    base_token_dict = create_base_token_dictionary(tokenizer_directory_path, token_dictionary_file, adata, species)
    
    # Create nichejepa token dictionary
    _, n_gene_tokens = create_nichejepa_token_dictionary(nichejepa_token_dictionary_file, adata, base_token_dict, add_cls_tokens, use_separate_cell_and_neighborhood_tokens)

    # Normalize gene counts
    adata = normalize_gene_counts(adata, n_batches, norm_factor)

    # Compute means, non-zero means, and log-means
    cell_gene_means, neighborhood_gene_means = compute_cell_and_neighborhood_gene_means(adata)
    cell_gene_nzmeans, neighborhood_gene_nzmeans = compute_cell_and_neighborhood_gene_nzmeans(adata)
    cell_gene_logmeans, neighborhood_gene_logmeans = compute_cell_and_neighborhood_gene_logmeans(adata)
            
    # Create norm factor dictionaries for tokenizer and store on disk
    cell_gene_means_dict = {}
    cell_gene_nzmeans_dict = {}
    cell_gene_logmeans_dict = {}
    neighborhood_gene_means_dict = {}
    neighborhood_gene_nzmeans_dict = {}
    neighborhood_gene_logmeans_dict = {}
    
    for (gene,
         cell_gene_mean,
         cell_gene_nzmedian,
         cell_gene_logmean,
         neighborhood_gene_mean,
         neighborhood_gene_nzmedian,
         neighborhood_gene_logmean) in zip(
            adata.var["ensembl_id"].values,
            cell_gene_means,
            cell_gene_nzmeans,
            cell_gene_logmeans,
            neighborhood_gene_means,
            neighborhood_gene_nzmeans,
            neighborhood_gene_logmeans):
        cell_gene_means_dict[gene] = cell_gene_mean
        cell_gene_nzmeans_dict[gene] = cell_gene_nzmedian
        cell_gene_logmeans_dict[gene] = cell_gene_logmean
        neighborhood_gene_means_dict[gene] = neighborhood_gene_mean
        neighborhood_gene_nzmeans_dict[gene] = neighborhood_gene_nzmedian
        neighborhood_gene_logmeans_dict[gene] = neighborhood_gene_logmean
        
    # Save norm_factor dictionaries to disk
    with open(cell_gene_means_dictionary_file, "wb") as file:
        pickle.dump(cell_gene_means_dict, file)
    with open(cell_gene_nzmeans_dictionary_file, "wb") as file:
        pickle.dump(cell_gene_nzmeans_dict, file)
    with open(cell_gene_logmeans_dictionary_file, "wb") as file:
        pickle.dump(cell_gene_logmeans_dict, file)
    with open(neighborhood_gene_means_dictionary_file, "wb") as file:
        pickle.dump(neighborhood_gene_means_dict, file)
    with open(neighborhood_gene_nzmeans_dictionary_file, "wb") as file:
        pickle.dump(neighborhood_gene_nzmeans_dict, file)
    with open(neighborhood_gene_logmeans_dictionary_file, "wb") as file:
        pickle.dump(neighborhood_gene_logmeans_dict, file)    

    # Set output file prefix    
    model_input_size = n_gene_tokens * 2
    output_file_prefix = f"{dataset_name}_{norm_factor}_{norm_method}_{model_input_size}"
    
    start = time.time()
    
    # Tokenize data
    tk = tokenizers.CellNeighborhoodRankTokenizer(
            model_input_size=model_input_size,
            norm_method=norm_method,
            norm_factor=norm_factor,
            use_separate_cell_and_neighborhood_tokens=use_separate_cell_and_neighborhood_tokens,
            cell_gene_means_file=cell_gene_means_dictionary_file,
            cell_gene_nzmeans_file=cell_gene_nzmeans_dictionary_file,
            cell_gene_logmeans_file=cell_gene_logmeans_dictionary_file,
            neighborhood_gene_means_file=neighborhood_gene_means_dictionary_file,
            neighborhood_gene_nzmeans_file=neighborhood_gene_nzmeans_dictionary_file,
            neighborhood_gene_logmeans_file=neighborhood_gene_logmeans_dictionary_file,
            token_dictionary_file=nichejepa_token_dictionary_file,
            custom_attr_name_dict=custom_attr_name_dict)
        
    tk.tokenize_data(input_directory=input_directory,
                        output_directory=output_directory,
                        output_file_prefix=output_file_prefix)
    end = time.time()
    
    wandb.log({"tokenization_time": end - start})

    if config["wandb"]["save_artifact"]:
        # Create wandb artifact
        artifact = wandb.Artifact(name=f"{config['experiment']['dataset_name']}-tokenized",
                                type=f"{config['experiment']['tokenizer']}",
                                metadata=config)
        # add files to artifact
        artifact.add_file(local_path=nichejepa_token_dictionary_file, name="NicheJEPA Token Dictionary")
        artifact.add_dir(local_path=st_data_silver_folder_path, name=f"AnnData (silver)")
        artifact.add_dir(local_path=output_directory, name=f"Hugging Face Dataset (gold)")
        
        # save artifact
        artifact.save()
        
    wandb.finish()
    
    
if __name__ == "__main__":
    main()