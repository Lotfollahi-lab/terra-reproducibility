"""
Adapted from cell_neighborhood_rank_tokenizer.ipynb.

Usage
----------
.. terminal
    >>> python analysis/tokenization/create_cell_tokenized_dataset.py --config_file config/cell-neighborhood-tokenizer/cell-neighborhood-tokenizer_sss2-1b_1p_read-depth_shifted-log.yaml --override nproc=1 overwrite=True

Description
----------
This script tokenizes the gene expression data from the preprocessed AnnData (silver) dataset using the Cell Neighborhood Tokenizer or the Cell Graph Tokenizer. It logs tokenization time and CPU statistics to wandb. It also optionally creates a wandb artifact containing the tokenized dataset and the token dictionary. The `override` parameter in the usage command overrides number of processes to use for tokenization and whether to rebuild and overwrite the intermediate files required for creating the tokenized data.
"""

import os
import yaml
import time
import wandb
import pickle
import pstats
import cProfile
import argparse
import numpy as np
import scanpy as sc
import squidpy as sq
import anndata as ad
from pathlib import Path
from typing import List, Dict, Tuple, Union, Literal, Optional

from nichejepa.tokenizers.cell_tokenizers import CellNeighborhoodTokenizer, CellGraphTokenizer
from nichejepa.preprocessors.aggregators import aggregate_neighbors
from nichejepa.preprocessors.normalizers import normalize_by_read_depth
from nichejepa.preprocessors.filters import filter_poor_quality_cells



def process_anndata_batches(input_directory: Union[Path, str],
                                gene_panel_ID_to_gene_panel_dict_file: Union[Path, str],
                                file_path_to_gene_panel_ID_dict_file: Union[Path, str],
                                gene_list_file: Union[Path, str],
                                overwrite: Literal[True,False] = False) \
                                -> Tuple[List[sc.AnnData], List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Capture set of unique genes, gene panels, batches, assays, tissues, and species across all batches of anndata in the input directory. These are then used to create the token mappings (vocabularies) for the tokenizer.

    Parameters
    ----------
    - input_directory (Union[Path, str]): The path to the input directory.
    - gene_panel_ID_to_gene_panel_dict_file (Union[Path, str]): The file path to save the gene panel ID to gene panel dictionary.
    - file_path_to_gene_panel_ID_dict_file (Union[Path, str]): The file path to save the file path to gene panel ID dictionary.
    - gene_list_file (Union[Path, str]): The file path to save the gene list.
    - overwrite (Literal[True,False]): Whether to overwrite the gene panel mappings.

    Returns:
    ----------
    - List[sc.AnnData]: A list of AnnData objects, one per batch.
    - List[str]: A list of batch IDs, one per batch.
    - List[str]: A list of gene panel IDs, one per unique gene panel across batches.
    - List[str]: A list of assay IDs, one per unique assay across batches.
    - List[str]: A list of species IDs, one per unique species across batches.
    - List[str]: A list of tissue IDs, one per unique tissue across batches.
    - List[str]: A list of all unique gene IDs across batches.
    """
    file_paths = list(input_directory.glob(f"**/*.h5ad"))
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No .h5ad files found in {input_directory}")
    
    adata_batch_list = []

    # If gene panel ID mappings exist and overwrite is False, load them from disk
    if os.path.exists(gene_list_file) and \
            os.path.exists(gene_panel_ID_to_gene_panel_dict_file) and \
            os.path.exists(file_path_to_gene_panel_ID_dict_file) \
            and not overwrite:
        print("Loading gene panel mappings from disk.")
        with open(gene_list_file, "rb") as file:
            gene_list = pickle.load(file)
        with open(gene_panel_ID_to_gene_panel_dict_file, "rb") as file:
            gene_panel_ID_to_gene_panel_dict = pickle.load(file)
        with open(file_path_to_gene_panel_ID_dict_file, "rb") as file:
            file_path_to_gene_panel_ID_dict = pickle.load(file)

        for file_path in file_paths:
            print(f"Loading data from {file_path}")
            adata = sc.read_h5ad(file_path)
            adata_batch_list.append(adata)
              
    # If gene panel ID mappings do not exist or overwrite is True, create them
    else:
        print("Creating gene panel mappings.")
                
        # Initialize lists for capturing names for special tokens related to batch, gene panel, assay, species, and tissue from batched data
        batch_id_key_list = []
        gene_panel_id_key_list = []
        assay_key_list = []
        species_key_list = []
        tissue_key_list = []

        # Initialize gene panel mappings
        unique_gene_panels_count = 0
        unique_gene_panels_list = []
        gene_panel_ID_to_gene_panel_dict = {}
        file_path_to_gene_panel_ID_dict = {}
                    
        for file_path in file_paths:
            print(f"Loading data from {file_path}")
            adata = sc.read_h5ad(file_path)

            # ------------------------------------------------------------------------------------
            # Capture dataset_id for the current batch of AnnData
            # e.g. dataset_id = 0 or 1 or 2 or 3...
            dataset_id = adata.uns['dataset_id']
            print(f"Dataset ID: {dataset_id}")
            # ------------------------------------------------------------------------------------
            # Capture batch_id_key for the current batch of AnnData
            # e.g. batch_id_key = "0_batch1" or "1_batch1" or "0_batch2"...
            batch_id = adata.uns['batch']
            batch_id_key = f"spv_{dataset_id}_{batch_id}"    
            print(f"Batch ID Key: {batch_id_key}")
            batch_id_key_list.append(batch_id_key)
            # ------------------------------------------------------------------------------------ 
            # Capture assay keys for the current batch of AnnData
            # e.g. assay_key = "merfish" or "starmap" or "sim"...
            assay_key = f"spv_{adata.uns['assay']}"
            if assay_key not in assay_key_list:
                assay_key_list.append(assay_key)
            # ------------------------------------------------------------------------------------ 
            # Capture species keys for the current batch of AnnData
            # e.g. species_key = "mus_musculus" or "homo_sapiens" or "sim"...
            species_key = f"spv_{adata.uns['species']}"
            if species_key not in species_key_list:
                species_key_list.append(species_key)          
            # ------------------------------------------------------------------------------------ 
            # Capture tissue keys for the current batch of AnnData
            # e.g. tissue_key = "brain" or "sim"...
            tissue_key = f"spv_{adata.uns['tissue']}"
            if tissue_key not in tissue_key_list:
                tissue_key_list.append(tissue_key)            
            # ------------------------------------------------------------------------------------
            # Capture gene panel id key and gene panel mappings for the current batch of AnnData
            # Creates gene panel as an intersection of Ensembl gene set and genes in adata
            gene_panel = set(adata.var["ensembl_id"].values.tolist())
            
            # If gene panel is unique, assign a unique ID to it
            if gene_panel not in unique_gene_panels_list:
                unique_gene_panels_list.append(gene_panel)
                gene_panel_ID = f"gene_panel{unique_gene_panels_count}"
                gene_panel_ID_to_gene_panel_dict[gene_panel_ID] = gene_panel
                file_path_to_gene_panel_ID_dict[str(file_path)] = gene_panel_ID
                unique_gene_panels_count += 1
                
                # Creates gene panel id key list for use in token dictionary
                # e.g. gene_panel_id_key_list = ["spv_gene_panel0", "gene_panel1", ...]
                gene_panel_id_key = f"spv_{gene_panel_ID}"
                gene_panel_id_key_list.append(gene_panel_id_key)
                
            # If gene panel is not unique, assign find its ID and append file path
            else:
                gene_panel_ID = next((k for k, v in gene_panel_ID_to_gene_panel_dict.items() if v == gene_panel), None)
                file_path_to_gene_panel_ID_dict[str(file_path)] = gene_panel_ID
            # ------------------------------------------------------------------------------------
            # Append processed and loaded adata to adata_batch_list
            adata_batch_list.append(adata)

        # Build a complete list of genes in the dataset across all batches of anndata
        # e.g. gene_list = ['ENSMUSG00000020734','ENSMUSG00000021099', 'ENSMUSG00000019787']
        gene_list = list(set().union(*unique_gene_panels_list))
        gene_list = sorted(gene_list)
            
        # Save gene panel mappings to disk
        with open(gene_list_file, "wb") as file:
            print(f"Saving gene list to {gene_list_file}")
            pickle.dump(gene_list, file)
        with open(gene_panel_ID_to_gene_panel_dict_file, "wb") as file:
            print(f"Saving gene panel ID to gene panel dictionary to {gene_panel_ID_to_gene_panel_dict_file}")
            pickle.dump(gene_panel_ID_to_gene_panel_dict, file)
        with open(file_path_to_gene_panel_ID_dict_file, "wb") as file:
            print(f"Saving file path to gene panel ID dictionary to {file_path_to_gene_panel_ID_dict_file}")
            pickle.dump(file_path_to_gene_panel_ID_dict, file)
    
    return adata_batch_list, \
            batch_id_key_list, \
            gene_panel_id_key_list, \
            assay_key_list, \
            species_key_list, \
            tissue_key_list, \
            gene_list


def build_token_dictionary(token_dictionary_file: str,
                                tokenizer: str,
                                gene_list: List[str],
                                batch_id_key_list: List[str],
                                gene_panel_id_key_list: List[str],
                                assay_key_list: List[str],
                                species_key_list: List[str],
                                tissue_key_list: List[str],
                                overwrite: Literal[True,False]=True) -> Dict:
    """
    Build token dictionary for NicheJEPA.
    
    Parameters:
    ----------
    - token_dictionary_file (str): The file path to save the token dictionary.
    - tokenizer (str): The tokenizer to build token dictionary for.
    - gene_list (List[str]): The list of gene IDs.
    - batch_id_key_list (List[str]): A list of batch IDs, one per batch.
    - gene_panel_id_key_list (List[str]): A list of gene panel IDs, one per unique gene panel across batches.
    - assay_key_list (List[str]): A list of assay IDs, one per unique assay across batches.
    - species_key_list (List[str]): A list of species IDs, one per unique species across batches.
    - tissues (List[str]): A list of tissue IDs, one per unique tissue across batches.
    - overwrite (Literal[True,False]): Whether to overwrite the existing token dictionary.
    
    Returns:
    ----------
    - Dict: The token dictionary.
    """
    # Load tokenulary dictionary if it exists and not overwrite, else create it
    if os.path.exists(token_dictionary_file) and not overwrite:
        print(f"Loading token dictionary from {token_dictionary_file}")
        with open(token_dictionary_file, "rb") as file:
            token_dict = pickle.load(file)
    else:
        print(f"Creating token dictionary for NicheJEPA and saving to {token_dictionary_file}.")
        
        # Initialize with mandatory tokens
        token_dict = {'<pad>': 0, '<mask>': 1}
        
        # Add cls tokens depending on the tokenizer
        if tokenizer == "cell-neighborhood-tokenizer":
            token_dict.update({'<cls_cell>': 2, '<cls_neighborhood>': 3})
        elif tokenizer == "cell-graph-tokenizer":
            # cls_0: 2 is for cell, cls_1: 3 is for 1st neighbor, cls_2: 4 is for 2nd neighbor, ..., upto 100 neighbors
            token_dict.update({f'<cls_{i}>' : i + 2 for i in range(100)})
        
        # Add gene tokens
        # e.g. token_dict = {'ENSG00000139618': 4, ...}
        curr_token_dict_size = len(token_dict)
        token_dict.update({gene_id: idx + 4 for idx, gene_id in enumerate(gene_list)})
        
        # Add 1 token per batch_id in batch_id_key_list
        # e.g. token_dict = {'0_batch1': 4 + len(gene_list), ...}
        curr_token_dict_size = len(token_dict)
        token_dict.update({batch_id_key: idx + curr_token_dict_size for idx, batch_id_key in enumerate(batch_id_key_list)})
        
        # Add 1 token per gene_panel_id in gene_panel_id_key_list
        # e.g. token_dict = {'gene_panel0': 4 + len(gene_list) + len(batch_id_key_list), ...}
        curr_token_dict_size = len(token_dict)
        token_dict.update({gene_panel_id_key: idx + curr_token_dict_size for idx, gene_panel_id_key in enumerate(gene_panel_id_key_list)})
        
        # Add 1 token per assay in assay_key_list
        # e.g. token_dict = {'merfish': 4 + len(gene_list) + len(batch_id_key_list) + len(gene_panel_id_key_list), ...}
        curr_token_dict_size = len(token_dict)
        token_dict.update({assay_key: idx + curr_token_dict_size for idx, assay_key in enumerate(assay_key_list)})
        
        # Add 1 token per species in species_key_list
        # e.g. token_dict = {'mus_musculus': 4 + len(gene_list) + len(batch_id_key_list) + len(gene_panel_id_key_list) + len(assay_key_list), ...}
        curr_token_dict_size = len(token_dict)
        token_dict.update({species_key: idx + curr_token_dict_size for idx, species_key in enumerate(species_key_list)})
        
        # Add 1 token per tissue in tissues_key_list
        # e.g. token_dict = {'brain': 4 + len(gene_list) + len(batch_id_key_list) + len(gene_panel_id_key_list) + len(assay_key_list) + len(species), ...}
        curr_token_dict_size = len(token_dict)
        token_dict.update({tissue_key: idx + curr_token_dict_size for idx, tissue_key in enumerate(tissue_key_list)})
        
        # Add meta tokens
        curr_token_dict_size = len(token_dict)
        token_dict.update({'spt_batch': curr_token_dict_size, 
                           'spt_gene_panel': curr_token_dict_size + 1,
                           'spt_assay': curr_token_dict_size + 2,
                           'spt_species': curr_token_dict_size + 3,
                           'spt_tissue': curr_token_dict_size + 4})

        with open(token_dictionary_file, "wb") as file:
            pickle.dump(token_dict, file)

    print(token_dict)
    print("")
    
    return token_dict


def compute_max_neighbors(adata: sc.AnnData) -> int:
    """
    Compute the maximum number of neighbors in the dataset.

    Parameters:
    ----------
    - adata (sc.AnnData): Annotated data matrix with gene expression information.

    Returns:
    ----------
    - int: The maximum number of neighbors.
    """
    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key="spatial",
                            radius=27.50,
                            )

    max_neighbors = int(np.max(adata.obsp["spatial_connectivities"].sum(1)))

    return max_neighbors
    

def normalize_gene_counts(adata: sc.AnnData, 
                          norm_factor: str) -> sc.AnnData:
    """
    Normalize gene counts within cells (row-wise) in the given AnnData object.

    Parameters:
    ----------
    - adata (sc.AnnData): Annotated data matrix with gene expression information.
    - norm_factor (str): List of normalization factors.

    Returns:
    ----------
    - sc.AnnData: The cell-wise gene-count normalized AnnData object.

    Raises:
    ----------
    - NotImplementedError: If `norm_factor` other than `read_depth` is not implemented.
    - NotImplementedError: If simulated data with multiple batches is not implemented.
    """
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


def obtain_cumulative_times(profiler_stats_fname: str,
                            func_list_to_profile: Optional[List[str]] = ['tokenize_data', 'tokenize_files', 'create_dataset']) -> Dict:
    """
    Obtain cumulative times from the cProfile stats file for a specified list of functions.

    Parameters:
    ----------
    - profiler_stats_fname (str): The file path to the cProfile stats file.
    - func_list_to_profile (Optional[List[str]]): The list of functions to profile.

    Returns:
    ----------
    - Dict: A dictionary containing the cumulative times for each specified function.
    """
    # Create stats object
    stats = pstats.Stats(profiler_stats_fname)
    stats.strip_dirs().sort_stats('cumulative')

    cumulative_times = {}
    for func, stat in stats.stats.items():
        file_name, line_num, func_name = func
        if func_name in func_list_to_profile:
            cc, nc, tt, ct, callers = stat
            print(f"Function: {func_name} | Cumulative Time: {ct}")
            cumulative_times[func_name] = ct

    return cumulative_times


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
            if key == 'nproc':
                config['experiment'][key] = int(value)
            elif key == 'overwrite':
                if value == 'True':
                    config['experiment'][key] = True
                elif value == 'False':
                    config['experiment'][key] = False
                    
    return config


def main(config: Dict):
        
    # ------------------------------------------------------------

    # Configure experiment arguments
    experiment_name = config["experiment"]["experiment_name"]
    experiment_description = config["experiment"]["experiment_description"]
    tokenizer = config["experiment"]["tokenizer"]
    nproc = config["experiment"]["nproc"]
    overwrite = config["experiment"]["overwrite"]
    print(f"Experiment Name: {experiment_name}")
    print(f"Experiment Description: {experiment_description}")
    print(f"Tokenizer: {tokenizer}")
    
    # Configure tokenization parameters
    use_generator = config["tokenization"]["use_generator"]
    norm_method = config["tokenization"]["norm_method"]
    norm_factor = config["tokenization"]["norm_factor"]
    prenormalize = config["tokenization"]["prenormalize"]
    processing_mode = config["tokenization"]["processing_mode"]
    num_shards = config["tokenization"]["num_shards"]
    keep_in_memory = config["tokenization"]["keep_in_memory"]
    radius = config["tokenization"]["radius"]
    delaunay_radius_union = config["tokenization"]["delaunay_radius_union"]

    # ------------------------------------------------------------
        
    # Configure data paths
    data_directory_path = Path(config["paths"]["data_directory_path"])
    st_data_silver_folder_path = data_directory_path / "silver"
    st_data_gold_folder_path = data_directory_path / "gold"
        
    # Configure cache path
    cache_directory_path = data_directory_path / "cache"
    os.makedirs(cache_directory_path, exist_ok=True)

    # Configure tokenizer paths
    # tokenizer_directory_path = Path(".") / "tokenizer"
    tokenizer_directory_path = data_directory_path / "tokenizer"
    tokenizer_folder_path = tokenizer_directory_path / tokenizer / experiment_name
    token_dictionary_file = tokenizer_folder_path / "token_dictionary.pkl"
    os.makedirs(tokenizer_folder_path, exist_ok=True)

    # Set input and output directories
    input_directory = st_data_silver_folder_path / experiment_name
    output_directory = st_data_gold_folder_path / tokenizer / experiment_name
    
    # Set file paths for gene panel mappings
    gene_list_file = tokenizer_folder_path / "gene_list.pkl"
    gene_panel_ID_to_gene_panel_dict_file = tokenizer_folder_path / "gene_panel_ID_to_gene_panel_dict.pkl"
    file_path_to_gene_panel_ID_dict_file = tokenizer_folder_path / "file_path_to_gene_panel_ID_dict.pkl"

    # ------------------------------------------------------------

    log_directory_path = Path("./logs") / experiment_name
        
    # Set profiler path
    profiler_path = log_directory_path / "cProfiler" / tokenizer
    os.makedirs(profiler_path, exist_ok=True)

    # Initialize wandb run
    wandb_tags = [experiment_name, tokenizer, norm_method, norm_factor, processing_mode]
    wandb.init(project=config["wandb"]["project"],
                        job_type=config["wandb"]["job_type"],
                        tags=wandb_tags,
                        mode=config["wandb"]["mode"],
                        config=config)
        
    # ------------------------------------------------------------

    # Load data and create gene panel mappings
    adata_batch_list, \
    batch_id_key_list, \
    gene_panel_id_key_list, \
    assay_key_list, \
    species_key_list, \
    tissue_key_list, \
    gene_list \
        = process_anndata_batches(input_directory,
                                    gene_panel_ID_to_gene_panel_dict_file,
                                    file_path_to_gene_panel_ID_dict_file,
                                    gene_list_file, 
                                    overwrite)

    # Build token dictionary
    token_dict = build_token_dictionary(token_dictionary_file,
                                                tokenizer,
                                                gene_list,
                                                batch_id_key_list,
                                                gene_panel_id_key_list,
                                                assay_key_list,
                                                species_key_list,
                                                tissue_key_list,
                                                overwrite)
    print(f"Number of Tokens: {len(token_dict)}.")
    
    # Set model input size based on tokenizer
    try:
        model_input_size = config["experiment"]["model_input_size"]
    except:
        if tokenizer == "cell-neighborhood-tokenizer":
            model_input_size = len(gene_list) * 2
        elif tokenizer == "cell-graph-tokenizer":
            try:
                tokens_per_cell = config["tokenization"]["tokens_per_cell"]
            except:
                raise ValueError("Please specify the number of tokens per cell.")
            adata = ad.concat(adata_batch_list, join="inner", index_unique="-")
            # TODO: Fix the following code block to handle multiple batches of data
            max_neighbors = compute_max_neighbors(adata)
            model_input_size = tokens_per_cell * min(16, max_neighbors) + 1

    print(f"Model Input Size: {model_input_size}")
    
    # Set output file prefix
    output_file_prefix = f"{experiment_name}_{num_shards}_{norm_factor}_{norm_method}_{model_input_size}"

    # ------------------------------------------------------------
    
    # TODO: Fix the following code block to handle multiple batches of data
    # Configure tokenized dictionary specific paths
    cell_gene_means_dictionary_file = tokenizer_folder_path / f"cell_gene_{norm_factor}_means_dictionary.pkl"
    cell_gene_nzmeans_dictionary_file = tokenizer_folder_path / f"cell_gene_{norm_factor}_nzmeans_dictionary.pkl"
    cell_gene_logmeans_dictionary_file = tokenizer_folder_path / f"cell_gene_{norm_factor}_logmeans_dictionary.pkl"
    neighborhood_gene_means_dictionary_file = tokenizer_folder_path / f"neighborhood_gene_{norm_factor}_means_dictionary.pkl"
    neighborhood_gene_nzmeans_dictionary_file = tokenizer_folder_path / f"neighborhood_gene_{norm_factor}_nzmeans_dictionary.pkl"
    neighborhood_gene_logmeans_dictionary_file = tokenizer_folder_path / f"neighborhood_gene_{norm_factor}_logmeans_dictionary.pkl"

    if prenormalize:
        # Concatenate AnnData objects
        adata = ad.concat(adata_batch_list, join="inner", index_unique="-")
        
        # Normalize gene counts
        adata = normalize_gene_counts(adata, norm_factor)

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
        del adata
            
    del adata_batch_list
    
    # ------------------------------------------------------------
    
    # Tokenize data
    with cProfile.Profile() as pr:
        start = time.time()
        
        if tokenizer == 'cell-neighborhood-tokenizer':
            Tokenizer = CellNeighborhoodTokenizer
        elif tokenizer == 'cell-graph-tokenizer':
            Tokenizer = CellGraphTokenizer
            
        # Instantiate tokenizer
        tk = Tokenizer(
                nproc=nproc,
                processing_mode=processing_mode,
                model_input_size=model_input_size,
                radius=radius,
                delaunay_radius_union=delaunay_radius_union,
                norm_method=norm_method,
                norm_factor=norm_factor,
                gene_panel_ID_to_gene_panel_dict_file=gene_panel_ID_to_gene_panel_dict_file,
                file_path_to_gene_panel_ID_dict_file=file_path_to_gene_panel_ID_dict_file,
                cell_gene_means_file=cell_gene_means_dictionary_file,
                cell_gene_nzmeans_file=cell_gene_nzmeans_dictionary_file,
                cell_gene_logmeans_file=cell_gene_logmeans_dictionary_file,
                neighborhood_gene_means_file=neighborhood_gene_means_dictionary_file,
                neighborhood_gene_nzmeans_file=neighborhood_gene_nzmeans_dictionary_file,
                neighborhood_gene_logmeans_file=neighborhood_gene_logmeans_dictionary_file,
                token_dictionary_file=token_dictionary_file)

        # Tokenize data 
        tk.tokenize_data(input_directory=input_directory,
                            output_directory=output_directory,
                            output_file_prefix=output_file_prefix,
                            use_generator=use_generator,
                            cache_directory_path=cache_directory_path,
                            keep_in_memory=keep_in_memory,
                            num_shards=num_shards)
    
        end = time.time()
    
        # Save time profiling stats to disk
        profiler_stats_fname = str(profiler_path / f"{output_file_prefix}_{nproc}_{processing_mode}.prof")
        pr.dump_stats(profiler_stats_fname)
        print(f"Profiling stats saved to: {profiler_stats_fname}")
                    
        # Extract cumulative times from profiler stats
        cumulative_times = obtain_cumulative_times(profiler_stats_fname)

    
    # Log tokenization and profiling times to wandb
    wandb.log({"nproc": nproc})
    wandb.log({"tokenization_time": end - start})
    for func_name, ct in cumulative_times.items():
        wandb.log({f"cumulative_time_{func_name}": ct})

    if config["wandb"]["save_artifact"]:
        # Create wandb artifact
        artifact = wandb.Artifact(name=f"{config['experiment']['experiment_name']}-tokenized",
                                    type=f"{config['experiment']['tokenizer']}",
                                    metadata=config)
        # add files to artifact
        artifact.add_file(local_path=token_dictionary_file, name="NicheJEPA Neighborhood Token Dictionary")
        artifact.add_file(local_path=profiler_stats_fname, name="cProfile Stats")
        artifact.add_dir(local_path=st_data_silver_folder_path, name=f"AnnData (silver)")
        artifact.add_dir(local_path=output_directory, name=f"Hugging Face Dataset (gold)")
        
        # save artifact
        artifact.save()
        
    wandb.finish()
    
    
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Collect configs 
    config = collect_configs(args)
        
    main(config)