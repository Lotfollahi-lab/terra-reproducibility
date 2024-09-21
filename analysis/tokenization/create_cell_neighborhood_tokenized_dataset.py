"""
Adapted from cell_neighborhood_rank_tokenizer.ipynb.

Usage
----------
.. terminal
    >>> python create_cell_neighborhood_tokenized_dataset.py --config_file config/cell-neighborhood-tokenization_sim1_shifted-log.yaml

Description
----------
This script tokenizes the gene expression data from the preprocessed AnnData (silver) dataset using the Cell Neighborhood Rank Tokenizer. It logs tokenization time and CPU statistics to wandb. It also optionally creates a wandb artifact containing the tokenized dataset and the token dictionary.
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
import anndata as ad
from pathlib import Path
from typing import Tuple, List, Dict, Union, Literal, Optional

from pyensembl import EnsemblRelease

from nichejepa import tokenizers
from nichejepa.aggregators.aggregate_neighbors import aggregate_neighbors
from nichejepa.normalizers.normalize_by_read_depth import normalize_by_read_depth
from nichejepa.preprocessors.filter_poor_quality_cells import filter_poor_quality_cells


def map_anndata_batches_to_gene_panels(input_directory: Union[Path, str], 
                                        gene_panel_ID_to_gene_panel_dict_file: Union[Path, str],
                                        file_path_to_gene_panel_ID_dict_file: Union[Path, str],
                                        gene_list_file: Union[Path, str],
                                        n_gene_panels: int = 1,
                                        overwrite: Literal[True,False] = False) -> Tuple[List[sc.AnnData], List, Dict, Dict]:
    """
    Create gene panel mappings for each batch of data and load the data into AnnData objects.

    Parameters
    ----------
    - input_directory (Union[Path, str]): The path to the input directory.
    - gene_panel_ID_to_gene_panel_dict_file (Union[Path, str]): The file path to save the gene panel ID to gene panel dictionary.
    - file_path_to_gene_panel_ID_dict_file (Union[Path, str]): The file path to save the file path to gene panel ID dictionary.
    - gene_list_file (Union[Path, str]): The file path to save the gene list.
    - n_gene_panels (int): The number of unique gene panels in the dataset.
    - overwrite (Literal[True,False]): Whether to overwrite the gene panel mappings.

    Returns:
    ----------
    - List[sc.AnnData]: A list of AnnData objects.
    - List: A list of all unique gene IDs across batches.
    - Dict: A dictionary mapping gene panel IDs to gene panels.
    - Dict: A dictionary mapping file paths to gene panel IDs.
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
        # Map gene panels to gene panel IDs and file paths
        if n_gene_panels == 1:
            # Load first batch of data
            print(f"Loading data from {file_paths[0]}")
            adata = sc.read_h5ad(file_paths[0])
            adata_batch_list.append(adata)
            
            # Extract gene ensembl IDs and create a gene panel
            gene_ids = adata.var["ensembl_id"]
            gene_list = gene_ids.values.tolist()
            gene_panel = set(gene_list)
            
            # Map gene panel ID to gene panel and file paths
            gene_panel_ID_to_gene_panel_dict = {0: gene_panel}
            file_path_to_gene_panel_ID_dict = {str(file_paths[0]): 0}
            
            # Load remaining batches of data
            for file_path in file_paths[1:]:
                print(f"Loading data from {file_path}")
                adata = sc.read_h5ad(file_path)
                adata_batch_list.append(adata)
                file_path_to_gene_panel_ID_dict[str(file_path)] = 0
            
        elif n_gene_panels > 1:
            # Initialize gene panel ID and mappings
            unique_gene_panels_count = 0
            unique_gene_panels_list = []
            gene_panel_ID_to_gene_panel_dict = {}
            file_path_to_gene_panel_ID_dict = {}

            # Iterate over each batch of data
            for file_path in file_paths:
                # Load each batch of data
                print(f"Loading data from {file_path}")
                adata = sc.read_h5ad(file_path)
                adata_batch_list.append(adata)
                
                # Extract gene ensembl IDs and create a gene panel
                gene_ids = adata.var["ensembl_id"]
                gene_panel = set(gene_ids.values.tolist())
                
                # If gene panel is unique, assign a unique ID to it
                if gene_panel not in unique_gene_panels_list:
                    unique_gene_panels_list.append(gene_panel)
                    gene_panel_ID_to_gene_panel_dict[unique_gene_panels_count] = gene_panel
                    file_path_to_gene_panel_ID_dict[str(file_path)] = unique_gene_panels_count
                    unique_gene_panels_count += 1
                # If gene panel is not unique, assign find its ID and append file path
                else:
                    gene_panel_ID = next((k for k, v in gene_panel_ID_to_gene_panel_dict.items() if v == gene_panel), None)
                    file_path_to_gene_panel_ID_dict[str(file_path)] = gene_panel_ID

            gene_list = list(set().union(*unique_gene_panels_list))

            if n_gene_panels != unique_gene_panels_count:
                raise ValueError(f"Number of gene panels {n_gene_panels} does not match the number of unique gene panels {unique_gene_panels_count}.")

        else:
            raise ValueError(f"Incorrect value for number of gene panels {n_gene_panels}")
        
        with open(gene_list_file, "wb") as file:
            print(f"Saving gene list to {gene_list_file}")
            pickle.dump(gene_list, file)
        with open(gene_panel_ID_to_gene_panel_dict_file, "wb") as file:
            print(f"Saving gene panel ID to gene panel dictionary to {gene_panel_ID_to_gene_panel_dict_file}")
            pickle.dump(gene_panel_ID_to_gene_panel_dict, file)
        with open(file_path_to_gene_panel_ID_dict_file, "wb") as file:
            print(f"Saving file path to gene panel ID dictionary to {file_path_to_gene_panel_ID_dict_file}")
            pickle.dump(file_path_to_gene_panel_ID_dict, file)
    
    return adata_batch_list, gene_list, gene_panel_ID_to_gene_panel_dict, file_path_to_gene_panel_ID_dict


def create_base_token_dictionary(token_dictionary_file: str,
                                    gene_list: List[str],
                                    species: str,
                                    overwrite: Literal[True, False] = True) -> Dict:
    """
    Create a base token dictionary for gene IDs and save it to a file.

    Parameters:
    ----------
    - token_dictionary_file (str): The file path to save the token dictionary.
    - gene_list (List[str]): The list of gene IDs.
    - species (str): The species of the dataset.
    - overwrite (bool): Whether to overwrite the base token dictionary file.
    
    Returns:
    ----------
    - Dict: The token dictionary with gene IDs as keys and indices as values.
    
    Raises:
    ----------
    - NotImplementedError: If species other than 'sim', 'homo_sapiens', or 'mus_musculus' is not implemented.
    """
    # Load base token dictionary if it exists, else create it
    if os.path.exists(token_dictionary_file) and not overwrite:
        with open(token_dictionary_file, "rb") as f:
            base_token_dict = pickle.load(f)
    else:        
        print(f"Species: {species}")
        if species == 'sim':
            # Extract Ensembl IDs of genes
            gene_list = gene_list
            
        elif species == 'homo_sapiens':
            # Extract Ensembl IDs of genes
            gene_list = gene_list

        elif species == 'mus_musculus':
            # Extract Ensembl IDs of protein coding and miRNA mouse genes
            ensembl = EnsemblRelease(111, species=species)
            ensembl.download()
            ensembl.index()
            all_genes = ensembl.genes()
            protein_coding_genes = [gene for gene in all_genes if gene.biotype == "protein_coding"]
            mirna_genes = [gene for gene in all_genes if gene.biotype == "miRNA"]
            all_relevant_genes = protein_coding_genes + mirna_genes
            gene_list = [gene.gene_id for gene in all_relevant_genes]
            
        else:
            raise NotImplementedError(f"Species {species} not implemented.")

        # Create token dictionary for gene IDs
        base_token_dict = {'<pad>': 0, '<mask>': 1}
        base_token_dict.update({gene_id: idx + 2 for idx, gene_id in enumerate(gene_list)})
            
        # Write to disk
        with open(token_dictionary_file, "wb") as f:
            pickle.dump(base_token_dict, f)
        
    return base_token_dict


def create_nichejepa_token_dictionary(nichejepa_token_dictionary_file: str,
                                      gene_list: List[str], 
                                      token_dict: Dict, 
                                      add_cls_tokens: Literal[True, False] = False,
                                      use_separate_cell_and_neighborhood_tokens: Literal[True, False] = False,
                                      overwrite: Literal[True, False] = False) -> Dict:
    """
    Create a NicheJEPA token dictionary.

    Parameters:
    ----------
    - nichejepa_token_dictionary_file (str): The file path to save the token dictionary.
    - gene_list (List[str]): The list of gene IDs.
    - token_dict (Dict): The initial token dictionary.
    - add_cls_tokens (bool): Whether to add cls tokens.
    - use_separate_cell_and_neighborhood_tokens (bool): Whether to separate cell and neighborhood tokens.
    - overwrite (bool): Whether to overwrite the NicheJEPA token dictionary file.

    Returns:
    ----------
    - Dict: The updated NicheJEPA token dictionary.
    - Int: The number of gene tokens.
    """
    # Load NichJEPA token dictionary if it exists, else create it
    if os.path.exists(nichejepa_token_dictionary_file) and not overwrite:
        with open(nichejepa_token_dictionary_file, "rb") as f:
            nichejepa_token_dict = pickle.load(f)
    else:
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
        token_dict_copy = token_dict.copy()
        for token in token_dict_copy.keys():
            if "ENS" in token and token not in gene_list:
                del(token_dict[token])
        del(token_dict_copy)

        # Use separate tokens for each gene for cell and neighborhood
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

        # Write to disk
        with open(nichejepa_token_dictionary_file, "wb") as file:
            pickle.dump(nichejepa_token_dict, file)

    # Count number of gene tokens 
    n_gene_tokens = len([token for token in nichejepa_token_dict.keys() if "ENS" in token])

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
    
    Usage:
    ----------
    .. terminal
        >>> python create_cell_neighborhood_tokenized_dataset.py --config_file config/cell-neighborhood-tokenization_sim1_1105fts_10000locs_read-depth_shifted-log.yaml --override nproc=1
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
                config['tokenization']['processing_mode'] = "sequential" if int(value) == 1 else "parallel"

    # Set wandb tags
    config['wandb']['tags'] += [config['tokenization']['processing_mode']]
            
    return config


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Collect configs 
    config = collect_configs(args)
            
    # Initialize wandb run
    wandb.init(project=config["wandb"]["project"],
                        job_type=config["wandb"]["job_type"],
                        tags=config["wandb"]["tags"],
                        mode=config["wandb"]["mode"],
                        config=config)
        
    # Configure experiment arguments
    tokenizer = config["experiment"]["tokenizer"]
    dataset_name = config["experiment"]["dataset_name"]
    species = config["experiment"]["species"]
    n_batches = config["experiment"]["n_batches"]
    nproc = config["experiment"]["nproc"]
    overwrite = config["experiment"]["overwrite"]
    print(f"Tokenizer: {tokenizer}")
    print(f"Dataset: {dataset_name}")
    
    try:
        n_gene_panels = config["experiment"]["n_gene_panels"]
    except:
        n_gene_panels = 1
    
    # Configure data paths
    data_directory_path = Path(config["paths"]["data_directory_path"])
    st_data_silver_folder_path = data_directory_path / "silver"
    st_data_gold_folder_path = data_directory_path / "gold"
    
    # Configure profiler path
    profiler_path = Path(config["paths"]["profiler_path"]) / dataset_name / "profiler" / tokenizer
    os.makedirs(profiler_path, exist_ok=True)
    
    # Configure cache path
    try:
        cache_directory_path = Path(config["paths"]["cache_directory_path"])
    except:
        cache_directory_path = data_directory_path / "cache"
    os.makedirs(cache_directory_path, exist_ok=True)    
    
    # Configure tokenization parameters
    use_generator = config["tokenization"]["use_generator"]
    add_cls_tokens = config["tokenization"]["add_cls_tokens"]
    use_separate_cell_and_neighborhood_tokens = config["tokenization"]["use_separate_cell_and_neighborhood_tokens"]
    norm_method = config["tokenization"]["norm_method"]
    norm_factor = config["tokenization"]["norm_factor"]
    prenormalize = config["tokenization"]["prenormalize"]
    processing_mode = config["tokenization"]["processing_mode"]
    custom_attr_name_dict = config["tokenization"]["custom_attr_name_dict"]
    num_shards = config["tokenization"]["num_shards"]
    keep_in_memory = config["tokenization"]["keep_in_memory"]

    # Configure tokenizer paths
    tokenizer_directory_path = Path(config["paths"]["tokenizer_directory_path"])
    tokenizer_folder_path = tokenizer_directory_path / tokenizer / dataset_name
    token_dictionary_file = tokenizer_folder_path / "token_dictionary.pkl"
    nichejepa_token_dictionary_file = tokenizer_folder_path / f"token_dictionary{'_cls' if add_cls_tokens else '_no_cls'}{'_sep' if use_separate_cell_and_neighborhood_tokens else '_no_sep'}.pkl"
    os.makedirs(tokenizer_folder_path, exist_ok=True)

    # Set input and output directories
    input_directory = st_data_silver_folder_path / dataset_name
    output_directory = st_data_gold_folder_path / tokenizer / dataset_name / f"{num_shards}-shards"

    # Configure tokenized dictionary specific paths
    cell_gene_means_dictionary_file = tokenizer_folder_path / f"cell_gene_{norm_factor}_means_dictionary.pkl"
    cell_gene_nzmeans_dictionary_file = tokenizer_folder_path / f"cell_gene_{norm_factor}_nzmeans_dictionary.pkl"
    cell_gene_logmeans_dictionary_file = tokenizer_folder_path / f"cell_gene_{norm_factor}_logmeans_dictionary.pkl"
    neighborhood_gene_means_dictionary_file = tokenizer_folder_path / f"neighborhood_gene_{norm_factor}_means_dictionary.pkl"
    neighborhood_gene_nzmeans_dictionary_file = tokenizer_folder_path / f"neighborhood_gene_{norm_factor}_nzmeans_dictionary.pkl"
    neighborhood_gene_logmeans_dictionary_file = tokenizer_folder_path / f"neighborhood_gene_{norm_factor}_logmeans_dictionary.pkl"
    
    # Set file paths for gene panel mappings
    gene_list_file = tokenizer_folder_path / "gene_list.pkl"
    gene_panel_ID_to_gene_panel_dict_file = tokenizer_folder_path / "gene_panel_ID_to_gene_panel_dict.pkl"
    file_path_to_gene_panel_ID_dict_file = tokenizer_folder_path / "file_path_to_gene_panel_ID_dict.pkl"
        
    # Load data
    adata_batch_list, gene_list, *_ \
        = map_anndata_batches_to_gene_panels(input_directory,
                                                gene_panel_ID_to_gene_panel_dict_file,
                                                file_path_to_gene_panel_ID_dict_file,
                                                gene_list_file, 
                                                n_gene_panels,
                                                overwrite)

    # Create token dictionary
    base_token_dict = create_base_token_dictionary(token_dictionary_file, 
                                                   gene_list, 
                                                   species, 
                                                   overwrite)
    
    # Create nichejepa token dictionary
    _, n_gene_tokens = create_nichejepa_token_dictionary(nichejepa_token_dictionary_file, 
                                                         gene_list, 
                                                         base_token_dict, 
                                                         add_cls_tokens, 
                                                         use_separate_cell_and_neighborhood_tokens, 
                                                         overwrite)

    if prenormalize:
        # Concatenate AnnData objects
        adata = ad.concat(adata_batch_list, join="inner", index_unique="-")
        
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
        del adata
            
    del adata_batch_list
    
    # Set model input size
    try:
        model_input_size = config["experiment"]["model_input_size"]
    except:
        model_input_size = n_gene_tokens * 2
    print(f"Model Input Size: {model_input_size}")
    
    # Set output file prefix
    output_file_prefix = f"{dataset_name}_{nproc}_{keep_in_memory}_{norm_factor}_{norm_method}_{model_input_size}"
    
    # Tokenize data
    with cProfile.Profile() as pr:
        start = time.time()
        
        # Instantiate tokenizer
        tk = tokenizers.CellNeighborhoodRankTokenizer(
                model_input_size=model_input_size,
                norm_method=norm_method,
                norm_factor=norm_factor,
                nproc=nproc,
                processing_mode=processing_mode,
                use_separate_cell_and_neighborhood_tokens=use_separate_cell_and_neighborhood_tokens,
                gene_panel_ID_to_gene_panel_dict_file=gene_panel_ID_to_gene_panel_dict_file,
                file_path_to_gene_panel_ID_dict_file=file_path_to_gene_panel_ID_dict_file,
                cell_gene_means_file=cell_gene_means_dictionary_file,
                cell_gene_nzmeans_file=cell_gene_nzmeans_dictionary_file,
                cell_gene_logmeans_file=cell_gene_logmeans_dictionary_file,
                neighborhood_gene_means_file=neighborhood_gene_means_dictionary_file,
                neighborhood_gene_nzmeans_file=neighborhood_gene_nzmeans_dictionary_file,
                neighborhood_gene_logmeans_file=neighborhood_gene_logmeans_dictionary_file,
                token_dictionary_file=nichejepa_token_dictionary_file,
                custom_attr_name_dict=custom_attr_name_dict)

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
        artifact = wandb.Artifact(name=f"{config['experiment']['dataset_name']}-tokenized",
                                type=f"{config['experiment']['tokenizer']}",
                                metadata=config)
        # add files to artifact
        artifact.add_file(local_path=nichejepa_token_dictionary_file, name="NicheJEPA Token Dictionary")
        artifact.add_file(local_path=profiler_stats_fname, name="cProfile Stats")
        artifact.add_dir(local_path=st_data_silver_folder_path, name=f"AnnData (silver)")
        artifact.add_dir(local_path=output_directory, name=f"Hugging Face Dataset (gold)")
        
        # save artifact
        artifact.save()
        
    wandb.finish()
    
    
if __name__ == "__main__":
    main()