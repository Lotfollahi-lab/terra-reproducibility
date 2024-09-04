"""
Adapted from cell_graph_rank_tokenizer.ipynb.

Usage
----------
.. terminal
    >>> python create_cell_graph_tokenized_dataset.py --config_file config/cell-graph-tokenization_sim1_1105fts_10000locs_read-depth_shifted-log.yaml

Description
----------
This script tokenizes the gene expression data from the preprocessed AnnData (silver) dataset using the Cell Graph Rank Tokenizer. It logs tokenization time and CPU statistics to wandb. It also optionally creates a wandb artifact containing the tokenized dataset and the token dictionary.
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
from pathlib import Path
from typing import List, Dict, Union, Literal, Optional

from pyensembl import EnsemblRelease

from nichejepa import tokenizers
from nichejepa.aggregators.aggregate_neighbors import aggregate_neighbors
from nichejepa.normalizers.normalize_by_read_depth import normalize_by_read_depth
from nichejepa.preprocessors.filter_poor_quality_cells import filter_poor_quality_cells


def load_anndata(st_data_silver_folder_path: Union[Path, str], 
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
        else:
            raise ValueError("Simulated data with multiple batches not implemented.")
    elif species == 'mus_musculus':
        # Load the first batch of data.
        if n_batches == 1:
            raise ValueError("Mouse data with one batch not implemented.")
        else:
            adata = sc.read_h5ad(f"{st_data_silver_folder_path}/{dataset_name}/{dataset_name}_batch1.h5ad")
    else:
        raise NotImplementedError(f"Species {species} not implemented.")

    return adata


def create_base_token_dictionary(tokenizer_directory_path: str,
                            token_dictionary_file: str, 
                            adata: Dict,
                            species: str,
                            overwrite: Literal[True, False] = True) -> Dict:
    """
    Create a base token dictionary for gene IDs and save it to a file.

    Parameters:
    ----------
    - tokenizer_directory_path (str): The path to the tokenizer directory. (Used for loading geneformer token dictionary)
    - token_dictionary_file (str): The file path to save the token dictionary.
    - adata (Dict): The dictionary containing the gene IDs.
    - species (str): The species of the dataset.
    - overwrite (bool): Whether to overwrite the base token dictionary file.
    
    Returns:
    ----------
    - Dict: The token dictionary with gene IDs as keys and indices as values.
    
    Raises:
    ----------
    - NotImplementedError: If species other than 'sim' or 'mus_musculus' is not implemented.
    """
    # Load base token dictionary if it exists, else create it
    if os.path.exists(token_dictionary_file) and not overwrite:
        with open(token_dictionary_file, "rb") as f:
            base_token_dict = pickle.load(f)
    else:        
        if species == 'sim':
            # Extract Ensembl IDs of genes
            gene_ids = adata.var["ensembl_id"]
            
            # Create token dictionary for simulated data
            base_token_dict = {'<pad>': 0, '<mask>': 1}
            base_token_dict.update({gene_id: idx + 2 for idx, gene_id in enumerate(gene_ids)})
        elif species == 'mus_musculus':
            # Extract Ensembl IDs of protein coding and miRNA mouse genes
            ensembl = EnsemblRelease(111, species=species)
            ensembl.download()
            ensembl.index()
            all_genes = ensembl.genes()
            protein_coding_genes = [gene for gene in all_genes if gene.biotype == "protein_coding"]
            mirna_genes = [gene for gene in all_genes if gene.biotype == "miRNA"]
            all_relevant_genes = protein_coding_genes + mirna_genes
            gene_ids = [gene.gene_id for gene in all_relevant_genes]
            
            # Create token dictionary for mouse
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
                                      add_cls_tokens: Literal[True, False] = False,
                                    #   use_separate_cell_and_neighborhood_tokens: Literal[True, False] = False,
                                      overwrite: Literal[True, False] = False) -> Dict:
    """
    Create a NicheJEPA token dictionary.

    Parameters:
    ----------
    - nichejepa_token_dictionary_file (str): The file path to save the token dictionary.
    - adata (sc.AnnData): The AnnData object containing the dataset.
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
            # if use_separate_cell_and_neighborhood_tokens:
                # token_dict = insert_at_index(token_dict, 3, "<cls_neighborhood>", 3)

        # Remove genes not in dataset
        gene_list = adata.var["ensembl_id"].tolist()
        token_dict_copy = token_dict.copy()
        for token in token_dict_copy.keys():
            if "ENS" in token and token not in gene_list:
                del(token_dict[token])
        del(token_dict_copy)

        # Separate gene tokens into cell and neighborhood tokens
        nichejepa_token_dict = {
            (key if "ENS" in key else key): value
            for value, key in enumerate(token_dict.keys())}

        # # Use separate tokens for each gene for cell and neighborhood
        # if use_separate_cell_and_neighborhood_tokens:
        #     nichejepa_token_dict = {
        #         (key + "_cell" if "ENS" in key else key): value
        #         for value, key in enumerate(token_dict.keys())}
        #     nichejepa_token_dict_copy = nichejepa_token_dict.copy()
        #     i = 0
        #     for key in nichejepa_token_dict_copy.keys():
        #         if "ENS" in key:
        #             nichejepa_token_dict[key.replace("cell", "neighborhood")] = len(nichejepa_token_dict_copy) + i
        #             i += 1
        #     del(nichejepa_token_dict_copy)
        # else:
        #     nichejepa_token_dict = {
        #         key: value
        #         for value, key in enumerate(token_dict.keys())}

        # Write to disk
        with open(nichejepa_token_dictionary_file, "wb") as file:
            pickle.dump(nichejepa_token_dict, file)

    # Count number of gene tokens 
    n_gene_tokens = len([token for token in nichejepa_token_dict.keys() if "ENS" in token])

    return nichejepa_token_dict, n_gene_tokens


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
        >>> python create_cell_graph_tokenized_dataset.py --config_file config/cell-graph-tokenization_sim1_1105fts_10000locs_read-depth_shifted-log.yaml --override nproc=1
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
    norm_method = config["tokenization"]["norm_method"]
    norm_factor = config["tokenization"]["norm_factor"]
    prenormalize = config["tokenization"]["prenormalize"]
    processing_mode = config["tokenization"]["processing_mode"]
    custom_attr_name_dict = config["tokenization"]["custom_attr_name_dict"]
    num_shards = config["tokenization"]["num_shards"]
    keep_in_memory = config["tokenization"]["keep_in_memory"]    
    label_cats = config["tokenization"]["label_cats"]

    # Configure tokenizer paths
    tokenizer_directory_path = Path(config["paths"]["tokenizer_directory_path"])
    tokenizer_folder_path = tokenizer_directory_path / tokenizer / dataset_name
    token_dictionary_file = tokenizer_folder_path / "token_dictionary.pkl"
    nichejepa_token_dictionary_file = tokenizer_folder_path / f"token_dictionary{'_cls' if add_cls_tokens else '_no_cls'}.pkl"
    os.makedirs(tokenizer_folder_path, exist_ok=True)

    # Set input and output directories
    input_directory = st_data_silver_folder_path / dataset_name
    output_directory = st_data_gold_folder_path / tokenizer / dataset_name / f"{num_shards}-shards"

    # Configure tokenized dictionary specific paths
    gene_means_dictionary_file = tokenizer_folder_path / f"gene_{norm_factor}_means_dictionary.pkl"
    gene_nzmeans_dictionary_file = tokenizer_folder_path / f"gene_{norm_factor}_nzmeans_dictionary.pkl"
    gene_logmeans_dictionary_file = tokenizer_folder_path / f"gene_{norm_factor}_logmeans_dictionary.pkl"
    
    # Load data
    adata = load_anndata(st_data_silver_folder_path, dataset_name, species, n_batches)

    # Create token dictionary
    base_token_dict = create_base_token_dictionary(tokenizer_directory_path, token_dictionary_file, adata, species, overwrite)
    
    # Create nichejepa token dictionary
    _, n_gene_tokens = create_nichejepa_token_dictionary(nichejepa_token_dictionary_file, adata, base_token_dict, add_cls_tokens, overwrite)

    if prenormalize:
        # Normalize gene counts
        adata = normalize_gene_counts(adata, n_batches, norm_factor)

        # Compute means, non-zero means, and log-means
        cell_gene_means, _ = compute_cell_and_neighborhood_gene_means(adata)
        cell_gene_nzmeans, _ = compute_cell_and_neighborhood_gene_nzmeans(adata)
        cell_gene_logmeans, _ = compute_cell_and_neighborhood_gene_logmeans(adata)
                
        # Create norm factor dictionaries for tokenizer and store on disk
        cell_gene_means_dict = {}
        cell_gene_nzmeans_dict = {}
        cell_gene_logmeans_dict = {}
        
        for (gene,
            cell_gene_mean,
            cell_gene_nzmedian,
            cell_gene_logmean) in zip(
                adata.var["ensembl_id"].values,
                cell_gene_means,
                cell_gene_nzmeans,
                cell_gene_logmeans):
            cell_gene_means_dict[gene] = cell_gene_mean
            cell_gene_nzmeans_dict[gene] = cell_gene_nzmedian
            cell_gene_logmeans_dict[gene] = cell_gene_logmean
            
        # Save norm_factor dictionaries to disk
        with open(gene_means_dictionary_file, "wb") as file:
            pickle.dump(cell_gene_means_dict, file)
        with open(gene_nzmeans_dictionary_file, "wb") as file:
            pickle.dump(cell_gene_nzmeans_dict, file)
        with open(gene_logmeans_dictionary_file, "wb") as file:
            pickle.dump(cell_gene_logmeans_dict, file)

    # Set model input size
    try:
        model_input_size = config["experiment"]["model_input_size"]
    except:
        tokens_per_cell = n_gene_tokens
        max_neighbors = compute_max_neighbors(adata) 
        model_input_size = n_gene_tokens * min(16, max_neighbors) + 1

    # Set output file prefix
    output_file_prefix = f"{dataset_name}_{nproc}_{keep_in_memory}_{norm_factor}_{norm_method}_{model_input_size}"
    
    # Tokenize data
    with cProfile.Profile() as pr:
        start = time.time()
        
        # Instantiate tokenizer
        tk = tokenizers.CellGraphRankTokenizer(
                model_input_size=model_input_size,
                tokens_per_cell=tokens_per_cell,
                norm_method=norm_method,
                norm_factor=norm_factor,
                nproc=nproc,
                processing_mode=processing_mode,
                gene_means_file=gene_means_dictionary_file,
                gene_nzmeans_file=gene_nzmeans_dictionary_file,
                gene_logmeans_file=gene_logmeans_dictionary_file,
                token_dictionary_file=nichejepa_token_dictionary_file,
                custom_attr_name_dict=custom_attr_name_dict,
                special_tokens = (None if not add_cls_tokens else ["<cls>"]),
                special_tokens_idx = (None if not add_cls_tokens else [0])
                )

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
        artifact.add_file(local_path=nichejepa_token_dictionary_file, name="NicheJEPA Graph Token Dictionary")
        artifact.add_file(local_path=profiler_stats_fname, name="cProfile Stats")
        artifact.add_dir(local_path=st_data_silver_folder_path, name=f"AnnData (silver)")
        artifact.add_dir(local_path=output_directory, name=f"Hugging Face Dataset (gold)")
        
        # save artifact
        artifact.save()
        
    wandb.finish()
    
    
if __name__ == "__main__":
    main()