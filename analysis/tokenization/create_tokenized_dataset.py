"""
Sample Usage
----------
CellNeighborhoodTokenizer
.. terminal
    >>> python analysis/tokenization/create_tokenized_dataset.py --config_file config/cell-neighborhood-tokenizer/cell-neighborhood-tokenizer_sss2-1b_1p.yaml

CellGraphTokenizer
.. terminal
    >>> python analysis/tokenization/create_tokenized_dataset.py --config_file config/cell-graph-tokenizer/cell-graph-tokenizer_sss2-1b_1p.yaml

Description
----------
This script tokenizes the gene expression data from the harmonized AnnData
(silver) datasets using the Cell Neighborhood Tokenizer or the Cell Graph
Tokenizer. It logs tokenization time and CPU statistics to wandb. It also
optionally creates a wandb artifact containing the tokenized dataset and the
token dictionary.
"""


import argparse
import cProfile
import os
import pickle
import pstats
import time
import wandb
import yaml
from typing import Optional
from pathlib import Path

from nichejepa.tokenizers.cell_tokenizers import (CellNeighborhoodTokenizer,
                                                  CellGraphTokenizer)


def collect_config(args: argparse.Namespace) -> dict:
    """
    Collect config from the config file based on command line arguments.
    
    Parameters:
    ----------
    args:
        The command line arguments.
    
    Returns:
    ----------
    config:
        The configuration dictionary.
    """
    # Read parameters from config file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
                    
    return config


def obtain_cumulative_times(profiler_stats_fname: str,
                            func_list_to_profile: Optional[
                                list[str]] = ['tokenize_data',
                                              'tokenize_files',
                                              'create_dataset']) -> dict:
    """
    Obtain cumulative times from the cProfile stats file for a specified list of
    functions.

    Parameters:
    ----------
    profiler_stats_fname:
        The file path to the cProfile stats file.
    func_list_to_profile:
        The list of functions to profile.

    Returns:
    ----------
    cumulative_times:
        A dictionary containing the cumulative times for each specified
        function.
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
    argparse.Namespace
        The command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Process command line arguments.')
    
    parser.add_argument('--config_file', 
                        type=str, 
                        help='Path to the config file.')
    
    return parser.parse_args()


def main(config: dict):

    # ------------------------------------------------------------

    # Retrieve experiment config params
    experiment_name = config["experiment"]["experiment_name"]
    experiment_description = config["experiment"]["experiment_description"]
    tokenizer = config["experiment"]["tokenizer"]
    model_input_size = config["experiment"]["model_input_size"]
    print(f"Experiment name: {experiment_name}.")
    print(f"Experiment description: {experiment_description}.")
    print(f"Tokenizer: {tokenizer}.")
    print(f"Model input size: {model_input_size}.")

    # Retrieve tokenization config params
    species = config["tokenization"]["species"]
    use_generator = config["tokenization"]["use_generator"]
    rank_cell_norm_method = config["tokenization"]["rank_cell_norm_method"]
    rank_gene_norm_method = config["tokenization"]["rank_gene_norm_method"]
    rank_count_norm_method = config["tokenization"]["rank_count_norm_method"]
    count_cell_norm_method = config["tokenization"]["count_cell_norm_method"]
    count_gene_norm_method = config["tokenization"]["count_gene_norm_method"]
    count_count_norm_method = config["tokenization"]["count_count_norm_method"]
    processing_mode = config["tokenization"]["processing_mode"]
    num_shards = config["tokenization"]["num_shards"]
    keep_in_memory = config["tokenization"]["keep_in_memory"]
    include_zero_expr_genes = config["tokenization"]["include_zero_expr_genes"]
    n_neighs = config["tokenization"]["n_neighs"]
    radius = config["tokenization"]["radius"]
    delaunay = config["tokenization"]["delaunay"]

    nproc = int(os.environ.get("LSB_DJOB_NUMPROC", 1))

    # Retrieve paths from config
    token_dictionary_file_path = config["paths"]["token_dictionary_file_path"]
    norm_factor_file_path = config["paths"]["norm_factor_file_path"]
        
    # Set data paths
    data_directory_path = Path(config["paths"]["data_directory_path"])
    st_data_silver_folder_path = data_directory_path / "silver"
    st_data_gold_folder_path = data_directory_path / "gold"
        
    # Set cache path
    cache_directory_path = data_directory_path / "cache"
    os.makedirs(cache_directory_path, exist_ok=True)

    # Set input and output directories
    input_directory = st_data_silver_folder_path / experiment_name
    output_directory = st_data_gold_folder_path / tokenizer / experiment_name

    # Set log path
    log_directory_path = Path("./logs") / experiment_name
        
    # Set profiler path
    profiler_path = log_directory_path / "cProfiler" / tokenizer
    os.makedirs(profiler_path, exist_ok=True)

    # Set output file prefix
    neighborhood_graph = ""
    if n_neighs:
        neighborhood_graph += f"knn_{n_neighs}_"
    if radius:
        neighborhood_graph += f"radius_{radius}_"
    if delaunay:
        neighborhood_graph += "delaunay_"
    output_file_prefix = (f"{experiment_name}_{num_shards}_"
                          f"{rank_cell_norm_method}_{rank_gene_norm_method}_"
                          f"{rank_count_norm_method}_{count_cell_norm_method}_"
                          f"{count_gene_norm_method}_{count_count_norm_method}_"
                          f"{neighborhood_graph}{model_input_size}"
                          f"{'_incl_zero_expr' if include_zero_expr_genes else ''}")

    # Initialize wandb run
    wandb_tags = [experiment_name,
                  tokenizer]
    wandb.init(project=config["wandb"]["project"],
               job_type=config["wandb"]["job_type"],
               tags=wandb_tags,
               mode=config["wandb"]["mode"],
               config=config)
        
    # ------------------------------------------------------------

    # Load token dict
    print(f"Loading token dictionary from {token_dictionary_file_path}")
    with open(token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)
    print(f"Number of Gene Tokens: {sum('ENS' in key for key in token_dict.keys())}.")
    print(f"Number of Tokens: {len(token_dict)}.")
    
    # Tokenize data
    with cProfile.Profile() as pr:
        start = time.time()
        
        if tokenizer == 'cell-neighborhood-tokenizer':
            Tokenizer = CellNeighborhoodTokenizer
        elif tokenizer == 'cell-graph-tokenizer':
            Tokenizer = CellGraphTokenizer
            
        # Initialize tokenizer
        tk = Tokenizer(
            nproc=nproc,
            processing_mode=processing_mode,
            model_input_size=model_input_size,
            include_zero_expr_genes=include_zero_expr_genes,
            n_neighs=n_neighs,
            radius=radius,
            delaunay=delaunay,
            rank_cell_norm_method=rank_cell_norm_method,
            rank_gene_norm_method=rank_gene_norm_method,
            rank_count_norm_method=rank_count_norm_method,
            count_cell_norm_method=count_cell_norm_method,
            count_gene_norm_method=count_gene_norm_method,
            count_count_norm_method=count_count_norm_method,
            norm_factor_file_path=norm_factor_file_path,
            token_dictionary_file_path=token_dictionary_file_path)

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
        profiler_stats_fname = str(profiler_path /
            f"{output_file_prefix}_{nproc}_{processing_mode}.prof")
        pr.dump_stats(profiler_stats_fname)
        print(f"Profiling stats saved to: {profiler_stats_fname}.")
                    
        # Extract cumulative times from profiler stats
        cumulative_times = obtain_cumulative_times(profiler_stats_fname)

    # Log tokenization and profiling times to wandb
    wandb.log({"nproc": nproc})
    wandb.log({"tokenization_time": end - start})
    for func_name, ct in cumulative_times.items():
        wandb.log({f"cumulative_time_{func_name}": ct})

    if config["wandb"]["save_artifact"]:
        # Create wandb artifact
        artifact = wandb.Artifact(
            name=f"{config['experiment']['experiment_name']}-tokenized",
            type=f"{config['experiment']['tokenizer']}",
            metadata=config)
        # Add files to artifact
        artifact.add_file(
            local_path=token_dictionary_file,
            name="NicheJEPA Neighborhood Token Dictionary")
        artifact.add_file(
            local_path=profiler_stats_fname,
            name="cProfile Stats")
        artifact.add_dir(
            local_path=st_data_silver_folder_path,
            name=f"AnnData (silver)")
        artifact.add_dir(
            local_path=output_directory,
            name=f"Hugging Face Dataset (gold)")
        
        # Save artifact
        artifact.save()
        
    wandb.finish()
    
    
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Get config params
    config = collect_config(args)
        
    main(config)