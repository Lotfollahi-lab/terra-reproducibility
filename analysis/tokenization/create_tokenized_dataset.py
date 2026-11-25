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

The tokenizer options of the config's 'tokenization' section that the
script forwards are listed in `optional_tokenizer_kwargs` (including
`storage_format: compact`, which stores each cell's row without padding).
After tokenizing, the script checks that the saved dataset has the configured
storage format (columns and terra_tokenizer_params.json) and that every
recorded tokenizer setting equals the configured one, and fails otherwise.
"""


import argparse
import cProfile
import json
import os
import pickle
import pstats
import time
import wandb
import yaml
from typing import Optional
from pathlib import Path

from terra.tokenizers.cell_tokenizers import (CellNeighborhoodTokenizer,
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


def _leaf_dtype(feature) -> Optional[str]:
    """
    Value dtype of a feature of dataset_info.json, e.g. 'int16' for a list of
    int16 values ({'feature': {'dtype': 'int16', ...}, '_type': 'List'}).
    """
    while isinstance(feature, dict):
        if 'dtype' in feature:
            return feature['dtype']
        feature = feature.get('feature')
    return None


def check_written_dataset(dataset_path: Path,
                          expected_settings: dict,
                          storage_format: str,
                          norm_factor_file_path: str):
    """
    Check the saved tokenized dataset against the config it was made from:
    its columns and its tokenizer settings record
    (terra_tokenizer_params.json) must show the configured storage format,
    and every recorded setting must equal the configured one. Raises a
    RuntimeError otherwise (the job then fails instead of leaving a part
    that concatenate_datasets.py or train() would refuse later).

    Parameters:
    ----------
    dataset_path:
        The saved `.dataset` folder.
    expected_settings:
        Tokenizer settings as configured, by record key. Keys the record
        does not hold (e.g. chunk_size) are not compared.
    storage_format:
        The configured storage format ('padded' or 'compact').
    norm_factor_file_path:
        The configured norm-factor file.
    """
    dataset_path = Path(dataset_path)
    with open(dataset_path / 'dataset_info.json') as f:
        features = json.load(f)['features']
    record_path = dataset_path / 'terra_tokenizer_params.json'
    if not record_path.exists():
        raise RuntimeError(
            f"'{record_path}' was not written: the installed TERRA does not "
            "record the tokenizer settings.")
    record = json.loads(record_path.read_text())

    problems = []
    # The format is detected from the columns, as TERRA does
    # (terra.datasets.compact.is_compact): compact rows have segment lengths
    columns_format = 'compact' if 'segment_lengths' in features else 'padded'
    record_format = record.get('storage_format', 'padded')
    if columns_format != storage_format:
        problems.append(f"the columns are {columns_format}, but the config "
                        f"says storage_format: {storage_format}")
    if record_format != storage_format:
        problems.append(f"the record says storage_format "
                        f"{record_format!r}, the config {storage_format!r}")
    for key, value in expected_settings.items():
        if key == 'storage_format' or key not in record:
            continue
        if record[key] != value:
            problems.append(f"recorded {key} = {record[key]!r}, configured "
                            f"{value!r}")
    if 'add_neigh_cell_ids' in expected_settings and (
            ('cell_ids' in features)
            != bool(expected_settings['add_neigh_cell_ids'])):
        problems.append("the 'cell_ids' column does not match "
                        "add_neigh_cell_ids")
    if record.get('uses_norm_factors'):
        recorded_path = record.get('norm_factor_file_path')
        if (recorded_path is None
                or str(Path(recorded_path)) != str(Path(norm_factor_file_path))):
            problems.append(f"recorded norm_factor_file_path {recorded_path!r}"
                            f", configured {norm_factor_file_path!r}")
        if record.get('norm_factor_sha256') is None:
            problems.append('the norm-factor file was not found (no sha256 '
                            'recorded)')
    if problems:
        raise RuntimeError(
            f"The tokenized dataset '{dataset_path}' does not match its "
            "config:\n  " + "\n  ".join(problems))

    print(f"Written dataset: {dataset_path}")
    print(f"  storage format: {columns_format} (columns and record)")
    print(f"  columns: {list(features)}")
    print(f"  gene_tokens dtype: {_leaf_dtype(features['gene_tokens'])}, "
          f"gene_expr dtype: {_leaf_dtype(features['gene_expr'])}")
    print(f"  recorded settings: "
          f"{json.dumps(record, sort_keys=True)}")


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
    # Optional: per-file streaming assembly (lower peak memory). Default off
    # -> legacy single-dict assembly. Verify output equivalence before use.
    stream_per_file = config["tokenization"].get("stream_per_file", False)
    # Optional tokenizer settings, forwarded only when they are keys of the
    # 'tokenization' section (absent keys keep the tokenizer defaults):
    # - TERRA opt-in fixes. They are recorded with the dataset
    #   (terra_tokenizer_params.json), and train() checks the model config
    #   against them, so set them here, not only in the model config;
    # - storage_format: 'padded' (default: one row of model_input_size tokens
    #   per cell) or 'compact' (no padding stored; needs a TERRA with
    #   terra.datasets.compact to tokenize and to read the data). A TERRA
    #   without it rejects the key instead of writing padded rows;
    # - the special-token and neighbour-cell-id columns written per row;
    # - add_spatial_index: each cell's position in its file, its coordinates
    #   and its neighbours' offsets (TERRA exp/effective-exposures; the E0
    #   selection proxy, analysis/selection_e0).
    optional_tokenizer_kwargs = {
        key: config["tokenization"][key] for key in (
            "stable_rank_ties", "rank_tie_break_norm_factor",
            "zero_expr_from_raw_counts", "gcrd_basis_target_read_depth",
            "gcrd_target_read_depth_per_gene", "seurat_loess_span",
            "eliminate_stored_zeros", "pad_empty_neighbors",
            "aggregate_own_neighbors", "storage_format",
            "include_special_tokens", "special_tokens",
            "add_neigh_cell_ids", "chunk_size", "add_spatial_index")
        if key in config["tokenization"]}
    print(f"Optional tokenizer settings: {optional_tokenizer_kwargs}")
    storage_format = optional_tokenizer_kwargs.get("storage_format", "padded")
    print(f"Storage format: {storage_format}.")

    nproc = int(os.environ.get("LSB_DJOB_NUMPROC", 1))

    # Retrieve paths from config
    token_dictionary_file_path = config["paths"]["token_dictionary_file_path"]
    norm_factor_file_path = config["paths"]["norm_factor_file_path"]
    # Optional: frozen corpus-wide PFlog1pPF targets (s1, s2) produced by
    # compute_cohort_norm_factors.py. None -> per-file PF targets (legacy).
    pf_targets_file_path = config["paths"].get("pf_targets_file_path", None)
        
    # Set data paths
    data_directory_path = Path(config["paths"]["data_directory_path"])
    st_data_silver_folder_path = data_directory_path / "silver"
    st_data_gold_folder_path = data_directory_path / "gold"
        
    # Set cache path (paths.cache_directory_path, if given, gives a job its
    # own cache folder, which it can delete after the save; the shared
    # default keeps a copy of every part it builds)
    cache_directory_path = Path(
        config["paths"].get("cache_directory_path")
        or data_directory_path / "cache")
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
            pf_targets_file_path=pf_targets_file_path,
            stream_per_file=stream_per_file,
            token_dictionary_file_path=token_dictionary_file_path,
            **optional_tokenizer_kwargs)

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

    # Check what was written against this config (format and record)
    check_written_dataset(
        dataset_path=output_directory / f"{output_file_prefix}.dataset",
        expected_settings={
            "tokenizer": Tokenizer.__name__,
            "model_input_size": model_input_size,
            "n_neighs": n_neighs,
            "include_zero_expr_genes": include_zero_expr_genes,
            "rank_cell_norm_method": rank_cell_norm_method,
            "rank_gene_norm_method": rank_gene_norm_method,
            "rank_count_norm_method": rank_count_norm_method,
            "count_cell_norm_method": count_cell_norm_method,
            "count_gene_norm_method": count_gene_norm_method,
            "count_count_norm_method": count_count_norm_method,
            **optional_tokenizer_kwargs},
        storage_format=storage_format,
        norm_factor_file_path=norm_factor_file_path)

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
            local_path=token_dictionary_file_path,
            name="TERRA Neighborhood Token Dictionary")
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