"""
Compute statistics of a collection of batched AnnData objects.
"""

import os
import yaml
import wandb
import argparse
import concurrent
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, Union, Literal

from nichejepa.preprocessors.filter_poor_quality_cells import filter_poor_quality_cells


def load_anndata(filepath: str, 
                    file_format: str) -> sc.AnnData:
    """
    Load preprocessed data stored in an AnnData object.

    Parameters
    ----------
    - filepath (str): The path to the AnnData object.
    - file_format (str): The format of the file.

    Returns:
    ----------
    - sc.AnnData: The AnnData object.

    Raises:
    ----------
    - NotImplementedError: If file format other than `.h5ad` is im.
    """
    if file_format == "h5ad":
        adata = sc.read_h5ad(filepath)
        adata = filter_poor_quality_cells(adata)
    else:
        raise NotImplementedError(f"Reading file format {file_format} not implemented.")

    return adata


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
    ----------
    - argparse.Namespace: The command line arguments.
    
    Usage:
    ----------
    .. terminal
        >>> python compute_anndata_statistics.py --config_file config/compute_anndata_statistics_sim1_1105fts_10000locs.yaml --override mode=sequential
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
                config['analysis']['processing_mode'] = "sequential" if int(value) == 1 else "parallel"
            if key == 'mode':
                config['analysis']['processing_mode'] = value

    # Set wandb tags
    config['wandb']['tags'] += [config['analysis']['processing_mode']]
            
    return config


def collect_batch_stats(file_path: Union[Path, str],
                        file_format: Literal["h5ad"] = "h5ad",
                        cell_types_key: str = "cell_types",
                        niche_types_key: str = "niche_types") -> Dict:
    """
    Collect statistics of a single batch of data.
    
    Parameters:
    ----------
    - file_path (Union[Path, str]): The path to the batch file.
    - file_format (Literal["h5ad"]): The format of the file.
    - cell_types_key (str): The key for cell types in the AnnData object.
    - niche_types_key (str): The key for niche types in the AnnData object.
    
    Returns:
    ----------
    - batch_id (int): The batch ID.
    - batch_df (pd.DataFrame): The statistics of the batch.
    - cell_type_counts (Dict): The cell type counts.
    - niche_type_counts (Dict): The niche type counts.
    """
    # Extract batch_id from file name
    # example: merfish_mouse_brain_8-batch_batch1.h5ad -> 1
    # example: sim1_1105fts_10000locs.h5ad -> 1
    try:
        batch_id = int(file_path.stem.split('_')[-1][5:])
    except:
        batch_id = 1
    
    # Load the AnnData object and filter poor quality cells
    adata = load_anndata(file_path, file_format)
    
    # get number of cells and number of genes
    n_cells, n_genes = adata.shape

    # get cell and niche type counts
    # append 'cell_' and 'niche_' to the keys to identify them in the results
    cell_type_counts = {f"cell_{key}": value for key, value in adata.obs[cell_types_key].value_counts().to_dict().items()}
    niche_type_counts = {f"niche_{key}": value for key, value in adata.obs[niche_types_key].value_counts().to_dict().items()}
    
    # Create a dictionary for the row data corresponding to each batch
    batch_stats = {"batch_id": batch_id,
                "n_cells": n_cells,
                "n_genes": n_genes,
                **cell_type_counts,
                **niche_type_counts}
    
    # Create a single row DataFrame for the batch
    batch_df = pd.DataFrame(batch_stats, index=[0])
    
    return batch_id, batch_df, cell_type_counts, niche_type_counts
            

def collect_dataset_stats(input_directory: Union[Path, str],
                            nproc: int = 1,
                            file_format: Literal["h5ad"] = "h5ad",
                            processing_mode: Literal['sequential', 'parallel'] = 'sequential',
                            custom_attr_name_dict: Dict = None) -> pd.DataFrame:
    """
    Collect statistics of the dataset.
    
    Parameters:
    ----------
    - input_directory (Union[Path, str]): The path to the directory containing batch files.
    - nproc (int): The number of processes to use.
    - file_format (Literal["h5ad"]): The format of the file.
    - processing_mode (Literal['sequential', 'parallel']): The processing mode.
    - custom_attr_name_dict (Dict): The dictionary containing custom attribute names.
    
    Returns:
    ----------
    - pd.DataFrame: The statistics of the dataset.
    """    
    reverse_custom_attr_name_dict = {value: key for key, value in custom_attr_name_dict.items()}
    cell_types_key = reverse_custom_attr_name_dict['cell_types']
    niche_types_key = reverse_custom_attr_name_dict['niche_types']
    
    results = pd.DataFrame(columns=["batch_id", "n_cells", "n_genes"])
    
    if processing_mode == 'sequential':
        print(f"Processing files sequentially...")
        for file_path in input_directory.glob(f"*.{file_format}"):
            print(f"Reading {file_path}...")

            # Collect statistics of a single batch of data
            batch_id, batch_df, cell_type_counts, niche_type_counts = collect_batch_stats(file_path,
                                                                                            file_format,
                                                                                            cell_types_key,
                                                                                            niche_types_key)
            
            # Ensure all keys in cell_type_counts and niche_type_counts are present in the results
            for key in cell_type_counts.keys():
                if key not in results.columns:
                    results[key] = 0
            for key in niche_type_counts.keys():
                if key not in results.columns:
                    results[key] = 0
            
            print(f"Adding row data for Batch {batch_id} to results...")
            results = pd.concat([results,batch_df], ignore_index=True)
                        
    elif processing_mode == 'parallel':
        print(f"Processing files in parallel...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = []
            for file_path in input_directory.glob(f"*.{file_format}"):
                print(f"Reading {file_path}...")
                future = executor.submit(collect_batch_stats, file_path, file_format, cell_types_key, niche_types_key)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                # Collect statistics of a single batch of data
                batch_id, batch_df, cell_type_counts, niche_type_counts = future.result()

                # Ensure all keys in cell_type_counts and niche_type_counts are present in the results
                for key in cell_type_counts.keys():
                    if key not in results.columns:
                        results[key] = 0
                for key in niche_type_counts.keys():
                    if key not in results.columns:
                        results[key] = 0
                
                print(f"Adding row data for Batch {batch_id} to results...")
                results = pd.concat([results,batch_df], ignore_index=True)
    
    results.fillna(0, inplace=True)
    return results


def main(config: Dict) -> None:
    # Configure parameters
    dataset_name = config['experiment']['dataset_name']
    nproc = config['experiment']['nproc']
    processing_mode = config['analysis']['processing_mode']
    data_directory_path = Path(config['paths']['data_directory_path'])
    custom_attr_name_dict = config["analysis"]["custom_attr_name_dict"]

    try:
        file_format = config['experiment']['file_format']
    except:
        file_format = "h5ad"

    # Set path to directory containing AnnData batch files
    input_directory = data_directory_path / "silver" / dataset_name
    output_directory = data_directory_path / "silver_statistics" / dataset_name
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Collecting statistics for {dataset_name}...")
    results = collect_dataset_stats(input_directory=input_directory,
                                    nproc=nproc,
                                    file_format=file_format,
                                    processing_mode=processing_mode,
                                    custom_attr_name_dict=custom_attr_name_dict)
    
    print(results)
    
    output_filename = output_directory / f"{dataset_name}_statistics.csv"
    print(f"Saving statistics to {output_filename}...")
    results.to_csv(output_filename, index=False)
    
    # Save results as wandb artifact
    if config["wandb"]["save_artifact"]:
        # Create wandb artifact
        artifact = wandb.Artifact(name=f"{config['experiment']['dataset_name']}-anndata-statistics",
                                type="statistics-dataframe",
                                metadata=config)
        # add files to artifact
        artifact.add_file(local_path=results, name="DataFrame for Batchwise AnnData Statistics")
        
        # save artifact
        artifact.save()
    

if __name__ == '__main__':
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

    # Call main function
    main(config)

    wandb.finish()