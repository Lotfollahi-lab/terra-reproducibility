"""
For each batch of AnnData in the input directory, this script creates a list of shared gene symbols across all batches. It then creates a new batch of AnnData for each batch in the input directory with only the shared genes. The new batches are saved in the output directory.

Usage:
>>> python create_semi-synthetic-data_multiple-assays.py --config_file config/multiple-assays/multiple-assays_mmb0-1b_smb1-1b_2p_to_mmb0-1b_smb1-1b_1p.yaml
"""

import os
import yaml
import argparse
import scanpy as sc
import anndata as ad
from typing import Dict
from pathlib import Path


def identify_shared_genes(input_directory: Path) -> ad.AnnData:
    """
    Identify shared genes across AnnData batch objects.
    
    Parameters:
    ----------
    - input_directory (Path): The directory containing the input AnnData objects.
    
    Returns:
    ----------
    - shared_gene_symbols (set): The set of shared gene symbols.
    """
    shared_gene_symbols = set()
    
    # Iterate over all files in the input directory
    for file_path in input_directory.glob("*.h5ad"):
        adata = sc.read(file_path)
        print(f"File: {file_path}")
        # If shared_gene_symbols is empty, add all genes in the first file
        if set(shared_gene_symbols) == set():
            shared_gene_symbols = set(adata.var.index)
        # Otherwise, take the intersection of shared_gene_symbols and the genes in the current file
        else:
            shared_gene_symbols = shared_gene_symbols.intersection(set(adata.var.index))

    print(f"Number of shared gene symbols: {len(shared_gene_symbols)}")
    return shared_gene_symbols
    
    
def create_dataset_with_shared_genes(input_directory: Path, 
                                     output_directory: Path, 
                                     shared_gene_symbols: set):
    """
    Create a new dataset with shared genes from the input_directory.
    
    Parameters:
    ----------
    - input_directory (Path): The directory containing the input datasets.
    - output_directory (Path): The directory to write the output datasets.
    - shared_gene_symbols (set): The set of shared gene symbols.
    
    Returns:
    ----------
    - None
    """
    for file_path in input_directory.glob("*.h5ad"):
        adata = sc.read(file_path)
        print(f"Old adata shape: {adata.shape}")
        
        # Create a new adata object with only the shared genes and filter cells with less than 10 genes
        new_adata = adata[:, adata.var.index.isin(shared_gene_symbols)].copy()
        sc.pp.filter_cells(new_adata, min_genes=10)
        
        # Rename cell_id
        new_adata.obs['cell_id'] = [f"{new_adata.uns['dataset_id']}_{new_adata.uns['batch']}_{cell_idx}" for cell_idx in range(new_adata.shape[0])]
        
        print(f"New adata shape: {new_adata.shape}")
        print(new_adata.obs['cell_id'])
        
        # Write new_adata to output_directory with same file name
        new_file_name = f"{file_path.stem}_shared_genes{file_path.suffix}"
        output_file_path = output_directory / new_file_name
        new_adata.write(output_file_path)
            
    return
    

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
    
    return config


def main(config: Dict):
    
#    Configure experiment parameters
    input_experiment_name = config['experiment']['input_experiment_name']
    output_experiment_name = config['experiment']['output_experiment_name']
    
    # # Set data directory
    data_directory_path = Path(config['paths']['data_directory_path'])
    st_data_silver_folder_path = data_directory_path / "silver"

    # Set input data directory
    input_directory = st_data_silver_folder_path / input_experiment_name
        
    # Set output data directory
    output_directory = st_data_silver_folder_path / output_experiment_name
    os.makedirs(output_directory, exist_ok=True)

    # Identify shared genes
    shared_gene_symbols = identify_shared_genes(input_directory)
    
    # Create dataset with shared genes
    create_dataset_with_shared_genes(input_directory, 
                                     output_directory, 
                                     shared_gene_symbols)
    
    print(f"Harmonized data saved to {output_directory}")


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    # Collect configs 
    config = collect_configs(args)
    
    main(config=config)