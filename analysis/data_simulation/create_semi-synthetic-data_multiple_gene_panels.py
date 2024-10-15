"""
For each batch of AnnData in the input directory, this script creates `n_gene_panels` new batches of AnnData, one per new gene panel. The new gene panels are created by randomly selecting genes from the original gene list according to the following procedure:
1. A fixed number of genes are shared between all gene panels.
2. The remaining genes are divided into `n_gene_panels` subsets according to the `subset_ratios` parameter.
3. The shared genes and the genes in each subset are concatenated to form the gene panels.

Usage:
>>> python create_semi-synthetic-data_multiple_gene_panels.py --config_file config/multiple-gene-panels/multiple-gene-panels_mmb0-300k-2b_1p_mmb0-300k-2b_2p.yaml

"""

import os
import yaml
import argparse
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Dict, List


def create_dataset_with_multiple_gene_panels(input_directory: Path, 
                                                output_directory: Path,
                                                gene_panel_list: List[List[str]]):
    """
    Create a new dataset with multiple gene panels from the input_directory.
    
    Parameters:
    ----------
    - input_directory (Path): The directory containing the input datasets.
    - output_directory (Path): The directory to write the output datasets.
    - gene_panel_list (List[List[str]]): A list of gene panels.
    
    Returns:
    ----------
    - None
    """
    file_paths = list(input_directory.glob(f"**/*.h5ad"))
    offset = 1000
    
    for file_path in file_paths:
        print(f"Loading data from {file_path}")
        adata = sc.read_h5ad(file_path)
        print(f"Old adata shape: {adata.shape}")
        dataset_id = adata.uns['dataset_id']
        
        # Subset the adata object to include only the genes in the gene_panel
        for panel_idx, gene_panel in enumerate(gene_panel_list):
            new_adata = adata[:, adata.var['ensembl_id'].isin(gene_panel)].copy()

            original_batch_id = int(new_adata.uns['batch'][5:])
            new_batch_id = original_batch_id + offset * (panel_idx+1)
            new_adata.uns['batch'] = f"batch{new_batch_id}"
            
            # Update cell_id
            new_adata.obs['cell_id'] = [f"{dataset_id}_{new_adata.uns['batch']}_{cell_idx}" for cell_idx in range(new_adata.shape[0])]
                        
            print(new_adata.uns['batch'], new_adata.obs['cell_id'][2])
            
            print(f"Panel ID: {panel_idx} | New adata shape: {new_adata.shape}")
         
            # Write new_adata to output_directory with same file name
            new_file_name = f"{file_path.stem}_panel_{panel_idx}{file_path.suffix}"
            output_file_path = output_directory / new_file_name
            new_adata.write(output_file_path)
            print(f"Writing to {output_file_path}")
        print()
    

def create_gene_panels(gene_list: List[str], 
                       n_gene_panels: int = 2,
                       shared_genes_ratio: float = 0.4,
                       subset_ratios: List[float] = [0.4, 0.2],
                       seed: int = 42):
    """
    Create gene panels from the gene_list.
    
    Parameters:
    ----------
    - gene_list (List[str]): The list of genes.
    - n_gene_panels (int): The number of gene panels to create.
    - shared_genes_ratio (float): The ratio of shared genes between gene panels.
    - subset_ratios (List[float]): The ratio of genes in each gene panel.
    - seed (int): The random seed.
    
    Returns:
    ----------
    - List[List[str]]: A list of gene panels.
    """
    if subset_ratios is None or len(subset_ratios) != n_gene_panels:
        raise ValueError("subset_ratios must be a list of size equal to n_gene_panels")
    
    if not np.isclose(sum(subset_ratios), 1 - shared_genes_ratio):
        print(sum(subset_ratios), 1 - shared_genes_ratio)
        raise ValueError("The sum of subset_ratios must be equal to 1 - shared_genes_ratio")
    
    n_genes = len(gene_list)
    print(f"Total Number of Genes: {n_genes}")
    print()
    
    gene_idx = np.arange(n_genes, dtype=int)
    np.random.seed(seed)
    np.random.shuffle(gene_idx)
    
    gene_panel_list = []
    
    shared_genes_count = int(n_genes * shared_genes_ratio)
    shared_genes = np.array(gene_list)[gene_idx[:shared_genes_count]]
    
    start_idx = shared_genes_count
    for i in range(n_gene_panels):
        if i == n_gene_panels - 1:
            gene_panel = np.array(gene_list)[gene_idx[start_idx:]]
        else:
            subset_count = int(n_genes * subset_ratios[i])
            gene_panel = np.array(gene_list)[gene_idx[start_idx:start_idx + subset_count]]
            start_idx += subset_count
        
        gene_panel = np.concatenate([shared_genes, gene_panel])
        gene_panel_list.append(gene_panel.tolist())
        
        print(f"Panel {i+1}: {len(gene_panel)} genes")
        print()
    
    # Assert that the union of gene_panels is the same as the gene_list
    union_genes = np.concatenate(gene_panel_list)
    assert set(union_genes) == set(gene_list), "The union of gene_panels is not the same as the gene_list"
    
    return gene_panel_list


def get_gene_list(data_dir: Path) -> List[str]:
    """
    Get the gene list from the first file in the directory.
    
    Parameters:
    ----------
    - data_dir (Path): The directory containing the data files.
    
    Returns:
    ----------
    - List[str]: The list of genes.
    """
    print("Getting gene list from the first file in the directory")
    file_paths = list(data_dir.glob(f"**/*.h5ad"))
    adata = sc.read_h5ad(file_paths[0])    
    gene_list = adata.var["ensembl_id"].values.tolist()
    return gene_list
    
    
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
    
   # Configure experiment parameters
    input_experiment_name = config['experiment']['input_experiment_name']
    n_gene_panels = config['experiment']['n_gene_panels']
    output_experiment_name = config['experiment']['output_experiment_name']
    
    # Set input data directory
    data_directory_path = Path(config['paths']['data_directory_path'])
    st_data_silver_folder_path = data_directory_path / "silver"
    input_directory = st_data_silver_folder_path / input_experiment_name
    
    # Set output data directory
    output_directory = st_data_silver_folder_path / output_experiment_name
    os.makedirs(output_directory, exist_ok=True)

    # Set gene panel parameters
    shared_genes_ratio = config['gene_panel']['shared_genes_ratio']
    subset_ratios = config['gene_panel']['subset_ratios']
    seed = config['gene_panel']['seed']
    
    gene_list = get_gene_list(input_directory)
    
    gene_panel_list = create_gene_panels(gene_list, 
                                            n_gene_panels,
                                            shared_genes_ratio,
                                            subset_ratios,
                                            seed)

    create_dataset_with_multiple_gene_panels(input_directory, 
                                                output_directory,
                                                gene_panel_list)

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    # Collect configs 
    config = collect_configs(args)
    
    main(config=config)