import argparse
from pathlib import Path
import os
import yaml
from typing import Dict, List
import anndata as ad
import scanpy as sc
import pyensembl

def harmonize_obs(input_directory: Path = None) -> List[ad.AnnData]:
    """
    Harmonize obs of all datasets.
    
    Parameters
    ----------
    input_directory:
        Directory containing input (silver) preprocessed anndata files.
    
    Returns
    ----------
    adata_batch_list:
        List of AnnData objects containing the harmonized data.
    """
    starmap_file_path = Path('/lustre/scratch126/cellgen/team361/DATASETS/silver/mmb0-1b_smb1-1b_2p/harmonised_starmap_plus_mouse_cns_batch15.h5ad')
    adata_starmap = sc.read(starmap_file_path)
    print("Original starmap data:")
    print(adata_starmap)

    # adata_starmap.X = adata_starmap.layers["counts"]
    # Multiply the coordinates of starmap by 1000
    # adata_starmap.obsm["spatial"] = adata_starmap.obsm["spatial"] * 1000
    
    # sc.pp.filter_cells(adata_starmap, min_genes=10)
    # print("Filtered starmap data:")
    # print(adata_starmap)

    merfish_file_path = Path('/lustre/scratch126/cellgen/team361/DATASETS/silver/mmb0-1b_smb1-1b_2p/harmonised_merfish_mouse_brain_239-batch_batch82.h5ad')
    adata_merfish = sc.read(merfish_file_path)
    print("Original merfish data:")
    print(adata_merfish)

    # sc.pp.filter_cells(adata_merfish, min_genes=10)
    
    # print("Filtered merfish data:")
    # print(adata_merfish)

    # Ensure both adata_starmap and adata_merfish are initialized
    # if adata_starmap is None or adata_merfish is None:
        # raise ValueError("Both starmap and merfish datasets must be provided.")
        
    # Set donor_id in starmap
    # donor_id = list(adata_merfish.obs["donor_id"].value_counts().to_dict().keys())[0]
    # adata_starmap.obs["donor_id"] = donor_id

    # Set brain_section_label for starmap
    brain_section_label = list(adata_merfish.obs["brain_section_label"].value_counts().to_dict().keys())[0]
    adata_starmap.obs["brain_section_label"] = brain_section_label
    
    # Set sample_id for merfish and starmap
    # sample_id = list(adata_starmap.obs["batch"].value_counts().to_dict().keys())[0]
    # adata_starmap.obs["sample_id"] = sample_id
    # adata_merfish.obs["sample_id"] = sample_id
    
    # Set batch for starmap
    # adata_starmap.obs["batch"] = starmap_file_path.stem.split('_')[-1]
    
    # Set assay for merfish and starmap
    # adata_merfish.obs["assay"] = "merfish"
    # adata_starmap.obs["assay"] = "starmap"
        
    # Set major_brain_region in starmap
    # adata_starmap.obs["major_brain_region"] = adata_starmap.obs["Main_molecular_tissue_region"]
    
    # Set cell_type in starmap
    # adata_starmap.obs["cell_type"] = adata_starmap.obs["Main_molecular_cell_type"]
    
    # Set ccf_region_name in starmap to NaN
    # adata_starmap.obs["ccf_region_name"] = "NaN"
    
    # Set Sub_molecular_cell_type in merfish to NaN
    # adata_merfish.obs["Sub_molecular_cell_type"] = "NaN"
    
    # Set Sub_molecular_tissue_region in merfish to NaN
    # adata_merfish.obs["Sub_molecular_tissue_region"] = "NaN"
        
    return starmap_file_path, merfish_file_path, adata_starmap, adata_merfish


def fix_var(adata_starmap, adata_merfish) -> ad.AnnData:
    print(adata_starmap.var)
    print(adata_merfish.var)
    adata_starmap.var.index = [s.capitalize() for s in adata_starmap.var.index]
    starmap_gene_symbols = set([s.capitalize() for s in adata_starmap.var.index.tolist()])
    merfish_gene_symbols = set([s.capitalize() for s in adata_merfish.var.index.tolist()])
    shared_gene_symbols = starmap_gene_symbols.intersection(merfish_gene_symbols)
    print(f"Number of shared gene symbols: {len(shared_gene_symbols)}")
    
    adata_merfish_var_fixed = adata_merfish[:, adata_merfish.var.index.isin(shared_gene_symbols)].copy()
    adata_starmap_var_capital = adata_starmap[:, adata_starmap.var.index.isin(shared_gene_symbols)].copy()
    
    starmap_var_ordered = adata_starmap_var_capital.var.loc[adata_merfish_var_fixed.var.index]
    adata_starmap_var_fixed = adata_starmap_var_capital[:, starmap_var_ordered.index].copy()
    
    adata_starmap_var_fixed.var.index.name = 'gene_symbol'
    adata_starmap_var_fixed.var['ensembl_id'] = adata_merfish_var_fixed.var['ensembl_id']
    
    return adata_starmap_var_fixed, adata_merfish_var_fixed
    

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

    # harmonize obs of all datasets
    starmap_file_path, merfish_file_path, adata_starmap, adata_merfish = harmonize_obs()
    
    adata_starmap_var_fixed, adata_merfish_var_fixed = fix_var(adata_starmap, adata_merfish)
    
    sc.pp.filter_cells(adata_starmap_var_fixed, min_genes=10)
    sc.pp.filter_cells(adata_merfish_var_fixed, min_genes=10)
    
    print("Filtered starmap data:")
    print(adata_starmap_var_fixed)
    print(adata_starmap_var_fixed.obs['cell_id'][2])
    print("Filtered merfish data:")
    print(adata_merfish_var_fixed)
    print(adata_merfish_var_fixed.obs['cell_id'][2])
    
    
    # Save the harmonized data
    adata_starmap_var_fixed.write(output_directory / starmap_file_path.name)
    adata_merfish_var_fixed.write(output_directory / merfish_file_path.name)
    print(f"Harmonized data saved to {output_directory}")


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    # Collect configs 
    config = collect_configs(args)
    
    main(config=config)