import os
import pickle
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from pathlib import Path
import scipy.sparse as sp
from pyensembl import EnsemblRelease


def create_gene_name_to_ensembl_id_dict(species: str, 
                                        release: int = 111) -> dict:
    """
    Create a dictionary of gene symbol names to ensembl id obtained from Ensembl Release.
    
    Parameters:
    ----------
    - species (str): The species of the dataset.
    - release (int): The release version of the Ensembl database.
    
    Returns:
    ----------
    - Dict: A dictionary of gene symbol names to ensembl id.
    """
    fname = Path("/lustre/scratch126/cellgen/team361/am84/NicheJEPA/datasets") / f"{species}_gene_name_to_ensembl_id_dict.pkl"
    if os.path.exists(fname):
        print(f"Loading gene name to ensembl id dictionary for Species {args.species}...")
        with open(fname, 'rb') as f:
            gene_name_to_ensembl_id_dict = pickle.load(f)
    else:
        print(f"Creating gene name to ensembl id dictionary for Species {args.species}...")
        # Extract Ensembl IDs of protein coding and miRNA mouse genes
        ensembl = EnsemblRelease(release=release, species=species)
        ensembl.download()
        ensembl.index()
        all_genes = ensembl.genes()
        protein_coding_genes = [gene for gene in all_genes if gene.biotype == "protein_coding"]
        mirna_genes = [gene for gene in all_genes if gene.biotype == "miRNA"]
        all_relevant_genes = protein_coding_genes + mirna_genes
        gene_name_to_ensembl_id_dict = {gene.gene_name.capitalize(): gene.gene_id for gene in all_relevant_genes}
        with open(fname, 'wb') as f:
            pickle.dump(gene_name_to_ensembl_id_dict, f)

    return gene_name_to_ensembl_id_dict



def harmonize_X_gene_counts(args, adata):
    if adata.X is None and 'counts' not in adata.layers.keys():
        raise ValueError("Both adata.X and adata.layers['counts'] are missing.")

    if adata.X is None and 'counts' in adata.layers.keys():
        if not isinstance(adata.layers['counts'], sp.csr_matrix):
            adata.layers['counts'] = adata.layers['counts'].tocsr()
        adata.X = adata.layers['counts']
    elif adata.X is not None and 'counts' not in adata.layers.keys():
        if not isinstance(adata.X, sp.csr_matrix):
            adata.X = adata.X.tocsr()
                
    adata.X = adata.X.astype(np.float32)

    print(f"adata.X shape: {adata.X.shape}")
    print(f"type(adata.X): {type(adata.X)}")
    
    return adata
    

def harmonize_var_gene_names(args,
                             adata,
                             gene_name_to_ensembl_id_dict):
    if adata.var is None:
        raise ValueError("adata.var is missing.")
    
    if adata.var_names is None or len(adata.var_names) == 0:
        raise ValueError("Gene names are not available.")

    if adata.var.empty:
        print(adata.var.index)
        assert len(adata.var.index) == adata.shape[1], "Number of genes in adata.var is not equal to number of genes in adata.X."

    gene_names = [gene_name.capitalize() for gene_name in adata.var_names.tolist()]
    adata.var.index = gene_names
    
    harmonized_gene_names = []
    matching_ensembl_ids = []
    for gene_name in gene_names:
        if gene_name in gene_name_to_ensembl_id_dict.keys():
            harmonized_gene_names.append(gene_name)
            matching_ensembl_ids.append(gene_name_to_ensembl_id_dict[gene_name])
        # else:
            # print(f"Gene name {gene_name} not found in gene_name_to_ensembl_id_dict. Skipped...")

    print("Number of genes in bronze data:", len(gene_names))
    print("Number of genes with matching ensemble ids:", len(harmonized_gene_names))
    print(f"Number of genes skipped: {len(gene_names) - len(harmonized_gene_names)}")

    original_sum = adata.X.sum()
    original_selected_sum = adata[:, adata.var.index.isin(harmonized_gene_names)].X.sum()
    original_nonselected_sum = adata[:, ~adata.var.index.isin(harmonized_gene_names)].X.sum()

    adata = adata[:, adata.var.index.isin(harmonized_gene_names)].copy()
    
    adata.var = pd.DataFrame(index=pd.Index(harmonized_gene_names, name="gene_name"), data={"ensembl_id": matching_ensembl_ids})

    harmonized_sum = adata.X.sum()
    
    assert original_sum == original_selected_sum + original_nonselected_sum
    assert harmonized_sum == original_selected_sum
    
    # print("Sum of original adata.X:", original_sum)
    # print("Sum of original selected gene name columns:", original_selected_sum)
    # print("Sum of original non selected gene name columns:", original_nonselected_sum)
    # print(f"Original sum == original selected + original non selected: {original_sum == original_selected_sum + original_nonselected_sum}")
    # print(f"harmonized_sum == original_selected_sum: {harmonized_sum == original_selected_sum}")
    
    print("adata.var:")
    print(adata.var)

    return adata


def harmonize_obsm_spatial(args,
                            adata):
    if adata.obsm is None:
        raise ValueError("adata.obsm is missing.")
    
    if 'spatial' in adata.obsm.keys():
        assert adata.obsm['spatial'].shape[0] == adata.shape[0], "Number of spatial coordinates is not equal to number of cells."
        assert adata.obsm['spatial'].shape[1] == 2, "Number of spatial coordinates is not equal to 2."
        if not isinstance(adata.obsm['spatial'], np.ndarray):
            adata.obsm['spatial'] = np.array(adata.obsm['spatial'])
        
        if adata.obsm['spatial'].dtype != np.float32:
            adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
        print("adata.obsm['spatial'] is already present.")

    if 'spatial' not in adata.obsm.keys():
        if args.x_coords not in adata.obs.keys():
            raise ValueError(f"{args.x_coords} is missing in adata.obs.")
        if args.y_coords not in adata.obs.keys():
            raise ValueError(f"{args.y_coords} is missing in adata.obs.")
        x_coords = np.array(adata.obs[args.x_coords].tolist(), dtype=np.float32)
        y_coords = np.array(adata.obs[args.y_coords].tolist(), dtype=np.float32)
        adata.obsm['spatial'] = np.vstack([x_coords, y_coords]).T
        
    print("adata.obsm['spatial']:")
    print(adata.obsm['spatial'])
    
    return adata
    

def harmonize_uns_metadata(args,
                            adata):
    for key in ['dataset_id', 'assay', 'tissue', 'species', 'batch']:
        try:
            print(f"{key} is present in adata.uns: {adata.uns[key]}")
        except:
            try:
                if key in adata.obs.keys() and adata.obs[key].nunique() == 1:
                    print(f"{key} is present in adata.obs: {adata.obs[key].iloc[0]}")
                    adata.uns[key] = adata.obs[key].iloc[0]
            except:
                print(f"{key} is missing from adata. Setting it from args to {getattr(args,key)}")
                adata.uns[key] = getattr(args,key)

    if 'batch' not in adata.uns['batch']:
        adata.uns['batch'] = f"batch{adata.uns['batch']}"
    
    print("Metadata in adata.uns:")
    for key in ['dataset_id', 'assay', 'tissue', 'species', 'batch']:
        print(f"{key}: {adata.uns[key]}")
    return adata


def harmonize_batch(args,
                    file,
                    gene_name_to_ensembl_id_dict):
    adata = sc.read_h5ad(file)
    print(f"Raw adata shape: {adata.shape}")
    
    adata = harmonize_X_gene_counts(args, 
                                    adata)
    adata = harmonize_var_gene_names(args,
                                     adata, gene_name_to_ensembl_id_dict)
    adata = harmonize_obsm_spatial(args,
                                   adata)
    adata = harmonize_uns_metadata(args,
                                   adata)
    
    # filter cells with less than 10 genes
    sc.pp.filter_cells(adata, min_genes=10)
    
    # set cell id
    adata.obs['cell_id'] = [f"{adata.uns['dataset_id']}_{adata.uns['batch']}_{cell_idx}" for cell_idx in range(adata.shape[0])]
    print("adata.obs['cell_id']:")
    print(adata.obs['cell_id'])
    
    print(f"Harmonized adata shape: {adata.shape}")
    return adata    


def harmonize_bronze_data_to_silver_data(args,
                                         input_directory,
                                         output_directory,
                                         gene_name_to_ensembl_id_dict):    
    for file in input_directory.glob("**/*.h5ad"):
        print(f"Harmonizing: {file}")
        file_adata = harmonize_batch(args,
                                     file,
                                     gene_name_to_ensembl_id_dict)
        print(file_adata)
        print(f"Writing to: {output_directory / file.name}")
        file_adata.write(output_directory / file.name)
        
        print("")

    return


def main(args):
    data_directory_path = Path(args.data_directory_path)
    bronze_directory_path = data_directory_path / "bronze"
    silver_directory_path = data_directory_path / "silver"
    
    input_directory = bronze_directory_path / args.experiment_name
    output_directory = silver_directory_path / args.experiment_name
    
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Experiment Name: {args.experiment_name}")
    
    gene_name_to_ensembl_id_dict = create_gene_name_to_ensembl_id_dict(args.species)
    
    harmonize_bronze_data_to_silver_data(args,
                                         input_directory,
                                         output_directory,
                                         gene_name_to_ensembl_id_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    parser.add_argument('--data_directory_path', 
                        type=str,
                        default='/lustre/scratch126/cellgen/team361/DATASETS',
                        help='Top-level directory path for all datasets')
    parser.add_argument('--experiment_name',
                        type=str,
                        default="xhs1000-39b_1p",
                        help='Name of the experiment')
    parser.add_argument('--dataset_id',
                        type=str,
                        default="1000",
                        help='Dataset ID')
    parser.add_argument('--assay',
                        type=str,
                        default="xenium",
                        help='Assay or technology used')
    parser.add_argument('--tissue',
                        type=str,
                        default="skin",
                        help='Name of tissue')
    parser.add_argument('--species',
                        type=str,
                        default="homo_sapiens",
                        help='Name of species')
    parser.add_argument('--batch',
                        type=str,
                        default="batch0",
                        help='batch id')
    parser.add_argument('--x_coords',
                        type=str,
                        default="Raw_X",
                        help='Key in .obs to use as x coordinate')
    parser.add_argument('--y_coords',
                        type=str,
                        default="Raw_Y",
                        help='Key in .obs to use as y coordinate')
    args = parser.parse_args()    
    
    main(args)