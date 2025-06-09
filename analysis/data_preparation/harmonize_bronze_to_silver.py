import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Dict
from pathlib import Path
import scipy.sparse as sp
from pyensembl import EnsemblRelease

from argparse_utils import parse_arguments, collect_configs
from download_utils import download_zipfiles
from extract_utils import pre_process


def create_gene_ensembl_map_dict(
        data_directory_path: Path,
        species: str, 
        release: int=111
    ) -> dict:
    """
    Create a dictionary of gene symbol names to ensembl id obtained from Ensembl Release.
    
    Parameters:
    ----------
    data_directory_path:
        The path to the data directory.
    species:
        The species of the dataset.
    release:
        The release version of the Ensembl database.
    
    Returns:
    ----------
    gene_name_to_ensembl_id_dict:
        A dictionary of gene symbol names to ensembl id.
    """
    fname = data_directory_path / "genes" / f"{species}_gene_name_to_ensembl_id_dict.pkl"
    if os.path.exists(fname):
        print(f"Loading gene name to ensembl id dictionary from {fname}...")
        with open(fname, 'rb') as f:
            gene_name_to_ensembl_id_dict = pickle.load(f)
    else:
        print(f"Creating gene name to ensembl id dictionary at {fname}...")
        # Extract Ensembl IDs of protein coding and miRNA mouse genes
        ensembl = EnsemblRelease(release=release, species=species)
        ensembl.download()
        ensembl.index()
        all_genes = ensembl.genes()
        protein_coding_genes = [gene for gene in all_genes if gene.biotype == "protein_coding"]
        mirna_genes = [gene for gene in all_genes if gene.biotype == "miRNA"]
        all_relevant_genes = protein_coding_genes + mirna_genes
        gene_name_to_ensembl_id_dict = {gene.gene_name: gene.gene_id for gene in all_relevant_genes}
        with open(fname, 'wb') as f:
            pickle.dump(gene_name_to_ensembl_id_dict, f)

    return gene_name_to_ensembl_id_dict


def harmonize_X_gene_counts(adata):
    if adata.X is None and 'counts' not in adata.layers.keys():
        raise ValueError("Both adata.X and adata.layers['counts'] are missing.")

    elif adata.X is None and 'counts' in adata.layers.keys():
        print("adata.X is missing. adata.layers['counts'] is present.")
        
        # convert adata.layers['counts'] to sp.csr_matrix if it is not already
        if not isinstance(adata.layers['counts'], sp.csr_matrix):
            adata.layers['counts'] = adata.layers['counts'].tocsr()
        
        # check if all entries in adata.layers['counts'] are integers
        if not np.all(np.mod(adata.layers['counts'].data, 1) == 0):
            raise ValueError("Not all entries in adata.layers['counts'] are integers.")
        
        # if all entries are integers, set adata.X from adata.layers['counts']
        print("All entries in adata.layers['counts'] are integers.")
        print("Setting adata.X from adata.layers['counts']...")
        adata.X = adata.layers['counts']
        
    elif adata.X is not None:
        print("adata.X is present.")
        if not isinstance(adata.X, sp.csr_matrix):
            adata.X = sp.csr_matrix(adata.X)

        # Check if all entries of adata.X are integers even if the type is float
        if not np.all(np.mod(adata.X.data, 1) == 0):
            print("Not all entries in adata.X are integers.")
            if adata.layers['counts'] is None:
                raise ValueError("adata.layers['counts'] is missing and adata.X has float entries.")
            else:
                # convert adata.layers['counts'] to sp.csr_matrix if it is not already
                if not isinstance(adata.layers['counts'], sp.csr_matrix):
                    adata.layers['counts'] = adata.layers['counts'].tocsr()
                
                if not np.all(np.mod(adata.layers['counts'].data, 1) == 0):
                    raise ValueError("Not all entries in adata.layers['counts'] are integers.")
                else:
                    print("All entries in adata.layers['counts'] are integers.")
                    print("Setting adata.X from adata.layers['counts']...")
                    adata.X = adata.layers['counts']
        else:
            print("All entries in adata.X are integers.")

    # set adata.X to int dtype
    adata.X = adata.X.astype(int)
    print(adata.X)
    
    # check that adata.X is sp.csr_matrix
    assert isinstance(adata.X, sp.csr_matrix), "adata.X is not a sp.csr_matrix."
    assert np.all(np.mod(adata.X.data, 1) == 0), "Not all entries in adata.X are integers."
    assert adata.X.dtype == int, "adata.X is not of int dtype."

    print(f"adata.X shape: {adata.X.shape}")
    print(f"type(adata.X): {type(adata.X)}")
    print(f"adata.X.dtype: {adata.X.dtype}")

    return adata


def harmonize_var_gene_names(
        adata,
        gene_ensembl_map_dict,
    ):
    if adata.var is None:
        raise ValueError("adata.var is missing.")
    
    if adata.var_names is None or len(adata.var_names) == 0:
        raise ValueError("Gene names are not available.")

    if adata.var.empty:
        print(adata.var.index)
        assert len(adata.var.index) == adata.shape[1], "Number of genes in adata.var is not equal to number of genes in adata.X."

    gene_names = [gene_name for gene_name in adata.var_names.tolist()]
    adata.var.index = gene_names
    
    harmonized_gene_names = []
    matching_ensembl_ids = []
    for gene_name in gene_names:
        if gene_name in gene_ensembl_map_dict.keys():
            harmonized_gene_names.append(gene_name)
            matching_ensembl_ids.append(gene_ensembl_map_dict[gene_name])
        # else:
            # print(f"Gene name {gene_name} not found in gene_ensembl_map_dict. Skipped...")

    print("Number of genes in bronze data:", len(gene_names))
    print("Number of genes with matching ensembl ids:", len(harmonized_gene_names))
    print(f"Number of genes skipped: {len(gene_names) - len(harmonized_gene_names)}")

    adata = adata[:, adata.var.index.isin(harmonized_gene_names)].copy()
    
    adata.var = pd.DataFrame(index=pd.Index(harmonized_gene_names, name="gene_name"), data={"ensembl_id": matching_ensembl_ids})
    
    print("adata.var:")
    print(adata.var)

    return adata


def harmonize_obsm_spatial(adata):
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
    else:
        raise ValueError("Spatial coordinates are missing.")

    print("adata.obsm['spatial']:")
    print(adata.obsm['spatial'])
    
    return adata


def harmonize_uns_metadata(
        adata,
        uns_params,
    ):
    print(adata)
    for key, value in uns_params.items():
        try:
            if value != adata.uns[key]:
                print(f"{key} present in adata.uns.")
                print("Value is not harmonized. Setting from config...")
                adata.uns[f'original_{key}'] = adata.uns[key]
            else:
                print(f"{key} present in adata.uns.")
                print("Value is already harmonized.")
        except:
            print(f"{key} not present in adata.uns.")
            print("Setting from config...")

        adata.uns[key] = value
        print(f"adata.uns['{key}']: {value}")

    return adata


def harmonize_batch(
        adata_batch,
        uns_params,
        misc_params,
        gene_ensembl_map_dict,
    ):
    print(f"Raw adata shape: {adata_batch.shape}")
    adata_batch = adata_batch.copy()
    
    adata_batch = harmonize_X_gene_counts(
                    adata_batch
                    )
    adata_batch = harmonize_var_gene_names(
                    adata_batch,
                    gene_ensembl_map_dict,
                    )
    adata_batch = harmonize_obsm_spatial(
                    adata_batch
                    )
    adata_batch = harmonize_uns_metadata(
                    adata_batch,
                    uns_params,
                    )
    
    # filter cells with less than 10 genes
    sc.pp.filter_cells(
        adata_batch,
        min_genes=misc_params['min_genes']
    )
    
    # set cell id
    if 'cell_id' in adata_batch.obs.keys():
        adata_batch.obs['original_cell_id'] = adata_batch.obs['cell_id']
    adata_batch.obs['cell_id'] = [f"{adata_batch.uns['dataset_id']}_{adata_batch.uns['batch']}_{cell_idx}" for cell_idx in range(adata_batch.shape[0])]
    print("adata_batch.obs['cell_id']:")
    print(adata_batch.obs['cell_id'])
    
    print(f"Harmonized adata_batch shape: {adata_batch.shape}")
    return adata_batch


def harmonize_bronze_data_to_silver_data(
        data_directory_path,
        input_directory,
        uns_params,
        batch_params,
        key_mappings,
        misc_params,
        output_directory,
    ):

    # save or load gene_name_to_ensembl_id_dict
    gene_ensembl_map_dict = create_gene_ensembl_map_dict(
                                data_directory_path=data_directory_path,
                                species=uns_params["species"],
                            )
    
    files = list(input_directory.glob("**/*.h5ad"))

    if batch_params['split_by_batch']:
        print("Data is not split by batch. Splitting data by batch...")

        # TODO: Remove this hardcoding
        file = files[0]
        print(f"Harmonizing: {file}")

        adata = sc.read_h5ad(file)

        # check if the obs key for batch is specified in the config file
        assert batch_params['batch_key'] is not None, "Key for batch is not provided in key_mappings."
        # check if this key is present in adata.obs.keys()
        assert batch_params['batch_key'] in adata.obs.keys(), f"Key {batch_params['batch_key']} not found in adata.obs.keys()."

        for key in key_mappings.keys():
            if key in ['cell_type', 'niche_type']:
                print(f'Setting {key} in adata.obs')
                adata.obs[key] = adata.obs[key_mappings[key]]
            elif key == 'X_coordinate':
                x_coordinates = np.array(adata.obs[key_mappings[key]])
            elif key == 'Y_coordinate':
                y_coordinates = np.array(adata.obs[key_mappings[key]])
            
        try:
            adata.obsm['spatial'] = np.column_stack((x_coordinates, y_coordinates))
        except:
            pass
        
        sample_batch_map = batch_params['sample_batch_map']
        if len(sample_batch_map) > 0:            
            for sample_id, batch_id in batch_params['sample_batch_map'].items():
                print(f"Harmonizing Sample ID: {sample_id}")
                adata_batch = adata[adata.obs[batch_params['batch_key']] == sample_id]

                uns_params['batch'] = f"batch{batch_id}"

                adata_batch = harmonize_batch(
                                    adata_batch,
                                    uns_params,
                                    misc_params,
                                    gene_ensembl_map_dict,
                                )
                
                print(f"Final Harmonized Adata: Batch = {batch_id}")
                print(adata_batch)
                
                out_fname = f"adata_batch{batch_id}.h5ad"
                print(f"Writing to: {output_directory / out_fname}")
                adata_batch.write(output_directory / out_fname)
                print("")
        else:
            # TODO: Fix this error handling
            print("Raw Adata has the following keys:")
            print(adata)
            raise ValueError("Sample batch map is empty.")
    else:
        print("Data is already split by batch.")

        # NOTE: this is hardcoded to assume that we don't have a sample_batch_map when split_by_batch is False
        if len(batch_params['sample_batch_map']) > 0:
            sample_batch_map = batch_params['sample_batch_map']
        else:
            batch_id = 0
        
        for file in files:

            print(f"Harmonizing: {file}")
            adata_batch = sc.read_h5ad(file)

            # NOTE: this is hardcoded that all batches within the same dataset have the same key for keys in key_mappings such as cell_type, niche_type, etc.
            for key in key_mappings.keys():
                if key in ['cell_type', 'niche_type']:
                    print(f'Setting {key} in adata.obs')
                    adata_batch.obs[key] = adata_batch.obs[key_mappings[key]]
                elif key == 'X_coordinate':
                    x_coordinates = np.array(adata_batch.obs[key_mappings[key]])
                elif key == 'Y_coordinate':
                    y_coordinates = np.array(adata_batch.obs[key_mappings[key]])
                
            try:
                print("Setting spatial coordinates in adata.obsm based on keys provided in key_mappings...")
                adata_batch.obsm['spatial'] = np.column_stack((x_coordinates, y_coordinates))
            except:
                pass

            if len(batch_params['sample_batch_map']) > 0:
                batch_key = adata_batch.uns[batch_params['batch_key']]
                batch_id = sample_batch_map[batch_key]
                
            uns_params['batch'] = f"batch{batch_id}"

            adata_batch = harmonize_batch(
                                adata_batch,
                                uns_params,
                                misc_params,
                                gene_ensembl_map_dict,
                            )

            print(f"Final Harmonized Adata: Batch = {batch_id}")
            print(adata_batch)

            out_fname = f"{file.stem}.h5ad"
            print(f"Writing to: {output_directory / out_fname}")
            adata_batch.write(output_directory / out_fname)
            print("")
            
            batch_id += 1

    return


def main(config: Dict):
    # -------------------- Parse args from config ------------------------------
    # get experiment params
    experiment_name = config["experiment"]["name"]
    print(f"Experiment Name: {experiment_name}")
    
    # set data paths
    data_directory_path = Path(config["paths"]["data_directory_path"])
    bronze_directory_path = data_directory_path / "bronze"
    silver_directory_path = data_directory_path / "silver"
    
    # set input directory for bronze (raw) files for this dataset
    input_directory = bronze_directory_path / experiment_name
    os.makedirs(input_directory, exist_ok=True)
    print(f"Input Directory: {input_directory}")

    # set output directory for silver (harmonized) files for this dataset
    output_directory = silver_directory_path / experiment_name
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output Directory: {output_directory}")

    # get harmonization params
    uns_params = config['harmonization']['uns']
    batch_params = config['harmonization']['batch']
    key_mappings = config['harmonization']['key_mappings']
    misc_params = config['harmonization']['misc']

    # -------------------- Download (if necessary) ------------------------------
    # if files are online and have not been previously downloaded, download them
    if config["dataset"]["source"] == 'online':
        raw_file_paths = download_zipfiles(
            input_directory,
            **config["dataset"]["download_params"])
    elif config["dataset"]["source"] == 'directories':
        raw_file_paths = [path for path in input_directory.glob("*") if path.is_dir()]
    else:
        pass
    # else:
    #     # check if all files in input directory are zip files or directories
    #     if all([i.suffix == '.zip' for i in list(input_directory.glob('*'))]):
    #         raw_zipfile_paths = list(input_directory.glob("*"))
    #     elif all([i.is_dir() for i in list(input_directory.glob('*'))]):
    #         raw_zipfile_paths = [path for path in input_directory.glob("*") if path.is_dir()]
    #     elif all([i.suffix == '.h5ad' for i in list(input_directory.glob('*'))]):
    #         h5ad_files = list(input_directory.glob("**/*.h5ad"))
    #     else:
    #         raise ValueError("Input directory should contain either all zip files, all h5ad files or all directories.")

    # ------------------ Extract AnnData from ZipFiles --------------------------

    h5ad_files = list(input_directory.glob("**/*.h5ad"))

    # create raw h5ad from raw zip file or load it if already exists
    if len(h5ad_files) == 0 or config["dataset"]["overwrite"]:
        if len(h5ad_files) == 0:
            print("No bronze (raw) .h5ad files found.")
        elif config["dataset"]["overwrite"]:
            print("Overwriting previously existing bronze (raw) .h5ad files...")
        for raw_file_path in raw_file_paths:
            print(f"Extracting bronze (raw) .h5ad from {raw_file_path}...")
            adata = pre_process(
                        raw_file_path,
                        uns_params['assay'],
                    )
            print("Raw adata:")
            print(adata)
            raw_adata_fname = input_directory / f"{raw_file_path.stem}.h5ad"
            print(f"Writing to file: {raw_adata_fname}")
            adata.write(raw_adata_fname)
    else:
        print("Using existing bronze (raw) .h5ad files...")

    # ------------------ Harmonize --------------------------

    harmonize_bronze_data_to_silver_data(
        data_directory_path,
        input_directory,
        uns_params,
        batch_params,
        key_mappings,
        misc_params,
        output_directory,
    )


if __name__ == '__main__':

    args = parse_arguments()
    config = collect_configs(args)
    main(config)