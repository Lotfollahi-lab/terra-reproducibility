import os
import time
import tqdm
import zipfile
import tempfile
import requests
import yaml
import pickle
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Dict
from pathlib import Path
import scipy.sparse as sp
from pyensembl import EnsemblRelease

CHUNK_SIZE = 5 * 1024 * 1024


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
    fname = Path("/lustre/scratch126/cellgen/team361/DATASETS/genes") / f"{species}_gene_name_to_ensembl_id_dict.pkl"
    if os.path.exists(fname):
        print(f"Loading gene name to ensembl id dictionary for Species {species}...")
        with open(fname, 'rb') as f:
            gene_name_to_ensembl_id_dict = pickle.load(f)
    else:
        print(f"Creating gene name to ensembl id dictionary for Species {species}...")
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

    if adata.X is None and 'counts' in adata.layers.keys():
        if not isinstance(adata.layers['counts'], sp.csr_matrix):
            adata.layers['counts'] = adata.layers['counts'].tocsr()
        adata.X = adata.layers['counts']
    elif adata.X is not None and 'counts' not in adata.layers.keys():
        if isinstance(adata.X, np.ndarray) or isinstance(adata.X, pd.DataFrame):
            adata.X = sp.csr_matrix(adata.X)
                
    adata.X = adata.X.astype(np.float32)

    print(f"adata.X shape: {adata.X.shape}")
    print(f"type(adata.X): {type(adata.X)}")

    return adata


def harmonize_var_gene_names(adata,
                             gene_name_to_ensembl_id_dict):
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
        if gene_name in gene_name_to_ensembl_id_dict.keys():
            harmonized_gene_names.append(gene_name)
            matching_ensembl_ids.append(gene_name_to_ensembl_id_dict[gene_name])
        # else:
            # print(f"Gene name {gene_name} not found in gene_name_to_ensembl_id_dict. Skipped...")

    print("Number of genes in bronze data:", len(gene_names))
    print("Number of genes with matching ensembl ids:", len(harmonized_gene_names))
    print(f"Number of genes skipped: {len(gene_names) - len(harmonized_gene_names)}")

    original_sum = adata.X.sum()
    original_selected_sum = adata[:, adata.var.index.isin(harmonized_gene_names)].copy().X.sum()
    original_nonselected_sum = adata[:, ~adata.var.index.isin(harmonized_gene_names)].copy().X.sum()
    print(f"Original sum: {original_sum}")
    print(f"Original selected sum: {original_selected_sum}")
    print(f"Original nonselected sum: {original_nonselected_sum}")
    adata = adata[:, adata.var.index.isin(harmonized_gene_names)].copy()
    
    adata.var = pd.DataFrame(index=pd.Index(harmonized_gene_names, name="gene_name"), data={"ensembl_id": matching_ensembl_ids})

    harmonized_sum = adata.X.sum()
    # assert original_sum == original_selected_sum + original_nonselected_sum
    assert harmonized_sum == original_selected_sum
    
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


def harmonize_uns_metadata(uns_metadata,
                            adata):
    print(adata)
    for key in uns_metadata.keys():
        print(key)
        try:
            print(f"{key} is present in adata.uns: {adata.uns[key]}")
        except:
            print("Setting metadata from config file to adata.uns")
            try:
                assert key in adata.obs.keys() and adata.obs[key].nunique() == 1
                print(f"{key} is present in adata.obs: {adata.obs[key].iloc[0]}")
                adata.uns[key] = adata.obs[key].iloc[0]
            except:
                print(f"{key} is missing from adata. Setting it from config file to {uns_metadata[key]}")
                print(f"Setting adata.uns['{key}'] = {uns_metadata[key]}")
                adata.uns[key] = uns_metadata[key]

    if 'batch' not in adata.uns['batch']:
        adata.uns['batch'] = f"batch{adata.uns['batch']}"

    print("Metadata in adata.uns:")
    for key in uns_metadata.keys():
        print(f"{key}: {adata.uns[key]}")
    return adata


def harmonize_batch(uns_metadata,
                    file,
                    gene_name_to_ensembl_id_dict):
    adata = sc.read_h5ad(file)
    print(f"Raw adata shape: {adata.shape}")
    
    adata = harmonize_X_gene_counts(adata)
    adata = harmonize_var_gene_names(adata,
                                     gene_name_to_ensembl_id_dict)
    adata = harmonize_obsm_spatial(adata)
    adata = harmonize_uns_metadata(uns_metadata,
                                   adata)
    
    # filter cells with less than 10 genes
    sc.pp.filter_cells(adata, min_genes=10)
    
    # set cell id
    adata.obs['cell_id'] = [f"{adata.uns['dataset_id']}_{adata.uns['batch']}_{cell_idx}" for cell_idx in range(adata.shape[0])]
    print("adata.obs['cell_id']:")
    print(adata.obs['cell_id'])
    
    print(f"Harmonized adata shape: {adata.shape}")
    return adata    


def harmonize_bronze_data_to_silver_data(uns_metadata,
                                         input_directory,
                                         output_directory,
                                         gene_name_to_ensembl_id_dict):
    batch_count = 0
    for file in input_directory.glob("**/*.h5ad"):
        print(f"Harmonizing: {file}")
        uns_metadata["batch"] = str(batch_count)
        file_adata = harmonize_batch(uns_metadata,
                                     file,
                                     gene_name_to_ensembl_id_dict)
        print(file_adata)
        print(f"Writing to: {output_directory / file.name}")
        file_adata.write(output_directory / file.name)

        print("")
        batch_count += 1

    return


def pre_process_10x_xenium(raw_path):
    # What Xenium output looks like:
    # https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs
    raw_path = str(raw_path)
    print(f"[+] Reading {raw_path}")
    temp_dir = tempfile.TemporaryDirectory()
    if os.path.isfile(raw_path) and raw_path.endswith("zip"):
        print(f"[+] Input is a zip file. Extracting required files to {temp_dir.name}")
        with zipfile.ZipFile(raw_path, "r") as zip:
            zip.extract("cell_feature_matrix.h5", path=temp_dir.name)
            zip.extract("cells.parquet", path=temp_dir.name)
            zip.extract("experiment.xenium", path=temp_dir.name)
            zip.extract("gene_panel.json", path=temp_dir.name)
            zip.extract("analysis.zarr.zip", path=temp_dir.name)
        raw_path = temp_dir.name

    cells_parquet = os.path.join(raw_path, "cells.parquet")
    cell_feature_h5 = os.path.join(raw_path, "cell_feature_matrix.h5")

    # read cell-feature matrix
    print(f"[+] Reading cell-feature matrix {cell_feature_h5}")
    adata = sc.read_10x_h5(filename=cell_feature_h5)

    # make sure var_names is ensembl_id instead of gene name
    adata.var = adata.var.drop(columns=["feature_types","genome"])
    adata.var = adata.var.reset_index()
    adata.var.columns = ['gene_name','ensembl_id']
    adata.var = adata.var.set_index('gene_name')

    # read observations
    print(f"[+] Reading observations {cells_parquet}")
    df = pd.read_parquet(cells_parquet)
    df = df.set_index(adata.obs_names)
    adata.obs = df.copy()
    # get spatial coordintes in the embedings slot
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
    adata.obs = adata.obs.drop(columns=["x_centroid", "y_centroid"])

    # record xenium experiment and gene panel information
    experiment_xenium = os.path.join(raw_path, "experiment.xenium")
    print(f"[+] Reading experiment {experiment_xenium}")
    with open(experiment_xenium, "rt") as f:
        adata.uns["experiment.xenium"] = f.read()
    gene_panel = os.path.join(raw_path, "gene_panel.json")
    print(f"[+] Reading gene panel {gene_panel}")
    with open(gene_panel, "rt") as f:
        adata.uns["gene_panel.json"] = f.read()

    return adata


def pre_process_cosmx(raw_path):
    """
    Preprocesses raw data from NanoString CosMx

    Parameters:
    ----------
    raw_path (str): Path of the folder with NanoString CosMx zip file
    
    Returns:
    -------
    adata: AnnData object with the NanoString CosMx data
    """
    raw_path = str(raw_path)
    print(f"[+] Reading {raw_path}")
    if os.path.isfile(raw_path) and raw_path.endswith("zip"):
        temp_dir = tempfile.TemporaryDirectory()
        print(f"[+] Input is a zip file. Extracting required files to {temp_dir.name}")
        zip_file = zipfile.ZipFile(raw_path)
        for file in zip_file.namelist():
            if any([ext in file for ext in ['exprMat','metadata']]):
                if 'exprMat' in file:
                    exprfile = file
                elif 'metadata' in file:
                    metafile = file
                with zipfile.ZipFile(raw_path, "r") as zipp:
                    zipp.extract(file, path=temp_dir.name)
        raw_path = temp_dir.name
    else:
        print(f"[+] Input is a directory. Finding required files")
        for file in os.listdir(raw_path):
            if any([ext in file for ext in ['exprMat','metadata']]):
                if 'exprMat' in file:
                    exprfile = file
                elif 'metadata' in file:
                    metafile = file
                    
    exprfile = os.path.join(raw_path, exprfile)
    metafile = os.path.join(raw_path, metafile)

    print(f"[+] Reading cell-feature matrix {exprfile}")
    exp = pd.read_csv(exprfile)
    exp.index = [f"{i}_{j}" for i,j in zip(exp['fov'], exp['cell_ID'])]

    print(f"[+] Reading metadata matrix {metafile}")
    meta = pd.read_csv(metafile)
    meta.index = [f"{i}_{j}" for i,j in zip(meta['fov'], meta['cell_ID'])]

    exp = exp.drop(columns=['fov', 'cell_ID', 'cell'], errors='ignore')

    adata = sc.AnnData(exp)

    adata.obs_names = exp.index
    adata.var_names = exp.columns
    
    common_obsnames = adata.obs_names[adata.obs_names.isin(meta.index)]

    meta = meta.loc[common_obsnames]

    adata = adata[common_obsnames].copy()
    adata.obs = meta
    
    return adata


def pre_process(raw_path, assay):
    """
    Preprocesses raw data according to the assay type

    Parameters:
    ----------
    - raw_path (str) : Path of the folder with raw files, assuming the folder name is the sample name
    - assay (str) : Type of assay  
    
    Returns:
    -------
    - adata : AnnData object with the NanoString CosMx data
    """
    
    if assay == 'xenium':
        adata = pre_process_10x_xenium(raw_path)
    elif assay == 'cosmx':
        adata = pre_process_cosmx(raw_path)
    
    return adata


def download_zipfiles(urls: list[str] = [],
                      input_directory: Path = None,
                      redownload: str = "False") -> list[Path]:
    raw_zipfile_paths = []
    for url in urls:
        raw_zipfile_name = f"{os.path.basename(url)}"
        raw_zipfile_path = input_directory / raw_zipfile_name
        raw_zipfile_paths.append(raw_zipfile_path)

        # download files if not already downloaded
        if not os.path.exists(raw_zipfile_path) or redownload == "True":
            # download files
            print(f"Downloading files from {url}...")
            download_url(url,
                        raw_zipfile_path)
        else:
            print(f"{raw_zipfile_path} already exists. Skipping download...")
    return raw_zipfile_paths


def download_url(url: str = None,
                 raw_file_path: Path = None):
    # poor man's retry
    _ATTEMPTS=3
    for attempt in range(1,_ATTEMPTS+1):
        try:
            r = requests.get(url, stream=True)
            total_size = int(r.headers.get("content-length", 0))
            with tqdm.tqdm(
                total=total_size, unit="B", unit_scale=True, desc=os.path.basename(url)
            ) as p:
                with open(raw_file_path, "wb") as f:
                    for chunk in r.iter_content(CHUNK_SIZE):
                        p.update(len(chunk))
                        f.write(chunk)
            return True
        except Exception as ex:
            # log error, wait 60s and try agaain
            print(f"Attempt #{attempt} failed to download. Error :{repr(ex)}")
            time.sleep(60)
    print(f"ERROR. Fail all {_ATTEMPTS} attempts")


def main(config: Dict):
    # get experiment params
    experiment_name = config["experiment"]["name"]
    print(f"Experiment Name: {experiment_name}")
    
    # set data paths
    data_directory_path = Path(config["paths"]["data_directory_path"])
    bronze_directory_path = data_directory_path / "bronze"
    silver_directory_path = data_directory_path / "silver"
    
    # download files and put them in here
    input_directory = bronze_directory_path / experiment_name
    os.makedirs(input_directory, exist_ok=True)
    print(f"Input Directory: {input_directory}")
    h5ad_files = list(input_directory.glob("**/*.h5ad"))
    print(h5ad_files)
    

    # preprocess files saved in input_directory and save them in output_directory
    output_directory = silver_directory_path / experiment_name
    os.makedirs(output_directory, exist_ok=True)

    # get dataset params
    uns_metadata = {}
    uns_metadata["dataset_id"] = config["dataset"]["dataset_id"]
    uns_metadata["assay"] = config["dataset"]["assay"]
    uns_metadata["species"] = config["dataset"]["species"]
    uns_metadata["tissue"] = config["dataset"]["tissue"]
    
    if config["data_source"]["online"]:
        raw_zipfile_paths = download_zipfiles(config["data_source"]["urls"],
                                                input_directory,
                                                config["data_source"]["redownload"])
        
        # create raw adata from raw zip file or load it if already exists
        if len(h5ad_files) == 0 or config["experiment"]["overwrite"]:
            for raw_zipfile_path in raw_zipfile_paths:
                print("Creating raw adata from raw zip file...")
                adata = pre_process(raw_zipfile_path, uns_metadata["assay"])
                print("Raw adata:")
                print(adata)
                raw_adata_fname = input_directory / f"{raw_zipfile_path.stem}.h5ad"
                print(f"Writing raw adata to file: {raw_adata_fname}")
                adata.write(raw_adata_fname)
        else:
            print("Raw adata already exists.")

    # save or load gene_name_to_ensembl_id_dict
    gene_name_to_ensembl_id_dict = create_gene_name_to_ensembl_id_dict(uns_metadata["species"])

    harmonize_bronze_data_to_silver_data(uns_metadata,
                                        input_directory,
                                        output_directory,
                                        gene_name_to_ensembl_id_dict)


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
                        default='/lustre/scratch126/cellgen/team361/am84/NicheJEPA/reproducibility/config/bronze_to_silver/xhs1000-39b_1p.yaml',
                        help='Path to the config file')
    parser.add_argument('--override', 
                        nargs='+',
                        default=False, 
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
            if key == 'overwrite':
                if value == 'True':
                    config['experiment'][key] = True
                elif value == 'False':
                    config['experiment'][key] = False
    else:
        config["experiment"]["overwrite"] = False
                    
    return config


if __name__ == '__main__':

    args = parse_arguments()
    config = collect_configs(args)
    main(config)