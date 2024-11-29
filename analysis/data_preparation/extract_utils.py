import os
import zipfile
import tempfile
import scanpy as sc
import pandas as pd


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
    elif assay == 'merfish':
        adata = pre_process_merfish(raw_path)
    elif assay == 'starmap':
        adata = pre_process_starmap(raw_path)
    else:
        raise ValueError(f"Assay type {assay} not supported.")
    
    return adata


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

def pre_process_merfish(raw_merfish_dir):
    """
    Assumes the input directory is not a zip file but folder instead.
    """
    for file in os.listdir(raw_merfish_dir):
        if 'cell_by_gene' in file:
            expr_file = file
        elif 'cell_metadata' in file:
            meta_file = file
    
    expr_file = os.path.join(raw_merfish_dir, expr_file)
    meta_file = os.path.join(raw_merfish_dir, meta_file)
    
    sample = raw_merfish_dir.name
    region = "region_0"
    slide_id = f"{sample}_{region}"
    
    expr = pd.read_csv(expr_file, index_col=0, dtype={"cell": str})
    meta = pd.read_csv(meta_file, index_col=0, dtype={"EntityID": str})

    meta.index = meta.index.astype(str) + f"_{slide_id}"
    expr.index = expr.index.astype(str) + f"_{slide_id}"
    meta = meta.loc[expr.index]

    is_gene = ~expr.columns.str.lower().str.contains("blank")

    adata = sc.AnnData(expr.loc[:, is_gene], dtype=np.uint16, obs=meta)

    adata.obsm["spatial"] = adata.obs[["center_x", "center_y"]].values
    adata.obs["slide_id"] = pd.Series(slide_id, index=adata.obs_names, dtype="category")

    adata.X = sp.csr_matrix(adata.X)

    return adata


def pre_process_starmap(raw_starmap_dir):
    """
    Assumes the input directory is not a zip file but folder instead.
    """
    for file in os.listdir(raw_starmap_dir):
        if 'raw_expression' in file:
            expr_file = file
        elif 'cell_metadata' in file:
            meta_file = file
    
    expr_file = os.path.join(raw_starmap_dir, expr_file)
    meta_file = os.path.join(raw_starmap_dir, meta_file)
    
    sample = raw_starmap_dir.name
    region = "region_0"
    
    expr = pd.read_csv(expr_file, index_col=0)
    meta = pd.read_csv(meta_file, index_col=0)

    adata = sc.AnnData(expr, dtype=np.uint16, obs=meta)

    adata.obsm["spatial"] = adata.obs[["x", "y"]].values
    adata.obs["cell_id_orig"] = adata.obs_names
    adata.obs = adata.obs.reset_index(drop=True)

    adata.X = sp.csr_matrix(adata.X)

    return adata