"""
This script writes the shared-gene evaluation data of a dataset with two or more sections: each silver file restricted
to the genes measured in every section (by var['ensembl_id'], the ids the tokenizer reads), without the cells that then
have fewer than `--min_shared_genes` detected shared genes of the token dictionary (default 10). The files go to the
silver folder of the gene set, silver/<DATASET_NAME>_shared_genes (`ablation_utils.get_gene_set_dataset_name`), under
the source file names, with the same obs names, obs, obsm and uns as the source files, and without explicitly stored
zeros. The source files are not modified. The folder records the md5 of the source files and the minimum
(ablation_provenance.json), which check_setup.py (check 4) compares with the current silver files.

Why: each section of the training data is tokenized over its own gene panel, so a section's top genes include genes
that the other sections do not measure, and the integration metrics between the sections then partly measure panel
identity. The published mouse evaluation embedded shared-gene re-tokenizations of both sections instead
(stjepa_v2_ablation.ipynb: the `_shared_genes` files of mmb0-1b_smb1-1b_1p_coord_aligned, tokenized with
min_genes_per_cell 10). tokenize_dataset.py --gene_set shared_genes tokenizes these files with every tokenization of the
plan, the token dictionary and the normalization factor file of training (`submit_tokenization.sh` submits them with the
training datasets when EVAL_GENE_SET is shared_genes). The factor file covers all shared genes, since every section
measures them.

Usage (`source datasets/<dataset>.env` first; the silver files are the filtered training files):
```bash
python filter_shared_genes.py --silver_file_paths ${SILVER_FILE_PATHS} \
    --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}" \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}"
```
"""

import argparse
import os
import pickle
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ablation_utils import (PROVENANCE_FILE_NAME,
                            count_detected_vocabulary_genes,
                            get_gene_set_dataset_name,
                            get_gene_set_silver_file_paths,
                            get_shared_gene_provenance,
                            write_provenance)


def get_gene_ids(adata: ad.AnnData) -> pd.Series:
    """Ensembl ids of the features as strings, None for a feature without an id."""
    gene_ids = adata.var["ensembl_id"]
    valid = gene_ids.notna() & ~gene_ids.astype(str).str.strip().str.lower().isin(["", "nan", "none"])
    return gene_ids.astype(str).where(valid, None)


def main(args):
    with open(args.token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)
    source_file_paths = [Path(file_path).resolve() for file_path in args.silver_file_paths]
    if len(source_file_paths) < 2:
        raise ValueError("Shared genes need two or more sections; with one section, they are all genes.")
    output_file_paths = [Path(file_path) for file_path in get_gene_set_silver_file_paths(
        args.silver_file_paths, args.data_directory_path, args.dataset_name, "shared_genes")]
    output_folder_path = output_file_paths[0].parent.resolve()
    if len({file_path.name for file_path in source_file_paths}) != len(source_file_paths):
        raise ValueError(f"The silver files share file names: {[str(path) for path in source_file_paths]}.")
    if any(file_path.parent == output_folder_path for file_path in source_file_paths):
        raise ValueError(f"{output_folder_path} holds source files; the silver files are those of "
                         f"{args.dataset_name}.")
    # The tokenizer reads every h5ad file of the silver folder: it must hold exactly the written files
    output_file_names = {file_path.name for file_path in output_file_paths}
    other_files = sorted(str(file_path) for file_path in output_folder_path.rglob("*.h5ad")
                         if file_path.name not in output_file_names) if output_folder_path.exists() else []
    if other_files:
        raise ValueError(f"{output_folder_path} holds other h5ad files, which the tokenizer would also read: "
                         f"{other_files}.")

    # Genes measured in every section
    adatas = [ad.read_h5ad(file_path, backed="r") for file_path in source_file_paths]
    shared_gene_ids = set.intersection(*[set(get_gene_ids(adata).dropna()) for adata in adatas])
    n_vocabulary_genes = sum(bool(token_dict.get(gene_id, False)) for gene_id in shared_gene_ids)
    print(f"{len(shared_gene_ids)} genes measured in all {len(adatas)} sections ({n_vocabulary_genes} in the token "
          f"dictionary); minimum detected shared vocabulary genes per cell: {args.min_shared_genes}.")
    if n_vocabulary_genes == 0:
        raise ValueError("The sections share no gene of the token dictionary.")
    del adatas

    # Files are written next to the output folder first, then moved in, so that it never holds partial files
    temporary_folder_path = output_folder_path.parent / f".{output_folder_path.name}.tmp"
    temporary_folder_path.mkdir(parents=True, exist_ok=True)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    # Written last: a folder whose files are being replaced is never taken as made from these files
    (output_folder_path / PROVENANCE_FILE_NAME).unlink(missing_ok=True)
    for source_file_path, output_file_path in zip(source_file_paths, output_file_paths):
        adata = ad.read_h5ad(source_file_path)
        shared = get_gene_ids(adata).isin(shared_gene_ids).to_numpy()
        adata = adata[:, shared].copy()
        n_detected = count_detected_vocabulary_genes(adata, token_dict)
        keep = n_detected >= args.min_shared_genes
        if not keep.any():
            raise ValueError(f"{source_file_path}: no cell has {args.min_shared_genes} detected shared genes.")
        print(f"{source_file_path}: {int(shared.sum())} of {len(shared)} genes kept; {int(keep.sum())} of "
              f"{len(keep)} cells kept ({int((~keep).sum())} with fewer than {args.min_shared_genes} detected shared "
              f"genes; median detected shared genes {np.median(n_detected):g}).")
        adata = adata[keep].copy()
        if sp.issparse(adata.X):
            adata.X.eliminate_zeros()
        temporary_file_path = temporary_folder_path / output_file_path.name
        adata.write_h5ad(temporary_file_path)
        os.replace(temporary_file_path, output_file_path)
        print(f"Wrote {adata.n_obs} cells x {adata.n_vars} genes to {output_file_path}.")
    shutil.rmtree(temporary_folder_path)
    write_provenance(output_folder_path, get_shared_gene_provenance(args.silver_file_paths, args.min_shared_genes))
    print(f"The silver folder {output_folder_path} of {get_gene_set_dataset_name(args.dataset_name, 'shared_genes')} "
          f"holds {len(output_file_paths)} files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--silver_file_paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Silver AnnData files of the dataset, one per section (the filtered training files); '
                             'not modified.')
    parser.add_argument('--data_directory_path',
                        type=str,
                        required=True,
                        help='Data directory with the silver subdirectory.')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help='Name of the dataset; the files are written to silver/<dataset_name>_shared_genes.')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        required=True,
                        help='Token dictionary of the tokenization (detected genes are counted among its genes).')
    parser.add_argument('--min_shared_genes',
                        type=int,
                        default=10,
                        help='Minimum number of detected shared genes of the token dictionary per cell (default 10, '
                             'min_genes_per_cell of the published mouse evaluation).')
    args = parser.parse_args()

    main(args)
