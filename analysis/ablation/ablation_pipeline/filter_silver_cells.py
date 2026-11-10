"""
This script removes the cells without a detected gene of the token dictionary from silver AnnData files and writes the
remaining cells, with everything else unchanged (obs, var, uns, obsm, layers), to a new silver folder under the same
file names. The source files are not modified. check_setup.py (check 2) fails while the silver files hold such cells.

Why: with `zero_expr_from_raw_counts` such a cell has an empty ranked gene list in every tokenization. Its own segment
holds only padding (its embedding is empty), and on the sparse ranking path (most arms, including base) the pinned
tokenizer does not count it as a neighbor segment: each cell with k such neighbors gets longer segments, which no
longer start where the datasets read them, and loses its k farthest neighbors (TERRA's H2; fixed there in 7f6a30c by
removing these cells before tokenization). The filtered files are what the normalization factors, the tokenization and
the evaluation then read, so all three see the same cells.

It also removes zeros stored explicitly in a sparse X (same values): on the sparse ranking path the tokenizer would
turn them into gene tokens with a count of 0 (TERRA only warns), although such genes are not detected.

Usage (`source datasets/<dataset>.env` first; README.md, e.g. Mouse rerun, step 3):
```bash
python filter_silver_cells.py --source_silver_file_paths ${SOURCE_SILVER_FILE_PATHS} \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" \
    --output_folder_path "${DATA_DIRECTORY_PATH}/silver/${DATASET_NAME}"
```
"""

import argparse
import os
import pickle
import shutil
from pathlib import Path

import anndata as ad
import scipy.sparse as sp

from ablation_utils import cells_without_vocabulary_genes


def main(args):
    with open(args.token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)
    output_folder_path = Path(args.output_folder_path).resolve()
    source_file_paths = [Path(file_path).resolve() for file_path in args.source_silver_file_paths]

    # The tokenizer reads every h5ad file of the silver folder: it must hold exactly the filtered files
    file_names = [file_path.name for file_path in source_file_paths]
    if len(set(file_names)) != len(file_names):
        raise ValueError(f"The source files share file names: {file_names}.")
    missing = [str(file_path) for file_path in source_file_paths if not file_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source silver files not found: {missing}.")
    if any(file_path.parent == output_folder_path for file_path in source_file_paths):
        raise ValueError(f"{output_folder_path} holds source files; write the filtered files to a new folder.")
    other_files = sorted(str(file_path) for file_path in output_folder_path.rglob("*.h5ad")
                         if file_path.name not in file_names) if output_folder_path.exists() else []
    if other_files:
        raise ValueError(f"{output_folder_path} holds other h5ad files, which the tokenizer would also read: "
                         f"{other_files}.")

    # Files are written next to the output folder first, then moved in, so that it never holds partial files
    temporary_folder_path = output_folder_path.parent / f".{output_folder_path.name}.tmp"
    temporary_folder_path.mkdir(parents=True, exist_ok=True)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    n_removed = 0
    for source_file_path in source_file_paths:
        adata = ad.read_h5ad(source_file_path)
        empty = cells_without_vocabulary_genes(adata, token_dict)
        n_removed += int(empty.sum())
        print(f"{source_file_path}: {adata.n_obs} cells, {int(empty.sum())} without a detected gene of the token "
              f"dictionary{', e.g. ' + str(list(adata.obs_names[empty][:5])) if empty.any() else ''}.")
        if empty.all():
            raise ValueError(f"{source_file_path}: no cell has a detected gene of the token dictionary.")
        adata = adata[~empty].copy()
        if sp.issparse(adata.X):
            n_stored_zeros = int((adata.X.data == 0).sum())
            if n_stored_zeros:
                adata.X.eliminate_zeros()
                print(f"Removed {n_stored_zeros} explicitly stored zeros from X (same values).")
        temporary_file_path = temporary_folder_path / source_file_path.name
        adata.write_h5ad(temporary_file_path)
        os.replace(temporary_file_path, output_folder_path / source_file_path.name)
        print(f"Wrote {adata.n_obs} cells to {output_folder_path / source_file_path.name}.")
    shutil.rmtree(temporary_folder_path)
    print(f"Removed {n_removed} cells in total; the silver folder {output_folder_path} holds {len(file_names)} files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--source_silver_file_paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Silver AnnData files to filter (raw counts in X, var["ensembl_id"]); not modified.')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        required=True,
                        help='Token dictionary of the tokenization (its genes are the vocabulary).')
    parser.add_argument('--output_folder_path',
                        type=str,
                        required=True,
                        help='New silver folder of the dataset (silver/<DATASET_NAME>); written with the filtered files '
                             'under the source file names.')
    args = parser.parse_args()

    main(args)
