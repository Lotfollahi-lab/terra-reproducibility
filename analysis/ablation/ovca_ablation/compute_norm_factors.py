"""
This script writes the per-gene normalization factors of a cohort of silver AnnData files (a list of files or a folder)
with the gene-corrected read depth constants of the tokenizer (153.4768 + 0.0487 * n_genes), using `norm_factors.py`.
Cells are pooled over the cohort as by `stjepa-reproducibility/src/compute_norm_factors.py` (concatenated cells), so
that the six mouse gene-corrected read depth mean arms can be rerun with factors that are consistent with their
tokenization, on the same cohort as their original factor file (see `check_mouse_norm_factors.py`). The output has the
columns of `prepare_ovca_data.py`: gene_id, nonzero_mean, mean, gene_corrected_read_depth_nonzero_mean,
gene_corrected_read_depth_mean, n_cells, n_nonzero_cells.

Runs in the TERRA or the ST-JEPA v2 environment.

Usage:
```bash
python compute_norm_factors.py --silver_directory_path <silver>/<dataset_name> --norm_factor_file_path <output>.csv
python compute_norm_factors.py --silver_file_paths <a>.h5ad <b>.h5ad --norm_factor_file_path <output>.csv
```
"""

import argparse
import os
from pathlib import Path

from norm_factors import TOKENIZER_GCRD_CONSTANTS, compute_norm_factors


def main(args):
    if args.silver_directory_path is not None:
        # All h5ad files below the folder, as the tokenizer reads them
        silver_file_paths = sorted(Path(args.silver_directory_path).glob("**/*.h5ad"))
        if not silver_file_paths:
            raise FileNotFoundError(f"No h5ad files in {args.silver_directory_path}.")
    else:
        silver_file_paths = [Path(file_path) for file_path in args.silver_file_paths]
    print(f"Cohort: {len(silver_file_paths)} silver files; gene-corrected read depth constants "
          f"a = {args.basis_target_read_depth}, b = {args.target_read_depth_per_gene}.")

    norm_factor_df = compute_norm_factors(silver_file_paths,
                                          basis_target_read_depth=args.basis_target_read_depth,
                                          target_read_depth_per_gene=args.target_read_depth_per_gene,
                                          verbose=True)

    os.makedirs(Path(args.norm_factor_file_path).parent, exist_ok=True)
    norm_factor_df.to_csv(args.norm_factor_file_path, index=False)
    print(f"Saved normalization factors of {len(norm_factor_df)} genes to {args.norm_factor_file_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    cohort = parser.add_mutually_exclusive_group(required=True)
    cohort.add_argument('--silver_file_paths',
                        type=str,
                        nargs='+',
                        help='Silver AnnData files of the cohort (raw counts, var["ensembl_id"]).')
    cohort.add_argument('--silver_directory_path',
                        type=str,
                        help='Folder with the silver AnnData files of the cohort (all *.h5ad below it).')
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        required=True,
                        help='Output per-gene normalization factors (.csv).')
    parser.add_argument('--basis_target_read_depth',
                        type=float,
                        default=TOKENIZER_GCRD_CONSTANTS[0],
                        help='Read depth independent of the number of probed genes (default: the tokenizer\'s).')
    parser.add_argument('--target_read_depth_per_gene',
                        type=float,
                        default=TOKENIZER_GCRD_CONSTANTS[1],
                        help='Additional read depth per probed gene (default: the tokenizer\'s).')
    args = parser.parse_args()

    main(args)
