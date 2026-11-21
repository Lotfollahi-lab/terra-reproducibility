"""
Farm check before re-tokenizing: per silver .h5ad, how many cells have fewer
than --min_genes (default 10) expressed VOCABULARY genes (the genes the
tokenizer keeps: adata.var['ensembl_id'] is an 'ENS' key of the token
dictionary, count > 0)? Also prints the cells with 0 vocabulary genes.

Read-only. Opens each file backed (X is read in row blocks; obs/var are loaded).
Cells with filter_pass != 1 are skipped, as the tokenizer does
(terra/preprocessors/filters.py).

Usage (farm, terra venv, from the terra-reproducibility root):
    python analysis/data_preparation/checks/count_low_gene_cells.py \
        --token_dict ${TERRA_DATA_DIR}/tokenizer/token_dictionary_homo_sapiens.pkl \
        --names analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt \
        --silver ${TERRA_DATA_DIR}/silver [--min_genes 10]
Columns: cells (filter_pass), cells below min_genes, cells with 0 vocabulary
genes, has filter_pass, path. Exit code 1 if any cell is below min_genes.
"""
import argparse
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

ROWS = 100_000


def count_file(path, vocab, min_genes):
    a = ad.read_h5ad(path, backed='r')
    keep = (a.obs['filter_pass'].to_numpy() == 1
            if 'filter_pass' in a.obs.columns else np.ones(a.n_obs, bool))
    cols = np.flatnonzero([g in vocab for g in a.var['ensembl_id']])
    n_below = n_zero = 0
    for start in range(0, a.n_obs, ROWS):
        X = a.X[start:start + ROWS]
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
        Xv = X[:, cols]
        n_genes = np.asarray((Xv > 0).sum(axis=1)).ravel()
        n_genes = n_genes[keep[start:start + ROWS]]
        n_below += int((n_genes < min_genes).sum())
        n_zero += int((n_genes == 0).sum())
    a.file.close()
    return int(keep.sum()), n_below, n_zero, 'filter_pass' in a.obs.columns


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--token_dict', required=True)
    ap.add_argument('--names')
    ap.add_argument('--silver')
    ap.add_argument('--min_genes', type=int, default=10)
    args = ap.parse_args()
    with open(args.token_dict, 'rb') as f:
        vocab = {k for k in pickle.load(f) if 'ENS' in k}
    roots = [Path(p) for p in args.paths]
    if args.names:
        roots += [Path(args.silver) / l.split(',')[0].strip()
                  for l in open(args.names) if l.strip()]
    n_tot = below_tot = zero_tot = 0
    for r in roots:
        for path in (sorted(r.glob('**/*.h5ad')) if r.is_dir() else [r]):
            n, below, zero, has_fp = count_file(path, vocab, args.min_genes)
            n_tot += n
            below_tot += below
            zero_tot += zero
            print(f'{n}\t{below}\t{zero}\t{has_fp}\t{path}', flush=True)
    print(f'# total {n_tot} cells; below {args.min_genes} vocabulary genes: '
          f'{below_tot}; with 0: {zero_tot}')
    sys.exit(1 if below_tot else 0)


if __name__ == '__main__':
    main()
