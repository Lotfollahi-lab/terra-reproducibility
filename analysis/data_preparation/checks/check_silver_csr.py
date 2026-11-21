"""
G-3 farm check: is adata.X of every silver .h5ad in canonical sparse format?

Canonical = CSR (or CSC) with strictly increasing indices within every row
(column): sorted, and no gene stored twice. Reads X straight from the HDF5
file in row blocks, so it needs little memory and never loads obs/obsm.

Usage (farm, terra venv, any node, from the terra-reproducibility root):
    python analysis/data_preparation/checks/check_silver_csr.py \
        ${TERRA_DATA_DIR}/silver/<dataset>/... [more files or dirs]
    python analysis/data_preparation/checks/check_silver_csr.py \
        --names analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt \
        --silver ${TERRA_DATA_DIR}/silver
Directories are searched recursively for *.h5ad, as the tokenizer does.
Exit code 1 if any file is non-canonical.

Columns: verdict, format, dtype, rows (columns for CSC), nnz,
'<rows with unsorted indices>u/<rows with a gene stored twice>d',
explicitly stored zeros (removed by eliminate_stored_zeros: true), path.
Verdicts:
    OK          canonical sparse (or dense): tokenizes the same on dev and r4/terra
    BAD-VALUES  non-canonical and not float32: on dev 2a9a599 shifted_log and
                pflog1ppf attach values to the wrong genes (round-4 G-3)
    BAD-DUPS    float32 with duplicate entries: on dev a gene is tokenized twice
    UNSORTED    float32, unsorted but no duplicates: values correct on dev with
                shifted_log, but pflog1ppf is wrong and the tie order without
                stable_rank_ties differs; canonicalize anyway
    CSC-OK      CSC without duplicates: both layers become sorted CSR in the
                tokenizer, so tokens are as for canonical input
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

MAX_NNZ = 20_000_000  # stored entries per read


def _sparse_meta(g):
    enc = g.attrs.get('encoding-type', g.attrs.get('h5sparse_format'))
    if isinstance(enc, bytes):
        enc = enc.decode()
    return {'csr_matrix': 'csr', 'csc_matrix': 'csc',
            'csr': 'csr', 'csc': 'csc'}.get(enc)


def check_file(path):
    with h5py.File(path, 'r') as f:
        X = f['X']
        if isinstance(X, h5py.Dataset):
            return dict(verdict='OK', fmt='dense', dtype=str(X.dtype),
                        rows=X.shape[0], nnz=-1, bad_rows='-', zeros=-1)
        fmt = _sparse_meta(X)
        if fmt is None:
            return dict(verdict='UNKNOWN', fmt='?', dtype='?', rows=-1,
                        nnz=-1, bad_rows=-1, zeros=-1)
        indptr = X['indptr'][:].astype(np.int64)
        indices, data = X['indices'], X['data']
        n_major = len(indptr) - 1
        n_unsorted = n_dup = zeros = 0
        a = 0
        while a < n_major:
            # Blocks of about MAX_NNZ stored entries (at least one row)
            b = int(np.searchsorted(indptr, indptr[a] + MAX_NNZ,
                                    side='right')) - 1
            b = min(max(b, a + 1), n_major)
            p = indptr[a:b + 1]
            idx = indices[p[0]:p[-1]].astype(np.int64)
            zeros += int(np.count_nonzero(data[p[0]:p[-1]] == 0))
            a = b
            if idx.size < 2:
                continue
            d = np.diff(idx)
            # Differences across a row boundary do not count
            starts = p[1:-1] - p[0]
            starts = starts[(starts > 0) & (starts < idx.size)]
            within = np.ones(d.size, bool)
            within[starts - 1] = False
            ends = p[1:] - p[0]
            for kind, mask in (('unsorted', within & (d < 0)),
                               ('dup', within & (d == 0))):
                rows = np.unique(np.searchsorted(ends, np.flatnonzero(mask),
                                                 side='right'))
                if kind == 'unsorted':
                    n_unsorted += rows.size
                else:
                    n_dup += rows.size
        dtype = data.dtype
        if fmt == 'csc':
            # The tokenizer converts both layers to CSR, which sorts them;
            # only duplicate entries matter
            verdict = 'CSC-OK' if n_dup == 0 else 'BAD-DUPS'
        elif n_unsorted == 0 and n_dup == 0:
            verdict = 'OK'
        elif dtype != np.float32:
            verdict = 'BAD-VALUES'
        elif n_dup:
            verdict = 'BAD-DUPS'
        else:
            verdict = 'UNSORTED'
        bad_rows = f'{n_unsorted}u/{n_dup}d'
        return dict(verdict=verdict, fmt=fmt, dtype=str(dtype), rows=n_major,
                    nnz=int(indptr[-1]), bad_rows=bad_rows, zeros=zeros)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--names', help='dataset names file (one per line)')
    ap.add_argument('--silver', help='silver root for --names')
    args = ap.parse_args()
    roots = [Path(p) for p in args.paths]
    if args.names:
        names = [l.split(',')[0].strip() for l in open(args.names)
                 if l.strip()]
        roots += [Path(args.silver) / n for n in names]
    files, n_missing = [], 0
    for r in roots:
        if r.is_dir():
            found = sorted(r.glob('**/*.h5ad'))
            if not found:
                print(f'MISSING\t-\t-\t-\t-\t-\t-\t{r} (no .h5ad)')
                n_missing += 1
            files += found
        elif r.exists():
            files.append(r)
        else:
            print(f'MISSING\t-\t-\t-\t-\t-\t-\t{r}')
            n_missing += 1
    n_bad = n_missing
    for path in files:
        r = check_file(path)
        n_bad += r['verdict'] not in ('OK', 'CSC-OK')
        print('\t'.join(str(r[k]) for k in (
            'verdict', 'fmt', 'dtype', 'rows', 'nnz', 'bad_rows', 'zeros'))
            + f'\t{path}', flush=True)
    print(f'# {len(files)} files, {n_missing} missing, {n_bad} not OK '
          '(missing included)', file=sys.stderr)
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
