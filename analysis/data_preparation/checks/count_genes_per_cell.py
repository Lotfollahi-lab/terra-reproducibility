"""
How many cells would 512 genes per cell (instead of 256) actually help?

Per silver .h5ad, per dataset and over the whole corpus: the number of
detected VOCABULARY genes per cell, i.e. the gene tokens the cell-graph
tokenizer can store for that cell: stored entries of X whose column's
var['ensembl_id'] is an 'ENS' key of the token dictionary and whose value is
non-zero (the corpus is tokenized with eliminate_stored_zeros=True). Cells
with obs['filter_pass'] != 1 are left out, as the tokenizer does
(terra/preprocessors/filters.py), unless --no_filter_pass.

Reports, from exact per-cell histograms:
  cells, empty (0 genes), cells below 10 genes (the corpus floor),
  mean / median / p90 / p99 / max genes,
  share of cells with > 256 and > 512 genes (cells that a 256 / 512 segment
  truncates), share of the cells' gene tokens beyond the top 256 / 512
  (tokens that a 256 / 512 segment drops), and the mean filled length of a
  256 / 512 segment (mean min(g, 256), mean min(g, 512)): the non-padding
  tokens per segment, which set the context-encoder and predictor cost
  (the per-file column mean_filled_512 feeds tokenization_memory.py).

Memory-light: h5py only; X is read in blocks (CSR: indptr in full, then
indices/data in blocks of NNZ_BLOCK entries; CSC: one vocabulary column at
a time; dense: blocks of rows). Peak RAM is about 8 bytes x cells (indptr,
counts) + ~0.1 GB (one CSR block of NNZ_BLOCK entries), whatever the file
size. With --skip_values the CSR data array is not read (stored zeros then
count as genes); if every column is a vocabulary gene this needs only
indptr (np.diff).

Usage (farm, any env with h5py + numpy; read-only; on Lustre set
HDF5_USE_FILE_LOCKING=FALSE):
  whole corpus in one job:
    python count_genes_per_cell.py \
        --token_dict ${TERRA_DATA_DIR}/tokenizer/token_dictionary_homo_sapiens.pkl \
        --names analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt \
        --silver ${TERRA_DATA_DIR}/silver --out genes_per_cell
  one dataset per LSF array element (count_genes_per_cell.lsf), then merge:
    python count_genes_per_cell.py ... --names <names> --silver <silver> \
        --shard $((LSB_JOBINDEX - 1)) --n_shards 158 --out out/gpc_${LSB_JOBINDEX}
    python count_genes_per_cell.py --merge out/gpc_*_hist.npz --out genes_per_cell
  explicit files/folders:  python count_genes_per_cell.py --token_dict ... a.h5ad b.h5ad
Writes <out>_per_file.tsv, <out>_per_dataset.tsv (last row: ALL) and
<out>_hist.npz (per-dataset histograms, to re-aggregate; --merge sums the
histograms of datasets that appear in several files). Exit code 1 if a file
could not be read or a dataset folder has no .h5ad (listed in the per-file
table with status ERROR / MISSING); 2 for an empty shard.
"""
import argparse
import pickle
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np

MAX_BIN = 40_000          # genes per cell above this share one histogram bin
NNZ_BLOCK = 4_000_000     # stored entries per CSR block: ~100 MB peak, fastest (bench)
ROW_BLOCK = 20_000        # rows per block (dense)
COL_BLOCK = 2_000         # columns per block (CSC)


def _read_str_array(node):
    """A string column of an h5ad obs/var group (plain or categorical)."""
    if isinstance(node, h5py.Group):           # categorical (anndata >= 0.8)
        cats = _read_str_array(node['categories'])
        codes = node['codes'][...]
        out = np.empty(codes.shape, dtype=object)
        out[codes >= 0] = cats[codes[codes >= 0]]
        out[codes < 0] = None
        return out
    if node.dtype.kind in ('S', 'O'):
        return np.asarray(node.asstr()[...], dtype=object)
    return node[...]


def _read_column(f, frame, name):
    """Column `name` of obs/var, or None if absent. Handles group-based
    dataframes (anndata >= 0.7), legacy compound datasets and legacy
    categorical columns stored with a '__categories' group."""
    df = f[frame]
    if isinstance(df, h5py.Dataset):            # legacy compound dataset
        if name not in (df.dtype.names or ()):
            return None
        col = df[name]
        cats = f.get(f'uns/{name}_categories')
        if cats is not None:
            cats = np.asarray(cats.asstr()[...] if cats.dtype.kind in 'SO'
                              else cats[...], dtype=object)
            return cats[col]
        if col.dtype.kind == 'S':
            return col.astype(str).astype(object)
        return col
    if name not in df:
        return None
    node = df[name]
    legacy_cats = df.get('__categories')
    if (isinstance(node, h5py.Dataset) and legacy_cats is not None
            and name in legacy_cats):
        cats = _read_str_array(legacy_cats[name])
        codes = node[...]
        out = np.empty(codes.shape, dtype=object)
        out[codes >= 0] = cats[codes[codes >= 0]]
        out[codes < 0] = None
        return out
    if isinstance(node, h5py.Group) and 'categories' not in node:
        # nullable integer / boolean arrays: {'values', 'mask'}
        values = node['values'][...]
        if 'mask' in node:
            values = np.where(node['mask'][...], np.nan, values)
        return values
    return _read_str_array(node)


def _n_obs(f):
    X = f['X']
    if isinstance(X, h5py.Dataset):
        return X.shape[0]
    return int(X.attrs['shape'][0]) if 'shape' in X.attrs else int(
        X.attrs['h5sparse_shape'][0])


def _sparse_format(X):
    enc = X.attrs.get('encoding-type', X.attrs.get('h5sparse_format'))
    if isinstance(enc, bytes):
        enc = enc.decode()
    if enc in ('csr_matrix', 'csr'):
        return 'csr'
    if enc in ('csc_matrix', 'csc'):
        return 'csc'
    raise ValueError(f'unknown sparse encoding {enc!r}')


def genes_per_cell(path, vocab, use_filter_pass=True, skip_values=False):
    """Detected vocabulary genes of every (kept) cell of one .h5ad."""
    with h5py.File(path, 'r') as f:
        ens = _read_column(f, 'var', 'ensembl_id')
        if ens is None:
            raise ValueError("var has no 'ensembl_id' column")
        in_vocab = np.fromiter((isinstance(g, str) and g in vocab for g in ens),
                               dtype=bool, count=len(ens))
        n_obs = _n_obs(f)
        X = f['X']
        counts = np.zeros(n_obs, dtype=np.int64)
        if isinstance(X, h5py.Dataset):                      # dense
            cols = np.flatnonzero(in_vocab)
            for s in range(0, n_obs, ROW_BLOCK):
                block = X[s:s + ROW_BLOCK]
                counts[s:s + ROW_BLOCK] = np.count_nonzero(
                    block[:, cols], axis=1)
            fmt = 'dense'
        elif _sparse_format(X) == 'csr':
            fmt = 'csr'
            indptr = X['indptr'][...].astype(np.int64)
            if in_vocab.all() and skip_values:
                counts = np.diff(indptr)
            else:
                indices, data = X['indices'], X['data']
                row = 0
                while row < n_obs:
                    start = indptr[row]
                    # rows whose entries end within NNZ_BLOCK of `start`
                    end_row = int(np.searchsorted(
                        indptr, start + NNZ_BLOCK, side='right')) - 1
                    end_row = min(max(end_row, row + 1), n_obs)
                    end = indptr[end_row]
                    keep = in_vocab[indices[start:end]]
                    if not skip_values:
                        keep &= data[start:end] != 0
                    cs = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))
                    local = indptr[row:end_row + 1] - start
                    counts[row:end_row] = cs[local[1:]] - cs[local[:-1]]
                    row = end_row
        else:                                                # csc
            fmt = 'csc'
            indptr = X['indptr'][...].astype(np.int64)
            indices, data = X['indices'], X['data']
            vcols = np.flatnonzero(in_vocab)
            for s in range(0, len(vcols), COL_BLOCK):
                for c in vcols[s:s + COL_BLOCK]:
                    a, b = indptr[c], indptr[c + 1]
                    if b == a:
                        continue
                    rows = indices[a:b]
                    if not skip_values:
                        rows = rows[data[a:b] != 0]
                    counts += np.bincount(rows, minlength=n_obs)
        n_all = n_obs
        if use_filter_pass:
            fp = _read_column(f, 'obs', 'filter_pass')
            if fp is not None:
                # As filters.filter_cells: obs['filter_pass'].values == 1
                # (a string '1' does not pass there either)
                fp = np.asarray(fp)
                if fp.dtype.kind in 'biuf':
                    keep_cells = fp == 1
                else:
                    keep_cells = np.fromiter(
                        (isinstance(v, (bool, int, float, np.number))
                         and v == 1 for v in fp), dtype=bool, count=len(fp))
                counts = counts[keep_cells]
        return counts, dict(format=fmt, n_obs=n_all, n_cols=len(in_vocab),
                            n_vocab_cols=int(in_vocab.sum()))


def summarize(hist):
    """Statistics from a histogram of genes per cell (last bin = overflow)."""
    g = np.arange(len(hist), dtype=np.float64)
    n = hist.sum()
    if n == 0:
        return dict(cells=0)
    cdf = np.cumsum(hist)

    def q(p):
        return int(np.searchsorted(cdf, p * n, side='left'))
    tot = (hist * g).sum()
    out = dict(
        cells=int(n),
        empty=int(hist[0]),
        cells_lt10=int(hist[:10].sum()),
        mean=(hist * g).sum() / n,
        median=q(0.5), p90=q(0.9), p99=q(0.99),
        max=int(np.flatnonzero(hist)[-1]),
        share_cells_gt256=hist[257:].sum() / n,
        share_cells_gt512=hist[513:].sum() / n,
        share_tokens_beyond256=(hist * np.maximum(g - 256, 0)).sum() / tot
        if tot else 0.0,
        share_tokens_beyond512=(hist * np.maximum(g - 512, 0)).sum() / tot
        if tot else 0.0,
        mean_filled_256=(hist * np.minimum(g, 256)).sum() / n,
        mean_filled_512=(hist * np.minimum(g, 512)).sum() / n,
    )
    if hist[-1]:
        out['max'] = f'>={MAX_BIN}'
    return out


COLUMNS = ['cells', 'empty', 'cells_lt10', 'mean', 'median', 'p90', 'p99', 'max',
           'share_cells_gt256', 'share_cells_gt512', 'share_tokens_beyond256',
           'share_tokens_beyond512', 'mean_filled_256', 'mean_filled_512']


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f'{v:.4f}'
    return str(v)


def write_per_dataset(path, hists, n_files):
    """Per-dataset table from {dataset: histogram}; last row ALL."""
    total = np.zeros(MAX_BIN + 1, dtype=np.int64)
    with open(path, 'w') as pd_:
        pd_.write('\t'.join(['dataset', 'files'] + COLUMNS) + '\n')
        for name, hist in hists.items():
            total += hist
            s = summarize(hist)
            pd_.write('\t'.join([name, str(n_files.get(name, ''))]
                                + [fmt(s.get(c, '')) for c in COLUMNS]) + '\n')
        s = summarize(total)
        pd_.write('\t'.join(['ALL', str(sum(v for v in n_files.values()
                                            if isinstance(v, int)))]
                            + [fmt(s.get(c, '')) for c in COLUMNS]) + '\n')
    return total


def merge(paths, out):
    """Sum the per-dataset histograms of several <out>_hist.npz files."""
    hists, n_files = {}, {}
    for p in paths:
        with np.load(p) as z:
            files = (dict(zip(z['_names'], z['_files']))
                     if '_names' in z.files else {})
            for name in z.files:
                if name.startswith('_') or name == 'ALL':
                    continue
                hists[name] = hists.get(name, 0) + z[name]
                n_files[name] = n_files.get(name, 0) + int(files.get(name, 0))
    hists = dict(sorted(hists.items()))
    total = write_per_dataset(f'{out}_per_dataset.tsv', hists, n_files)
    np.savez_compressed(f'{out}_hist.npz', ALL=total,
                        _names=np.array(list(n_files), dtype=str),
                        _files=np.array(list(n_files.values())), **hists)
    s = summarize(total)
    print(f"merged {len(paths)} files, {len(hists)} datasets: "
          f"{s.get('cells', 0)} cells, >256 {s.get('share_cells_gt256', 0):.4f}, "
          f">512 {s.get('share_cells_gt512', 0):.4f}", file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('paths', nargs='*',
                    help='.h5ad files or folders (with --merge: _hist.npz files)')
    ap.add_argument('--token_dict')
    ap.add_argument('--names', help='dataset names file (first column)')
    ap.add_argument('--silver', help='silver root (with --names)')
    ap.add_argument('--out', default='genes_per_cell')
    ap.add_argument('--no_filter_pass', action='store_true')
    ap.add_argument('--skip_values', action='store_true',
                    help='do not read X data (stored zeros count as genes)')
    ap.add_argument('--shard', type=int, default=0,
                    help='with --n_shards: process datasets shard, '
                    'shard + n_shards, ... of the list (0-based)')
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--merge', action='store_true',
                    help='merge the _hist.npz files given as paths')
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge(args.paths, args.out)
    if not args.token_dict:
        ap.error('--token_dict is required (except with --merge)')
    if args.names and not args.silver:
        ap.error('--names needs --silver')

    with open(args.token_dict, 'rb') as f:
        vocab = {k for k in pickle.load(f) if 'ENS' in k}

    groups = []                       # (dataset name, [files])
    for p in map(Path, args.paths):
        files = sorted(p.glob('**/*.h5ad')) if p.is_dir() else [p]
        groups.append((p.stem if p.is_file() else p.name, files))
    if args.names:
        for line in open(args.names):
            name = line.split(',')[0].strip()
            if name:
                groups.append((name, sorted(
                    (Path(args.silver) / name).glob('**/*.h5ad'))))
    groups = groups[args.shard::args.n_shards]
    if not groups:
        print(f'shard {args.shard} of {args.n_shards}: no datasets',
              file=sys.stderr)
        return 2

    hists, n_files, n_errors = {}, {}, 0
    with open(f'{args.out}_per_file.tsv', 'w') as pf:
        pf.write('\t'.join(['dataset', 'status', 'format', 'n_obs', 'n_cols',
                            'n_vocab_cols'] + COLUMNS + ['path']) + '\n')
        for name, files in groups:
            hist = np.zeros(MAX_BIN + 1, dtype=np.int64)
            if not files:
                pf.write(f'{name}\tMISSING' + '\t' * (4 + len(COLUMNS))
                         + '\t\n')
                n_errors += 1
            for path in files:
                try:
                    counts, info = genes_per_cell(
                        path, vocab, use_filter_pass=not args.no_filter_pass,
                        skip_values=args.skip_values)
                except Exception as e:
                    n_errors += 1
                    pf.write(f'{name}\tERROR {type(e).__name__}: {e}'
                             + '\t' * (4 + len(COLUMNS)) + f'\t{path}\n')
                    traceback.print_exc(file=sys.stderr)
                    continue
                h = np.bincount(np.minimum(counts, MAX_BIN),
                                minlength=MAX_BIN + 1)
                hist += h
                s = summarize(h)
                pf.write('\t'.join([name, 'OK', info['format'],
                                    str(info['n_obs']), str(info['n_cols']),
                                    str(info['n_vocab_cols'])]
                                   + [fmt(s.get(c, '')) for c in COLUMNS]
                                   + [str(path)]) + '\n')
                pf.flush()
            hists[name] = hists.get(name, 0) + hist
            n_files[name] = n_files.get(name, 0) + len(files)
            s = summarize(hist)
            print(f"{name}: {s.get('cells', 0)} cells, "
                  f">256 {s.get('share_cells_gt256', 0):.3f}, "
                  f">512 {s.get('share_cells_gt512', 0):.3f}", file=sys.stderr)
    total = write_per_dataset(f'{args.out}_per_dataset.tsv', hists, n_files)
    np.savez_compressed(f'{args.out}_hist.npz', ALL=total,
                        _names=np.array(list(n_files), dtype=str),
                        _files=np.array(list(n_files.values())),
                        **hists)
    return 1 if n_errors else 0


if __name__ == '__main__':
    sys.exit(main())
