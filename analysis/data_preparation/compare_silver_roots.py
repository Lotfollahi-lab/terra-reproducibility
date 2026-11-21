#!/usr/bin/env python
"""
Compare two copies of the silver data, e.g. the NFS and the Lustre DATASETS
roots. Read-only.

For every dataset folder it reports whether it exists in one copy only, and
for every file whether it exists in one copy only. For .h5ad files in both
copies it compares the content rather than the bytes (files rewritten by
another anndata version differ in bytes but can hold the same data):
- cells (obs 'cell_id', else obs_names) and genes (var 'ensembl_id', else
  var_names): present in one copy only, and order;
- the count matrix X, as a matrix: for every cell in both copies, the set of
  (gene, value) pairs over the genes in both copies, after summing duplicate
  entries and dropping stored zeros. Storage details (CSR/CSC/dense, dtypes,
  unsorted indices, stored zeros) do not count as differences; they are
  reported in the x_* columns;
- obs and var columns, value by value, aligned on the cell and gene keys
  (categorical and plain string columns with the same values are equal);
- everything else in the file (layers, obsm, obsp, uns, ...): paths in one
  copy only, shapes, dtypes and values.

Levels:
- listing: folders, file names, sizes and modification times only (seconds);
- meta: plus cells, genes, obs/var columns and the structure of the other
  groups, without reading X (minutes);
- full: plus X and the values of the other groups (reads every common file
  of both copies once).

Outputs <out_prefix>_files.tsv (one row per file), <out_prefix>_datasets.tsv
(one row per dataset folder) and <out_prefix>_summary.txt.

Example:
    python compare_silver_roots.py \\
        --root_a /nfs/team361/sb75/DATASETS/silver \\
        --root_b /lustre/scratch126/cellgen/lotfollahi/DATASETS/silver \\
        --dataset_names_file_path hst_corpus_110m/hst_corpus_110m_dataset_names.txt \\
        --level meta --out_prefix silver_compare_meta --n_workers 4
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem

# Elements read per chunk (per array)
CHUNK = 1 << 24
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)

FILE_COLUMNS = [
    'dataset', 'file', 'in_corpus', 'in_a', 'in_b', 'size_a', 'size_b',
    'mtime_a', 'mtime_b', 'status',
    'n_cells_a', 'n_cells_b', 'cells_only_a', 'cells_only_b',
    'cell_order_same', 'cell_key', 'n_genes_a', 'n_genes_b', 'genes_only_a',
    'genes_only_b', 'gene_order_same', 'gene_key',
    'x_format_a', 'x_format_b', 'x_dtype_a', 'x_dtype_b', 'x_stored_a',
    'x_stored_b', 'x_nonzero_a', 'x_nonzero_b', 'x_sum_a', 'x_sum_b',
    'x_cells_differ', 'x_nonzero_only_genes_a', 'x_nonzero_only_genes_b',
    'obs_cols_only_a', 'obs_cols_only_b', 'obs_cols_differ',
    'var_cols_only_a', 'var_cols_only_b', 'var_cols_differ',
    'other_only_a', 'other_only_b', 'other_differ', 'info', 'error']


def _stat(path):
    try:
        st = os.stat(path)
        return st.st_size, time.strftime(
            '%Y-%m-%d %H:%M', time.localtime(st.st_mtime))
    except OSError as e:
        return -1, f'stat failed: {e}'


def list_root(root):
    """{dataset folder: {relative file path: (size, mtime)}}."""
    out = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            out.setdefault('<files at root>', {})[name] = _stat(path)
            continue
        files = {}
        for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
            dirnames.sort()
            for f in sorted(filenames):
                fp = os.path.join(dirpath, f)
                files[os.path.relpath(fp, path)] = _stat(fp)
        out[name] = files
    return out


# ---------------------------------------------------------------- values --

def _as_str(a):
    a = np.asarray(a, dtype=object).ravel()
    return np.array([x.decode('utf-8', 'replace') if isinstance(x, bytes)
                     else str(x) for x in a], dtype=object)


def _values_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in 'SUO' or b.dtype.kind in 'SUO':
        return bool(np.array_equal(_as_str(a), _as_str(b)))
    if a.dtype.names or b.dtype.names:
        return a.tobytes() == b.tobytes() and a.dtype == b.dtype
    if a.dtype.kind in 'fc' or b.dtype.kind in 'fc':
        return bool(np.array_equal(a.astype(np.complex128 if 'c' in (
            a.dtype.kind + b.dtype.kind) else np.float64),
            b.astype(np.complex128 if 'c' in (a.dtype.kind + b.dtype.kind)
                     else np.float64), equal_nan=True))
    return bool(np.array_equal(a, b))


def datasets_equal(da, db):
    """Value equality of two h5py datasets, read in chunks."""
    if da.shape != db.shape:
        return False
    if da.shape == ():
        return _values_equal(da[()], db[()])
    n = da.shape[0]
    row = int(np.prod(da.shape[1:])) if len(da.shape) > 1 else 1
    step = max(1, CHUNK // max(row, 1))
    for i in range(0, n, step):
        if not _values_equal(da[i:i + step], db[i:i + step]):
            return False
    return True


def _norm_attr(v):
    if isinstance(v, bytes):
        return v.decode('utf-8', 'replace')
    if isinstance(v, np.ndarray):
        return tuple(_norm_attr(x) for x in v.tolist())
    if isinstance(v, (list, tuple)):
        return tuple(_norm_attr(x) for x in v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def walk(f, skip_top):
    """{path: (kind, attrs, shape, dtype)} for everything outside skip_top."""
    items = {}

    def visit(name, obj):
        if name.split('/')[0] in skip_top:
            return
        attrs = {k: _norm_attr(v) for k, v in obj.attrs.items()}
        if isinstance(obj, h5py.Dataset):
            items[name] = ('dataset', attrs, obj.shape, str(obj.dtype))
        else:
            items[name] = ('group', attrs, None, None)

    f.visititems(visit)
    return items


def _short(paths, n=8):
    paths = sorted(paths)
    s = ';'.join(paths[:n])
    return s + (f';...(+{len(paths) - n})' if len(paths) > n else '')


def compare_other(fa, fb, level, cell_align, gene_align):
    """
    Groups other than X, obs and var. Arrays under obsm (varm) whose first
    axis runs over cells (genes) are compared row by row on the common cells
    (genes), so reordered or filtered copies compare correctly; layers, obsp
    and varp are compared only when cells and genes are in the same order.
    """
    skip = {'X', 'obs', 'var'}
    wa, wb = walk(fa, skip), walk(fb, skip)
    only_a = set(wa) - set(wb)
    only_b = set(wb) - set(wa)
    differ, info = [], []
    root_attrs_a = {k: _norm_attr(v) for k, v in fa.attrs.items()}
    root_attrs_b = {k: _norm_attr(v) for k, v in fb.attrs.items()}
    if root_attrs_a != root_attrs_b:
        info.append('root attrs differ')
    for p in sorted(set(wa) & set(wb)):
        ka, aa, sa, ta = wa[p]
        kb, ab, sb, tb = wb[p]
        if ka != kb:
            differ.append(f'{p}(group vs dataset)')
            continue
        if aa != ab:
            info.append(f'{p}: attrs differ')
        top = p.split('/')[0]
        axis = {'obsm': cell_align, 'varm': gene_align}.get(top)
        if ka == 'dataset' and axis is not None and not axis[2]:
            pos_a, pos_b, _ = axis
            if len(sa) == 0 or sa[1:] != sb[1:]:
                differ.append(f'{p}(shape {sa} vs {sb})')
            elif level == 'full' and not _values_equal(
                    fa[p][:][pos_a], fb[p][:][pos_b]):
                differ.append(f'{p}(values on common rows)')
            continue
        if ka == 'dataset' and top in ('layers', 'obsp', 'varp') and not (
                cell_align[2] and gene_align[2]):
            info.append(f'{p}: not compared (cells or genes differ in '
                        'order or set)')
            continue
        if ka == 'dataset':
            if sa != sb:
                differ.append(f'{p}(shape {sa} vs {sb})')
            elif level == 'full' and not datasets_equal(fa[p], fb[p]):
                differ.append(f'{p}(values)')
            elif ta != tb:
                info.append(f'{p}: dtype {ta} vs {tb}')
    return only_a, only_b, differ, info


# -------------------------------------------------------------- obs / var --

def read_frame(f, key):
    df = read_elem(f[key])
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f'{key} is not a data frame')
    return df


def keys_of(df, column):
    """Alignment keys: the column if present, else the index."""
    if column in df.columns:
        return _as_str(df[column].astype(object).to_numpy()), column
    return _as_str(df.index.to_numpy()), 'index'


def align(keys_a, keys_b):
    """Positions of the common keys in a and b (in a's order)."""
    ia_index = pd.Index(keys_a)
    ib_index = pd.Index(keys_b)
    info = []
    if not ia_index.is_unique:
        info.append(f'{len(keys_a) - ia_index.nunique()} duplicate keys in a')
    if not ib_index.is_unique:
        info.append(f'{len(keys_b) - ib_index.nunique()} duplicate keys in b')
    if (not ia_index.is_unique or not ib_index.is_unique):
        if len(keys_a) == len(keys_b) and np.array_equal(keys_a, keys_b):
            pos = np.arange(len(keys_a))
            return pos, pos, 0, 0, True, info
        # first occurrences only
        ia_index = ia_index[~ia_index.duplicated()]
        ib_index = ib_index[~ib_index.duplicated()]
        first_a = ~pd.Index(keys_a).duplicated()
        first_b = ~pd.Index(keys_b).duplicated()
        pos_a_all = np.flatnonzero(first_a)
        pos_b_all = np.flatnonzero(first_b)
    else:
        pos_a_all = np.arange(len(keys_a))
        pos_b_all = np.arange(len(keys_b))
    loc = ib_index.get_indexer(ia_index)
    common = loc >= 0
    pos_a = pos_a_all[common]
    pos_b = pos_b_all[loc[common]]
    only_a = int((~common).sum())
    only_b = int(len(ib_index) - common.sum())
    same_order = (only_a == 0 and only_b == 0
                  and len(keys_a) == len(keys_b)
                  and np.array_equal(keys_a, keys_b))
    return pos_a, pos_b, only_a, only_b, same_order, info


def _column_values(s):
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    return s.to_numpy(dtype=object)


def _safe_eq(x, y):
    try:
        r = x == y
        return bool(np.all(r)) if isinstance(r, np.ndarray) else bool(r)
    except Exception:
        return False


def compare_frames(df_a, df_b, pos_a, pos_b):
    cols_a, cols_b = list(df_a.columns), list(df_b.columns)
    only_a = [c for c in cols_a if c not in cols_b]
    only_b = [c for c in cols_b if c not in cols_a]
    differ = []
    for c in cols_a:
        if c not in cols_b:
            continue
        va = _column_values(df_a[c])[pos_a]
        vb = _column_values(df_b[c])[pos_b]
        na, nb = pd.isna(va), pd.isna(vb)
        try:
            eq = va == vb
            if not isinstance(eq, np.ndarray) or eq.dtype != bool or \
                    eq.shape != va.shape:
                raise TypeError
        except Exception:  # e.g. pd.NA, whose comparisons are NA
            eq = np.array([_safe_eq(x, y) for x, y in zip(va, vb)],
                          dtype=bool)
        eq = eq | (na & nb)
        n_diff = int((~eq).sum())
        if n_diff:
            differ.append(f'{c}({n_diff})')
    return only_a, only_b, differ


# ---------------------------------------------------------------------- X --

def _splitmix64(x):
    with np.errstate(over='ignore'):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK64
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
        return x ^ (x >> np.uint64(31))


def _entry_hash(gene_ids, values):
    """Order-independent per-entry hash of (common gene id, value)."""
    bits = np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)
    with np.errstate(over='ignore'):
        return _splitmix64(_splitmix64(gene_ids.astype(np.uint64)) ^ bits)


def x_info(f):
    X = f['X']
    if isinstance(X, h5py.Dataset):
        return 'dense', tuple(X.shape), str(X.dtype)
    enc = _norm_attr(X.attrs.get('encoding-type', ''))
    if not enc:
        fmt = _norm_attr(X.attrs.get('h5sparse_format', ''))
        enc = {'csr': 'csr_matrix', 'csc': 'csc_matrix'}.get(fmt, fmt)
    shape = X.attrs.get('shape', X.attrs.get('h5sparse_shape'))
    return enc, tuple(int(s) for s in shape), str(X['data'].dtype)


def x_fingerprints(f, gene_map, n_cells):
    """
    Per cell: an order-independent hash of its (gene, value) pairs over the
    genes mapped to common ids (gene_map[col] >= 0), duplicates summed and
    zeros dropped; plus counts of stored entries, non-zero entries, the sum,
    and non-zero entries in genes of this copy only.
    """
    fmt, shape, _ = x_info(f)
    X = f['X']
    fp = np.zeros(n_cells, dtype=np.uint64)
    stored = nonzero = only_genes_nonzero = 0
    total = 0.0

    def add_block(block, row_offset=None, rows=None):
        nonlocal nonzero, only_genes_nonzero, total
        block = block.tocsr()
        block.sum_duplicates()
        block.eliminate_zeros()
        nonzero += block.nnz
        total += float(block.data.sum(dtype=np.float64))
        coo = block.tocoo()
        g = gene_map[coo.col]
        keep = g >= 0
        only_genes_nonzero += int((~keep).sum())
        r = coo.row[keep] + row_offset if rows is None else rows[coo.row[keep]]
        h = _entry_hash(g[keep], coo.data[keep])
        np.add.at(fp, r, h)

    if fmt == 'dense':
        step = max(1, CHUNK // max(shape[1], 1))
        for i in range(0, shape[0], step):
            block = np.asarray(X[i:i + step])
            stored += block.size
            add_block(sp.csr_matrix(block), row_offset=i)
        return fp, stored, nonzero, total, only_genes_nonzero
    if fmt not in ('csr_matrix', 'csc_matrix'):
        raise ValueError(f'unsupported X encoding {fmt!r}')
    indptr = X['indptr'][:].astype(np.int64)
    stored = int(indptr[-1])
    n_major = len(indptr) - 1
    i = 0
    while i < n_major:
        # rows (csr) or columns (csc) whose entries fit in one chunk
        j = int(np.searchsorted(indptr, indptr[i] + CHUNK, side='right')) - 1
        j = max(j, i + 1)
        j = min(j, n_major)
        lo, hi = int(indptr[i]), int(indptr[j])
        data = X['data'][lo:hi]
        idx = X['indices'][lo:hi].astype(np.int64)
        local_ptr = indptr[i:j + 1] - lo
        if fmt == 'csr_matrix':
            block = sp.csr_matrix((data, idx, local_ptr),
                                  shape=(j - i, shape[1]))
            add_block(block, row_offset=i)
        else:
            block = sp.csc_matrix((data, idx, local_ptr),
                                  shape=(shape[0], j - i))
            # columns i..j of the matrix: map local columns to genes
            coo = block.tocoo()
            sub = sp.csr_matrix((coo.data, (coo.row, coo.col + i)),
                                shape=shape)
            add_block(sub, row_offset=0)
        i = j
    return fp, stored, nonzero, total, only_genes_nonzero


# ------------------------------------------------------------- one file --

def compare_file(path_a, path_b, level):
    r = {}
    info = []
    with h5py.File(path_a, 'r') as fa, h5py.File(path_b, 'r') as fb:
        obs_a, obs_b = read_frame(fa, 'obs'), read_frame(fb, 'obs')
        var_a, var_b = read_frame(fa, 'var'), read_frame(fb, 'var')
        cells_a, ck_a = keys_of(obs_a, 'cell_id')
        cells_b, ck_b = keys_of(obs_b, 'cell_id')
        genes_a, gk_a = keys_of(var_a, 'ensembl_id')
        genes_b, gk_b = keys_of(var_b, 'ensembl_id')
        r['cell_key'] = ck_a if ck_a == ck_b else f'{ck_a}/{ck_b}'
        r['gene_key'] = gk_a if gk_a == gk_b else f'{gk_a}/{gk_b}'
        cpa, cpb, c_only_a, c_only_b, c_same, c_info = align(cells_a, cells_b)
        gpa, gpb, g_only_a, g_only_b, g_same, g_info = align(genes_a, genes_b)
        info += [f'cells: {x}' for x in c_info] + [f'genes: {x}' for x in g_info]
        r.update(n_cells_a=len(cells_a), n_cells_b=len(cells_b),
                 cells_only_a=c_only_a, cells_only_b=c_only_b,
                 cell_order_same=c_same, n_genes_a=len(genes_a),
                 n_genes_b=len(genes_b), genes_only_a=g_only_a,
                 genes_only_b=g_only_b, gene_order_same=g_same)
        oa, ob, od = compare_frames(obs_a, obs_b, cpa, cpb)
        va_, vb_, vd = compare_frames(var_a, var_b, gpa, gpb)
        r.update(obs_cols_only_a=_short(oa), obs_cols_only_b=_short(ob),
                 obs_cols_differ=_short(od), var_cols_only_a=_short(va_),
                 var_cols_only_b=_short(vb_), var_cols_differ=_short(vd))
        fmt_a, shape_a, dt_a = x_info(fa)
        fmt_b, shape_b, dt_b = x_info(fb)
        r.update(x_format_a=fmt_a, x_format_b=fmt_b, x_dtype_a=dt_a,
                 x_dtype_b=dt_b)
        if shape_a != (len(cells_a), len(genes_a)) or \
                shape_b != (len(cells_b), len(genes_b)):
            info.append('X shape does not match obs x var')
        x_differ = False
        if level == 'full':
            # common gene ids: position in a's common order
            map_a = np.full(len(genes_a), -1, dtype=np.int64)
            map_b = np.full(len(genes_b), -1, dtype=np.int64)
            map_a[gpa] = np.arange(len(gpa))
            map_b[gpb] = np.arange(len(gpa))
            fpa, st_a, nz_a, sum_a, og_a = x_fingerprints(fa, map_a, len(cells_a))
            fpb, st_b, nz_b, sum_b, og_b = x_fingerprints(fb, map_b, len(cells_b))
            n_cells_differ = int((fpa[cpa] != fpb[cpb]).sum())
            r.update(x_stored_a=st_a, x_stored_b=st_b, x_nonzero_a=nz_a,
                     x_nonzero_b=nz_b, x_sum_a=sum_a, x_sum_b=sum_b,
                     x_cells_differ=n_cells_differ,
                     x_nonzero_only_genes_a=og_a,
                     x_nonzero_only_genes_b=og_b)
            x_differ = n_cells_differ > 0
        o_a, o_b, o_d, o_info = compare_other(
            fa, fb, level, (cpa, cpb, c_same), (gpa, gpb, g_same))
        info += o_info
        r.update(other_only_a=_short(o_a), other_only_b=_short(o_b),
                 other_differ=_short(o_d))
    parts = []
    if c_only_a or c_only_b:
        parts.append('cells')
    if g_only_a or g_only_b:
        parts.append('genes')
    if x_differ:
        parts.append('X')
    if oa or ob or od:
        parts.append('obs')
    if va_ or vb_ or vd:
        parts.append('var')
    if o_a or o_b or o_d:
        parts.append('other')
    if parts:
        r['status'] = 'differs:' + ','.join(parts)
    else:
        same = 'same_content' if level == 'full' else 'same_meta'
        r['status'] = same if (c_same and g_same) else \
            same + '_different_order'
    if fmt_a != fmt_b or dt_a != dt_b:
        info.append(f'X storage {fmt_a}/{dt_a} vs {fmt_b}/{dt_b}')
    r['info'] = _short(info, 6)
    return r


def _worker(args):
    dataset, rel, path_a, path_b, level = args
    t0 = time.time()
    try:
        r = compare_file(path_a, path_b, level)
    except Exception as e:  # report and continue with the other files
        r = {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
    r['seconds'] = round(time.time() - t0, 1)
    return dataset, rel, r


# ------------------------------------------------------------------ main --

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--root_a', required=True, help='First silver folder.')
    p.add_argument('--root_b', required=True, help='Second silver folder.')
    p.add_argument('--level', choices=['listing', 'meta', 'full'],
                   default='meta')
    p.add_argument('--dataset_names_file_path', default=None,
                   help='Corpus dataset names (one per line); marks corpus '
                        'datasets in the outputs and the summary.')
    p.add_argument('--only_corpus', action='store_true',
                   help='Compare file contents of corpus datasets only.')
    p.add_argument('--datasets', default=None,
                   help='Comma-separated dataset folders to compare '
                        '(default: all).')
    p.add_argument('--n_workers', type=int, default=4)
    p.add_argument('--shard', type=int, default=0,
                   help='This shard (0-based) of --n_shards; for job arrays.')
    p.add_argument('--n_shards', type=int, default=1)
    p.add_argument('--out_prefix', default='silver_compare')
    args = p.parse_args()

    corpus = set()
    if args.dataset_names_file_path:
        with open(args.dataset_names_file_path) as fh:
            corpus = {l.strip() for l in fh if l.strip()}
    selected = set(args.datasets.split(',')) if args.datasets else None

    print(f'Listing {args.root_a} ...', flush=True)
    tree_a = list_root(args.root_a)
    print(f'Listing {args.root_b} ...', flush=True)
    tree_b = list_root(args.root_b)

    rows, jobs = [], []
    for ds in sorted(set(tree_a) | set(tree_b)):
        if selected is not None and ds not in selected:
            continue
        fa, fb = tree_a.get(ds, {}), tree_b.get(ds, {})
        for rel in sorted(set(fa) | set(fb)):
            sa, ma = fa.get(rel, (None, None))
            sb, mb = fb.get(rel, (None, None))
            row = dict(dataset=ds, file=rel, in_corpus=ds in corpus,
                       in_a=rel in fa, in_b=rel in fb, size_a=sa, size_b=sb,
                       mtime_a=ma, mtime_b=mb)
            if not (rel in fa and rel in fb):
                row['status'] = 'only_a' if rel in fa else 'only_b'
            elif args.level == 'listing' or not rel.endswith('.h5ad') or \
                    (args.only_corpus and ds not in corpus):
                row['status'] = 'size_equal' if sa == sb else 'size_differs'
            else:
                jobs.append((ds, rel, os.path.join(args.root_a, ds, rel),
                             os.path.join(args.root_b, ds, rel), args.level))
            rows.append(row)

    # biggest files first, then a deterministic shard split
    jobs.sort(key=lambda j: -(os.path.getsize(j[2]) + os.path.getsize(j[3])))
    jobs = jobs[args.shard::args.n_shards]
    if args.n_shards > 1:
        keep = {(j[0], j[1]) for j in jobs}
        rows = [r for r in rows if (r['dataset'], r['file']) in keep]
    by_key = {(r['dataset'], r['file']): r for r in rows}
    total_bytes = sum(os.path.getsize(j[2]) + os.path.getsize(j[3])
                      for j in jobs)
    print(f'{len(jobs)} files to compare ({total_bytes / 1e9:.1f} GB in both '
          f'copies), level {args.level}, {args.n_workers} workers', flush=True)

    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = [ex.submit(_worker, j) for j in jobs]
        for fut in as_completed(futures):
            ds, rel, r = fut.result()
            by_key[(ds, rel)].update(r)
            done += 1
            print(f'[{done}/{len(jobs)} {time.time() - t0:.0f}s] {ds}/{rel}: '
                  f'{r["status"]} ({r["seconds"]}s)', flush=True)

    files = pd.DataFrame(rows).reindex(
        columns=FILE_COLUMNS + ['seconds'])
    files.to_csv(f'{args.out_prefix}_files.tsv', sep='\t', index=False)

    # one row per dataset folder
    ds_rows = []
    for ds in sorted(set(tree_a) | set(tree_b)):
        if selected is not None and ds not in selected:
            continue
        g = files[files['dataset'] == ds]
        if args.n_shards > 1 and not len(g):
            continue
        st = g['status'].fillna('')
        in_a, in_b = ds in tree_a, ds in tree_b
        if not in_a or not in_b:
            status = 'only_a' if in_a else 'only_b'
        elif (st == 'error').any():
            status = 'error'
        elif st.str.startswith('only_').any() or \
                st.str.startswith('differs').any() or (st == 'size_differs').any():
            status = 'differs'
        elif st.str.endswith('different_order').any():
            status = 'same_different_order'
        else:
            status = 'same'
        ds_rows.append(dict(
            dataset=ds, in_corpus=ds in corpus, in_a=in_a, in_b=in_b,
            n_files_a=int(g['in_a'].sum()), n_files_b=int(g['in_b'].sum()),
            files_only_a=int((st == 'only_a').sum()),
            files_only_b=int((st == 'only_b').sum()),
            files_differ=int(st.str.startswith('differs').sum()
                             + (st == 'size_differs').sum()),
            files_error=int((st == 'error').sum()),
            gb_a=round(g['size_a'].fillna(0).clip(lower=0).sum() / 1e9, 3),
            gb_b=round(g['size_b'].fillna(0).clip(lower=0).sum() / 1e9, 3),
            status=status))
    dsdf = pd.DataFrame(ds_rows)
    dsdf.to_csv(f'{args.out_prefix}_datasets.tsv', sep='\t', index=False)

    lines = [f'root_a = {args.root_a}', f'root_b = {args.root_b}',
             f'level = {args.level}', '']
    if args.n_shards == 1 and selected is None:
        for label, side in (('a', 'in_a'), ('b', 'in_b')):
            other = 'in_b' if side == 'in_a' else 'in_a'
            only = dsdf[dsdf[side] & ~dsdf[other]]
            lines.append(f'Dataset folders only in {label}: {len(only)}'
                         + (f' (corpus: {int(only["in_corpus"].sum())})'
                            if corpus else ''))
            lines += [f'  {d}{" [corpus]" if c else ""}'
                      for d, c in zip(only['dataset'], only['in_corpus'])]
        if corpus:
            missing = sorted(corpus - set(tree_a) - set(tree_b))
            lines.append(f'Corpus datasets in neither copy: {len(missing)} '
                         + ' '.join(missing))
        lines.append('')
    both = dsdf[dsdf['in_a'] & dsdf['in_b']]
    lines.append(f'Dataset folders in both: {len(both)}')
    for s, n in both['status'].value_counts().items():
        lines.append(f'  {s}: {n}')
    if corpus:
        cb = both[both['in_corpus']]
        lines.append(f'Corpus dataset folders in both: {len(cb)}')
        for s, n in cb['status'].value_counts().items():
            lines.append(f'  {s}: {n}')
    lines.append('')
    lines.append('Files by status:')
    for s, n in files['status'].fillna('').value_counts().items():
        lines.append(f'  {s}: {n}')
    bad = files[files['status'].fillna('').str.startswith(('differs', 'error'))]
    if len(bad):
        lines.append('')
        lines.append('Files that differ or failed (first 50):')
        for _, r in bad.head(50).iterrows():
            lines.append(f'  {r["dataset"]}/{r["file"]}: {r["status"]} '
                         f'{r["error"] if isinstance(r["error"], str) else ""}')
    text = '\n'.join(lines)
    with open(f'{args.out_prefix}_summary.txt', 'w') as fh:
        fh.write(text + '\n')
    print('\n' + text)


if __name__ == '__main__':
    sys.exit(main())
