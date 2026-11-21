#!/usr/bin/env python
"""
Validate the outputs of terra-reproducibility
analysis/data_preparation/compute_cohort_norm_factors.py against the silver
data they were computed from.

Inputs: {prefix}_norm_factors.csv and .json (and, optionally,
{prefix}_batchwise_norm_factors.pkl and {prefix}_pf_targets.csv).

Checks (each prints PASS / FAIL / WARN; exit code 0 = PASS, 1 = a check failed,
2 = the validator itself crashed):

  cohort    the dataset names file has no duplicates and every dataset has files
  heldout   json excluded_batch_ids == VAL_BATCH_IDS + TEST_BATCH_IDS of
            heldout_batch_ids.py; VAL and TEST are disjoint; every held-out id
            matches cells of the cohort
  json      n_files, n_cells (training cells) and n_cells_excluded equal the
            recounts from obs; n_genes, n_genes_* and tie_break_mean_fill equal
            the values implied by the csv and the data
  rows      every gene of every cohort file and every vocabulary gene ('ENS'
            keys of the token dictionary) has exactly one row; no other rows;
            gene ids are strings without '_' (the script splits keys on '_')
  tiebreak  tie_break_mean is finite for every row; == probing_cell_mean where
            n_nonzero_cells > 0; == the json fill (= max of those + 1) elsewhere
  probing   n_probing_cells == number of training cells of the files whose var
            holds the gene (from obs/var alone, all genes); genes only in
            held-out files have n_probing_cells == 0
  recompute (X pass) every statistic column recomputed independently from the
            raw counts of the training cells (pooled sums, definitions below)
            matches the csv; with --max_files N < all files, the per-file
            statistics of N files are recomputed and compared with the
            batchwise pickle, and the csv is re-aggregated from the pickle
  pf        (if a pf_targets csv is given/found) s1 and s2 recomputed

Definitions (per gene g, pooled over the training cells of every processed
file whose var holds g; a 'processed' file has at least one training cell):
  nonzero_mean      = sum of positive counts / number of positive cells
  mean              = sum of counts / training cells, over the files in which
                      g is detected at least once (legacy definition)
  probing_cell_mean = sum of counts / training cells of all files probing g
  gene_corrected_read_depth_*: the same after x / rowsum * (basis + n_var *
                      per_gene), rowsum over all genes of the cell's file
  n_cells           = training cells of the files in which g is detected
  n_nonzero_cells   = positive cells; n_probing_cells = probing training cells
  tie_break_mean    = probing_cell_mean if n_nonzero_cells > 0 else fill,
                      fill = max(probing_cell_mean of those) + 1
  PF: s1 = mean training-cell depth; s2 = mean over training cells of
      sum_j log1p(x_ij * s1 / depth_i)

Memory: obs/var are read with h5py + anndata read_elem; X is read per file as
CSR row blocks of at most --nnz_block stored entries; nothing is densified.
"""

import argparse
import importlib.util
import json
import math
import os
import pickle
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from anndata.io import read_elem
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem

# Defaults of compute_cohort_norm_factors.py (they are not recorded in the
# json, so a run with other values must pass them here too)
DEFAULT_BASIS_TARGET_READ_DEPTH = 151.1754621380761
DEFAULT_TARGET_READ_DEPTH_PER_GENE = 0.04705053555567159

STAT_FLOAT_COLUMNS = [
    'nonzero_mean', 'mean', 'gene_corrected_read_depth_nonzero_mean',
    'gene_corrected_read_depth_mean', 'probing_cell_mean']
STAT_INT_COLUMNS = ['n_cells', 'n_nonzero_cells', 'n_probing_cells']
BATCHWISE_STATS = ['nonzero_means', 'means',
                   'gene_corrected_read_depth_nonzero_means',
                   'gene_corrected_read_depth_means', 'n_cells',
                   'n_nonzero_cells', 'n_probing_cells']


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail=''):
        self.rows.append((status, name, detail))
        print(f'[{status}] {name}' + (f': {detail}' if detail else ''),
              flush=True)

    def check(self, ok, name, detail_ok='', detail_fail=''):
        self.add('PASS' if ok else 'FAIL', name,
                 detail_ok if ok else (detail_fail or detail_ok))
        return ok

    def fail(self, name, detail=''):
        self.add('FAIL', name, detail)

    def warn(self, name, detail=''):
        self.add('WARN', name, detail)

    def info(self, name, detail=''):
        self.add('INFO', name, detail)

    @property
    def n_fail(self):
        return sum(r[0] == 'FAIL' for r in self.rows)

    def summary(self):
        counts = {s: sum(r[0] == s for r in self.rows)
                  for s in ('PASS', 'FAIL', 'WARN', 'INFO')}
        print('\n' + '=' * 78)
        for status, name, detail in self.rows:
            if status in ('FAIL', 'WARN'):
                print(f'[{status}] {name}: {detail}')
        print(f"SUMMARY: {counts['PASS']} passed, {counts['FAIL']} failed, "
              f"{counts['WARN']} warnings")
        print('RESULT: ' + ('FAIL' if counts['FAIL'] else 'PASS'))
        return counts


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # bytes on macOS, kilobytes on Linux
    return r / 1e9 if sys.platform == 'darwin' else r / 1e6


def short(items, n=10):
    items = list(items)
    return (f'{items[:n]}' + (f' ... (+{len(items) - n})'
                              if len(items) > n else ''))


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def load_heldout(module_dir):
    path = Path(module_dir) / 'heldout_batch_ids.py'
    spec = importlib.util.spec_from_file_location(
        '_validate_heldout_batch_ids', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.VAL_BATCH_IDS), list(module.TEST_BATCH_IDS), path


def find_heldout_dir(args):
    if args.heldout_module_dir:
        return args.heldout_module_dir
    candidates = [Path(args.cohort_dataset_names_file_path).resolve().parent.parent,
                  Path(__file__).resolve().parent.parent,
                  Path(__file__).resolve().parent, Path.cwd()]
    for c in candidates:
        if (c / 'heldout_batch_ids.py').exists():
            return str(c)
    raise SystemExit('heldout_batch_ids.py not found; pass --heldout_module_dir '
                     '(terra-reproducibility/analysis/data_preparation).')


def batch_ids_of(cell_ids):
    """Batch id '_'.join(cell_id.split('_')[:2]) (as the script derives it)."""
    ids = np.asarray(cell_ids).astype(str)
    if ids.size == 0:
        return ids, 0
    p1 = np.char.partition(ids, '_')
    p2 = np.char.partition(p1[:, 2], '_')
    batch = np.where(p1[:, 1] == '', p1[:, 0],
                     np.char.add(np.char.add(p1[:, 0], '_'), p2[:, 0]))
    n_malformed = int((np.char.count(ids, '_') < 2).sum())
    return batch, n_malformed


def cohort_files(args, report):
    silver = Path(args.data_directory_path) / 'silver'
    # The script reads the names with pandas (index_col=0, header=None)
    names = list(pd.read_csv(args.cohort_dataset_names_file_path,
                             index_col=0, header=None).index)
    dup = sorted({n for n in names if names.count(n) > 1})
    report.check(not dup, 'cohort: dataset names unique',
                 f'{len(names)} datasets',
                 f'duplicated names {dup}: the script processes their files '
                 'twice (double-counted statistics)')
    files, empty = [], []
    for name in names:
        found = sorted((silver / name).glob('*.h5ad'))
        if not found:
            empty.append(name)
        files.extend(found)
    if empty:
        report.warn('cohort: datasets without .h5ad files',
                    f'{len(empty)}: {short(empty)}')
    report.info('cohort: files', f'{len(files)} files in {len(names)} datasets')
    return files


# ----------------------------------------------------------------------------
# Pass 1: obs / var of every file
# ----------------------------------------------------------------------------
def scan_files(files, excluded, report):
    excluded_sorted = np.array(sorted(excluded))
    meta = []
    found_batches = {}
    t0 = time.time()
    n_malformed_total = 0
    for k, path in enumerate(files):
        with h5py.File(path, 'r') as f:
            cell_ids = np.asarray(read_elem(f['obs/cell_id']))
            raw_genes = np.asarray(read_elem(f['var/ensembl_id']), dtype=object)
            X = f['X']
            if isinstance(X, h5py.Dataset):
                shape, enc = X.shape, 'dense'
            else:
                shape = tuple(X.attrs['shape'])
                enc = X.attrs.get('encoding-type', '?')
                if isinstance(enc, bytes):
                    enc = enc.decode()
        bad_genes = [g for g in raw_genes if not isinstance(g, str)]
        genes = np.array([str(g) for g in raw_genes], dtype=object)
        batch, n_malformed = batch_ids_of(cell_ids)
        n_malformed_total += n_malformed
        is_excl = (np.isin(batch, excluded_sorted) if len(excluded_sorted)
                   else np.zeros(len(batch), dtype=bool))
        ub, uc = np.unique(batch, return_counts=True)
        for b, c in zip(ub, uc):
            found_batches[b] = found_batches.get(b, 0) + int(c)
        n_train = int((~is_excl).sum())
        m = dict(path=path, n_obs=len(cell_ids), n_train=n_train,
                 n_excl=int(is_excl.sum()), genes=genes, shape=shape, enc=enc,
                 # Rows of training cells (None: all rows)
                 train_rows=None if not is_excl.any() else ~is_excl,
                 processed=not bool(is_excl.all()),  # script: skip if all()
                 cell_ids_unique=bool(pd.Index(cell_ids).is_unique),
                 bad_genes=bad_genes)
        if shape[0] != len(cell_ids) or shape[1] != len(genes):
            report.fail('scan: X shape matches obs/var',
                        f'{path}: X {shape}, obs {len(cell_ids)}, var {len(genes)}')
        meta.append(m)
        if (k + 1) % 50 == 0:
            print(f'  scanned {k + 1}/{len(files)} files '
                  f'({time.time() - t0:.0f}s, max RSS {rss_gb():.1f} GB)',
                  flush=True)
    if n_malformed_total:
        report.warn('scan: cell ids with fewer than two "_"',
                    f'{n_malformed_total} cells')
    not_unique = [str(m['path']) for m in meta if not m['cell_ids_unique']]
    if not_unique:
        report.warn('scan: duplicated cell ids within a file', short(not_unique, 5))
    bad = [(str(m['path']), short(m['bad_genes'], 3)) for m in meta if m['bad_genes']]
    report.check(not bad, 'scan: var ensembl_id values are strings',
                 '', f'non-string (e.g. NaN) gene ids in {short(bad, 3)}')
    encs = sorted({m['enc'] for m in meta})
    report.info('scan: X encodings', f'{encs}')
    return meta, found_batches


# ----------------------------------------------------------------------------
# Pass 2: X
# ----------------------------------------------------------------------------
def column_stats(m, target_depth, nnz_block, pf_s1=None):
    """
    Per var column statistics of the training cells of one file: sums of all
    and of positive counts, number of positive cells, the same sums after
    gene-corrected read depth scaling; plus PF sums.
    """
    path, n_obs, n_var = m['path'], m['shape'][0], m['shape'][1]
    train_rows = m['train_rows']
    out = {k: np.zeros(n_var, dtype=np.float64)
           for k in ('s_all', 's_pos', 'nz', 'g_all', 'g_pos')}
    extra = dict(n_zero_depth=0, n_negative=0, n_noninteger=0,
                 depth_sum=0.0, pf_logsum=0.0)

    def accumulate(row_of, cols, vals, n_rows, r0):
        rowsum = np.bincount(row_of, weights=vals, minlength=n_rows)
        if train_rows is None:
            keep_row = np.ones(n_rows, dtype=bool)
        else:
            keep_row = train_rows[r0:r0 + n_rows]
        extra['n_zero_depth'] += int((rowsum[keep_row] == 0).sum())
        extra['depth_sum'] += float(rowsum[keep_row].sum())
        keep = keep_row[row_of]
        r, c, v = row_of[keep], cols[keep], vals[keep]
        extra['n_negative'] += int((v < 0).sum())
        extra['n_noninteger'] += int((v != np.round(v)).sum())
        pos = v > 0
        out['s_all'] += np.bincount(c, weights=v, minlength=n_var)
        out['s_pos'] += np.bincount(c[pos], weights=v[pos], minlength=n_var)
        out['nz'] += np.bincount(c[pos], minlength=n_var)
        with np.errstate(divide='ignore', invalid='ignore'):
            g = v / rowsum[r] * target_depth
        out['g_all'] += np.bincount(c, weights=g, minlength=n_var)
        out['g_pos'] += np.bincount(c[pos], weights=g[pos], minlength=n_var)
        if pf_s1 is not None:
            safe = np.where(rowsum > 0, rowsum, 1.0)
            extra['pf_logsum'] += float(np.log1p(v * (pf_s1 / safe[r])).sum())

    with h5py.File(path, 'r') as f:
        X = f['X']
        if isinstance(X, h5py.Dataset):
            step = max(1, nnz_block // max(n_var, 1))
            for r0 in range(0, n_obs, step):
                block = np.asarray(X[r0:r0 + step], dtype=np.float64)
                rr, cc = np.nonzero(block)
                accumulate(rr, cc, block[rr, cc], block.shape[0], r0)
        elif m['enc'] == 'csr_matrix':
            indptr = X['indptr'][:].astype(np.int64)
            data_ds, ind_ds = X['data'], X['indices']
            r0 = 0
            while r0 < n_obs:
                r1 = int(np.searchsorted(indptr, indptr[r0] + nnz_block,
                                         side='right')) - 1
                r1 = min(max(r1, r0 + 1), n_obs)
                p0, p1 = indptr[r0], indptr[r1]
                vals = np.asarray(data_ds[p0:p1], dtype=np.float64)
                cols = np.asarray(ind_ds[p0:p1], dtype=np.int64)
                row_of = np.repeat(np.arange(r1 - r0),
                                   np.diff(indptr[r0:r1 + 1]))
                accumulate(row_of, cols, vals, r1 - r0, r0)
                r0 = r1
        else:  # csc or other: load the matrix (rare; memory = the file's X)
            import scipy.sparse as sp
            mat = sp.csr_matrix(read_elem(X))
            coo = mat.tocoo()
            accumulate(coo.row.astype(np.int64), coo.col.astype(np.int64),
                       coo.data.astype(np.float64), n_obs, 0)
    return out, extra


# ----------------------------------------------------------------------------
# Comparison helpers
# ----------------------------------------------------------------------------
def compare_float(report, name, expected, observed, genes, rtol, atol=1e-12):
    expected = np.asarray(expected, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    nan_e, nan_o = np.isnan(expected), np.isnan(observed)
    close = np.isclose(expected, observed, rtol=rtol, atol=atol, equal_nan=True)
    bad = ~close
    if not bad.any():
        with np.errstate(invalid='ignore', divide='ignore'):
            both = ~nan_e & ~nan_o & (expected != 0)
            rel = (np.abs(observed[both] - expected[both]) / np.abs(expected[both])
                   if both.any() else np.array([0.0]))
        report.add('PASS', name, f'{len(expected)} genes, max rel diff '
                   f'{rel.max():.2e}')
        return True
    idx = np.nonzero(bad)[0][:5]
    examples = '; '.join(f'{genes[i]}: expected {expected[i]!r}, csv '
                         f'{observed[i]!r}' for i in idx)
    report.fail(name, f'{int(bad.sum())} of {len(expected)} genes differ '
                f'(NaN pattern differs for {int((nan_e != nan_o).sum())}); {examples}')
    return False


def compare_int(report, name, expected, observed, genes):
    expected = np.asarray(expected, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    bad = ~((expected == observed) | (np.isnan(expected) & np.isnan(observed)))
    if not bad.any():
        report.add('PASS', name, f'{len(expected)} genes')
        return True
    idx = np.nonzero(bad)[0][:5]
    examples = '; '.join(f'{genes[i]}: expected {expected[i]:g}, csv '
                         f'{observed[i]:g}' for i in idx)
    report.fail(name, f'{int(bad.sum())} of {len(expected)} genes differ; '
                f'{examples}')
    return False


def expected_from_sums(acc):
    with np.errstate(divide='ignore', invalid='ignore'):
        exp = {
            'nonzero_mean': np.where(acc['nz'] > 0, acc['s_pos'] / acc['nz'], np.nan),
            'mean': np.where(acc['n_cells'] > 0, acc['mean_num'] / acc['n_cells'], np.nan),
            'gene_corrected_read_depth_nonzero_mean':
                np.where(acc['nz'] > 0, acc['g_pos'] / acc['nz'], np.nan),
            'gene_corrected_read_depth_mean':
                np.where(acc['n_cells'] > 0, acc['g_mean_num'] / acc['n_cells'], np.nan),
            'probing_cell_mean':
                np.where(acc['n_probing'] > 0, acc['probe_num'] / acc['n_probing'], np.nan),
            'n_cells': acc['n_cells'],
            'n_nonzero_cells': acc['nz'],
            'n_probing_cells': acc['n_probing'],
        }
    return exp


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(args):
    report = Report()
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    csv_path = Path(args.norm_factor_csv)
    json_path = Path(args.json) if args.json else Path(
        str(csv_path).removesuffix('.csv') + '.json')
    prefix = csv_path.name.removesuffix('_norm_factors.csv')
    print(f'csv:  {csv_path}\njson: {json_path}')

    # ---------------- held-out lists
    heldout_dir = find_heldout_dir(args)
    val_ids, test_ids, heldout_path = load_heldout(heldout_dir)
    heldout = sorted(set(val_ids) | set(test_ids))
    report.info('heldout: module', f'{heldout_path} ({len(val_ids)} val, '
                f'{len(test_ids)} test ids)')
    overlap = sorted(set(val_ids) & set(test_ids))
    report.check(not overlap, 'heldout: VAL and TEST disjoint', '',
                 f'in both lists: {overlap}')
    dups = sorted({b for b in val_ids + test_ids
                   if (val_ids + test_ids).count(b) > 1})
    if dups:
        report.warn('heldout: duplicated ids in the lists', short(dups))

    # ---------------- json
    with open(json_path) as f:
        rec = json.load(f)
    excluded = set(rec.get('excluded_batch_ids') or [])
    report.check(sorted(excluded) == heldout,
                 'json: excluded_batch_ids == VAL + TEST of heldout_batch_ids.py',
                 f'{len(excluded)} ids',
                 f'only in json {short(sorted(excluded - set(heldout)))}; '
                 f'only in lists {short(sorted(set(heldout) - excluded))}')

    # ---------------- csv
    # As the tokenizer reads it (round-trip float parsing so that values
    # equal those the script wrote)
    df = pd.read_csv(csv_path, float_precision='round_trip')
    df_str = pd.read_csv(csv_path, dtype={'gene_id': str},
                         keep_default_na=False, usecols=['gene_id'])
    required = ['gene_id'] + STAT_FLOAT_COLUMNS + STAT_INT_COLUMNS + ['tie_break_mean']
    missing_cols = [c for c in required if c not in df.columns]
    if not report.check(not missing_cols, 'csv: columns present',
                        f'{len(df)} rows, columns {list(df.columns)}',
                        f'missing {missing_cols}'):
        return finish(report, args, t_start)
    gene_na = df['gene_id'].isna()
    report.check(not gene_na.any(), 'csv: gene_id never parsed as NaN',
                 '', f'{int(gene_na.sum())} gene ids read as NaN, e.g. '
                 f'{list(df_str["gene_id"][gene_na.to_numpy()][:5])}')
    csv_genes = df_str['gene_id'].to_numpy(dtype=object)
    dup_rows = pd.Index(csv_genes)[pd.Index(csv_genes).duplicated()].unique()
    report.check(len(dup_rows) == 0, 'rows: one row per gene (no duplicates)',
                 '', f'{len(dup_rows)} genes with several rows: {short(dup_rows)}')
    with_underscore = [g for g in csv_genes if '_' in g]
    report.check(not with_underscore, 'rows: gene ids without "_"', '',
                 f'{short(with_underscore)} (the script groups keys by '
                 "gene_id.split('_')[0])")

    # ---------------- vocabulary
    vocab_path = args.token_dictionary_file_path or rec.get('token_dictionary_file_path')
    vocab = set()
    if vocab_path:
        if (args.token_dictionary_file_path and rec.get('token_dictionary_file_path')
                and os.path.abspath(args.token_dictionary_file_path)
                != os.path.abspath(rec['token_dictionary_file_path'])):
            report.warn('json: token dictionary differs from --token_dictionary_file_path',
                        f"json {rec['token_dictionary_file_path']}")
        with open(vocab_path, 'rb') as f:
            vocab = {k for k in pickle.load(f) if 'ENS' in str(k)}
        report.info('vocabulary', f'{len(vocab)} genes from {vocab_path}')
    else:
        report.warn('vocabulary', 'no token dictionary given or recorded: '
                    'vocabulary rows not checked')

    # ---------------- pass 1: obs/var
    files = cohort_files(args, report)
    print(f'Scanning obs/var of {len(files)} files...', flush=True)
    meta, found_batches = scan_files(files, excluded, report)
    unmatched = sorted(set(heldout) - set(found_batches))
    report.check(not unmatched, 'heldout: every held-out id matches cells',
                 f'{len(heldout)} ids, {sum(found_batches.get(b, 0) for b in heldout)} cells',
                 f'{len(unmatched)} ids match no cell: {short(unmatched)}')
    n_train = sum(m['n_train'] for m in meta)
    n_excl = sum(m['n_excl'] for m in meta)
    report.check(rec.get('n_files') == len(files), 'json: n_files == recount',
                 f'{len(files)}', f"json {rec.get('n_files')}, recount {len(files)}")
    report.check(rec.get('n_cells') == n_train,
                 'json: n_cells == training cells recounted from obs',
                 f'{n_train}', f"json {rec.get('n_cells')}, recount {n_train}")
    report.check(rec.get('n_cells_excluded') == n_excl,
                 'json: n_cells_excluded == held-out cells recounted from obs',
                 f'{n_excl}', f"json {rec.get('n_cells_excluded')}, recount {n_excl}")
    if args.expected_n_cells_total is not None:
        report.check(n_train + n_excl == args.expected_n_cells_total,
                     'cohort: total cells == --expected_n_cells_total',
                     f'{n_train + n_excl}',
                     f'{n_train + n_excl} != {args.expected_n_cells_total}')

    # ---------------- gene index (cohort genes + vocabulary + csv rows)
    cohort_genes = set()
    train_genes = set()
    for m in meta:
        cohort_genes.update(m['genes'])
        if m['processed']:
            train_genes.update(m['genes'])
    all_genes = sorted(cohort_genes | vocab | set(csv_genes))
    gindex = pd.Index(all_genes)
    G = len(gindex)
    for m in meta:
        m['gidx'] = gindex.get_indexer(m['genes'])

    csv_set = set(csv_genes)
    missing_cohort = sorted(cohort_genes - csv_set)
    missing_vocab = sorted(vocab - csv_set)
    extra_rows = sorted(csv_set - cohort_genes - vocab)
    report.check(not missing_cohort, 'rows: every gene of every cohort file has a row',
                 f'{len(cohort_genes)} cohort genes',
                 f'{len(missing_cohort)} missing: {short(missing_cohort)}')
    report.check(not missing_vocab, 'rows: every vocabulary gene has a row',
                 f'{len(vocab)} vocabulary genes',
                 f'{len(missing_vocab)} missing: {short(missing_vocab)}')
    report.check(not extra_rows, 'rows: no rows outside cohort + vocabulary', '',
                 f'{len(extra_rows)} extra rows: {short(extra_rows)}')

    # csv aligned to the gene index (first row of a gene, as TERRA reads it)
    first = df.assign(gene_id=csv_genes).drop_duplicates('gene_id').set_index('gene_id')
    csvg = first.reindex(gindex)

    # ---------------- tie-break
    tb = csvg['tie_break_mean'].to_numpy(np.float64)
    pcm = csvg['probing_cell_mean'].to_numpy(np.float64)
    nnz_csv = csvg['n_nonzero_cells'].to_numpy(np.float64)
    has_row = gindex.isin(csv_set)
    nonfinite = has_row & ~np.isfinite(tb)
    report.check(not nonfinite.any(), 'tiebreak: tie_break_mean finite for every row',
                 f'{int(has_row.sum())} rows',
                 f'{int(nonfinite.sum())} rows not finite: {short(gindex[nonfinite])}')
    has_key = has_row & (nnz_csv > 0) & np.isfinite(pcm)
    eq = tb[has_key] == pcm[has_key]
    report.check(eq.all(), 'tiebreak: == probing_cell_mean where n_nonzero_cells > 0',
                 f'{int(has_key.sum())} genes',
                 f'{int((~eq).sum())} differ: {short(gindex[has_key][~eq])}')
    fill_expected = float(pcm[has_key].max()) + 1.0 if has_key.any() else float('nan')
    fill_json = rec.get('tie_break_mean_fill')
    report.check(fill_json == fill_expected,
                 'tiebreak: json fill == max(probing_cell_mean of detected) + 1',
                 f'{fill_json!r}', f'json {fill_json!r}, expected {fill_expected!r}')
    rest = has_row & ~has_key
    eq_fill = tb[rest] == fill_json
    report.check(eq_fill.all(), 'tiebreak: == json fill for genes not detected in training',
                 f'{int(rest.sum())} genes',
                 f'{int((~eq_fill).sum())} differ: {short(gindex[rest][~eq_fill])}')
    # Detected genes whose tie-break key equals another gene's: the tokenizer
    # orders those by token id (information only)
    tb_det = pd.Series(tb[has_key])
    n_shared = int(tb_det.duplicated(keep=False).sum())
    report.info('tiebreak: detected genes sharing their tie_break_mean with another gene',
                f'{n_shared} of {int(has_key.sum())} (ordered by token id among themselves)')
    for key, val in (('n_genes', int(len(df))),
                     ('n_genes_not_detected_in_training', int(rest.sum())),
                     ('n_genes_only_in_excluded_batches',
                      len(cohort_genes - train_genes)),
                     ('n_genes_only_in_vocabulary', len(vocab - cohort_genes))):
        report.check(rec.get(key) == val, f'json: {key}', f'{val}',
                     f'json {rec.get(key)}, expected {val}')

    # ---------------- probing counts from obs/var (all genes, no X)
    n_probing = np.zeros(G)
    for m in meta:
        if m['processed']:
            np.add.at(n_probing, m['gidx'], m['n_train'])
    rows_mask = has_row
    compare_int(report, 'probing: n_probing_cells == training cells of probing files (obs/var)',
                n_probing[rows_mask], csvg['n_probing_cells'].to_numpy(np.float64)[rows_mask],
                gindex[rows_mask])
    only_excl = gindex.isin(sorted(cohort_genes - train_genes))
    if only_excl.any():
        z = csvg['n_probing_cells'].to_numpy(np.float64)[only_excl]
        report.check((z == 0).all(), 'probing: genes only in held-out files have n_probing_cells 0',
                     f'{int(only_excl.sum())} genes',
                     f'{short(gindex[only_excl][z != 0])}')
    else:
        report.info('probing: genes only in held-out files', 'none')

    # ---------------- pass 2: X
    target = lambda m: (args.basis_target_read_depth
                        + m['shape'][1] * args.target_read_depth_per_gene)
    processed = [m for m in meta if m['processed']]
    max_files = args.max_files if args.max_files and args.max_files < len(processed) else 0

    pf_path = Path(args.pf_targets_csv) if args.pf_targets_csv else (
        csv_path.parent / f'{prefix}_pf_targets.csv')
    pf = None
    if pf_path.exists():
        pf = pd.read_csv(pf_path).iloc[0]
        if int(pf['n_cells']) != n_train:
            report.fail('pf: n_cells == training cells',
                        f"{pf_path}: {int(pf['n_cells'])} != {n_train} (stale file?)")
    pf_s1 = float(pf['pf_depth_target']) if pf is not None else None

    if args.skip_x:
        report.warn('recompute', 'skipped (--skip_x)')
        return finish(report, args, t_start)

    if not max_files:
        acc = {k: np.zeros(G) for k in ('s_pos', 'nz', 'g_pos', 'n_cells', 'mean_num',
                                        'g_mean_num', 'probe_num', 'n_probing')}
        totals = dict(n_zero_depth=0, n_negative=0, n_noninteger=0,
                      depth_sum=0.0, pf_logsum=0.0, n=0)
        t0 = time.time()
        print(f'Reading X of {len(processed)} files (full recomputation)...', flush=True)
        for k, m in enumerate(processed):
            st, ex = column_stats(m, target(m), args.nnz_block, pf_s1)
            gi, T = m['gidx'], m['n_train']
            det = st['nz'] > 0
            np.add.at(acc['s_pos'], gi, st['s_pos'])
            np.add.at(acc['nz'], gi, st['nz'])
            np.add.at(acc['g_pos'], gi, st['g_pos'])
            np.add.at(acc['n_cells'], gi, np.where(det, T, 0))
            np.add.at(acc['mean_num'], gi, np.where(det, st['s_all'], 0.0))
            np.add.at(acc['g_mean_num'], gi, np.where(det, st['g_all'], 0.0))
            np.add.at(acc['probe_num'], gi, st['s_all'])
            np.add.at(acc['n_probing'], gi, T)
            for key in ('n_zero_depth', 'n_negative', 'n_noninteger', 'depth_sum', 'pf_logsum'):
                totals[key] += ex[key]
            totals['n'] += T
            if (k + 1) % 10 == 0 or k + 1 == len(processed):
                print(f'  X {k + 1}/{len(processed)} files, {totals["n"]} cells '
                      f'({time.time() - t0:.0f}s, max RSS {rss_gb():.1f} GB)', flush=True)
        report.check(totals['n_zero_depth'] == 0, 'data: no training cell with zero depth',
                     '', f"{totals['n_zero_depth']} cells (their GCRD values are NaN in the script)")
        report.check(totals['n_negative'] == 0, 'data: no negative counts', '',
                     f"{totals['n_negative']} negative entries")
        if totals['n_noninteger']:
            report.warn('data: non-integer counts', f"{totals['n_noninteger']} entries")
        exp = expected_from_sums(acc)
        sel = rows_mask.copy()
        if args.sample_genes:
            cand = np.nonzero(rows_mask)[0]
            pick = rng.choice(cand, size=min(args.sample_genes, len(cand)), replace=False)
            sel = np.zeros(G, dtype=bool)
            sel[pick] = True
            sel |= rows_mask & ~has_key  # undetected / held-out-only / vocabulary rows
        report.info('recompute: genes compared', f'{int(sel.sum())} of {int(rows_mask.sum())}')
        for col in STAT_FLOAT_COLUMNS:
            compare_float(report, f'recompute: {col}', exp[col][sel],
                          csvg[col].to_numpy(np.float64)[sel], gindex[sel], args.rtol)
        for col in STAT_INT_COLUMNS:
            compare_int(report, f'recompute: {col}', exp[col][sel],
                        csvg[col].to_numpy(np.float64)[sel], gindex[sel])
        exp_tb = np.where(exp['n_nonzero_cells'] > 0, exp['probing_cell_mean'], np.nan)
        exp_fill = np.nanmax(exp_tb[rows_mask]) + 1.0
        exp_tb = np.where(np.isnan(exp_tb), exp_fill, exp_tb)
        compare_float(report, 'recompute: tie_break_mean', exp_tb[sel],
                      tb[sel], gindex[sel], args.rtol)
        if pf is not None:
            s1 = totals['depth_sum'] / max(totals['n'], 1)
            s2 = totals['pf_logsum'] / max(totals['n'], 1)
            report.check(math.isclose(s1, pf_s1, rel_tol=args.rtol),
                         'pf: pf_depth_target (s1)', f'{s1!r}', f'recomputed {s1!r}, csv {pf_s1!r}')
            report.check(math.isclose(s2, float(pf['pf_logsum_target']), rel_tol=args.rtol),
                         'pf: pf_logsum_target (s2)', f'{s2!r}',
                         f"recomputed {s2!r}, csv {float(pf['pf_logsum_target'])!r}")
    else:
        check_with_batchwise(report, args, meta, processed, gindex, csvg, has_row,
                             max_files, target, rng, csv_path, prefix)
    return finish(report, args, t_start)


def check_with_batchwise(report, args, meta, processed, gindex, csvg, has_row,
                         max_files, target, rng, csv_path, prefix):
    """--max_files N: per-file stats of N files vs the batchwise pickle, and
    the csv re-aggregated from the pickle."""
    bw_path = Path(args.batchwise_pkl) if args.batchwise_pkl else (
        csv_path.parent / f'{prefix}_batchwise_norm_factors.pkl')
    if not bw_path.exists():
        report.fail('recompute', f'--max_files needs the batchwise pickle ({bw_path}); '
                    'run without --max_files for the full recomputation')
        return
    print(f'Loading {bw_path}...', flush=True)
    with open(bw_path, 'rb') as f:
        bw = pickle.load(f)
    report.info('batchwise: loaded', f'{len(bw)} keys (max RSS {rss_gb():.1f} GB)')
    G = len(gindex)
    # Expected number of entries of each gene, in file order
    n_entries = np.zeros(G, dtype=np.int64)
    offsets = {}
    for m in processed:
        gi = m['gidx']
        # occurrence number of each column's gene within the file
        order = pd.Series(gi).groupby(gi).cumcount().to_numpy()
        offsets[id(m)] = n_entries[gi] + order
        np.add.at(n_entries, gi, 1)
    bw_genes = sorted({k[:-len('_n_probing_cells')] for k in bw
                       if k.endswith('_n_probing_cells')})
    exp_genes = list(gindex[n_entries > 0])
    report.check(bw_genes == exp_genes,
                 'batchwise: genes == genes of processed files', f'{len(bw_genes)} genes',
                 f'only in pickle {short(sorted(set(bw_genes) - set(exp_genes)))}; '
                 f'missing {short(sorted(set(exp_genes) - set(bw_genes)))}')
    arrays = {}
    bad_len = []
    for i in np.nonzero(n_entries > 0)[0]:
        g = gindex[i]
        lists = [bw.get(f'{g}_{s}') for s in BATCHWISE_STATS]
        if any(l is None or len(l) != n_entries[i] for l in lists):
            bad_len.append(g)
            continue
        arrays[g] = np.array(lists, dtype=np.float64)  # (7, n_entries)
    report.check(not bad_len, 'batchwise: one entry per processed file holding the gene',
                 f'{len(arrays)} genes', f'{len(bad_len)} genes with other lengths: {short(bad_len)}')
    # n_probing entries == training cells of those files (all genes)
    exp_seq = {g: [] for g in arrays}
    for m in processed:
        for g in m['genes']:
            if g in exp_seq:
                exp_seq[g].append(m['n_train'])
    bad_probe = [g for g in arrays if not np.array_equal(arrays[g][6], exp_seq[g])]
    report.check(not bad_probe, 'batchwise: n_probing_cells entries == training cells per file',
                 '', f'{len(bad_probe)} genes: {short(bad_probe)}')
    # Re-aggregate the csv from the pickle
    agg = {c: np.full(G, np.nan) for c in STAT_FLOAT_COLUMNS + STAT_INT_COLUMNS}
    for g, a in arrays.items():
        i = gindex.get_loc(g)
        nzm, mn, gnzm, gmn, nc, nnz, npr = a
        with np.errstate(divide='ignore', invalid='ignore'):
            agg['nonzero_mean'][i] = (nzm * nnz).sum() / nnz.sum()
            agg['mean'][i] = (mn * nc).sum() / nc.sum()
            agg['gene_corrected_read_depth_nonzero_mean'][i] = (gnzm * nnz).sum() / nnz.sum()
            agg['gene_corrected_read_depth_mean'][i] = (gmn * nc).sum() / nc.sum()
            agg['probing_cell_mean'][i] = (mn * npr).sum() / npr.sum()
        agg['n_cells'][i], agg['n_nonzero_cells'][i], agg['n_probing_cells'][i] = (
            nc.sum(), nnz.sum(), npr.sum())
    for c in ('n_cells', 'n_nonzero_cells', 'n_probing_cells'):
        agg[c] = np.where(has_row & np.isnan(agg[c]), 0.0, agg[c])
    sel = has_row
    for col in STAT_FLOAT_COLUMNS:
        compare_float(report, f'aggregate: csv {col} == pickle aggregate', agg[col][sel],
                      csvg[col].to_numpy(np.float64)[sel], gindex[sel], args.rtol)
    for col in STAT_INT_COLUMNS:
        compare_int(report, f'aggregate: csv {col} == pickle aggregate', agg[col][sel],
                    csvg[col].to_numpy(np.float64)[sel], gindex[sel])
    # Per-file recomputation of N files: mixed files first, then random ones
    mixed = [m for m in processed if m['train_rows'] is not None]
    others = [m for m in processed if m['train_rows'] is None]
    pick = mixed[:max_files]
    if len(pick) < max_files:
        idx = rng.choice(len(others), size=min(max_files - len(pick), len(others)), replace=False)
        pick += [others[i] for i in sorted(idx)]
    report.info('per-file: files recomputed',
                f'{len(pick)} of {len(processed)} ({len(mixed)} mixed held-out/training files exist)')
    n_bad = 0
    examples = []
    for m in pick:
        st, ex = column_stats(m, target(m), args.nnz_block)
        T = m['n_train']
        with np.errstate(divide='ignore', invalid='ignore'):
            exp = np.vstack([
                np.where(st['nz'] > 0, st['s_pos'] / st['nz'], 0.0),
                st['s_all'] / T,
                np.where(st['nz'] > 0, st['g_pos'] / st['nz'], 0.0),
                st['g_all'] / T,
                np.where(st['nz'] > 0, T, 0),
                st['nz'],
                np.full(len(st['nz']), T)])
        cols = np.arange(len(m['genes']))
        if args.sample_genes and args.sample_genes < len(cols):
            cols = np.sort(rng.choice(cols, size=args.sample_genes, replace=False))
        off = offsets[id(m)]
        for j in cols:
            g = m['genes'][j]
            if g not in arrays:
                continue
            obs = arrays[g][:, off[j]]
            ok = np.isclose(exp[:, j], obs, rtol=args.rtol, atol=1e-12)
            if not ok.all():
                n_bad += 1
                if len(examples) < 5:
                    examples.append(f'{Path(m["path"]).name}:{g} expected '
                                    f'{exp[:, j].round(6).tolist()}, pickle {obs.round(6).tolist()}')
        print(f'  per-file {Path(m["path"]).name}: {T} training cells, max RSS {rss_gb():.1f} GB',
              flush=True)
    report.check(n_bad == 0, 'per-file: recomputed statistics == batchwise pickle entries',
                 f'{len(pick)} files', f'{n_bad} gene entries differ; {examples}')
    report.warn('recompute: partial', f'--max_files {max_files}: the csv is checked against '
                'the batchwise pickle, and the pickle against the data for these files only; '
                'run without --max_files for the full recomputation')


def finish(report, args, t_start):
    counts = report.summary()
    print(f'Runtime {time.time() - t_start:.0f}s, max RSS {rss_gb():.2f} GB')
    if args.report_json:
        with open(args.report_json, 'w') as f:
            json.dump({'counts': counts, 'checks': report.rows}, f, indent=1)
    return 1 if report.n_fail else 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--norm_factor_csv', required=True)
    p.add_argument('--json', default=None,
                   help='Default: the csv path with .json.')
    p.add_argument('--data_directory_path', required=True,
                   help='Directory holding silver/<dataset_name>/*.h5ad.')
    p.add_argument('--cohort_dataset_names_file_path', required=True)
    p.add_argument('--token_dictionary_file_path', default=None,
                   help='Default: the path recorded in the json.')
    p.add_argument('--heldout_module_dir', default=None,
                   help='Directory with heldout_batch_ids.py (default: the parent '
                        'of the names file directory, the parent of this '
                        'directory, this directory, cwd).')
    p.add_argument('--batchwise_pkl', default=None,
                   help='Default: {prefix}_batchwise_norm_factors.pkl next to the csv '
                        '(used with --max_files).')
    p.add_argument('--pf_targets_csv', default=None,
                   help='Default: {prefix}_pf_targets.csv next to the csv, if present.')
    p.add_argument('--max_files', type=int, default=0,
                   help='Recompute the per-file statistics of N files only and check '
                        'them against the batchwise pickle (0: all files, full pooled '
                        'recomputation).')
    p.add_argument('--sample_genes', type=int, default=0,
                   help='Compare K random genes (plus every gene not detected in '
                        'training) instead of all (0: all).')
    p.add_argument('--full', action='store_true',
                   help='Same as --max_files 0 --sample_genes 0.')
    p.add_argument('--skip_x', action='store_true',
                   help='Only the obs/var checks (no X pass).')
    p.add_argument('--basis_target_read_depth', type=float,
                   default=DEFAULT_BASIS_TARGET_READ_DEPTH)
    p.add_argument('--target_read_depth_per_gene', type=float,
                   default=DEFAULT_TARGET_READ_DEPTH_PER_GENE)
    p.add_argument('--expected_n_cells_total', type=int, default=None)
    p.add_argument('--nnz_block', type=int, default=20_000_000,
                   help='Stored entries of X read at once (memory ~ 60 B each).')
    p.add_argument('--rtol', type=float, default=1e-7)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--report_json', default=None)
    a = p.parse_args()
    if a.full:
        a.max_files, a.sample_genes = 0, 0
    try:
        code = main(a)
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        print('RESULT: ERROR (the validator crashed; this is not a validation result)')
        code = 2
    sys.exit(code)
