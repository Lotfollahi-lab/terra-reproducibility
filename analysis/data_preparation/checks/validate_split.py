#!/usr/bin/env python
"""
Validate the precomputed splits and n_nonzero_tokens lists of a tokenized
corpus (terra-reproducibility split_dataset.py and
save_dataset_n_nonzero_tokens.py outputs) against the tokenized dataset.
Both row layouts are accepted: the padded layout (every row model_input_size
tokens) and the compact layout (TERRA storage_format 'compact',
terra/datasets/compact.py: gene_tokens without padding plus the columns
segment_lengths and model_input_size). The layout is detected from the
segment_lengths column.

Training on compact rows needs no n_nonzero_tokens lists (TERRA derives the
lengths of the layout a model reads from the segment lengths), so
save_dataset_n_nonzero_tokens.py is skipped for compact corpora: pass
--no_n_nonzero_lists to accept missing lists (lists that exist are still
checked).

Files checked, in --tokenizer_folder_path, with NAME = the dataset folder name
without '.dataset' (as the scripts name them):
  --splits heldout (default): NAME_precomputed_split_{train,validation,test}.pkl
                              NAME_n_nonzero_tokens_{train,validation,test}.pkl
  --splits full:              NAME_precomputed_split_full.pkl
                              NAME_n_nonzero_tokens_full.pkl
  --splits both:              all of them

Checks (PASS / FAIL / WARN; exit code 0 = PASS, 1 = FAIL, 2 = the validator
itself crashed):
  dataset   cell ids present, well formed and (by default) unique; the arrow
            shards hold len(dataset) rows
  layout    padded or compact; the recorded storage_format agrees; optional
            --expect_param KEY=VALUE tokenizer settings
  heldout   VAL and TEST of heldout_batch_ids.py disjoint; every id matches rows
  load      each split loads through TERRA (terra.datasets.utils
            _read_precomputed_split, which load_precomputed_split wraps): TERRA
            format (not a legacy bare list), made for this dataset (row count
            and recorded cell ids)
  indices   integers in range, no duplicates; the three splits are disjoint and
            cover every row (full: the full split is every row once)
  content   validation == the rows of VAL batches; test == the rows of TEST
            batches; train holds no held-out row (independent batch-id
            parsing), and TERRA's own training-time scan agrees
  record    train records excluded_batch_ids == VAL + TEST
  n_nonzero each list has one entry per split row and equals the dataset's
            n_nonzero_tokens at the split's rows (all rows); for --sample_rows
            rows per split (0 / --full: all rows) n_nonzero_tokens equals the
            recount of non-pad gene tokens of gene_tokens (the stored entries of
            a compact row; the padding a compact row leaves out is token 0, so
            the count is the same), and the rows are well formed:
              padded:  len(gene_tokens) == len(gene_expr) == model_input_size
              compact: model_input_size column == the recorded one; 1 <=
                       segments <= n_neighs + 1; every segment length <=
                       int(model_input_size / segments); sum(segment_lengths)
                       == len(gene_tokens) == len(gene_expr)
  stale     no other split / n_nonzero pickle with the NAME_ prefix that is a
            legacy bare list or does not fit this dataset
  tokenizer (optional, --norm_factor_csv) the dataset was tokenized with that
            norm-factor file (sha256 in terra_tokenizer_params.json) and with
            stable_rank_ties + rank_tie_break_norm_factor
  json      (optional, --norm_factor_json) its excluded batches are the
            held-out batches; train rows vs its n_cells
  config    (optional, --training_config) a model config's data paths are the
            validated dataset / split / n_nonzero list, and train()'s held-out
            check (terra.datasets.utils._check_split_excludes_heldout with the
            config's test/val ids) accepts its split; WARN when the config's
            ids are fewer than the held-out lists (train() checks only those)

Memory: the arrow shards are memory-mapped one at a time and read record batch
by record batch (only the buffers of the columns used are touched; a shard is
unmapped when done); cell-id uniqueness is checked on 64-bit hashes (8 bytes
per row; exact re-check of colliding hashes). Per-row arrays: ~14 bytes per
row (110M rows: ~1.6 GB). The split / n_nonzero pickles are Python lists: a
100M-entry list takes ~4 GB while it is loaded (it is converted to an int64
array and freed at once). Request 24 GB for the 110M-cell corpus.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from datasets import load_from_disk

HELDOUT_SPLITS = ('train', 'validation', 'test')


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail=''):
        self.rows.append((status, name, detail))
        print(f'[{status}] {name}' + (f': {detail}' if detail else ''), flush=True)

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
    return r / 1e9 if sys.platform == 'darwin' else r / 1e6


def short(items, n=10):
    items = list(items)
    return f'{items[:n]}' + (f' ... (+{len(items) - n})' if len(items) > n else '')


def load_heldout(module_dir):
    path = Path(module_dir) / 'heldout_batch_ids.py'
    spec = importlib.util.spec_from_file_location('_validate_heldout_batch_ids', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.VAL_BATCH_IDS), list(module.TEST_BATCH_IDS), path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Reading the dataset
# ----------------------------------------------------------------------------
def dataset_shards(data_path):
    """Arrow files of a saved dataset, in row order (state.json), or None."""
    state_path = Path(data_path) / 'state.json'
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    files = [Path(data_path) / f['filename'] for f in state.get('_data_files', [])]
    return files if files and all(f.exists() for f in files) else None


def iter_batches(ds, data_path, columns, max_rows):
    """
    (start_row, pyarrow.Table of `columns`) chunks of at most `max_rows` rows,
    in row order. Shards are memory-mapped one at a time and read record batch
    by record batch, so only the buffers of `columns` of one shard are touched
    at a time; falls back to the dataset's table if the shards are unknown.
    """
    shards = dataset_shards(data_path) if ds._indices is None else None
    if shards is None:
        sub = ds.select_columns(columns).with_format('arrow')
        for start in range(0, len(ds), max_rows):
            yield start, sub[start:start + max_rows]
        return
    start = 0
    for shard in shards:
        with pa.memory_map(str(shard), 'r') as source:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                cols = [batch.column(batch.schema.get_field_index(c)) for c in columns]
                table = pa.Table.from_arrays(cols, names=list(columns))
                for s in range(0, table.num_rows, max_rows):
                    part = table.slice(s, max_rows)
                    yield start, part
                    start += part.num_rows
                del cols, table, batch
            del reader


def list_parts(arr):
    """(values ndarray, offsets ndarray starting at 0) of a list array."""
    if isinstance(arr, pa.ChunkedArray):
        if arr.num_chunks == 1:
            arr = arr.chunk(0)
        else:
            arr = arr.combine_chunks()
    offsets = np.asarray(arr.offsets, dtype=np.int64)
    values = arr.values.slice(int(offsets[0]), int(offsets[-1] - offsets[0]))
    return (np.asarray(values.to_numpy(zero_copy_only=False)),
            offsets - offsets[0])


def list_lengths(arr):
    if isinstance(arr, pa.ChunkedArray):
        return np.concatenate([list_lengths(c) for c in arr.chunks]) \
            if arr.num_chunks else np.zeros(0, dtype=np.int64)
    return np.diff(np.asarray(arr.offsets, dtype=np.int64))


def row_sums(values, offsets):
    csum = np.concatenate([[0], np.cumsum(values, dtype=np.int64)])
    return csum[offsets[1:]] - csum[offsets[:-1]]


def count_nonpad(gene_tokens, pad):
    """Non-pad, nonzero entries per row of a list array (as
    terra.tokenizers.tokenize.process_gene_tokens counts them)."""
    values, offsets = list_parts(gene_tokens)
    return row_sums(((values != 0) & (values != pad)).astype(np.int64), offsets)


class RowChecker:
    """Recount of n_nonzero_tokens and structure of padded / compact rows."""

    def __init__(self, compact, pad, model_input_size, n_neighs):
        self.compact, self.pad = compact, pad
        self.mis, self.n_neighs = model_input_size, n_neighs
        self.columns = ['gene_tokens', 'gene_expr'] + (
            ['segment_lengths', 'model_input_size'] if compact else [])
        self.problems = Counter()
        self.examples = {}
        self.padded_lengths = Counter()
        self.n_rows = 0

    def _flag(self, name, bad, rows):
        n = int(bad.sum())
        if n:
            self.problems[name] += n
            self.examples.setdefault(name, rows[bad][:5].tolist())

    def __call__(self, table, rows):
        """Recount of the rows of `table` (row numbers `rows`)."""
        rows = np.asarray(rows)
        self.n_rows += len(rows)
        tok = table.column('gene_tokens')
        recount = count_nonpad(tok, self.pad)
        n_tok = list_lengths(tok)
        n_expr = list_lengths(table.column('gene_expr'))
        self._flag('len(gene_expr) != len(gene_tokens)', n_expr != n_tok, rows)
        if not self.compact:
            for value, count in zip(*np.unique(n_tok, return_counts=True)):
                self.padded_lengths[int(value)] += int(count)
            if self.mis is not None:
                self._flag('padded len(gene_tokens) != model_input_size',
                           n_tok != self.mis, rows)
            return recount
        seg, seg_off = list_parts(table.column('segment_lengths'))
        seg = seg.astype(np.int64)
        n_seg = np.diff(seg_off)
        mis_col = np.asarray(table.column('model_input_size').to_numpy(
            zero_copy_only=False), dtype=np.int64)
        if self.mis is not None:
            self._flag('compact model_input_size column != recorded model_input_size',
                       mis_col != self.mis, rows)
        max_seg = (self.n_neighs + 1) if self.n_neighs is not None else None
        bad_nseg = n_seg < 1
        if max_seg is not None:
            bad_nseg |= n_seg > max_seg
        self._flag('compact number of segments outside [1, n_neighs + 1]', bad_nseg, rows)
        ok = n_seg >= 1
        seg_len = np.zeros(len(n_seg), dtype=np.int64)
        seg_len[ok] = (mis_col[ok] / n_seg[ok]).astype(np.int64)  # int(mis / n_seg)
        longest = np.zeros(len(n_seg), dtype=np.int64)
        if ok.any() and len(seg):
            nonempty = n_seg > 0
            longest[nonempty] = np.maximum.reduceat(seg, seg_off[:-1][nonempty])
        self._flag('compact segment length > int(model_input_size / segments)',
                   ok & (longest > seg_len), rows)
        self._flag('compact sum(segment_lengths) != len(gene_tokens)',
                   row_sums(seg, seg_off) != n_tok, rows)
        if (seg < 0).any():
            self._flag('compact negative segment length',
                       row_sums((seg < 0).astype(np.int64), seg_off) > 0, rows)
        return recount

    def report(self, report, label):
        report.check(not self.problems, f'n_nonzero: rows well formed ({label})',
                     f'{self.n_rows} rows ({"compact" if self.compact else "padded"})',
                     '; '.join(f'{k}: {v} rows, e.g. {self.examples[k]}'
                               for k, v in self.problems.items()))
        if not self.compact and self.mis is None:
            report.check(len(self.padded_lengths) <= 1,
                         f'n_nonzero: one padded row length ({label})',
                         f'{dict(self.padded_lengths)}',
                         f'padded rows of several lengths: {dict(self.padded_lengths)}')


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_expect(items):
    out = {}
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f'--expect_param {item!r}: use KEY=VALUE')
        key, value = item.split('=', 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main(args):
    report = Report()
    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    from terra.datasets import utils as tu

    data_path = args.tokenized_data_folder_path.rstrip('/')
    # Same name as split_dataset.py / save_dataset_n_nonzero_tokens.py
    name = data_path.rsplit('/', 1)[-1].removesuffix('.dataset')
    folder = Path(args.tokenizer_folder_path)
    want_heldout = args.splits in ('heldout', 'both')
    want_full = args.splits in ('full', 'both')
    print(f'dataset: {data_path}\nname:    {name}\nfolder:  {folder}\nsplits:  {args.splits}')

    # ---------------- held-out lists
    # Default: analysis/data_preparation (the parent of checks/), else the
    # validator's own directory
    here = Path(__file__).resolve().parent
    module_dir = args.heldout_module_dir or str(next(
        (d for d in (here.parent, here)
         if (d / 'heldout_batch_ids.py').exists()), here))
    val_ids, test_ids, heldout_path = load_heldout(module_dir)
    report.info('heldout: module', f'{heldout_path} ({len(val_ids)} val, {len(test_ids)} test)')
    overlap = sorted(set(val_ids) & set(test_ids))
    report.check(not overlap, 'heldout: VAL and TEST disjoint', '', f'in both: {overlap}')
    val_set, test_set = set(val_ids), set(test_ids)
    heldout = sorted(val_set | test_set)

    # ---------------- dataset, layout, tokenizer record
    ds = load_from_disk(data_path)
    n = len(ds)
    compact = 'segment_lengths' in ds.column_names
    report.info('dataset', f'{n} rows, columns {ds.column_names}')
    for col in ('cell_id', 'n_nonzero_tokens', 'gene_tokens', 'gene_expr') + (
            ('model_input_size',) if compact else ()):
        if col not in ds.column_names:
            report.fail('dataset: columns', f'no {col} column')
            return finish(report, args, t_start)
    params_path = Path(data_path) / 'terra_tokenizer_params.json'
    params = json.loads(params_path.read_text()) if params_path.exists() else None
    recorded_format = (params or {}).get('storage_format', 'padded')
    report.info('layout', f"{'compact' if compact else 'padded'} rows "
                f"(gene_tokens {ds.features['gene_tokens'].feature.dtype}, "
                f"gene_expr {ds.features['gene_expr'].feature.dtype})")
    if params is not None:
        report.check(recorded_format == ('compact' if compact else 'padded'),
                     'layout: recorded storage_format matches the columns',
                     recorded_format,
                     f"terra_tokenizer_params.json storage_format {recorded_format!r}, "
                     f"columns are {'compact' if compact else 'padded'}")
    mis = (params or {}).get('model_input_size')
    n_neighs = (params or {}).get('n_neighs')
    for key, value in parse_expect(args.expect_param).items():
        got = (params or {}).get(key, '<absent>')
        report.check(params is not None and got == value, f'layout: tokenizer setting {key}',
                     f'{value!r}', f'recorded {got!r}, expected {value!r}')

    # ---------------- cell ids and n_nonzero_tokens column (one pass)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    nnz_col = np.zeros(n, dtype=np.int32)
    hashes = np.zeros(n if args.check_unique_cell_ids else 0, dtype=np.uint64)
    batch_counts = Counter()
    n_null_ids = n_no_underscore = 0
    nnz_min, nnz_max = np.iinfo(np.int64).max, np.iinfo(np.int64).min
    val_arr = pa.array(sorted(val_set), type=pa.string())
    test_arr = pa.array(sorted(test_set), type=pa.string())
    t0 = time.time()
    n_seen = 0
    for start, tbl in iter_batches(ds, data_path, ['cell_id', 'n_nonzero_tokens'],
                                   args.chunk_rows):
        cid = tbl.column('cell_id')
        n_null_ids += cid.null_count
        b = batch_ids_arrow(cid)
        n_no_underscore += b.null_count - cid.null_count
        stop = start + len(tbl)
        val_mask[start:stop] = np.asarray(pc.fill_null(pc.is_in(b, value_set=val_arr), False))
        # split_dataset.py: a row goes to validation first, else to test
        test_mask[start:stop] = (np.asarray(pc.fill_null(pc.is_in(b, value_set=test_arr), False))
                                 & ~val_mask[start:stop])
        nnz = np.asarray(tbl.column('n_nonzero_tokens').to_numpy(), dtype=np.int64)
        if len(nnz):
            nnz_min, nnz_max = min(nnz_min, int(nnz.min())), max(nnz_max, int(nnz.max()))
        nnz_col[start:stop] = nnz
        if args.check_unique_cell_ids:
            hashes[start:stop] = pd.util.hash_array(
                np.asarray(cid.to_numpy(zero_copy_only=False), dtype=object))
        for item in pc.value_counts(pc.drop_null(b)).to_pylist():
            batch_counts[item['values']] += item['counts']
        n_seen = stop
    print(f'  read cell_id / n_nonzero_tokens ({time.time() - t0:.0f}s, '
          f'max RSS {rss_gb():.1f} GB)', flush=True)
    report.check(n_seen == n, 'dataset: arrow shards hold len(dataset) rows', f'{n}',
                 f'{n_seen} rows read, dataset has {n}')
    report.check(n_null_ids == 0 and n_no_underscore == 0, 'dataset: cell ids well formed',
                 f'{len(batch_counts)} batches',
                 f'{n_null_ids} null cell ids, {n_no_underscore} without "_"')
    if n:
        report.check(0 <= nnz_min and nnz_max <= np.iinfo(np.int32).max
                     and (mis is None or nnz_max <= mis),
                     'dataset: n_nonzero_tokens in [0, model_input_size]',
                     f'[{nnz_min}, {nnz_max}]',
                     f'range [{nnz_min}, {nnz_max}], model_input_size {mis}')
    if args.check_unique_cell_ids:
        check_unique(report, ds, data_path, hashes, args.chunk_rows)
        del hashes
    unmatched = sorted(set(heldout) - set(batch_counts))
    detail_ok = (f'{len(heldout)} ids; {int(val_mask.sum())} validation rows, '
                 f'{int(test_mask.sum())} test rows')
    detail_fail = f'{len(unmatched)} ids match no row: {short(unmatched)}'
    if want_heldout:
        report.check(not unmatched, 'heldout: every held-out id matches rows',
                     detail_ok, detail_fail)
    else:  # the full split ignores the held-out lists
        report.info('heldout: held-out ids vs rows', detail_fail if unmatched else detail_ok)

    reader = getattr(tu, '_read_precomputed_split', None)
    if reader is None:  # older TERRA: load_precomputed_split + the raw record
        def reader(path, dataset):
            indices = tu.load_precomputed_split(path, dataset)
            with open(path, 'rb') as f:
                raw = pickle.load(f)
            return indices, raw if isinstance(raw, dict) else None

    split_idx = {}
    # ---------------- held-out splits
    if want_heldout:
        label = np.full(n, -1, dtype=np.int8)
        records = {}
        for k, s in enumerate(HELDOUT_SPLITS):
            arr, record = load_split(report, reader, folder / f'{name}_precomputed_split_{s}.pkl',
                                     ds, s, n)
            if arr is None:
                continue
            records[s] = record
            taken = label[arr] != -1
            label[arr[~taken]] = k
            n_unique_new = int((label == k).sum())
            report.check(n_unique_new == len(arr),
                         f'indices: {s} no duplicates / no overlap with earlier splits',
                         '', f'{len(arr) - n_unique_new} rows duplicated or already in another split')
            split_idx[s] = arr
        if all(s in split_idx for s in HELDOUT_SPLITS):
            uncovered = int((label == -1).sum())
            report.check(uncovered == 0, 'indices: splits cover every row', f'{n} rows',
                         f'{uncovered} rows in no split')
            heldout_mask = val_mask | test_mask
            in_val, in_test, in_train = label == 1, label == 2, label == 0
            report.check(np.array_equal(in_val, val_mask),
                         'content: validation == rows of VAL batches',
                         f'{int(val_mask.sum())} rows',
                         f'{int((in_val & ~val_mask).sum())} extra, '
                         f'{int((val_mask & ~in_val).sum())} missing')
            report.check(np.array_equal(in_test, test_mask),
                         'content: test == rows of TEST batches',
                         f'{int(test_mask.sum())} rows',
                         f'{int((in_test & ~test_mask).sum())} extra, '
                         f'{int((test_mask & ~in_test).sum())} missing')
            n_bad = int((in_train & heldout_mask).sum())
            report.check(n_bad == 0, 'content: train holds no held-out row',
                         f'{int(in_train.sum())} train rows',
                         f'{n_bad} train rows are in held-out batches, e.g. rows '
                         f'{np.nonzero(in_train & heldout_mask)[0][:5].tolist()}')
            del label
            # TERRA's own training-time scan (forced: record not trusted)
            try:
                tu._check_split_excludes_heldout(
                    {'data': {'test_batch_ids': test_ids, 'val_batch_ids': val_ids}},
                    ds, [split_idx['train']], 'train split', record=None)
                report.add('PASS', 'content: TERRA training-time held-out scan of train')
            except ValueError as e:
                report.fail('content: TERRA training-time held-out scan of train', str(e)[:300])
        rec = records.get('train')
        if rec is not None:
            excl = rec.get('excluded_batch_ids')
            report.check(excl is not None and sorted(excl) == sorted(val_ids + test_ids),
                         'record: train excluded_batch_ids == VAL + TEST', f'{len(excl or [])} ids',
                         f'recorded {short(sorted(excl)) if excl is not None else None}; '
                         f'missing {short(sorted(set(heldout) - set(excl or [])))}; '
                         f'extra {short(sorted(set(excl or []) - set(heldout)))}')
        for s in ('validation', 'test'):
            if records.get(s) is not None and records[s].get('excluded_batch_ids') is not None:
                report.info(f'record: {s} excluded_batch_ids',
                            f"{short(records[s]['excluded_batch_ids'])}")

    # ---------------- full split (TERRA-112M)
    if want_full:
        arr, record = load_split(report, reader, folder / f'{name}_precomputed_split_full.pkl',
                                 ds, 'full', n)
        if arr is not None:
            report.check(len(arr) == n and np.array_equal(np.sort(arr), np.arange(n)),
                         'indices: full split holds every row exactly once', f'{n} rows',
                         f'{len(arr)} indices, {len(np.unique(arr))} distinct, dataset {n} rows')
            split_idx['full'] = arr
            if record is not None and record.get('excluded_batch_ids'):
                report.warn('record: full split records excluded batches',
                            short(record['excluded_batch_ids']))

    # ---------------- n_nonzero lists
    for s in split_idx:
        path = folder / f'{name}_n_nonzero_tokens_{s}.pkl'
        if not path.exists():
            if args.no_n_nonzero_lists:
                report.info(f'n_nonzero: {s} list not made',
                            f'{path.name} absent (--no_n_nonzero_lists)')
            else:
                report.fail(f'n_nonzero: {s}', f'{path} missing')
            continue
        with open(path, 'rb') as f:
            lst = pickle.load(f)
        try:
            arr = np.asarray(lst, dtype=np.int64)
        except Exception as e:
            report.fail(f'n_nonzero: {s} is a list of integers', str(e))
            continue
        del lst
        idx = split_idx[s]
        if not report.check(arr.ndim == 1 and len(arr) == len(idx),
                            f'n_nonzero: {s} one entry per split row', f'{len(arr)}',
                            f'{arr.shape} entries for {len(idx)} rows'):
            continue
        diff = arr != nnz_col[idx]
        report.check(not diff.any(), f'n_nonzero: {s} == dataset n_nonzero_tokens at split rows',
                     f'{len(arr)} rows', f'{int(diff.sum())} differ, first at positions '
                     f'{np.nonzero(diff)[0][:5].tolist()}')
        del arr, diff
        split_path = folder / f'{name}_precomputed_split_{s}.pkl'
        if split_path.exists() and os.path.getmtime(path) < os.path.getmtime(split_path):
            report.warn(f'n_nonzero: {s} older than its split', f'{path.name} is older than the split')

    # ---------------- recount of non-pad tokens + row structure
    pad = 0
    if args.token_dictionary_file_path:
        with open(args.token_dictionary_file_path, 'rb') as f:
            pad = pickle.load(f).get('<pad>', 0)
    checker = RowChecker(compact, pad, mis, n_neighs)
    if args.sample_rows:
        pools = [split_idx[s] for s in split_idx if len(split_idx[s])] or [np.arange(n)]
        rows = np.unique(np.concatenate(
            [rng.choice(p, size=min(args.sample_rows, len(p)), replace=False) for p in pools]
            + ([np.array([0, n - 1])] if n else [])))
        tbl = ds.select_columns(checker.columns).select(rows.tolist()) \
            .with_format('arrow')[:]
        recount = checker(tbl, rows)
        bad = recount != nnz_col[rows]
        report.check(not bad.any(), 'n_nonzero: recount of non-pad gene tokens (sampled rows)',
                     f'{len(rows)} rows', f'{int(bad.sum())} rows differ, e.g. rows '
                     f'{rows[bad][:5].tolist()}: column {nnz_col[rows][bad][:5].tolist()}, '
                     f'recount {recount[bad][:5].tolist()}')
        checker.report(report, 'sampled rows')
    else:
        n_bad, first = 0, []
        t0, last = time.time(), 0.0
        for start, tbl in iter_batches(ds, data_path, checker.columns, args.token_chunk_rows):
            rows = np.arange(start, start + len(tbl))
            recount = checker(tbl, rows)
            bad = recount != nnz_col[start:start + len(tbl)]
            n_bad += int(bad.sum())
            first += rows[bad][:5].tolist()[:5 - len(first)]
            if time.time() - last > 60:
                last = time.time()
                print(f'  recount {start + len(tbl)}/{n} rows ({last - t0:.0f}s, '
                      f'max RSS {rss_gb():.1f} GB)', flush=True)
        report.check(n_bad == 0, 'n_nonzero: recount of non-pad gene tokens (all rows)',
                     f'{checker.n_rows} rows', f'{n_bad} rows differ, e.g. {first}')
        checker.report(report, 'all rows')
        report.check(checker.n_rows == n, 'n_nonzero: every row recounted', f'{n}',
                     f'{checker.n_rows} rows read of {n}')

    # ---------------- stale pickles with the dataset's prefix
    wanted = (HELDOUT_SPLITS if want_heldout else ()) + (('full',) if want_full else ())
    expected = {f'{name}_precomputed_split_{s}.pkl' for s in wanted} | \
               {f'{name}_n_nonzero_tokens_{s}.pkl' for s in wanted}
    others = sorted(p for p in folder.glob(f'{name}_*')
                    if p.name not in expected
                    and (p.name[len(name) + 1:].startswith('precomputed')
                         or p.name[len(name) + 1:].startswith('n_nonzero')))
    n_stale = 0
    for p in others:
        rest = p.name[len(name) + 1:]
        try:
            with open(p, 'rb') as f:
                obj = pickle.load(f)
        except Exception as e:
            report.fail('stale: unreadable pickle', f'{p.name}: {e}')
            n_stale += 1
            continue
        if rest.startswith('precomputed_split') or rest.startswith('precomputed_epoch'):
            fmt_ok = isinstance(obj, dict) and obj.get('format') == tu.PRECOMPUTED_SPLIT_FORMAT
            del obj
            if not fmt_ok:
                report.fail('stale: legacy (bare list) split pickle', p.name)
                n_stale += 1
                continue
            try:
                reader(str(p), ds)
                report.info('stale: extra split file fits this dataset', p.name)
            except Exception as e:
                report.fail('stale: split pickle made for another tokenization', f'{p.name}: {e}')
                n_stale += 1
        elif rest.startswith('n_nonzero'):
            suffix = rest.removeprefix('n_nonzero_tokens_').removesuffix('.pkl')
            arr = np.asarray(obj) if not isinstance(obj, dict) else None
            del obj
            if suffix == 'full' and arr is not None and arr.ndim == 1:
                ok = len(arr) == n and np.array_equal(arr.astype(np.int64), nnz_col)
                if not ok:
                    report.fail('stale: n_nonzero_tokens_full does not fit this dataset', p.name)
                    n_stale += 1
                else:
                    report.info('stale: extra n_nonzero_tokens_full fits this dataset', p.name)
                continue
            split_path = folder / f'{name}_precomputed_split_{suffix}.pkl'
            if split_path.exists() and arr is not None and arr.ndim == 1:
                try:
                    sidx = np.asarray(reader(str(split_path), ds)[0], dtype=np.int64)
                    if len(sidx) == len(arr) and np.array_equal(arr.astype(np.int64), nnz_col[sidx]):
                        report.info('stale: extra n_nonzero list fits its split', p.name)
                        continue
                except Exception:
                    pass
            report.fail('stale: n_nonzero pickle that fits no split of this dataset', p.name)
            n_stale += 1
        else:
            report.warn('stale: other pickle with the dataset prefix', p.name)
    report.check(n_stale == 0, 'stale: no legacy or stale split / n_nonzero pickle',
                 f'{len(others)} other files with the prefix', f'{n_stale} stale files')
    json_files = list(Path(data_path).glob('*.json'))
    ds_mtime = max(os.path.getmtime(p) for p in json_files) if json_files else 0.0
    for s in split_idx:
        p = folder / f'{name}_precomputed_split_{s}.pkl'
        if p.exists() and os.path.getmtime(p) < ds_mtime:
            report.warn(f'stale: {s} split older than the dataset', p.name)

    # ---------------- tokenizer settings (optional)
    if params is None:
        report.warn('tokenizer: terra_tokenizer_params.json', 'absent; settings not checked')
    else:
        report.info('tokenizer: settings', f"stable_rank_ties={params.get('stable_rank_ties')}, "
                    f"rank_tie_break_norm_factor={params.get('rank_tie_break_norm_factor')}, "
                    f"norm_factor_file_path={params.get('norm_factor_file_path')}")
        if args.norm_factor_csv:
            report.check(params.get('stable_rank_ties') is True
                         and params.get('rank_tie_break_norm_factor') == args.expect_tie_break,
                         'tokenizer: stable_rank_ties + rank_tie_break_norm_factor',
                         f'{args.expect_tie_break}',
                         f"stable_rank_ties={params.get('stable_rank_ties')}, "
                         f"rank_tie_break_norm_factor={params.get('rank_tie_break_norm_factor')}")
            digest = sha256(args.norm_factor_csv)
            report.check(params.get('norm_factor_sha256') == digest,
                         'tokenizer: tokenized with this norm-factor csv (sha256)', digest[:12],
                         f"recorded {str(params.get('norm_factor_sha256'))[:12]} "
                         f"({params.get('norm_factor_file_path')}), csv {digest[:12]}")
        if args.token_dictionary_file_path and params.get('token_dictionary_sha256'):
            digest = sha256(args.token_dictionary_file_path)
            report.check(params['token_dictionary_sha256'] == digest,
                         'tokenizer: token dictionary sha256', digest[:12],
                         f"recorded {params['token_dictionary_sha256'][:12]}, given {digest[:12]}")

    # ---------------- training configs (optional)
    for cfg_path in args.training_config:
        check_training_config(report, cfg_path, args, tu, reader, ds, data_path, folder,
                              name, split_idx, heldout, batch_counts)

    # ---------------- norm-factor json (optional)
    if args.norm_factor_json:
        with open(args.norm_factor_json) as f:
            nf = json.load(f)
        report.check(sorted(nf.get('excluded_batch_ids') or []) == heldout,
                     'json: norm factors exclude exactly the held-out batches', f'{len(heldout)} ids',
                     f"json excludes {len(nf.get('excluded_batch_ids') or [])} ids; "
                     f"lists {len(heldout)}")
        if 'train' in split_idx:
            if nf.get('n_cells') == len(split_idx['train']) and \
                    nf.get('n_cells_excluded') == int((val_mask | test_mask).sum()):
                report.add('PASS', 'json: train / held-out rows == norm-factor n_cells / n_cells_excluded',
                           f"{nf.get('n_cells')} / {nf.get('n_cells_excluded')}")
            else:
                report.warn('json: train / held-out rows vs norm-factor n_cells / n_cells_excluded',
                            f"rows {len(split_idx['train'])} / {int((val_mask | test_mask).sum())}, "
                            f"json {nf.get('n_cells')} / {nf.get('n_cells_excluded')} (differs if "
                            'the tokenized corpus holds other cells than the silver cohort)')
    return finish(report, args, t_start)


def batch_ids_arrow(cell_ids):
    """Batch ids '<part0>_<part1>' as split_dataset.py derives them
    (cell_id.split('_')[0] + '_' + cell_id.split('_')[1]); null if the id has
    no '_' (split_dataset.py would fail on it)."""
    m = pc.extract_regex(cell_ids, r'^(?P<b>[^_]*_[^_]*)')
    return pc.struct_field(m, [0])


def check_unique(report, ds, data_path, hashes, chunk_rows):
    """Cell ids unique: duplicated 64-bit hashes, then the exact ids behind them."""
    sorted_h = np.sort(hashes)
    dup_mask = sorted_h[1:] == sorted_h[:-1]
    if not dup_mask.any():
        report.add('PASS', 'dataset: cell ids unique', f'{len(hashes)} rows (64-bit hashes)')
        return
    dup_hashes = np.unique(sorted_h[1:][dup_mask])
    del sorted_h
    counts = Counter()
    for start, tbl in iter_batches(ds, data_path, ['cell_id'], chunk_rows):
        m = np.isin(hashes[start:start + len(tbl)], dup_hashes)
        if m.any():
            ids = tbl.column('cell_id').to_numpy(zero_copy_only=False)[m]
            counts.update(ids.tolist())
    dups = {k: v for k, v in counts.items() if v > 1}
    report.check(not dups, 'dataset: cell ids unique',
                 f'{len(hashes)} rows ({len(dup_hashes)} hash collisions, ids distinct)',
                 f'{sum(dups.values()) - len(dups)} duplicated rows, e.g. '
                 f'{short(sorted(dups.items())[:5], 5)}')


def load_split(report, reader, path, ds, s, n):
    """Indices (int64 array) and record (without the indices) of one split."""
    if not path.exists():
        report.fail(f'load: {s}', f'{path} missing')
        return None, None
    try:
        idx, record = reader(str(path), ds)
    except Exception as e:  # TERRA refuses the split for this dataset
        report.fail(f'load: {s} split loads for this dataset (TERRA)', f'{type(e).__name__}: {e}')
        return None, None
    if record is None:
        report.fail(f'load: {s} split in TERRA format', f'{path.name} is a legacy bare list')
    else:
        report.add('PASS', f'load: {s} split loads for this dataset (TERRA)',
                   f"{record.get('format')}, n_rows {record.get('n_rows')}")
        # Keep the record without its index list (a 100M-entry list is ~4 GB)
        record = {k: v for k, v in record.items() if k != 'indices'}
    try:
        arr = np.asarray(idx)
    except Exception as e:
        report.fail(f'indices: {s}', f'not an index list: {e}')
        return None, record
    del idx
    if arr.ndim != 1 or not (np.issubdtype(arr.dtype, np.integer) or len(arr) == 0):
        report.fail(f'indices: {s} is one list of integers', f'ndim {arr.ndim}, dtype {arr.dtype}')
        return None, record
    arr = arr.astype(np.int64)
    in_range = (arr >= 0) & (arr < n)
    if not report.check(in_range.all(), f'indices: {s} in range', f'{len(arr)} rows',
                        f'{int((~in_range).sum())} out of [0, {n})'):
        return None, record
    if not (np.diff(arr) > 0).all():
        report.warn(f'indices: {s} ascending', 'not sorted (split_dataset.py writes ascending rows)')
    print(f'  {s}: {len(arr)} rows (max RSS {rss_gb():.1f} GB)', flush=True)
    return arr, record


def check_training_config(report, cfg_path, args, tu, reader, ds, data_path, folder,
                          name, split_idx, heldout, batch_counts):
    """A TERRA model config trains on this dataset with these files, and
    train()'s own held-out check (data.test_batch_ids / val_batch_ids) passes."""
    import yaml
    label = f'config {Path(cfg_path).name}'
    cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
    data = dict(cfg.get('data') or {})
    root = args.terra_data_dir or os.environ.get('TERRA_DATA_DIR')

    def expand(value):
        if isinstance(value, str) and '${TERRA_DATA_DIR}' in value:
            if root is None:
                return None
            value = value.replace('${TERRA_DATA_DIR}', root)
        return value

    def same(a, b):
        return a is not None and os.path.realpath(str(a).rstrip('/')) == \
            os.path.realpath(str(b).rstrip('/'))

    if root is None:
        report.warn(f'{label}: paths', 'TERRA_DATA_DIR unset and no --terra_data_dir: '
                    '${TERRA_DATA_DIR} paths not compared')
    tdp = expand(data.get('tokenized_data_folder_path'))
    if tdp is not None:
        report.check(same(tdp, data_path), f'{label}: tokenized_data_folder_path is this dataset',
                     '', f'config {tdp}, validated {data_path}')
    split_file = expand(data.get('precomputed_split'))
    split_name = None
    if split_file is not None:
        for s in split_idx:
            if same(split_file, folder / f'{name}_precomputed_split_{s}.pkl'):
                split_name = s
        report.check(split_name is not None, f'{label}: precomputed_split is a validated split',
                     f'{split_name}', f'config {split_file}; validated '
                     f'{[f"{name}_precomputed_split_{s}.pkl" for s in split_idx]} in {folder}')
    nnz_file = expand(data.get('precomputed_n_nonzero_tokens'))
    if nnz_file is not None and split_name is not None:
        report.check(same(nnz_file, folder / f'{name}_n_nonzero_tokens_{split_name}.pkl'),
                     f'{label}: precomputed_n_nonzero_tokens belongs to that split',
                     '', f'config {nnz_file}, split {split_name}')
    cfg_ids = set(data.get('test_batch_ids') or []) | set(data.get('val_batch_ids') or [])
    if split_name is not None:
        with_rows = sorted(b for b in cfg_ids if batch_counts.get(b))
        try:
            idx, record = reader(str(folder / f'{name}_precomputed_split_{split_name}.pkl'), ds)
            # train() trusts the record when it excludes the config's ids
            tu._check_split_excludes_heldout({'data': data}, ds, [split_idx[split_name]],
                                             f'{split_name} split', record=record)
            report.add('PASS', f"{label}: train()'s held-out check accepts the {split_name} split",
                       f'{len(cfg_ids)} config ids ({len(with_rows)} with rows)')
        except ValueError as e:
            report.fail(f"{label}: train()'s held-out check accepts the {split_name} split",
                        str(e)[:300])
    missing = sorted(set(heldout) - cfg_ids)
    if cfg_ids and missing and split_name != 'full':
        report.warn(f'{label}: data.test_batch_ids / val_batch_ids hold {len(cfg_ids)} of the '
                    f'{len(heldout)} held-out ids',
                    f'train() checks a split only against the config ids; the split itself '
                    f'excludes all {len(heldout)} (checked above). Missing e.g. {short(missing, 5)}')


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
    p.add_argument('--tokenized_data_folder_path', required=True)
    p.add_argument('--tokenizer_folder_path', required=True)
    p.add_argument('--splits', choices=('heldout', 'full', 'both'), default='heldout',
                   help='heldout: train/validation/test (TERRA-96M); full: the full split '
                        '(TERRA-112M); both.')
    p.add_argument('--heldout_module_dir', default=None,
                   help='Directory with heldout_batch_ids.py (default: the parent of '
                        'this directory, i.e. analysis/data_preparation, '
                        'else this directory).')
    p.add_argument('--token_dictionary_file_path', default=None,
                   help='For the <pad> token id (default 0) and the sha256 check.')
    p.add_argument('--norm_factor_csv', default=None,
                   help='Check that the dataset was tokenized with this csv.')
    p.add_argument('--norm_factor_json', default=None)
    p.add_argument('--expect_tie_break', default='tie_break_mean')
    p.add_argument('--expect_param', action='append', default=[], metavar='KEY=VALUE',
                   help='Tokenizer setting the dataset must record (VALUE as JSON, e.g. '
                        'n_neighs=16, pad_empty_neighbors=true, rank_gene_norm_method=null). '
                        'Repeatable.')
    p.add_argument('--training_config', action='append', default=[], metavar='YAML',
                   help='TERRA model config to check against the validated files: its '
                        'tokenized_data_folder_path, precomputed_split and '
                        'precomputed_n_nonzero_tokens, and train()\'s held-out check with its '
                        'data.test_batch_ids / val_batch_ids. Repeatable.')
    p.add_argument('--terra_data_dir', default=None,
                   help='Value of ${TERRA_DATA_DIR} in configs (default: the environment).')
    p.add_argument('--sample_rows', type=int, default=20000,
                   help='Rows per split whose gene tokens are recounted (0: all rows).')
    p.add_argument('--full', action='store_true', help='Same as --sample_rows 0.')
    p.add_argument('--check_unique_cell_ids', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--chunk_rows', type=int, default=1_000_000)
    p.add_argument('--token_chunk_rows', type=int, default=2_000,
                   help='Rows per recount chunk (padded 16x512 rows: ~70 KB of tokens each).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_n_nonzero_lists', action='store_true',
                   help='Do not require the n_nonzero_tokens lists (compact '
                        'corpora: training derives the lengths).')
    p.add_argument('--report_json', default=None)
    a = p.parse_args()
    if a.full:
        a.sample_rows = 0
    try:
        code = main(a)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        print('RESULT: ERROR (the validator crashed; this is not a validation result)')
        code = 2
    sys.exit(code)
