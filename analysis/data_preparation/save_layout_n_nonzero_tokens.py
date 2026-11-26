#!/usr/bin/env python
"""
Precompute, once per corpus and split, the per-cell lengths that TERRA's
length-grouped sampler groups the cells of a compact corpus by, for the
layouts the model configs read (e.g. 10 neighbours x 256 genes of the
16 x 512 hst_corpus_110m), so that training is given them with
data.precomputed_n_nonzero_tokens.

Why: without a list, every training and evaluation job computes them at
start-up from every row of the corpus (terra.datasets.compact.
layout_n_nonzero_tokens in CellBaseDataset.__init__), one small read per
record batch, one at a time. On Lustre that took about 2 h 50 min for the
115,488 record batches of hst_corpus_110m (job 246066, 2026-11-26), in every
rank. This job reads them once, with --workers processes in parallel, for all
layouts at once.

The lengths are TERRA's own: layout_n_nonzero_tokens of each arrow file of
the corpus (a row's lengths depend on that row only), in the corpus's row
order, then the split's rows in the split's order (the rows of
dataset.select(split) that prepare_dataset trains on). Each list passes the
check train() runs with a given list (check_n_nonzero_tokens, 1000 rows)
before it is written.

Layouts are given as <neighbours>x<genes> (e.g. 10x256): n_segments =
neighbours + 1, seq_len_cell = genes, tokenized_seq_len_cell = the corpus's
segment length (model_input_size / (n_neighs + 1) of its tokenizer record),
truncate_neighbors = neighbours < the corpus's n_neighs. They must match the
model config; make_test_run_configs.py checks that against the .json record
written next to each list.

Outputs, in --output_folder_path (default: --tokenizer_folder_path, next to
the split pickles), per layout:
  <corpus name>_n_nonzero_tokens_<split>_<layout>.pkl
      a pickled array.array('i') (4 bytes per cell): train() loads it with
      pickle.load and the cell dataset tests it with `if list:`, which a
      numpy array fails
  <corpus name>_n_nonzero_tokens_<split>_<layout>.pkl.json
      what it was made from (corpus, split, layout, counts)
Existing outputs are kept unless --overwrite.

It also reports how many rows do not have n_neighs + 1 segments. With none,
TERRA's stride-from-record change takes the corpus's segment length from its
record instead of reading every row; otherwise TERRA still reads every row
for it at start-up.

usage (farm22 CPU node, TERRA venv): see save_layout_n_nonzero_tokens.lsf
"""
import argparse
import array
import datetime
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import platform
import re
import subprocess
import sys
import time

import numpy as np

logger = logging.getLogger('save_layout_n_nonzero_tokens')

FORMAT = 'terra_layout_n_nonzero_tokens_v1'
LAYOUT_KEYS = ('seq_len_cell', 'n_segments', 'tokenized_seq_len_cell',
               'truncate_neighbors')


def fail(message):
    logger.error(message)
    sys.exit(1)


def parse_layout(text, n_neighs, stride):
    """The cell-dataset layout settings of '<neighbours>x<genes>'."""
    match = re.fullmatch(r'(\d+)x(\d+)', text)
    if not match:
        raise ValueError(f'layout {text!r}: use <neighbours>x<genes>, '
                         'e.g. 10x256.')
    neighbours, genes = int(match.group(1)), int(match.group(2))
    if neighbours > n_neighs:
        raise ValueError(f'layout {text!r}: the corpus has {n_neighs} '
                         'neighbours per cell.')
    if not 1 <= genes <= stride:
        raise ValueError(f'layout {text!r}: the corpus stores {stride} genes '
                         'per segment.')
    return {'name': text, 'seq_len_cell': genes,
            'n_segments': neighbours + 1, 'tokenized_seq_len_cell': stride,
            'truncate_neighbors': neighbours < n_neighs}


def layout_kwargs(layout):
    return {k: layout[k] for k in LAYOUT_KEYS}


def file_lengths(task):
    """
    Lengths of every row of one arrow file in every layout, with TERRA's
    layout_n_nonzero_tokens, plus what the rows look like (segments per row,
    rows that TERRA counts one by one).
    """
    path, layouts, n_seg_full, stride = task
    import datasets
    import pyarrow.compute as pc
    from terra.datasets.compact import layout_n_nonzero_tokens
    t0 = time.time()
    ds = datasets.Dataset.from_file(path)
    lengths = {layout['name']: np.asarray(
        layout_n_nonzero_tokens(ds, **layout_kwargs(layout)), dtype=np.int32)
        for layout in layouts}
    table = ds.data.table
    seg = table.column('segment_lengths')
    n_seg = np.asarray(pc.list_value_length(seg).to_numpy(
        zero_copy_only=False), dtype=np.int64)
    values = np.asarray(pc.list_flatten(seg).to_numpy(zero_copy_only=False),
                        dtype=np.int64)
    n = len(n_seg)
    stored = np.bincount(np.repeat(np.arange(n), n_seg), weights=values,
                         minlength=n).astype(np.int64)
    nnz = np.asarray(table.column('n_nonzero_tokens').to_numpy(),
                     dtype=np.int64)
    mis = np.asarray(table.column('model_input_size').to_numpy(),
                     dtype=np.int64)
    not_full = n_seg != n_seg_full
    # rows TERRA does not count from the segment lengths alone (it rebuilds
    # their padded rows): other segment lengths than the stride, or stored
    # entries other than the gene tokens
    not_exact = (not_full | (mis // np.maximum(n_seg, 1) != stride)
                 | (stored != nnz))
    # the condition of the stride-from-record change, per record batch
    # (terra.datasets.compact._segment_length_without_scan)
    batches_ok = all(c.offset == 0 and not c.null_count
                     and len(c.values) == n_seg_full * len(c)
                     for c in seg.chunks)
    return {'path': path, 'n_rows': n, 'lengths': lengths,
            'seg_min': int(n_seg.min()) if n else None,
            'seg_max': int(n_seg.max()) if n else None,
            'rows_not_full': int(not_full.sum()),
            'rows_not_exact': int(not_exact.sum()),
            'model_input_sizes': sorted(int(x) for x in np.unique(mis)),
            'batches_ok': batches_ok, 'seconds': time.time() - t0}


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def git_head(folder):
    try:
        return subprocess.run(
            ['git', '-C', folder, 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_atomic(path, write):
    tmp = f'{path}.tmp-{os.getpid()}'
    try:
        with open(tmp, 'wb') as f:
            write(f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    import datasets
    from terra.datasets.compact import check_n_nonzero_tokens, is_compact
    from terra.datasets.utils import load_precomputed_split
    from terra.tokenizers.params import (TOKENIZER_PARAMS_FILE,
                                         read_tokenizer_params)

    corpus = os.path.abspath(args.tokenized_data_folder_path.rstrip('/'))
    name = os.path.basename(corpus).removesuffix('.dataset')
    split_path = args.precomputed_split or os.path.join(
        args.tokenizer_folder_path,
        f'{name}_precomputed_split_{args.split}.pkl')
    out_dir = args.output_folder_path or args.tokenizer_folder_path
    if not os.path.isfile(split_path):
        fail(f"the split '{split_path}' does not exist.")

    # ---- the corpus's layout, from its tokenizer record
    record = read_tokenizer_params(corpus)
    if record is None:
        fail(f"'{corpus}' has no {TOKENIZER_PARAMS_FILE}.")
    if (record.get('tokenizer') != 'CellGraphTokenizer'
            or record.get('storage_format') != 'compact'
            or not record.get('n_neighs')
            or not record.get('model_input_size')):
        fail(f"'{corpus}' is not a compact cell-graph corpus with a k-NN "
             f"layout (record: tokenizer {record.get('tokenizer')!r}, "
             f"storage_format {record.get('storage_format')!r}, n_neighs "
             f"{record.get('n_neighs')!r}).")
    n_neighs = int(record['n_neighs'])
    n_seg_full = n_neighs + 1
    stride = int(record['model_input_size'] / n_seg_full)
    try:
        layouts = [parse_layout(t, n_neighs, stride) for t in args.layouts]
    except ValueError as e:
        fail(str(e))
    if len({layout['name'] for layout in layouts}) != len(layouts):
        fail('--layouts lists a layout twice.')
    outputs = {layout['name']: os.path.join(
        out_dir, f"{name}_n_nonzero_tokens_{args.split}_{layout['name']}.pkl")
        for layout in layouts}
    for path in outputs.values():
        for p in (path, path + '.json'):
            if os.path.exists(p) and not args.overwrite:
                fail(f"'{p}' exists (--overwrite replaces it).")
    if not os.path.isdir(out_dir):
        fail(f"--output_folder_path '{out_dir}' is not a folder.")
    logger.info(f'corpus {corpus}: {n_neighs} neighbours, {stride} genes per '
                f'segment; layouts {[layout["name"] for layout in layouts]}')

    # ---- corpus rows and the split (checked against the corpus as training
    # checks it: number of rows and the recorded cell ids)
    t0 = time.time()
    dataset = datasets.load_from_disk(corpus)
    if not is_compact(dataset):
        fail(f"'{corpus}' does not have compact rows.")
    files = [f['filename'] for f in dataset.cache_files]
    logger.info(f'{len(dataset):,} rows in {len(files):,} files '
                f'({time.time() - t0:.0f} s)')
    t0 = time.time()
    indices = load_precomputed_split(split_path, dataset)
    if len(indices) and isinstance(indices[0], (list, tuple, np.ndarray)):
        fail(f"'{split_path}' holds one index list per epoch; give a single "
             'split.')
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0 or idx.min() < 0 or idx.max() >= len(dataset):
        fail(f"'{split_path}' has no rows or rows outside the corpus.")
    logger.info(f'split {split_path}: {len(idx):,} rows '
                f'({time.time() - t0:.0f} s)')

    # ---- every row of every file, all layouts, files in parallel
    t0 = time.time()
    tasks = [(f, layouts, n_seg_full, stride) for f in files]
    results, rows_done = {}, 0
    step = max(1, len(files) // 40)
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(file_lengths, tasks), 1):
            results[res['path']] = res
            rows_done += res['n_rows']
            if i % step == 0 or i == len(files):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(files) - i)
                logger.info(f'{i:,}/{len(files):,} files, {rows_done:,} rows, '
                            f'{elapsed:.0f} s elapsed, ~{eta:.0f} s left')
    full = {layout['name']: np.empty(len(dataset), dtype=np.int32)
            for layout in layouts}
    pos = 0
    for f in files:
        res = results[f]
        for key in full:
            full[key][pos:pos + res['n_rows']] = res['lengths'][key]
        pos += res['n_rows']
    if pos != len(dataset):
        fail(f'the files hold {pos:,} rows, the corpus {len(dataset):,}.')
    rows_not_full = sum(r['rows_not_full'] for r in results.values())
    rows_not_exact = sum(r['rows_not_exact'] for r in results.values())
    seg_min = min(r['seg_min'] for r in results.values() if r['n_rows'])
    seg_max = max(r['seg_max'] for r in results.values() if r['n_rows'])
    mis_values = sorted({m for r in results.values()
                         for m in r['model_input_sizes']})
    batches_ok = all(r['batches_ok'] for r in results.values())
    per_file = np.array([r['seconds'] for r in results.values()])
    logger.info(f'all rows read in {time.time() - t0:.0f} s with '
                f'{args.workers} workers (per file: median '
                f'{np.median(per_file):.1f} s, max {per_file.max():.1f} s)')
    logger.info(f'segments per row: {seg_min}-{seg_max}; rows without '
                f'{n_seg_full} segments: {rows_not_full:,}; rows counted one '
                f'by one: {rows_not_exact:,}; model_input_size values: '
                f'{mis_values}')
    if batches_ok:
        logger.info(f'every row has {n_seg_full} segments: TERRA with the '
                    'stride-from-record change reads the segment length from '
                    'the corpus record, without reading the rows.')
    else:
        logger.warning(f'not every row has {n_seg_full} segments: TERRA still '
                       'reads every row at start-up for the segment length '
                       '(data_segment_length).')

    # ---- the split's rows, checked as train() checks a given list
    t0 = time.time()
    train_ds = dataset.select(indices)
    del indices
    logger.info(f'selected the split rows ({time.time() - t0:.0f} s)')
    meta = {
        'format': FORMAT,
        'tokenized_data_folder_path': corpus,
        'corpus_n_rows': len(dataset),
        'corpus_record_sha256': sha256(
            os.path.join(corpus, TOKENIZER_PARAMS_FILE)),
        'precomputed_split': os.path.abspath(split_path),
        'split': args.split,
        'split_size_bytes': os.path.getsize(split_path),
        'rows_without_full_segments': rows_not_full,
        'rows_counted_one_by_one': rows_not_exact,
        'stride_from_record_applies': batches_ok,
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec='seconds'),
        'host': platform.node(),
        'lsf_job_id': os.environ.get('LSB_JOBID'),
        'terra_commit': git_head(os.path.dirname(
            sys.modules['terra'].__file__)),
        'script_commit': git_head(os.path.dirname(os.path.abspath(__file__))),
    }
    for layout in layouts:
        key = layout['name']
        lengths = full[key][idx]
        out = array.array('i')
        if out.itemsize != 4:
            fail(f"array.array('i') has {out.itemsize} bytes per item here.")
        out.frombytes(lengths.astype(np.int32).tobytes())
        t1 = time.time()
        # the check train() runs at start-up with a given list
        check_n_nonzero_tokens(out, train_ds, **layout_kwargs(layout))
        path = outputs[key]
        write_atomic(path, lambda f: pickle.dump(
            out, f, protocol=pickle.HIGHEST_PROTOCOL))
        record_out = dict(meta, layout=dict(
            layout_kwargs(layout), name=key, n_neighs=layout['n_segments'] - 1),
            n_cells=len(lengths),
            sum=int(lengths.sum(dtype=np.int64)),
            mean=float(lengths.mean()), min=int(lengths.min()),
            max=int(lengths.max()))
        write_atomic(path + '.json', lambda f: f.write(
            json.dumps(record_out, indent=2).encode()))
        logger.info(f'{key}: {len(lengths):,} cells (mean {lengths.mean():.1f}, '
                    f'max {lengths.max()}), checked at 1000 rows in '
                    f'{time.time() - t1:.0f} s, wrote {path}')
    print('\nAdd to a model config that reads one of these layouts (data '
          'section):')
    for layout in layouts:
        print(f"  {layout['name']}: precomputed_n_nonzero_tokens: "
              f"{outputs[layout['name']]}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tokenized_data_folder_path', required=True,
                   help='The compact corpus (.dataset folder).')
    p.add_argument('--tokenizer_folder_path', required=True,
                   help='Folder of the split pickles (<corpus name>_'
                        'precomputed_split_<split>.pkl).')
    p.add_argument('--split', default='train',
                   help='Split name in the split pickle name (default: train).')
    p.add_argument('--precomputed_split', default=None,
                   help='Split pickle (default: from --tokenizer_folder_path '
                        'and --split).')
    p.add_argument('--layouts', nargs='+', required=True,
                   help='<neighbours>x<genes>, e.g. 16x512 10x256.')
    p.add_argument('--output_folder_path', default=None,
                   help='Default: --tokenizer_folder_path.')
    p.add_argument('--workers', type=int, default=32,
                   help='Processes reading the corpus files (default 32).')
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
