#!/usr/bin/env python
"""
Check one tokenized dataset in the compact storage format (a per-dataset
part written by create_tokenized_dataset.py, or the concatenated corpus) and
print a short report. Read-only.

Checks (FAIL makes the exit code 1):
- the settings record (terra_tokenizer_params.json) says compact, and the
  columns are compact;
- every row has one segment length per segment, no segment is longer than
  model_input_size / n_segments, and n_nonzero_tokens equals the stored
  entries of the row;
- cell ids are unique;
- on --n_sample random rows: every stored token is a gene token of the
  token dictionary, unique within its segment; every stored value is finite
  and > 0; within each segment the values do not increase (genes ranked by
  count); genes with equal values (equal counts) are ordered by the
  norm-factor column the record names (rank_tie_break_norm_factor,
  ascending), then by token id; each row rebuilds (TERRA
  expand_compact_item) to model_input_size tokens with as many non-zero
  tokens as stored entries;
- with --silver_dir: the part has no more rows than the silver files have
  cells.

Reports: rows, size on disk and bytes per row, genes per cell (index
segment: mean, median, share at the segment length), neighbour segments
(mean genes, share empty), share of stored genes that never occur in the
training cells (tie-break key = fill value), and the dropped cells.

Usage (farm, TERRA venv, from the terra-reproducibility root):
    D=/lustre/scratch126/cellgen/lotfollahi/DATASETS
    python analysis/data_preparation/checks/check_tokenized_part.py \\
        --dataset_path $D/gold/cell-graph-tokenizer/xhs49-1b_1p/xhs49-1b_1p_32_None_None_None_None_None_shifted_log_knn_16_8704.dataset \\
        --silver_dir $D/silver/xhs49-1b_1p
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TOKEN_DICT = ('/lustre/scratch126/cellgen/lotfollahi/DATASETS/'
                      'tokenizer/token_dictionary_homo_sapiens.pkl')


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1000 or unit == 'TB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{n} B'
        n /= 1000.0


def list_lengths(table, name):
    """Per-row lengths of a list column of an Arrow table, read chunk by chunk
    from the list offsets (the values are not read). Combining the chunks
    first would overflow the 32-bit list offsets of a part with more than
    2^31 - 1 stored entries."""
    lengths = [c.value_lengths().to_numpy(zero_copy_only=False)
               for c in table.column(name).chunks]
    return (np.concatenate(lengths) if lengths
            else np.zeros(0, dtype=np.int32))


def list_column(table, name):
    """(flat values, per-row lengths) of a list column of an Arrow table,
    chunk by chunk (see list_lengths)."""
    chunks = table.column(name).chunks
    if not chunks:
        return np.zeros(0), np.zeros(0, dtype=np.int32)
    values = np.concatenate([c.flatten().to_numpy(zero_copy_only=False)
                             for c in chunks])
    return values, list_lengths(table, name)


def silver_cells(silver_dir):
    import h5py
    total, files = 0, sorted(Path(silver_dir).glob('*.h5ad'))
    for f in files:
        with h5py.File(f, 'r') as h:
            obs = h['obs']
            if isinstance(obs, h5py.Group) and '_index' in obs.attrs:
                idx = obs.attrs['_index']
                idx = idx.decode() if isinstance(idx, bytes) else idx
                total += len(obs[idx])
            else:
                X = h['X']
                shape = (X.shape if isinstance(X, h5py.Dataset)
                         else X.attrs['shape'])
                total += int(shape[0])
    return total, len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--token_dictionary_file_path', default=DEFAULT_TOKEN_DICT)
    ap.add_argument('--norm_factor_csv', default=None,
                    help='Default: the norm-factor file the record names.')
    ap.add_argument('--silver_dir', default=None,
                    help='Silver folder of the dataset (compare cell counts).')
    ap.add_argument('--n_sample', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from datasets import load_from_disk

    path = Path(args.dataset_path)
    fails, notes = [], []
    record = json.loads((path / 'terra_tokenizer_params.json').read_text())
    features = json.loads((path / 'dataset_info.json').read_text())['features']
    if record.get('storage_format') != 'compact':
        fails.append(f"record storage_format = {record.get('storage_format')!r}")
    if 'segment_lengths' not in features:
        fails.append('no segment_lengths column: the rows are not compact')
        print('\n'.join('FAIL ' + f for f in fails))
        return 1

    keys = ('storage_format', 'n_neighs', 'model_input_size',
            'count_count_norm_method', 'stable_rank_ties',
            'rank_tie_break_norm_factor', 'pad_empty_neighbors',
            'eliminate_stored_zeros', 'include_zero_expr_genes',
            'norm_factor_file_path', 'norm_factor_sha256', 'terra_version')
    print(f'Dataset: {path}')
    print('Record: ' + ', '.join(f'{k}={record.get(k)}' for k in keys))

    ds = load_from_disk(str(path))
    n = len(ds)
    disk = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    print(f'Rows: {n:,}; on disk {human(disk)} ({disk / max(n, 1):,.0f} B '
          'per row)')
    gt = features['gene_tokens']
    print(f"Columns: {list(features)}; gene_tokens "
          f"{gt.get('feature', gt).get('dtype')}")

    table = ds.data.table
    mis = int(record['model_input_size'])
    n_seg = int(record['n_neighs']) + 1
    seq = mis // n_seg

    # Segment lengths of every row
    seg_vals, seg_rows = list_column(table, 'segment_lengths')
    if not (seg_rows == n_seg).all():
        bad = int((seg_rows != n_seg).sum())
        notes.append(f'{bad} rows do not have {n_seg} segment lengths (rows '
                     'with fewer neighbours, pad_empty_neighbors false)')
    seg = seg_vals.astype(np.int64)
    if seg.size and seg.max() > seq:
        fails.append(f'a segment holds {seg.max()} entries, more than the '
                     f'segment length {seq}')
    cs = np.r_[0, np.cumsum(seg)]
    ends = np.cumsum(seg_rows)
    row_sums = cs[ends] - cs[ends - seg_rows]
    nnz = table.column('n_nonzero_tokens').to_numpy()
    tok_rows = list_lengths(table, 'gene_tokens')
    if not np.array_equal(tok_rows, row_sums):
        fails.append(f'{int((tok_rows != row_sums).sum())} rows store another '
                     'number of tokens than their segment lengths add up to')
    if not np.array_equal(nnz, row_sums):
        fails.append(f'n_nonzero_tokens differs from the stored entries in '
                     f'{int((nnz != row_sums).sum())} rows')

    # Genes per cell (index segment) and neighbour segments
    if (seg_rows == n_seg).all() and n:
        seg2 = seg.reshape(n, n_seg)
        idx_seg, nb = seg2[:, 0], seg2[:, 1:]
        print(f'Index cell: mean {idx_seg.mean():.1f} genes, median '
              f'{np.median(idx_seg):.0f}, min {idx_seg.min()}, max '
              f'{idx_seg.max()}; {100 * (idx_seg == seq).mean():.1f}% of '
              f'cells fill all {seq} positions; '
              f'{100 * (idx_seg < 10).mean():.2f}% have < 10 genes')
        print(f'Neighbour segments: mean {nb.mean():.1f} genes, '
              f'{100 * (nb == 0).mean():.2f}% empty')
        if (idx_seg == 0).any():
            notes.append(f'{int((idx_seg == 0).sum())} cells have no gene')

    # Unique cell ids
    cell_ids = table.column('cell_id').to_pandas()
    n_dup = int(cell_ids.duplicated().sum())
    if n_dup:
        fails.append(f'{n_dup} duplicated cell ids')

    # Sampled rows: tokens, values, ranking and tie-break order
    with open(args.token_dictionary_file_path, 'rb') as fh:
        token_dict = pickle.load(fh)
    n_tok = max(token_dict.values()) + 1
    is_gene = np.zeros(n_tok, dtype=bool)
    gene_of = np.empty(n_tok, dtype=object)
    for k, v in token_dict.items():
        if str(k).startswith('ENS'):
            is_gene[v] = True
            gene_of[v] = k
    tb_col = record.get('rank_tie_break_norm_factor')
    nf_path = args.norm_factor_csv or record.get('norm_factor_file_path')
    tb, fill = None, None
    if tb_col and nf_path:
        nf = pd.read_csv(nf_path)
        nf = nf.set_index('gene_id' if 'gene_id' in nf.columns
                          else nf.columns[0])
        tb = np.full(n_tok, np.nan)
        for g, val in nf[tb_col].items():
            if g in token_dict:
                tb[token_dict[g]] = val
        # Key of genes never detected in the training cells (max + 1)
        fill = None
        nf_json = Path(nf_path).with_suffix('.json')
        if nf_json.exists():
            fill = json.loads(nf_json.read_text()).get('tie_break_mean_fill')
    elif record.get('stable_rank_ties'):
        notes.append('no tie-break column recorded: ties checked by token id')

    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(n, size=min(args.n_sample, n), replace=False))
    sub = ds.select(sample)
    counts = dict(not_gene=0, out_of_range=0, dup_in_segment=0,
                  bad_value=0, not_ranked=0, tie_order=0, no_key=0,
                  expand=0, entries=0, untrained=0)
    examples = []
    try:
        from terra.datasets.compact import expand_compact_item
    except Exception as e:  # pragma: no cover
        expand_compact_item = None
        notes.append(f'TERRA not importable ({e}); rebuild check skipped')
    for i, row in zip(sample, sub):
        toks = np.asarray(row['gene_tokens'], dtype=np.int64)
        vals = np.asarray(row['gene_expr'], dtype=np.float64)
        lens = np.asarray(row['segment_lengths'], dtype=np.int64)
        counts['entries'] += toks.size
        if ((toks <= 0) | (toks >= n_tok)).any():
            counts['out_of_range'] += 1
            examples.append(f'row {i}: token out of range')
            continue
        counts['not_gene'] += int((~is_gene[toks]).sum())
        if not np.isfinite(vals).all() or (vals <= 0).any():
            counts['bad_value'] += 1
            examples.append(f'row {i}: non-finite or non-positive value')
        if tb is not None:
            k = tb[toks]
            counts['no_key'] += int(np.isnan(k).sum())
            if fill is not None:
                counts['untrained'] += int((k == fill).sum())
        start = 0
        for s, L in enumerate(lens):
            t, v = toks[start:start + L], vals[start:start + L]
            start += L
            if len(np.unique(t)) != len(t):
                counts['dup_in_segment'] += 1
                examples.append(f'row {i} segment {s}: duplicated token')
            dv = np.diff(v)
            if (dv > 0).any():
                counts['not_ranked'] += 1
                j = int(np.argmax(dv > 0))
                examples.append(f'row {i} segment {s}: value rises at '
                                f'position {j + 1} ({v[j]:.4f} -> '
                                f'{v[j + 1]:.4f})')
            tie = dv == 0
            if tie.any():
                a, b = t[:-1][tie], t[1:][tie]
                if tb is not None:
                    ka, kb = tb[a], tb[b]
                    wrong = (ka > kb) | ((ka == kb) & (a > b))
                else:
                    wrong = a > b
                if wrong.any():
                    counts['tie_order'] += 1
                    j = int(np.flatnonzero(wrong)[0])
                    examples.append(
                        f'row {i} segment {s}: tied genes {gene_of[a[j]]} '
                        f'(key {ka[j] if tb is not None else "-"}) before '
                        f'{gene_of[b[j]]} '
                        f'(key {kb[j] if tb is not None else "-"})')
        if expand_compact_item is not None:
            item = expand_compact_item({
                'gene_tokens': toks, 'gene_expr': vals.astype(np.float32),
                'segment_lengths': lens,
                'model_input_size': row['model_input_size']})
            g = item['gene_tokens'].numpy()
            if g.size != mis or int((g != 0).sum()) != toks.size:
                counts['expand'] += 1
                examples.append(f'row {i}: rebuilt row has {g.size} tokens, '
                                f'{int((g != 0).sum())} non-zero')
    print(f'Sampled {len(sample):,} rows ({counts["entries"]:,} stored '
          'entries)')
    if tb is not None and counts['entries']:
        print(f'Stored genes never seen in training (tie-break fill): '
              f'{100 * counts["untrained"] / counts["entries"]:.2f}%')
    for key, label in (('out_of_range', 'rows with tokens out of range'),
                       ('not_gene', 'stored tokens that are not genes'),
                       ('dup_in_segment', 'segments with a duplicated token'),
                       ('bad_value', 'rows with non-finite or <= 0 values'),
                       ('not_ranked', 'segments whose values rise'),
                       ('tie_order', 'segments with tied genes out of '
                                     'tie-break order'),
                       ('no_key', 'stored genes without a tie-break key'),
                       ('expand', 'rows that do not rebuild correctly')):
        if counts[key]:
            fails.append(f'{counts[key]} {label}')

    if args.silver_dir:
        cells, n_files = silver_cells(args.silver_dir)
        print(f'Silver: {cells:,} cells in {n_files} files; {cells - n:,} '
              f'cells not tokenized ({100 * (cells - n) / max(cells, 1):.2f}%'
              ', e.g. < 10 vocabulary genes)')
        if n > cells:
            fails.append(f'{n} rows but only {cells} silver cells')

    for x in notes:
        print('NOTE ' + x)
    for x in examples[:8]:
        print('  example: ' + x)
    if fails:
        for x in fails:
            print('FAIL ' + x)
        return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
