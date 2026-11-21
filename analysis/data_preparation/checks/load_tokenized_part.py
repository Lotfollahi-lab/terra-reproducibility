#!/usr/bin/env python
"""
Load a tokenized cell-graph dataset (a per-dataset part or the corpus) and
print one row as stored and as TERRA's CellGraphDataset hands it to the
model, at the 16 x 512 layout (knn16_seq512) and sliced to 10 x 256
(knn10_seq256). Read-only; a quick look, not a check (see
check_tokenized_part.py).

Usage (farm, TERRA venv, from the terra-reproducibility root):
    D=/lustre/scratch126/cellgen/lotfollahi/DATASETS
    python analysis/data_preparation/checks/load_tokenized_part.py \\
        --dataset_path $D/gold/cell-graph-tokenizer/xhs49-1b_1p/xhs49-1b_1p_32_None_None_None_None_None_shifted_log_knn_16_8704.dataset
"""
import argparse
import pickle

import datasets
import numpy as np

from terra.datasets.cell_datasets import CellGraphDataset

DEFAULT_TOKEN_DICT = ('/lustre/scratch126/cellgen/lotfollahi/DATASETS/'
                      'tokenizer/token_dictionary_homo_sapiens.pkl')
# data keys of the hst_corpus_110m training configs (16 x 512 corpus)
LAYOUTS = {
    '16 x 512 (knn16_seq512)': dict(seq_len_cell=512, seq_len_neighborhood=8192,
                                    truncate_neighbors=False),
    '10 x 256 (knn10_seq256)': dict(seq_len_cell=256, seq_len_neighborhood=2560,
                                    truncate_neighbors=True),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--token_dictionary_file_path', default=DEFAULT_TOKEN_DICT)
    ap.add_argument('--row', type=int, default=0)
    args = ap.parse_args()

    ds = datasets.load_from_disk(args.dataset_path)
    print(ds)
    row = ds[args.row]
    print(f"\nRow {args.row} as stored: cell {row['cell_id']}; "
          f"{len(row['gene_tokens'])} stored entries; segment_lengths "
          f"{row['segment_lengths']}")
    print(f"  first genes {row['gene_tokens'][:5]}, values "
          f"{np.round(row['gene_expr'][:5], 3).tolist()}")

    ds.set_format(type='torch')
    with open(args.token_dictionary_file_path, 'rb') as fh:
        vocab = len(pickle.load(fh))
    for name, kw in LAYOUTS.items():
        cds = CellGraphDataset(gt_type='combined', cell_pos_enc='coord',
                               dataset=ds, vocab_size=vocab, special_tokens=[],
                               nz_spc=False, tokenized_seq_len_cell=512, **kw)
        item = cds[args.row]
        print(f'\n{name}: {len(cds):,} items; row {args.row} as the model '
              'gets it:')
        for k, v in item.items():
            print(f'  {k:16s} {tuple(v.shape) if hasattr(v, "shape") else v} '
                  f'{getattr(v, "dtype", "")}')
        t = item['tokens']
        print(f'  gene tokens: {int((t != 0).sum())} of {t.numel()} positions; '
              f'first 5 {t[:5].tolist()}, values '
              f'{[round(x, 3) for x in item["values"][:5].tolist()]}')


if __name__ == '__main__':
    main()
