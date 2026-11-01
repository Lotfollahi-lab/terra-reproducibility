"""Were the benchmark cells part of TERRA's training data?

Each benchmark dataset has a split file listing its row indices in the
tokenized corpus (``..._precomputed_split_<dataset>.pkl``). This reports which
fraction of them is in the corpus-level train / validation / test / full
splits, for the current files and the older ``*_old`` versions (Oct 2025),
with each file's modification date: TERRA-96M (trained 1 Nov 2025) names a
``train`` split; the TERRA-112M bundle config names ``full``, while its
training run's params.yaml names ``train``.

Output: ``<results>/token_diagnostics/training_split_overlap.csv``.
Run in any environment with numpy/pandas (~16 GB memory).
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from bench_utils import DATASETS, method_output_dir  # noqa: E402

TOKENIZER = Path('/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer')
PREFIX = 'hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_10_precomputed_split'


def load(name):
    return np.asarray(pickle.load(open(TOKENIZER / f'{PREFIX}_{name}.pkl', 'rb')),
                      dtype=np.int64)


SPLITS = ['train', 'validation', 'test', 'full', 'train_old', 'validation_old', 'test_old']


def main():
    splits = {name: load(name) for name in SPLITS}
    for name, idx in splits.items():
        mtime = time.strftime('%Y-%m-%d', time.localtime(
            (TOKENIZER / f'{PREFIX}_{name}.pkl').stat().st_mtime))
        print(f'{name}: {len(idx):,} rows (file dated {mtime})')
    rows = []
    for dataset in DATASETS:
        idx = load(dataset)
        rows.append(dict(dataset=dataset, n_cells=len(idx),
                         **{f'frac_in_{name}': float(np.isin(idx, split).mean())
                            for name, split in splits.items()}))
    table = pd.DataFrame(rows)
    table.to_csv(method_output_dir('token_diagnostics') / 'training_split_overlap.csv',
                 index=False)
    print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
