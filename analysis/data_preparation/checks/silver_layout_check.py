"""
Silver layout check before re-tokenizing hst_corpus_110m (read-only; reads obs
only, backed).

For the silver folders of the names file it reports:
  MISSING       a folder without .h5ad files
  SUBFOLDER     .h5ad files below the top level of a folder. The tokenizer reads
                silver/<ds>/**/*.h5ad (cell_tokenizers.py:761), but
                compute_cohort_norm_factors.py reads only silver/<ds>/*.h5ad, so
                such files would be tokenized without norm factors.
  DUP_IDS       repeated cell ids within one file
  SHARED_BATCH  a batch id (<dataset_id>_<batch>, as split_dataset.py derives it)
                found in more than one file (duplicate-id risk at concatenation)
  SMALL_FILE    a file with <= --n_neighs cells (default 16): the tokenizer
                builds the kNN graph per file and raises for it (TERRA
                preprocessors/graph.py, "at least n_neighs + 1 are needed")
  SMALL_BATCH   a batch with <= --n_neighs cells (information only)
  UNMATCHED     a held-out id of heldout_batch_ids.py with no cell (both
                compute_cohort_norm_factors.py and split_dataset.py raise on it)
  SHOW          the cell count of each --show batch id (default 1010_batch26)
With --table, writes '<batch id>\t<cells>\t<first file>' for every batch (the
file -> batch-id map to keep with the corpus: held-out ids are batch numbers).
Cells with filter_pass != 1 are skipped, as the tokenizer does.

Usage (farm, terra venv, from the terra-reproducibility root):
    python analysis/data_preparation/checks/silver_layout_check.py \
        --names analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt \
        --silver ${TERRA_DATA_DIR}/silver \
        --heldout analysis/data_preparation/heldout_batch_ids.py --n_neighs 16
Exit code 1 if any MISSING, SUBFOLDER, SMALL_FILE, DUP_IDS, SHARED_BATCH or
UNMATCHED line.
"""
import argparse
import collections
import runpy
import sys
from pathlib import Path

import anndata as ad
import numpy as np


def batch_ids(cell_ids):
    p1 = np.char.partition(cell_ids, '_')
    p2 = np.char.partition(p1[:, 2], '_')
    return np.char.add(np.char.add(p1[:, 0], '_'), p2[:, 0])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--names', required=True,
                    help='Dataset names file (first field of each line).')
    ap.add_argument('--silver', required=True,
                    help='Silver root holding one folder per dataset.')
    ap.add_argument('--heldout', default=None,
                    help='heldout_batch_ids.py (VAL_BATCH_IDS, TEST_BATCH_IDS).')
    ap.add_argument('--n_neighs', type=int, default=16,
                    help='Neighbours per cell of the tokenization (a file '
                         'needs more cells than this).')
    ap.add_argument('--show', nargs='*', default=['1010_batch26'],
                    help='Batch ids whose cell counts are printed.')
    ap.add_argument('--table', default=None,
                    help='Write <batch id>, <cells>, <first file> per batch.')
    args = ap.parse_args()
    names = [l.split(',')[0].strip() for l in open(args.names) if l.strip()]
    owner, n_cells, problems = {}, collections.Counter(), 0
    for name in names:
        root = Path(args.silver) / name
        files = sorted(root.glob('**/*.h5ad'))
        n_top = len(sorted(root.glob('*.h5ad')))
        if not files:
            print(f'MISSING\t{root}')
            problems += 1
            continue
        if len(files) != n_top:
            print(f'SUBFOLDER\t{name}\t{len(files) - n_top} of {len(files)} files')
            problems += 1
        for f in files:
            a = ad.read_h5ad(f, backed='r')
            ids = a.obs['cell_id'].astype(str).to_numpy().astype('U')
            keep = (a.obs['filter_pass'].to_numpy() == 1
                    if 'filter_pass' in a.obs.columns
                    else np.ones(len(ids), bool))
            a.file.close()
            if int(keep.sum()) <= args.n_neighs:
                print(f'SMALL_FILE\t{f}\t{int(keep.sum())} cells')
                problems += 1
            n_dup = len(ids) - len(np.unique(ids))
            if n_dup:
                print(f'DUP_IDS\t{f}\t{n_dup}')
                problems += 1
            for b, c in zip(*np.unique(batch_ids(ids[keep]), return_counts=True)):
                b = str(b)
                n_cells[b] += int(c)
                if b in owner and owner[b] != str(f):
                    print(f'SHARED_BATCH\t{b}\t{owner[b]}\t{f}')
                    problems += 1
                owner.setdefault(b, str(f))
    for b, c in sorted(n_cells.items()):
        if c <= args.n_neighs:
            print(f'SMALL_BATCH\t{b}\t{c}')
    if args.heldout:
        lists = runpy.run_path(args.heldout)
        for b in list(lists['VAL_BATCH_IDS']) + list(lists['TEST_BATCH_IDS']):
            if n_cells[b] == 0:
                print(f'UNMATCHED\t{b}')
                problems += 1
    if args.table:
        with open(args.table, 'w') as f:
            for b, c in sorted(n_cells.items()):
                f.write(f'{b}\t{c}\t{owner[b]}\n')
    for b in args.show:
        print(f'SHOW\t{b}\t{n_cells[b]}\t{owner.get(b, "-")}')
    print(f'# {len(names)} datasets, {len(owner)} batches, '
          f'{sum(n_cells.values())} cells, {problems} problems')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
