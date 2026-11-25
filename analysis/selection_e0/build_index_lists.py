"""
Index lists of the E0 arms (`data.epoch_index_lists`, TERRA
exp/effective-exposures) for the tokenized selection proxy
(analysis/selection_e0/README.md).

N is the number of rows of the proxy (its training split is every row), and
the full budget C is n_epochs x N niche presentations. Every list has
n_epochs rows of equal length, so a run's learning-rate, weight-decay and EMA
schedules end at its own budget (with optimization.ipe_scale: 1.0).

Arms
  uniform_c1, uniform_c2, uniform_c4, uniform_c8
      Every row, budget C/k: consecutive passes over all rows, each in a fresh
      random order, cut into n_epochs epochs of N // k rows. uniform_c1 lists
      every row in order each epoch, which trains exactly as without lists.
  fixed_2, fixed_8, fixed_32
      Unique views at matched compute: a random 1/k of the rows (N // k),
      fixed for the run, budget C. Every epoch is N rows of passes over the
      subset. The other rows stay in the data as the subset's neighbours.
  sections_8
      Diversity at matched size: as fixed_8, but the subset is whole sections
      (batches), a few per assay. Each assay gets its share of N // 8 rows,
      and its last section only part of its rows.
  rs2_c4
      Budget C/4: each epoch an independent random N // 4 rows (repeated
      sampling of random subsets; rows can repeat across epochs).
  poisson_c4
      Budget C/4: each epoch N // 4 centres of a fresh random maximal
      independent set of the symmetric j-nearest-neighbour graph, i.e. a
      Poisson-disk sample whose radius is each cell's distance to its j-th
      neighbour. j is the largest of 1..--max_j whose independent sets hold
      N // 4 rows; an epoch whose set is smaller is topped up at random (the
      summary counts those rows).

Seeds (the E0 matrix): three for uniform_c1, uniform_c4, rs2_c4 and
poisson_c4, two for the others, 24 lists in all (--seeds overrides). A list's
seed is also its run's training seed.

Output, in --out_dir:
    <arm>_seed<s>.npy    int32 [n_epochs, length]
    lists_summary.tsv    per list: rows per epoch, budget, unique rows,
                         copies, sections, assay shares, j, top-up rows

Usage (farm, terra venv with the pinned TERRA code first on PYTHONPATH; see
README.md):
    python analysis/selection_e0/build_index_lists.py \
        --dataset <gold>/hst_corpus_110m_e0_proxy/<...>.dataset \
        --files_tsv <artifacts>/e0/proxy/proxy_files.tsv \
        --out_dir <artifacts>/e0/index_lists
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

ARMS = ('uniform_c1', 'uniform_c2', 'uniform_c4', 'uniform_c8', 'fixed_2',
        'fixed_8', 'fixed_32', 'sections_8', 'rs2_c4', 'poisson_c4')
DEFAULT_SEEDS = {arm: (0, 1, 2) if arm in ('uniform_c1', 'uniform_c4',
                                           'rs2_c4', 'poisson_c4')
                 else (0, 1) for arm in ARMS}


def read_proxy(dataset_path: str, chunk_rows: int = 1_000_000):
    """Section (batch id) of every row and the neighbour offsets."""
    from datasets import load_from_disk
    view = load_from_disk(dataset_path).select_columns(
        ['cell_id', 'neighbor_offsets']).with_format('arrow')
    sections, offsets, lengths = [], [], []
    for start in range(0, len(view), chunk_rows):
        table = view[start:start + chunk_rows]
        parts = pc.split_pattern(table.column('cell_id'), '_', max_splits=2)
        sections.append(pc.binary_join_element_wise(
            pc.list_element(parts, 0), pc.list_element(parts, 1),
            '_').to_numpy(zero_copy_only=False))
        lists = table.column('neighbor_offsets').combine_chunks()
        offsets.append(lists.flatten().to_numpy().astype(np.int64))
        lengths.append(pc.list_value_length(lists).to_numpy())
    lengths = np.concatenate(lengths).astype(np.int64)
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    return np.concatenate(sections), np.concatenate(offsets), indptr


def assay_of_sections(files_tsv: str) -> dict:
    """Assay of every section (batch id) from select_proxy.py's manifest."""
    assay = {}
    with open(files_tsv) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            for section in row['unit'].split('+'):
                assay[section] = row['assay']
    return assay


def knn_graph(offsets, indptr, j: int):
    """Symmetric graph of each row and its j nearest neighbours (CSR)."""
    n = len(indptr) - 1
    degree = np.diff(indptr)
    # Position of every offset within its row (nearest neighbour first)
    position = np.arange(len(offsets)) - np.repeat(indptr[:-1], degree)
    take = position < j
    src = np.repeat(np.arange(n), degree)[take]
    dst = src + offsets[take]
    # Rows outside the dataset and the row itself are no neighbours
    keep = (dst >= 0) & (dst < n) & (dst != src)
    src, dst = src[keep], dst[keep]
    src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    order = np.argsort(src, kind='stable')
    graph_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=graph_indptr[1:])
    return graph_indptr, dst[order]


def random_mis(indptr, indices, rng) -> np.ndarray:
    """
    A random maximal independent set (Luby's rounds with fixed random
    priorities: the greedy set in a random order).
    """
    n = len(indptr) - 1
    priority = rng.random(n)
    degree = np.diff(indptr)
    with_edges = degree > 0
    starts = indptr[:-1][with_edges]
    source = np.repeat(np.arange(n), degree)
    active = np.ones(n, dtype=bool)
    selected = np.zeros(n, dtype=bool)
    while active.any():
        neighbour_priority = np.where(active[indices], priority[indices], -1.0)
        best = np.full(n, -1.0)
        if starts.size:
            best[with_edges] = np.maximum.reduceat(neighbour_priority, starts)
        winners = active & (priority > best)
        if not winners.any():
            # The highest-priority active row always wins (ties have
            # probability 0), unless the graph has a self-loop
            raise RuntimeError('No progress in the independent-set rounds.')
        selected |= winners
        active &= ~winners
        active[indices[winners[source]]] = False
    return selected


def passes(rows: np.ndarray, length: int, rng) -> np.ndarray:
    """`length` entries of passes over `rows`, each in a fresh random order."""
    n_passes = -(-length // len(rows))
    return np.concatenate([rng.permutation(rows)
                           for _ in range(n_passes)])[:length]


def few_sections(codes: dict, size: int, rng) -> np.ndarray:
    """Rows of a few whole sections per assay (see sections_8)."""
    section, assay = codes['section'], codes['assay']
    n = len(section)
    chosen = []
    for a in np.unique(assay):
        target = round(size * np.count_nonzero(assay == a) / n)
        for s in rng.permutation(np.unique(section[assay == a])):
            if target <= 0:
                break
            rows = np.flatnonzero(section == s)
            if len(rows) > target:
                rows = rng.choice(rows, target, replace=False)
            chosen.append(rows)
            target -= len(rows)
    return np.sort(np.concatenate(chosen))


def build(arm: str, seed: int, n: int, n_epochs: int, codes: dict,
          offsets, indptr, max_j: int, calibration: dict):
    """The [n_epochs, length] list of an arm and seed, and its notes."""
    rng = np.random.default_rng([seed, ARMS.index(arm)])
    notes = {}
    kind, _, k = arm.rpartition('_')
    if kind == 'uniform':
        k = int(k[1:])
        length = n // k
        if k == 1:
            return np.tile(np.arange(n), (n_epochs, 1)), notes
        total = n_epochs * length
        lists = np.concatenate([rng.permutation(n)
                                for _ in range(-(-total // n))])[:total]
        return lists.reshape(n_epochs, length), notes
    if kind in ('fixed', 'sections'):
        k = int(k)
        if kind == 'fixed':
            subset = np.sort(rng.choice(n, n // k, replace=False))
        else:
            subset = few_sections(codes, n // k, rng)
        notes['subset_rows'] = len(subset)
        return np.stack([passes(subset, n, rng)
                         for _ in range(n_epochs)]), notes
    if arm == 'rs2_c4':
        return np.stack([rng.choice(n, n // 4, replace=False)
                         for _ in range(n_epochs)]), notes
    if arm == 'poisson_c4':
        length = n // 4
        if 'j' not in calibration:
            sizes = {}
            for j in range(1, max_j + 1):
                graph = knn_graph(offsets, indptr, j)
                sizes[j] = int(random_mis(*graph, np.random.default_rng(
                    [12345, j])).sum())
            fitting = [j for j, size in sizes.items()
                       if size >= 1.02 * length]
            calibration['j'] = max(fitting) if fitting else 1
            calibration['sizes'] = sizes
            calibration['graph'] = knn_graph(offsets, indptr,
                                             calibration['j'])
        epochs, topped_up = [], 0
        for _ in range(n_epochs):
            selected = np.flatnonzero(random_mis(*calibration['graph'], rng))
            if len(selected) >= length:
                epochs.append(rng.choice(selected, length, replace=False))
            else:
                rest = np.setdiff1d(np.arange(n), selected)
                extra = rng.choice(rest, length - len(selected),
                                   replace=False)
                topped_up += len(extra)
                epochs.append(np.concatenate([selected, extra]))
        notes.update(j=calibration['j'], topped_up_rows=topped_up)
        return np.stack(epochs), notes
    raise ValueError(f'Unknown arm {arm!r}.')


def summary(arm, seed, lists, n, n_epochs, codes, notes) -> dict:
    flat = lists.reshape(-1)
    copies = max(int(np.bincount(epoch, minlength=n).max())
                 for epoch in lists)
    presented = np.bincount(codes['assay'][flat],
                            minlength=len(codes['assay_names']))
    sections = np.bincount(codes['section'][flat],
                           minlength=len(codes['section_names']))
    return dict(
        list=f'{arm}_seed{seed}.npy', arm=arm, seed=seed, n_rows=n,
        epochs=n_epochs, rows_per_epoch=lists.shape[1],
        budget_fraction=round(flat.size / (n_epochs * n), 4),
        unique_rows=int(np.count_nonzero(np.bincount(flat, minlength=n))),
        max_copies_per_epoch=copies,
        sections=int(np.count_nonzero(sections)),
        assay_shares=';'.join(f'{a}={c / flat.size:.3f}' for a, c in zip(
            codes['assay_names'], presented) if c),
        subset_rows=notes.get('subset_rows', ''), j=notes.get('j', ''),
        topped_up_rows=notes.get('topped_up_rows', ''))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True,
                        help='the tokenized proxy (.dataset folder)')
    parser.add_argument('--files_tsv', required=True,
                        help="select_proxy.py's proxy_files.tsv")
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_epochs', type=int, default=6)
    parser.add_argument('--arms', nargs='+', default=list(ARMS),
                        choices=ARMS)
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='seeds of every arm (default: the E0 matrix)')
    parser.add_argument('--max_j', type=int, default=10)
    parser.add_argument('--skip_neighbor_check', action='store_true')
    args = parser.parse_args()

    sections, offsets, indptr = read_proxy(args.dataset)
    n = len(sections)
    if not args.skip_neighbor_check:
        # The neighbour offsets must name rows of the same file
        from datasets import load_from_disk
        from terra.training.exposures import check_neighbor_rows
        pairs = check_neighbor_rows(load_from_disk(args.dataset))
        print(f'Neighbour rows checked: {pairs} pairs.')
    assay = assay_of_sections(args.files_tsv)
    section_names, section_code = np.unique(sections, return_inverse=True)
    missing = sorted(set(section_names) - set(assay))
    if missing:
        raise SystemExit(f'{len(missing)} sections of the dataset are not in '
                         f'{args.files_tsv}, e.g. {missing[:3]}.')
    assay_names, assay_of_section = np.unique(
        [assay[s] for s in section_names], return_inverse=True)
    codes = dict(section=section_code, section_names=section_names,
                 assay=assay_of_section[section_code],
                 assay_names=assay_names)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration, rows = {}, []
    for arm in args.arms:
        for seed in (args.seeds if args.seeds is not None
                     else DEFAULT_SEEDS[arm]):
            lists, notes = build(arm, seed, n, args.n_epochs, codes,
                                 offsets, indptr, args.max_j, calibration)
            np.save(out_dir / f'{arm}_seed{seed}.npy', lists.astype(np.int32))
            rows.append(summary(arm, seed, lists, n, args.n_epochs, codes,
                                notes))
            print('\t'.join(str(v) for v in rows[-1].values()))
    if 'sizes' in calibration:
        print('poisson_c4: independent-set sizes by j: '
              + ', '.join(f'{j}: {s} ({s / n:.3f})'
                          for j, s in calibration['sizes'].items())
              + f'; j = {calibration["j"]}.')
    with open(out_dir / 'lists_summary.tsv', 'w') as f:
        f.write('\t'.join(rows[0]) + '\n')
        for row in rows:
            f.write('\t'.join(str(v) for v in row.values()) + '\n')
    print(f'Wrote {len(rows)} lists to {out_dir} ({n} rows, '
          f'{args.n_epochs} epochs).')


if __name__ == '__main__':
    main()
