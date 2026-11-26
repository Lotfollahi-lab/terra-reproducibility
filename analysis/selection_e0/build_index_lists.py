"""
Index lists of the E0 arms (`data.epoch_index_lists`, TERRA
exp/effective-exposures) for the tokenized selection proxy
(analysis/selection_e0/README.md).

N is the number of rows of the proxy (its training split is every row), and
the full budget C is n_epochs x N niche presentations. Every list has
n_epochs rows of equal length, so a run's learning-rate, weight-decay and EMA
schedules end at its own budget (with optimization.ipe_scale: 1.0). The
sampler shuffles each epoch's entries itself (seeded by the training seed), so
a list sets which rows an epoch presents and how often, not their order.

Arms
  uniform_c1, uniform_c2, uniform_c4, uniform_c8
      Every row, budget C/k: consecutive passes over all rows, each in a fresh
      random order, cut into n_epochs epochs of N // k rows. uniform_c1 lists
      every row each epoch, which trains exactly as without lists.
  fixed_2, fixed_8, fixed_32
      Unique views at matched compute: a random 1/k of the rows (N // k),
      fixed for the run, budget C. Every epoch is N rows of passes over the
      subset. The other rows stay in the data as the subset's neighbours.
  sections_8
      Diversity at matched size: as fixed_8, but the subset is whole sections
      (batches), a few per assay. Each assay gets its share of N // 8 rows,
      and its last section only part of its rows.
  poisson_c4
      Coverage, budget C/4: each epoch N // 4 centres of a fresh random
      maximal independent set of the symmetric j-nearest-neighbour graph, i.e.
      a Poisson-disk sample whose radius is each cell's distance to its j-th
      neighbour. j is the largest of 1..--max_j whose independent sets hold
      N // 4 rows; an epoch whose set is smaller is topped up at random (the
      summary counts those rows).
  poisson_fixed_c4
      The matched-coverage control of poisson_c4, budget C/4: poisson_c4's
      first-epoch centres (same seed), kept for every epoch. Each epoch covers
      the same cells as a poisson_c4 epoch would; only the re-drawing differs.
  rs2_c4 (not in the E0 matrix; --arms rs2_c4)
      Budget C/4: each epoch an independent random N // 4 rows (repeated
      sampling of random subsets). At about 1.5 passes it is close to
      uniform_c4, so E0 leaves it out; it stays available as an E3 null.

Shared random numbers: a seed draws the same random stream in every arm of a
family, so the arms of one seed are nested. uniform_c8's entries are the first
half of uniform_c4's, which are the first half of uniform_c2's; fixed_32's
subset lies in fixed_8's, which lies in fixed_2's; poisson_fixed_c4 keeps
poisson_c4's first epoch. Paired comparisons within a seed then differ by the
arm, not by which rows were drawn.

Seeds (the E0 matrix): three for uniform_c1, uniform_c4, fixed_8 and
poisson_c4, two for the others, 24 lists in all (--seeds overrides). A list's
seed is also its run's training seed. E0's six other runs reuse these lists:
the learning-rate checks (uniform_c1 and uniform_c4, seed 0, at 0.5x and 2x
the rate) and length grouping off (uniform_c4, seeds 0 and 1, with
data.mega_batch_mult_max: 1, so each mega-batch is one batch).

Output, in --out_dir:
    <arm>_seed<s>.npy    int32 [n_epochs, length]
    lists_summary.tsv    per list: rows per epoch, budget, unique rows,
                         distinct centres per epoch, copies, sections, assay
                         shares, cells covered (as a centre or one of its
                         first --n_context neighbours) per epoch and over the
                         run, j, top-up rows

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
        'fixed_8', 'fixed_32', 'sections_8', 'rs2_c4', 'poisson_c4',
        'poisson_fixed_c4')
E0_ARMS = tuple(arm for arm in ARMS if arm != 'rs2_c4')
DEFAULT_SEEDS = {arm: (0, 1, 2) if arm in ('uniform_c1', 'uniform_c4',
                                           'fixed_8', 'poisson_c4')
                 else (0, 1) for arm in ARMS}
# The random stream of each arm family (shared random numbers, see above)
STREAMS = {'uniform': 1, 'fixed': 2, 'sections': 3, 'rs2': 4, 'poisson': 5}


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


def neighbour_pairs(offsets, indptr, j: int):
    """(row, neighbour row, rank) of each row's first j neighbours."""
    n = len(indptr) - 1
    degree = np.diff(indptr)
    # Rank of every offset within its row (nearest neighbour first)
    rank = np.arange(len(offsets)) - np.repeat(indptr[:-1], degree)
    take = rank < j
    src = np.repeat(np.arange(n), degree)[take]
    dst = src + offsets[take]
    rank = rank[take]
    # Rows outside the dataset and the row itself are no neighbours
    keep = (dst >= 0) & (dst < n) & (dst != src)
    return src[keep], dst[keep], rank[keep]


def neighbour_rows(offsets, indptr, j: int) -> np.ndarray:
    """[N, j] rows of each row's first j neighbours (-1: none)."""
    n = len(indptr) - 1
    src, dst, rank = neighbour_pairs(offsets, indptr, j)
    rows = np.full((n, j), -1,
                   dtype=np.int32 if n < 2**31 else np.int64)
    rows[src, rank] = dst
    return rows


def knn_graph(offsets, indptr, j: int):
    """Symmetric graph of each row and its j nearest neighbours (CSR)."""
    n = len(indptr) - 1
    src, dst, _ = neighbour_pairs(offsets, indptr, j)
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


def poisson_graph(offsets, indptr, max_j: int, length: int,
                  calibration: dict):
    """The j-nearest-neighbour graph of the Poisson-disk arms (j set once)."""
    if 'graph' not in calibration:
        sizes = {}
        for j in range(1, max_j + 1):
            graph = knn_graph(offsets, indptr, j)
            sizes[j] = int(random_mis(*graph, np.random.default_rng(
                [12345, j])).sum())
        fitting = [j for j, size in sizes.items() if size >= 1.02 * length]
        calibration['j'] = max(fitting) if fitting else 1
        calibration['sizes'] = sizes
        calibration['graph'] = knn_graph(offsets, indptr, calibration['j'])
    return calibration['graph']


def poisson_epoch(graph, n: int, length: int, rng):
    """
    `length` centres of a fresh random maximal independent set, topped up at
    random if the set is smaller; and the number of top-up rows.
    """
    selected = np.flatnonzero(random_mis(*graph, rng))
    if len(selected) >= length:
        return rng.choice(selected, length, replace=False), 0
    rest = np.setdiff1d(np.arange(n), selected)
    extra = rng.choice(rest, length - len(selected), replace=False)
    return np.concatenate([selected, extra]), len(extra)


def build(arm: str, seed: int, n: int, n_epochs: int, codes: dict,
          offsets, indptr, max_j: int, calibration: dict):
    """The [n_epochs, length] list of an arm and seed, and its notes."""
    family, _, k = arm.partition('_')
    # One stream per family and seed: the arms of a seed are nested
    rng = np.random.default_rng([seed, STREAMS[family]])
    notes = {}
    if family == 'uniform':
        k = int(k[1:])
        length = n // k
        if k == 1:
            return np.tile(np.arange(n), (n_epochs, 1)), notes
        total = n_epochs * length
        lists = np.concatenate([rng.permutation(n)
                                for _ in range(-(-total // n))])[:total]
        return lists.reshape(n_epochs, length), notes
    if family in ('fixed', 'sections'):
        k = int(k)
        if family == 'fixed':
            # The first N // k rows of one permutation
            subset = np.sort(rng.permutation(n)[:n // k])
        else:
            subset = few_sections(codes, n // k, rng)
        notes['subset_rows'] = len(subset)
        order_rng = np.random.default_rng([seed, STREAMS[family], k])
        return np.stack([passes(subset, n, order_rng)
                         for _ in range(n_epochs)]), notes
    if arm == 'rs2_c4':
        return np.stack([rng.choice(n, n // 4, replace=False)
                         for _ in range(n_epochs)]), notes
    if arm in ('poisson_c4', 'poisson_fixed_c4'):
        length = n // 4
        graph = poisson_graph(offsets, indptr, max_j, length, calibration)
        epochs, extras = [], []
        for _ in range(n_epochs):
            if arm == 'poisson_fixed_c4' and epochs:
                # poisson_c4's first epoch (same seed), kept
                epochs.append(epochs[0])
                extras.append(extras[0])
            else:
                rows, extra = poisson_epoch(graph, n, length, rng)
                epochs.append(rows)
                extras.append(extra)
        notes.update(j=calibration['j'], topped_up_rows=sum(extras))
        return np.stack(epochs), notes
    raise ValueError(f'Unknown arm {arm!r}.')


def coverage(lists, context) -> tuple:
    """
    Mean distinct centres per epoch, and the fraction of rows covered as a
    centre or one of its `context` neighbours: mean per epoch, and over the
    run.
    """
    n = len(context)
    run = np.zeros(n, dtype=bool)
    centres, per_epoch = [], []
    for epoch in lists:
        covered = np.zeros(n, dtype=bool)
        covered[epoch] = True
        rows = np.flatnonzero(covered)
        centres.append(len(rows))
        for chunk in np.array_split(rows, max(1, len(rows) // 1_000_000)):
            neighbours = context[chunk].reshape(-1)
            covered[neighbours[neighbours >= 0]] = True
        per_epoch.append(covered.mean())
        run |= covered
    return float(np.mean(centres)), float(np.mean(per_epoch)), \
        float(run.mean())


def summary(arm, seed, lists, n, n_epochs, codes, context, notes) -> dict:
    flat = lists.reshape(-1)
    centres, covered_epoch, covered_run = coverage(lists, context)
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
        centres_per_epoch=round(centres, 1),
        max_copies_per_epoch=copies,
        sections=int(np.count_nonzero(sections)),
        assay_shares=';'.join(f'{a}={c / flat.size:.3f}' for a, c in zip(
            codes['assay_names'], presented) if c),
        cells_covered_per_epoch=round(covered_epoch, 4),
        cells_covered_run=round(covered_run, 4),
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
    parser.add_argument('--arms', nargs='+', default=list(E0_ARMS),
                        choices=ARMS,
                        help='arms to build (default: the E0 matrix)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='seeds of every arm (default: the E0 matrix)')
    parser.add_argument('--max_j', type=int, default=10)
    parser.add_argument('--n_context', type=int, default=10,
                        help='neighbours a niche reads (for the coverage '
                             'columns of the summary)')
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
    context = neighbour_rows(offsets, indptr, args.n_context)
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
                                context, notes))
            print('\t'.join(str(v) for v in rows[-1].values()))
    if 'sizes' in calibration:
        print('Poisson-disk arms: independent-set sizes by j: '
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
