"""
E0 selection proxy (analysis/selection_e0/README.md): about 48 training batches
of hst_corpus_110m, stratified by assay x tissue and each capped at about 100k
cells by a square crop, written as the silver folder that the proxy's tokenizer
config (config/cell-graph-tokenizer/cell-graph-tokenizer_<proxy>.yaml)
tokenizes.

Dry run by default: reads only HDF5 metadata of the silver files (cell ids,
obs filter_pass, uns assay and tissue, and the coordinates of the files to
crop), prints the strata, the chosen batches and the crops, and writes
nothing. With --run it also writes

    <data>/silver/<proxy>/<dataset>__<file>.h5ad   a link to the silver file,
                                                   or the cells of its crop
    <out_dir>/proxy_files.tsv                      one row per proxy file
    <out_dir>/proxy_strata.tsv                     batches and cells per stratum
    <out_dir>/proxy_selection.json                 settings and totals

Units. A unit is a training batch (the cell id prefix <dataset_id>_<batch>, as
split_dataset.py derives it) of the corpus datasets with all its files; a
file that holds cells of several batches is a unit of its own. Left out:
held-out batches (heldout_batch_ids.py) and every file with a held-out cell,
and units with fewer than --min_cells cells that pass the tokenizer's filter
(obs filter_pass == 1, as terra.preprocessors.filters.filter_cells).

Strata are the (assay, tissue) pairs of the files' uns. Every stratum gets one
unit (with more strata than --n_batches, strata are drawn without replacement
with probability proportional to the square root of their passing cells);
the remaining units go one at a time to the stratum furthest below its share
of the total, shares being proportional to that square root (at most the
stratum's units). Within a stratum, units are drawn
at random, cycling over its datasets in random order, so they come from as
many datasets as possible.

Crops. A unit with more than --max_cells passing cells keeps, in each of its
files, the passing cells nearest (maximum norm, i.e. a square window) to a
random passing cell, the file's share of --max_cells being proportional to
its passing cells, plus every other cell inside that window. The tokenizer
builds the neighbour graph inside the window, so cells at its border lose
their neighbours outside it.

Usage (farm, terra venv, from the terra-reproducibility root; see README.md):
    python analysis/selection_e0/select_proxy.py \
        --data_directory_path /lustre/scratch126/cellgen/lotfollahi/DATASETS \
        --names analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt \
        --out_dir /nfs/team361/sb75/terra-reproducibility/artifacts/e0/proxy [--run]
"""
import argparse
import datetime
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

try:
    from anndata.io import read_elem
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'analysis' / 'data_preparation'))
from heldout_batch_ids import TEST_BATCH_IDS, VAL_BATCH_IDS  # noqa: E402

PROXY_NAME = 'hst_corpus_110m_e0_proxy'


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.size == 1:
        return _text(value.reshape(-1)[0])
    return str(value)


def batch_counts(cell_ids) -> dict:
    """Cells per batch id (<dataset_id>_<batch>) of an array of cell ids."""
    parts = pc.split_pattern(pa.array(np.asarray(cell_ids, dtype=str)), '_',
                             max_splits=2)
    prefix = pc.binary_join_element_wise(
        pc.list_element(parts, 0), pc.list_element(parts, 1), '_')
    counts = pc.value_counts(prefix)
    return {v.as_py(): c.as_py() for v, c in zip(counts.field('values'),
                                                 counts.field('counts'))}


def passing(group) -> np.ndarray | None:
    """Mask of the cells that pass the tokenizer's filter (None: all)."""
    if 'filter_pass' not in group:
        return None
    values = read_elem(group['filter_pass'])
    values = np.asarray(getattr(values, 'to_numpy', lambda: values)())
    return values == 1


def scan_file(path: Path) -> dict:
    """Cells, passing cells, batches, assay and tissue of a silver file."""
    with h5py.File(path, 'r') as f:
        cell_ids = read_elem(f['obs/cell_id'])
        mask = passing(f['obs'])
        uns = f['uns'] if 'uns' in f else {}
        assay = _text(read_elem(uns['assay'])) if 'assay' in uns else 'unknown'
        tissue = (_text(read_elem(uns['tissue'])) if 'tissue' in uns
                  else 'unknown')
    n_cells = len(cell_ids)
    return dict(path=path, n_cells=n_cells,
                n_pass=int(n_cells if mask is None else mask.sum()),
                batches=batch_counts(cell_ids), assay=assay, tissue=tissue)


def scan(silver: Path, names: list[str]) -> tuple[list[dict], list[str]]:
    """Scan every file of the datasets; returns the files and the problems."""
    files, problems = [], []
    for dataset in names:
        paths = sorted((silver / dataset).glob('*.h5ad'))
        if not paths:
            problems.append(f'{dataset}: no h5ad file in {silver / dataset}')
        for path in paths:
            info = scan_file(path)
            info['dataset'] = dataset
            files.append(info)
    return files, problems


def build_units(files: list[dict], heldout: set, min_cells: int):
    """Training units (see the module docstring) and the left-out files."""
    units, left_out = {}, defaultdict(int)
    for info in files:
        batches = set(info['batches'])
        if batches & heldout:
            left_out['files with held-out cells'] += 1
            continue
        key = (next(iter(batches)) if len(batches) == 1
               else '+'.join(sorted(batches)))
        unit = units.setdefault(key, dict(key=key, files=[], n_cells=0,
                                          n_pass=0, strata=set()))
        unit['files'].append(info)
        unit['n_cells'] += info['n_cells']
        unit['n_pass'] += info['n_pass']
        unit['strata'].add((info['assay'], info['tissue']))
    out = []
    for key in sorted(units):
        unit = units[key]
        if unit['n_pass'] < min_cells:
            left_out[f'units with fewer than {min_cells} passing cells'] += 1
            continue
        # Files of one batch normally agree; otherwise the most common pair
        strata = sorted(unit['strata'])
        unit['stratum'] = max(strata, key=lambda s: sum(
            f['n_pass'] for f in unit['files']
            if (f['assay'], f['tissue']) == s))
        unit['dataset'] = unit['files'][0]['dataset']
        out.append(unit)
    return out, dict(left_out)


def allot(strata: dict, n_batches: int, rng) -> dict:
    """Units per stratum (see the module docstring)."""
    names = sorted(strata)
    weight = np.array([math.sqrt(sum(u['n_pass'] for u in strata[s]))
                       for s in names])
    capacity = np.array([len(strata[s]) for s in names])
    if len(names) >= n_batches:
        chosen = rng.choice(len(names), size=n_batches, replace=False,
                            p=weight / weight.sum())
        return {names[i]: 1 for i in sorted(chosen)}
    # One unit each, then one at a time to the stratum furthest below its
    # share of the total (strata without units left are skipped)
    count = np.ones(len(names), dtype=int)
    total = min(n_batches, int(capacity.sum()))
    target = weight / weight.sum() * total
    while count.sum() < total:
        deficit = np.where(count < capacity, target - count, -np.inf)
        count[int(np.argmax(deficit))] += 1
    return {s: int(c) for s, c in zip(names, count)}


def draw_units(units: list[dict], n: int, rng) -> list[dict]:
    """`n` units at random, cycling over their datasets in random order."""
    by_dataset = defaultdict(list)
    for unit in units:
        by_dataset[unit['dataset']].append(unit)
    queues = [list(rng.permutation(np.array(by_dataset[d], dtype=object)))
              for d in rng.permutation(sorted(by_dataset))]
    chosen = []
    while len(chosen) < n and any(queues):
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen


def crop_file(info: dict, n_keep: int, rng) -> dict:
    """The square window of a file's crop (see the module docstring)."""
    with h5py.File(info['path'], 'r') as f:
        xy = np.asarray(read_elem(f['obsm/spatial']), dtype=np.float64)[:, :2]
        mask = passing(f['obs'])
    if mask is None:
        mask = np.ones(len(xy), dtype=bool)
    candidates = np.flatnonzero(mask)
    centre = xy[rng.choice(candidates)]
    distance = np.abs(xy - centre).max(axis=1)
    n_keep = min(max(int(n_keep), 1), len(candidates))
    half_width = float(np.partition(distance[candidates], n_keep - 1)[
        n_keep - 1])
    keep = distance <= half_width
    return dict(keep=keep, centre=centre.tolist(), half_width=half_width,
                n_cells_kept=int(keep.sum()),
                n_pass_kept=int((keep & mask).sum()))


def write_file(info: dict, crop: dict | None, destination: Path):
    if crop is None:
        os.symlink(Path(info['path']).resolve(), destination)
        return
    import anndata as ad
    adata = ad.read_h5ad(info['path'], backed='r')
    adata[np.flatnonzero(crop['keep'])].to_memory().write_h5ad(destination)
    adata.file.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_directory_path', required=True)
    parser.add_argument('--names', required=True,
                        help='dataset names of the corpus, one per line')
    parser.add_argument('--out_dir', required=True,
                        help='folder of the manifest files (written with --run)')
    parser.add_argument('--proxy_name', default=PROXY_NAME)
    parser.add_argument('--n_batches', type=int, default=48)
    parser.add_argument('--max_cells', type=int, default=100_000)
    parser.add_argument('--min_cells', type=int, default=5_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run', action='store_true',
                        help='write the proxy folder and the manifests')
    args = parser.parse_args()

    data = Path(args.data_directory_path)
    silver = data / 'silver'
    names = [line.strip() for line in open(args.names) if line.strip()]
    proxy_folder = silver / args.proxy_name
    if args.run and proxy_folder.exists() and any(proxy_folder.iterdir()):
        sys.exit(f'{proxy_folder} exists and is not empty: move it away '
                 'first (the tokenized proxy and its index lists belong to '
                 'its files).')

    rng = np.random.default_rng(args.seed)
    heldout = set(VAL_BATCH_IDS) | set(TEST_BATCH_IDS)
    files, problems = scan(silver, names)
    if problems:
        sys.exit('Silver problems:\n  ' + '\n  '.join(problems))
    units, left_out = build_units(files, heldout, args.min_cells)
    strata = defaultdict(list)
    for unit in units:
        strata[unit['stratum']].append(unit)
    allotted = allot(strata, args.n_batches, rng)
    chosen = []
    for stratum in sorted(allotted):
        chosen += draw_units(strata[stratum], allotted[stratum], rng)

    # Crops and the rows of the manifest
    rows = []
    for unit in sorted(chosen, key=lambda u: u['key']):
        for info in unit['files']:
            crop = None
            if unit['n_pass'] > args.max_cells and info['n_pass'] > 0:
                share = args.max_cells * info['n_pass'] / unit['n_pass']
                crop = crop_file(info, round(share), rng)
            rows.append(dict(
                file=f"{info['dataset']}__{Path(info['path']).name}",
                dataset=info['dataset'], unit=unit['key'],
                assay=unit['stratum'][0], tissue=unit['stratum'][1],
                source=str(info['path']), n_cells_source=info['n_cells'],
                n_pass_source=info['n_pass'],
                n_cells=crop['n_cells_kept'] if crop else info['n_cells'],
                n_pass=crop['n_pass_kept'] if crop else info['n_pass'],
                cropped=int(crop is not None),
                crop_centre=crop['centre'] if crop else '',
                crop_half_width=crop['half_width'] if crop else '',
                _crop=crop, _info=info))

    # Report
    print(f'Corpus: {len(names)} datasets, {len(files)} files, {len(units)} '
          f'training units; left out: {left_out or "nothing"}.')
    print(f'{len(strata)} strata (assay, tissue); {len(chosen)} units chosen '
          f'(--n_batches {args.n_batches}).')
    strata_rows = []
    for stratum in sorted(strata):
        mine = [r for r in rows if (r['assay'], r['tissue']) == stratum]
        strata_rows.append(dict(
            assay=stratum[0], tissue=stratum[1], units=len(strata[stratum]),
            pass_cells=sum(u['n_pass'] for u in strata[stratum]),
            chosen=allotted.get(stratum, 0),
            datasets=len({r['dataset'] for r in mine}),
            proxy_pass_cells=sum(r['n_pass'] for r in mine)))
    header = list(strata_rows[0]) if strata_rows else []
    print('\t'.join(header))
    for row in strata_rows:
        print('\t'.join(str(row[k]) for k in header))
    total = sum(r['n_pass'] for r in rows)
    print(f'Proxy: {len(rows)} files, {sum(r["cropped"] for r in rows)} '
          f'cropped, {total} passing cells ({sum(r["n_cells"] for r in rows)}'
          ' cells).')
    if not args.run:
        print('Dry run: nothing written (add --run).')
        return

    proxy_folder.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        write_file(row['_info'], row['_crop'], proxy_folder / row['file'])
    columns = [k for k in rows[0] if not k.startswith('_')]
    with open(out_dir / 'proxy_files.tsv', 'w') as f:
        f.write('\t'.join(columns) + '\n')
        for row in rows:
            f.write('\t'.join(str(row[k]) for k in columns) + '\n')
    with open(out_dir / 'proxy_strata.tsv', 'w') as f:
        f.write('\t'.join(header) + '\n')
        for row in strata_rows:
            f.write('\t'.join(str(row[k]) for k in header) + '\n')
    with open(out_dir / 'proxy_selection.json', 'w') as f:
        json.dump(dict(
            created=datetime.datetime.now().isoformat(timespec='seconds'),
            proxy_folder=str(proxy_folder),
            settings={k: v for k, v in vars(args).items() if k != 'run'},
            n_files=len(rows), n_units=len(chosen),
            n_cropped=sum(r['cropped'] for r in rows),
            n_cells=sum(r['n_cells'] for r in rows), n_pass_cells=total,
            left_out=left_out, n_strata=len(strata)), f, indent=2)
    print(f'Wrote {proxy_folder} and the manifests in {out_dir}.')


if __name__ == '__main__':
    main()
