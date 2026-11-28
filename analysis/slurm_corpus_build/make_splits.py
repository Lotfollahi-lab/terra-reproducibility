#!/usr/bin/env python
"""
Precomputed training / validation / test splits of a concatenated corpus, one
set per split variant: one model each, all trained on the same tokenized
corpus, e.g. one on every dataset and one per gene panel.

Inputs
- --corpus_path: the corpus (concatenate_datasets.py);
- --split_samples (optional): '<dataset name> <sample> <val|test>' per line,
  tab- or space-separated, '#' comments (corpus_tools.py
  propose-split-samples writes a random proposal); the samples' batch ids
  (<dataset_id>_batch<k>) come from the silver manifests
  (<data_directory_path>/silver/<dataset>/silver_manifest.tsv). Without it
  nothing is held out: each training split holds every row of its datasets,
  and the validation and test files are empty;
- --splits: the variants, 'name=dataset,dataset name=dataset ...' (default
  'all=' every dataset of --names_file).

For a variant with the datasets D:
- validation: the rows of the validation samples of D;
- test: the rows of the test samples of D;
- train: every other row of D.
Rows of other datasets are in none of the three, and a sample is held out
from every variant that holds its dataset, so models trained on different
variants are evaluated on the same samples.

The files are written with TERRA's save_precomputed_split (it records the
corpus' row count and the cell ids of evenly spaced rows, so training refuses
them for another corpus), as
  <out_dir>/<variant>/<corpus>_precomputed_split_{train,validation,test}.pkl
(<corpus>: the corpus folder name without '.dataset', as split_dataset.py
names them). The training split records every validation and test batch of
--split_samples as excluded, which holds for every variant: training then
trusts the record when a config lists the variant's held-out batches as
data.val_batch_ids / data.test_batch_ids, instead of scanning every cell id.
<out_dir>/<variant>/split_summary.json gives the datasets, the held-out
samples and batch ids, the row counts and 'config_data': the data-section
entries of a TERRA training config for the variant (paths below
--data_directory_path written as ${TERRA_DATA_DIR}/...).

Checks before a variant is moved into place: each file loads through TERRA
for this corpus (load_precomputed_split) with the written indices; TERRA's
training-time check finds no held-out row in the training split, once by a
scan of the cell ids and once from the record; the three splits are disjoint
and cover exactly the rows of D; every held-out sample has rows. Batches
whose row count differs from their silver manifest (cells the tokenizer
dropped) are reported.

A variant folder written for the same corpus (row count and recorded cell
ids), datasets and held-out batches is kept; another one is refused unless
--overwrite. Each variant is written to <variant>.partial-<id> (<id>: the
Slurm job id, or the process id) and renamed into place when complete.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc

ROLES = {'val': 'validation', 'validation': 'validation', 'test': 'test'}
SPLIT_FILES = ('train', 'validation', 'test')
VARIANT_NAME = re.compile(r'^[A-Za-z0-9_-]+$')


class SplitError(Exception):
    """An input or result that the splits must not be written with."""


def log(message: str):
    print(message, flush=True)


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', str(text))]


def read_names(names_file) -> list[str]:
    return [line.strip() for line in Path(names_file).read_text().splitlines()
            if line.strip()]


def parse_variants(spec: str, names: list[str]) -> dict[str, list[str]]:
    """{variant: datasets} from 'name=dataset,dataset ...'."""
    if not spec.strip():
        return {'all': list(names)}
    variants = {}
    for entry in spec.split():
        name, sep, members = entry.partition('=')
        if not sep:
            raise SplitError(f'split variant {entry!r}: expected '
                             'name=dataset,dataset')
        if not VARIANT_NAME.match(name):
            raise SplitError(f'split variant name {name!r}: use letters, '
                             "digits, '_' and '-'")
        if name in variants:
            raise SplitError(f'split variant {name!r} given twice')
        datasets = list(dict.fromkeys(d for d in members.split(',') if d))
        if not datasets:
            raise SplitError(f'split variant {name!r} has no datasets')
        unknown = [d for d in datasets if d not in names]
        if unknown:
            raise SplitError(f'split variant {name!r}: {unknown} are not '
                             f'datasets of the corpus {names}')
        variants[name] = datasets
    return variants


def read_manifests(data_directory_path: Path,
                   names: list[str]) -> dict[str, pd.DataFrame]:
    manifests = {}
    for name in names:
        path = data_directory_path / 'silver' / name / 'silver_manifest.tsv'
        if not path.exists():
            raise SplitError(f'no silver manifest {path}')
        manifests[name] = pd.read_csv(path, sep='\t',
                                      dtype={'sample': str, 'batch_id': str})
    return manifests


def read_split_samples(path, manifests: dict) -> list[dict]:
    entries, seen = [], {}
    for k, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise SplitError(f'{path}:{k}: expected <dataset> <sample> '
                             f'<val|test>, got {line!r}')
        dataset, sample, role = fields
        if dataset not in manifests:
            raise SplitError(f'{path}:{k}: {dataset!r} is not a dataset of '
                             f'the corpus {sorted(manifests)}')
        if role.lower() not in ROLES:
            raise SplitError(f'{path}:{k}: role {role!r}: use val or test')
        table = manifests[dataset]
        match = table.loc[table['sample'] == sample, 'batch_id']
        if match.empty:
            close = [s for s in table['sample']
                     if s.strip().lower() == sample.strip().lower()]
            raise SplitError(
                f'{path}:{k}: sample {sample!r} is not in the silver '
                f'manifest of {dataset}'
                + (f' (case-insensitive matches: {close})' if close else ''))
        if (dataset, sample) in seen:
            raise SplitError(f'{path}:{k}: {dataset} {sample} is listed '
                             f'twice (also on line {seen[dataset, sample]})')
        seen[dataset, sample] = k
        entries.append({'dataset': dataset, 'sample': sample,
                        'role': ROLES[role.lower()],
                        'batch_id': match.iloc[0]})
    return entries


def corpus_batch_codes(ds) -> tuple[np.ndarray, list[str]]:
    """Batch-id code of every row and the batch ids of the codes. A batch id
    is the first two '_' fields of the cell id, as in TERRA
    (terra.datasets.utils._batch_ids_from_cell_ids)."""
    if ds._indices is not None:
        ds = ds.flatten_indices()
    cell_ids = ds.data.column('cell_id')
    batch_ids = pc.replace_substring_regex(
        cell_ids, pattern=r'^([^_]*)_([^_]*).*$', replacement=r'\1_\2')
    encoded = pc.dictionary_encode(batch_ids).combine_chunks()
    codes = np.asarray(encoded.indices.to_numpy(zero_copy_only=False),
                       dtype=np.int64)
    return codes, [str(b) for b in encoded.dictionary.to_pylist()]


def with_data_dir(path, data_directory_path: Path) -> str:
    """`path` as ${TERRA_DATA_DIR}/... when below the data directory."""
    path = Path(path).resolve()
    try:
        return '${TERRA_DATA_DIR}/' + str(path.relative_to(
            data_directory_path.resolve()))
    except ValueError:
        return str(path)


def existing_variant_fits(folder: Path, signature: dict, ds) -> bool:
    """Whether `folder` holds the variant of `signature` for this corpus."""
    from terra.datasets.utils import load_precomputed_split
    summary_file = folder / 'split_summary.json'
    if not summary_file.exists():
        return False
    summary = json.loads(summary_file.read_text())
    if summary.get('signature') != signature:
        return False
    try:
        for name in SPLIT_FILES:
            load_precomputed_split(str(folder / summary['files'][name]), ds)
    except (OSError, KeyError, ValueError) as error:
        log(f'  {folder.name}: the existing files do not fit the corpus '
            f'({error})')
        return False
    return True


def check_variant(ds, files: dict, indices: dict, in_variant: np.ndarray,
                  val_batches: list, test_batches: list):
    """Raise unless the written files hold the splits TERRA should read."""
    from terra.datasets.utils import (_check_split_excludes_heldout,
                                      _read_precomputed_split,
                                      load_precomputed_split)
    for name in SPLIT_FILES:
        loaded = np.asarray(load_precomputed_split(str(files[name]), ds),
                            dtype=np.int64)
        if not np.array_equal(loaded, indices[name]):
            raise SplitError(f'{files[name]} does not load with the written '
                             'indices')
    union = np.concatenate([indices[name] for name in SPLIT_FILES])
    if len(np.unique(union)) != len(union):
        raise SplitError('the training, validation and test splits overlap')
    if not np.array_equal(np.sort(union), np.flatnonzero(in_variant)):
        raise SplitError('the splits do not cover exactly the rows of the '
                         "variant's datasets")
    config = {'data': {'val_batch_ids': val_batches,
                       'test_batch_ids': test_batches}}
    train = [indices['train'].tolist()]
    # TERRA's training-time check: a scan of every cell id (no record) ...
    _check_split_excludes_heldout(config, ds, train, str(files['train']),
                                  None)
    # ... and from the record that the training split carries
    _, record = _read_precomputed_split(str(files['train']), ds)
    _check_split_excludes_heldout(config, ds, train, str(files['train']),
                                  record)


def main(args) -> int:
    from datasets import load_from_disk
    from terra.datasets.utils import save_precomputed_split

    t0 = time.time()
    data_dir = Path(args.data_directory_path)
    corpus = Path(args.corpus_path)
    names = read_names(args.names_file)
    variants = parse_variants(args.splits or '', names)
    manifests = read_manifests(data_dir, names)
    entries = (read_split_samples(args.split_samples, manifests)
               if args.split_samples else [])
    val_ids = {e['batch_id'] for e in entries if e['role'] == 'validation'}
    test_ids = {e['batch_id'] for e in entries if e['role'] == 'test'}
    heldout_ids = sorted(val_ids | test_ids, key=natural_key)
    log(f'Corpus: {corpus}')
    if args.split_samples:
        log(f'Split samples: {args.split_samples} ({len(val_ids)} '
            f'validation, {len(test_ids)} test)')
    else:
        log('No split samples: nothing is held out')
    log(f'Variants: {json.dumps(variants)}')

    ds = load_from_disk(str(corpus))
    if 'cell_id' not in ds.column_names:
        raise SplitError(f'{corpus} has no cell_id column')
    codes, batch_ids = corpus_batch_codes(ds)
    code_of = {b: k for k, b in enumerate(batch_ids)}
    rows_per_code = np.bincount(codes, minlength=len(batch_ids))
    log(f'{len(ds):,} rows in {len(batch_ids)} batches '
        f'({time.time() - t0:.0f} s)')

    # Every row belongs to a batch of the silver manifests
    known = {b: name for name, table in manifests.items()
             for b in table['batch_id']}
    unknown = [b for b in batch_ids if b not in known]
    if unknown:
        raise SplitError(f'{len(unknown)} batch ids of the corpus are in no '
                         f'silver manifest of {names}, e.g. {unknown[:5]}')
    differ = []
    for name, table in manifests.items():
        for batch, sample, n_cells in zip(table['batch_id'], table['sample'],
                                          table['n_cells']):
            n_rows = int(rows_per_code[code_of[batch]]) if batch in code_of else 0
            if n_rows != n_cells:
                differ.append(f'{batch} ({name} {sample}): {n_rows} rows, '
                              f'{n_cells} cells')
    if differ:
        log(f'NOTE: {len(differ)} batches have other row counts than their '
            f'silver manifest, e.g. {differ[:5]}')
    empty = [f"{e['dataset']} {e['sample']} ({e['batch_id']})"
             for e in entries if e['batch_id'] not in code_of]
    if empty:
        raise SplitError(f'held-out samples without rows in the corpus: '
                         f'{empty}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = corpus.name.removesuffix('.dataset')
    job_id = os.environ.get('SLURM_JOB_ID') or str(os.getpid())
    for variant, datasets in variants.items():
        batches = {b for d in datasets for b in manifests[d]['batch_id']}
        val_batches = sorted(val_ids & batches, key=natural_key)
        test_batches = sorted(test_ids & batches, key=natural_key)
        signature = {
            'corpus_path': str(corpus.resolve()),
            'n_rows': len(ds),
            'datasets': datasets,
            'validation_batch_ids': val_batches,
            'test_batch_ids': test_batches,
            'excluded_batch_ids': heldout_ids,
        }
        folder = out_dir / variant
        if folder.exists():
            if existing_variant_fits(folder, signature, ds) and not args.overwrite:
                log(f'{variant}: kept {folder} (same corpus, datasets and '
                    'held-out samples; --overwrite rewrites it)')
                continue
            if not args.overwrite:
                raise SplitError(
                    f'{folder} holds splits of another corpus, other '
                    'datasets or other held-out samples. Pass --overwrite '
                    'to replace them (a training run that is already going '
                    'has read its split at start-up).')

        def mask(selected) -> np.ndarray:
            return np.isin(codes, np.array([code_of[b] for b in selected
                                            if b in code_of], dtype=np.int64))
        in_variant = mask(batches)
        is_val, is_test = mask(val_batches), mask(test_batches)
        indices = {'train': np.flatnonzero(in_variant & ~is_val & ~is_test),
                   'validation': np.flatnonzero(is_val),
                   'test': np.flatnonzero(is_test)}
        if not len(indices['train']):
            raise SplitError(f'{variant}: no training rows')
        if args.split_samples and not val_batches and not test_batches:
            log(f'WARNING: {variant} has no validation or test samples')

        partial = out_dir / f'{variant}.partial-{job_id}'
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir()
        files = {name: partial / f'{stem}_precomputed_split_{name}.pkl'
                 for name in SPLIT_FILES}
        save_precomputed_split(ds, indices['train'].tolist(), str(files['train']),
                               excluded_batch_ids=heldout_ids)
        save_precomputed_split(ds, indices['validation'].tolist(),
                               str(files['validation']))
        save_precomputed_split(ds, indices['test'].tolist(), str(files['test']))
        check_variant(ds, files, indices, in_variant, val_batches,
                      test_batches)

        samples = {role: [f"{e['dataset']} {e['sample']} ({e['batch_id']})"
                          for e in entries if e['role'] == role
                          and e['dataset'] in datasets]
                   for role in ('validation', 'test')}
        summary = {
            'variant': variant,
            'datasets': datasets,
            'dataset_ids': sorted({b.split('_', 1)[0] for b in batches},
                                  key=natural_key),
            'n_rows': {name: int(len(indices[name])) for name in SPLIT_FILES},
            'n_rows_corpus': len(ds),
            'samples': samples,
            'files': {name: files[name].name for name in SPLIT_FILES},
            'config_data': {
                'tokenized_data_folder_path': with_data_dir(corpus, data_dir),
                'precomputed_split': with_data_dir(
                    folder / files['train'].name, data_dir),
                **({'norm_factor_file_path': with_data_dir(
                    args.norm_factor_csv, data_dir)}
                   if args.norm_factor_csv else {}),
                **({'token_dict_folder_path': with_data_dir(
                    args.token_dictionary_file_path, data_dir)}
                   if args.token_dictionary_file_path else {}),
                'val_batch_ids': val_batches,
                'test_batch_ids': test_batches,
            },
            'split_samples_file': (str(Path(args.split_samples).resolve())
                                   if args.split_samples else None),
            'signature': signature,
        }
        (partial / 'split_summary.json').write_text(
            json.dumps(summary, indent=1) + '\n')

        replaced = None
        if folder.exists():
            replaced = out_dir / f'{variant}.replaced-{job_id}'
            os.rename(folder, replaced)
        os.rename(partial, folder)
        if replaced is not None:
            shutil.rmtree(replaced, ignore_errors=True)
        n = summary['n_rows']
        log(f"{variant}: datasets {datasets}; rows: train {n['train']:,}, "
            f"validation {n['validation']:,} (samples: {len(val_batches)}), "
            f"test {n['test']:,} (samples: {len(test_batches)}) -> {folder}")
    log(f'Splits checked and written ({time.time() - t0:.0f} s).')
    return 0


def parse_arguments(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus_path', required=True)
    ap.add_argument('--data_directory_path', required=True)
    ap.add_argument('--names_file', required=True,
                    help='Dataset names of the corpus, one per line.')
    ap.add_argument('--split_samples', default=None,
                    help='Held-out samples (default: none).')
    ap.add_argument('--splits', default='',
                    help="Variants, 'name=dataset,dataset ...' (default: "
                         "'all=' every dataset).")
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--norm_factor_csv', default=None,
                    help='For config_data: the norm factors the corpus was '
                         'tokenized with.')
    ap.add_argument('--token_dictionary_file_path', default=None,
                    help='For config_data.')
    ap.add_argument('--overwrite', action='store_true')
    return ap.parse_args(argv)


if __name__ == '__main__':
    try:
        sys.exit(main(parse_arguments()))
    except SplitError as error:
        log(f'ERROR: {error}')
        sys.exit(1)
