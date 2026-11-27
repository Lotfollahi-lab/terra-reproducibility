#!/usr/bin/env python
"""
Helpers of the Slurm corpus build (corpus_job.sh). Subcommands:

tokenizer-config
    Write the create_tokenized_dataset.py config of one silver dataset with
    the tokenizer settings of the re-tokenized hst_corpus_110m (16 neighbours
    x 512 genes, model_input_size 8704, shifted_log, stable rank ties broken
    by tie_break_mean, pad_empty_neighbors, eliminate_stored_zeros, compact
    rows; config/cell-graph-tokenizer/cell-graph-tokenizer_xhs49-1b_1p.yaml),
    except special_tokens: [] (no special-token values are looked up, so the
    token dictionary needs no entries for the new batches; the special-value
    columns are written as 0, the padding value; the model recipe uses no
    metatokens). Prints the path of the tokenized part it will write.

nproc
    Number of tokenization processes that fits the job memory, from the
    memory model of analysis/data_preparation/checks/tokenization_memory.py:
    per input file a dict of n_cells x (24 x (n_neighs + 1) x mean
    min(genes, 512) + 1024) bytes; a parallel job holds up to 3 x nproc files;
    estimate = base_gb + margin x (sum of the 3 x nproc largest file dicts).
    Reads the silver manifests; prints the chosen nproc.

heldout-ids
    Batch ids (<dataset_id>_batch<k>) of the samples listed in a file
    (<dataset name> <sample> per line, tab- or space-separated; a third
    column, the role in a split-samples file, is ignored), from the silver
    manifests. Prints them space-separated (for compute_cohort_norm_factors.py
    --exclude_batch_ids).

part-status
    State of a tokenized part: 'absent'; 'incomplete' (no settings record:
    killed while writing); 'stale' (recorded with another norm-factor file or
    token dictionary than the current ones); 'done'.

norm-status
    Whether existing norm factors fit the current silver files: 'absent',
    'current' (their record counts the files and cells of the silver
    manifests, the same excluded batches and token dictionary, and they are
    newer than every silver folder), or 'outdated: <reason>'.

corpus-status
    Whether the concatenated corpus fits the tokenized parts. Fails (exit 1)
    unless every part of the corpus is 'done'. Prints '<status> <corpus
    path>': 'absent'; 'current' (its settings record has the current norm
    factors, is newer than every part's record, and the corpus has as many
    rows as the parts); 'outdated' (the reason goes to stderr).

propose-split-samples
    A random validation and test set of samples per dataset (seeded; at
    least one sample each when the fraction is above 0), written as a
    split-samples file for make_splits.py: '<dataset> <sample> <val|test>'
    per line. Refuses to replace an existing file (edits are kept) unless
    --overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TOKENIZER = 'cell-graph-tokenizer'
# Tokenization section of the hst_corpus_110m tokenizer configs (16 x 512,
# compact), as in config/cell-graph-tokenizer/cell-graph-tokenizer_*.yaml
# on feat/compact-tokenized-data
CORPUS_TOKENIZATION = {
    'count_cell_norm_method': None,
    'count_count_norm_method': 'shifted_log',
    'count_gene_norm_method': None,
    'delaunay': False,
    'eliminate_stored_zeros': True,
    'include_zero_expr_genes': False,
    'keep_in_memory': False,
    'n_neighs': 16,
    'num_shards': 32,
    'pad_empty_neighbors': True,
    'processing_mode': 'parallel',
    'radius': None,
    'rank_cell_norm_method': None,
    'rank_count_norm_method': None,
    'rank_gene_norm_method': None,
    'rank_tie_break_norm_factor': 'tie_break_mean',
    'species': 'homo_sapiens',
    'stable_rank_ties': True,
    'storage_format': 'compact',
    'stream_per_file': True,
    'use_generator': True,
    'special_tokens': [],
}
SEQ_LEN_CELL = 512


def part_name(dataset_name: str, tokenization: dict,
              model_input_size: int) -> str:
    """File name create_tokenized_dataset.py gives the tokenized part."""
    t = tokenization
    graph = ''
    if t['n_neighs']:
        graph += f"knn_{t['n_neighs']}_"
    if t['radius']:
        graph += f"radius_{t['radius']}_"
    if t['delaunay']:
        graph += 'delaunay_'
    return (f"{dataset_name}_{t['num_shards']}_"
            f"{t['rank_cell_norm_method']}_{t['rank_gene_norm_method']}_"
            f"{t['rank_count_norm_method']}_{t['count_cell_norm_method']}_"
            f"{t['count_gene_norm_method']}_{t['count_count_norm_method']}_"
            f"{graph}{model_input_size}"
            f"{'_incl_zero_expr' if t['include_zero_expr_genes'] else ''}"
            '.dataset')


def tokenizer_config(args) -> int:
    tokenization = dict(CORPUS_TOKENIZATION)
    tokenization['species'] = args.species
    model_input_size = (tokenization['n_neighs'] + 1) * SEQ_LEN_CELL
    manifest = (Path(args.data_directory_path) / 'silver'
                / args.dataset_name / 'silver_manifest.tsv')
    n_batches = (len(pd.read_csv(manifest, sep='\t'))
                 if manifest.exists() else '?')
    config = {
        'experiment': {
            'experiment_description': (
                f'{args.dataset_name} (dataset-id {args.dataset_id}) | '
                f'{n_batches} batches | corpus {args.corpus_name}'),
            'experiment_name': args.dataset_name,
            'model_input_size': model_input_size,
            'tokenizer': TOKENIZER,
        },
        'paths': {
            'data_directory_path': str(args.data_directory_path),
            'norm_factor_file_path': str(args.norm_factor_csv),
            'pf_targets_file_path': None,
            'token_dictionary_file_path': str(args.token_dictionary_file_path),
        },
        'tokenization': tokenization,
        'wandb': {
            'job_type': 'create-tokenized-dataset',
            'mode': 'disabled',
            'project': 'TERRA',
            'save_artifact': False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=True)
    part = (Path(args.data_directory_path) / 'gold' / TOKENIZER
            / args.dataset_name
            / part_name(args.dataset_name, tokenization, model_input_size))
    print(f'Wrote {out}', file=sys.stderr)
    print(part)
    return 0


def slurm_memory_gb() -> float | None:
    """Memory of this Slurm job in GB, or None outside Slurm."""
    per_node = os.environ.get('SLURM_MEM_PER_NODE')
    if per_node:
        return float(per_node) / 1024
    per_cpu = os.environ.get('SLURM_MEM_PER_CPU')
    cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if per_cpu and cpus:
        return float(per_cpu) * int(cpus) / 1024
    return None


def nproc(args) -> int:
    manifest = (Path(args.data_directory_path) / 'silver'
                / args.dataset_name / 'silver_manifest.tsv')
    table = pd.read_csv(manifest, sep='\t')
    bytes_per_entry = 24 * (args.n_neighs + 1)
    dicts_gb = sorted(
        (table['n_cells'] * (bytes_per_entry * table['mean_filled_512']
                             + 1024) / 1e9).tolist(), reverse=True)
    mem_gb = args.mem_gb if args.mem_gb else slurm_memory_gb()

    def estimate(cores):
        return args.base_gb + args.margin * sum(dicts_gb[:3 * cores])

    chosen = 1
    if mem_gb is None:
        chosen = args.cpus
        message = 'job memory unknown; using every CPU'
    else:
        budget = args.headroom * mem_gb
        fitting = [c for c in range(args.cpus, 0, -1)
                   if estimate(c) <= budget]
        chosen = fitting[0] if fitting else 1
        message = (f'estimate {estimate(chosen):.0f} GB of {mem_gb:.0f} GB '
                   f'(budget {budget:.0f} GB)')
        if not fitting:
            message += ('; WARNING: even 1 process is estimated above the '
                        'budget, request more memory if the job is killed')
    print(f'{len(dicts_gb)} files, largest file dict {dicts_gb[0]:.1f} GB; '
          f'{chosen} of {args.cpus} processes: {message}', file=sys.stderr)
    print(chosen)
    return 0


def heldout_ids(args) -> int:
    pairs = []
    with open(args.heldout_samples) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) not in (2, 3):
                raise SystemExit(f'ERROR: expected <dataset> <sample> '
                                 f'(tab- or space-separated; a third column '
                                 f'is ignored): {line!r}')
            pairs.append((fields[0].strip(), fields[1].strip()))
    batch_ids, missing = [], []
    manifests = {}
    for dataset, sample in pairs:
        if dataset not in manifests:
            path = (Path(args.data_directory_path) / 'silver' / dataset
                    / 'silver_manifest.tsv')
            if not path.exists():
                raise SystemExit(f'ERROR: no manifest {path}')
            table = pd.read_csv(path, sep='\t', dtype={'sample': str})
            manifests[dataset] = dict(zip(table['sample'], table['batch_id']))
        batch_id = manifests[dataset].get(sample)
        if batch_id is None:
            missing.append(f'{dataset}/{sample}')
        else:
            batch_ids.append(batch_id)
    if missing:
        raise SystemExit(f'ERROR: held-out samples not in the silver '
                         f'manifests: {missing}')
    print(f'{len(batch_ids)} held-out batches', file=sys.stderr)
    print(' '.join(batch_ids))
    return 0


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def part_state(part: Path, norm_factor_sha256: str,
               token_dictionary_sha256: str) -> str:
    """'absent', 'incomplete', 'stale' or 'done' (see part-status)."""
    record = part / 'terra_tokenizer_params.json'
    if not part.exists():
        return 'absent'
    if not record.exists():
        return 'incomplete'
    recorded = json.loads(record.read_text())
    same = (recorded.get('norm_factor_sha256') == norm_factor_sha256
            and recorded.get('token_dictionary_sha256')
            == token_dictionary_sha256)
    return 'done' if same else 'stale'


def part_status(args) -> int:
    print(part_state(Path(args.part), file_sha256(args.norm_factor_csv),
                     file_sha256(args.token_dictionary_file_path)))
    return 0


def read_names(names_file) -> list[str]:
    """Dataset names of the corpus, one per line."""
    return [line.strip() for line in Path(names_file).read_text().splitlines()
            if line.strip()]


def corpus_dataset_path(data_directory_path, name: str) -> Path:
    """Path of the tokenized part of dataset `name`, or of the corpus when
    `name` is the corpus name (concatenate_datasets.py names it alike)."""
    tokenization = dict(CORPUS_TOKENIZATION)
    model_input_size = (tokenization['n_neighs'] + 1) * SEQ_LEN_CELL
    return (Path(data_directory_path) / 'gold' / TOKENIZER / name
            / part_name(name, tokenization, model_input_size))


def dataset_rows(path: Path) -> int:
    from datasets import load_from_disk
    return len(load_from_disk(str(path)))


def corpus_status(args) -> int:
    names = read_names(args.names_file)
    norm_sha = file_sha256(args.norm_factor_csv)
    dictionary_sha = file_sha256(args.token_dictionary_file_path)
    not_done, newest_part, n_part_rows = [], 0.0, 0
    for name in names:
        part = corpus_dataset_path(args.data_directory_path, name)
        state = part_state(part, norm_sha, dictionary_sha)
        if state != 'done':
            not_done.append(f'{name}: {state}')
            continue
        newest_part = max(newest_part, (part / 'terra_tokenizer_params.json')
                          .stat().st_mtime)
        n_part_rows += dataset_rows(part)
    if not_done:
        print(f'ERROR: parts not tokenized with the current norm factors and '
              f'token dictionary: {not_done}', file=sys.stderr)
        return 1
    corpus = corpus_dataset_path(args.data_directory_path, args.corpus_name)
    record = corpus / 'terra_tokenizer_params.json'
    reasons = []
    if not corpus.exists():
        status = 'absent'
    elif not record.exists():
        status = 'outdated'
        reasons.append('no settings record')
    else:
        recorded = json.loads(record.read_text())
        if recorded.get('norm_factor_sha256') != norm_sha:
            reasons.append('other norm factors')
        if record.stat().st_mtime < newest_part:
            reasons.append('a part is newer')
        n_rows = dataset_rows(corpus)
        if n_rows != n_part_rows:
            reasons.append(f'{n_rows:,} rows, the parts have {n_part_rows:,}')
        status = 'outdated' if reasons else 'current'
    print(f'{len(names)} parts done ({n_part_rows:,} rows); corpus {status}'
          + (f': {"; ".join(reasons)}' if reasons else ''), file=sys.stderr)
    print(f'{status} {corpus}')
    return 0


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', str(text))]


def propose_split_samples(args) -> int:
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f'ERROR: {out} exists (it may hold edits); pass --overwrite to '
              'replace it.', file=sys.stderr)
        return 1

    def count(fraction: float, n: int) -> int:
        return 0 if fraction <= 0 else max(1, int(fraction * n + 0.5))

    lines = [
        '# Validation and test samples of the corpus splits (make_splits.py):',
        '# <dataset> <sample> <val|test>; samples not listed are trained on.',
        f'# Proposed by corpus_tools.py propose-split-samples (seed '
        f'{args.seed}, validation fraction {args.val_fraction}, test fraction '
        f'{args.test_fraction}, at least one sample each). Edit freely.',
    ]
    summary = []
    for name in read_names(args.names_file):
        manifest = (Path(args.data_directory_path) / 'silver' / name
                    / 'silver_manifest.tsv')
        if not manifest.exists():
            print(f'ERROR: no manifest {manifest}', file=sys.stderr)
            return 1
        samples = sorted(pd.read_csv(manifest, sep='\t',
                                     dtype={'sample': str})['sample'],
                         key=natural_key)
        n = len(samples)
        n_val, n_test = count(args.val_fraction, n), count(args.test_fraction, n)
        if n_val + n_test >= n:
            print(f'ERROR: {name} has {n} samples, too few for {n_val} '
                  f'validation and {n_test} test samples.', file=sys.stderr)
            return 1
        # One stream per dataset: adding a dataset keeps the others' choice
        rng = np.random.default_rng([args.seed, zlib.crc32(name.encode())])
        order = rng.permutation(n)
        val = sorted((samples[i] for i in order[:n_val]), key=natural_key)
        test = sorted((samples[i] for i in order[n_val:n_val + n_test]),
                      key=natural_key)
        lines += [f'{name}\t{s}\tval' for s in val]
        lines += [f'{name}\t{s}\ttest' for s in test]
        summary.append(f'{name}: {n} samples, validation {val}, test {test}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(summary), file=sys.stderr)
    print(f'Wrote {out}', file=sys.stderr)
    return 0


def norm_status(args) -> int:
    csv = Path(args.norm_factor_csv)
    record = csv.with_suffix('.json')
    if not csv.exists():
        print('absent')
        return 0
    if not record.exists():
        print(f'outdated: no record {record.name}')
        return 0
    recorded = json.loads(record.read_text())
    excluded = set(args.excluded_batch_ids or [])
    names = [line.strip() for line in Path(args.names_file).read_text()
             .splitlines() if line.strip()]
    n_files = n_cells = 0
    newest_silver = 0.0
    for name in names:
        folder = Path(args.data_directory_path) / 'silver' / name
        table = pd.read_csv(folder / 'silver_manifest.tsv', sep='\t')
        n_files += len(table)
        n_cells += int(table.loc[~table['batch_id'].isin(excluded),
                                 'n_cells'].sum())
        newest_silver = max(newest_silver,
                            (folder / 'silver_prep.json').stat().st_mtime)
    reasons = []
    if recorded.get('n_files') != n_files:
        reasons.append(f"{recorded.get('n_files')} files recorded, "
                       f'{n_files} in the silver manifests')
    if recorded.get('n_cells') != n_cells:
        reasons.append(f"{recorded.get('n_cells')} cells recorded, "
                       f'{n_cells} in the silver manifests')
    if set(recorded.get('excluded_batch_ids', [])) != excluded:
        reasons.append('other excluded batches')
    if (str(recorded.get('token_dictionary_file_path'))
            != str(args.token_dictionary_file_path)):
        reasons.append('another token dictionary')
    if csv.stat().st_mtime < newest_silver:
        reasons.append('a silver folder is newer')
    print('current' if not reasons else 'outdated: ' + '; '.join(reasons))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('tokenizer-config')
    p.add_argument('--dataset_name', required=True)
    p.add_argument('--dataset_id', required=True)
    p.add_argument('--corpus_name', required=True)
    p.add_argument('--data_directory_path', required=True)
    p.add_argument('--token_dictionary_file_path', required=True)
    p.add_argument('--norm_factor_csv', required=True)
    p.add_argument('--species', default='homo_sapiens')
    p.add_argument('--out', required=True)
    p.set_defaults(func=tokenizer_config)

    p = sub.add_parser('nproc')
    p.add_argument('--dataset_name', required=True)
    p.add_argument('--data_directory_path', required=True)
    p.add_argument('--cpus', type=int, required=True)
    p.add_argument('--mem_gb', type=float, default=None,
                   help='Default: the Slurm job memory.')
    p.add_argument('--n_neighs', type=int, default=16)
    p.add_argument('--base_gb', type=float, default=16.0)
    p.add_argument('--margin', type=float, default=1.3)
    p.add_argument('--headroom', type=float, default=0.9)
    p.set_defaults(func=nproc)

    p = sub.add_parser('heldout-ids')
    p.add_argument('--heldout_samples', required=True)
    p.add_argument('--data_directory_path', required=True)
    p.set_defaults(func=heldout_ids)

    p = sub.add_parser('part-status')
    p.add_argument('--part', required=True)
    p.add_argument('--norm_factor_csv', required=True)
    p.add_argument('--token_dictionary_file_path', required=True)
    p.set_defaults(func=part_status)

    p = sub.add_parser('norm-status')
    p.add_argument('--norm_factor_csv', required=True)
    p.add_argument('--names_file', required=True)
    p.add_argument('--data_directory_path', required=True)
    p.add_argument('--token_dictionary_file_path', required=True)
    p.add_argument('--excluded_batch_ids', nargs='*', default=[])
    p.set_defaults(func=norm_status)

    p = sub.add_parser('corpus-status')
    p.add_argument('--names_file', required=True)
    p.add_argument('--data_directory_path', required=True)
    p.add_argument('--corpus_name', required=True)
    p.add_argument('--norm_factor_csv', required=True)
    p.add_argument('--token_dictionary_file_path', required=True)
    p.set_defaults(func=corpus_status)

    p = sub.add_parser('propose-split-samples')
    p.add_argument('--names_file', required=True,
                   help='Dataset names of the corpus (tokenizer/<corpus>/'
                        '<corpus>_dataset_names.txt).')
    p.add_argument('--data_directory_path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--val_fraction', type=float, default=0.05)
    p.add_argument('--test_fraction', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    p.set_defaults(func=propose_split_samples)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
