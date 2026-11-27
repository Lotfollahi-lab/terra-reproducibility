#!/usr/bin/env python
"""
Compare the embeddings of the same cells across gene panel runs: runs of gene_panel_subsampling.py with different
models, e.g. TERRA-96M, a model trained on the full panel and models trained on smaller panels (panel_training.py).

Each run is read from its folder (run_config.json, cells.tsv.gz, panels.json, embeddings/, evaluation/). A run whose
model folder holds panel_model.json (written by `panel_training.py package`) used a model trained here on one panel:
'trained_full' if that panel is the full panel, else 'trained_panel'. Any other run used a 'pretrained' model.

Pairs (column pair_type), for every TERRA representation (cell and neighborhood embeddings):
- same_model_panel_vs_full: a condition of a run against the full panel of the same run (one model, one embedding
  space: the inference-only, "zero-shot" panel effect);
- same_panel_across_models: a condition against the same condition in another run (the same genes, two models); for
  two 'trained_full' runs on the full panel, this is the training-seed ceiling of every cross-model metric;
- trained_panel_vs_full: the panel a 'trained_panel' model was trained on against the full panel of every other run
  (a model trained on the panel against a full-panel model); column same_seed marks two models trained here with
  the same seed, which share their initialization (the figure leaves them out: the fair contrast uses the other
  seed, as the training-seed ceiling does);
- panel_vs_other_model_full: a condition of a 'trained_full' run against the full panel of another 'trained_full' run
  (the zero-shot counterpart of trained_panel_vs_full: also two models with the same training data, but the panel
  is only applied at inference).

Metrics (column metric; see embedding_similarity.py), on the cells that every run evaluates:
- any pair: knn_overlap (k nearest neighbors within the section, k = the runs' knn_k), linear_cka,
  representational_similarity (Spearman correlation of pairwise cosine distances, 1,500 cells per section),
  cluster_ari_kmatched / cluster_nmi_kmatched (K-matched Leiden partitions of the evaluate step, mean over seed
  pairs), and after an orthogonal Procrustes alignment fit on a random half of each section's cells and scored on
  the other half: procrustes_cosine (matched cosine), procrustes_foscttm, procrustes_top1, procrustes_top10;
- pairs in one embedding space (same model folder) also: cosine and cosine_median (matched cosine), cosine_centered
  (after subtracting each embedding's mean), cosine_null (a random other cell of the section), cosine_gap (cosine
  minus cosine_null), and re-identification without alignment on the same held-out half: foscttm, top1, top10.
Values are given per section and over all cells (section 'all').

Outputs (--output_folder_path): runs.tsv, pairs.tsv, similarity_long.tsv, similarity_summary.tsv (mean, sd over
random panel draws), biology_long.tsv (the evaluate step's annotation metrics of every run), compare_done.json and
figures/ (plot_panel_comparison.py).

Example:
  python compare_panel_runs.py --output_folder_path .../panel_comparison/xhs1000_skin_panels \\
      --runs terra96m=.../panel_robustness/xhs1000_3sections_terra96m \\
             full_s0=.../panel_robustness/xhs1000_skin_full_s0 full_s1=.../panel_robustness/xhs1000_skin_full_s1 \\
             random_1000_r0_s0=.../panel_robustness/xhs1000_skin_random_1000_r0_s0
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_similarity import (matched_cosine, null_cosine, procrustes_fit,  # noqa: E402
                                  procrustes_transform, reidentification, representational_similarity,
                                  split_halves)
from panel_metrics import exact_knn_indices, knn_overlap, linear_cka, mean_pairwise_agreement  # noqa: E402

COMPARE_VERSION = 1
TERRA_REPRESENTATIONS = {'terra_neighborhood': 'neighborhood_emb', 'terra_cell': 'cell_emb',
                         'terra_spatial_cell': 'spatial_cell_emb'}
PAIR_TYPES = ('same_model_panel_vs_full', 'same_panel_across_models', 'trained_panel_vs_full',
              'panel_vs_other_model_full', 'leiden_seed_ceiling')


def log(message: str):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}', file=sys.stderr, flush=True)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj):
    tmp_path = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp_path, path)


def hash_list(values) -> str:
    return hashlib.sha1('\n'.join(map(str, values)).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------------------------------

class Run:
    """One gene_panel_subsampling.py run folder."""

    def __init__(self, label: str, folder: str):
        self.label = label
        self.folder = Path(folder).resolve()
        self.cfg = read_json(self.folder / 'run_config.json')
        self.model_folder = Path(self.cfg['model_folder_path']).resolve()
        panel_model_file = self.model_folder / 'panel_model.json'
        self.panel_model = read_json(panel_model_file) if panel_model_file.is_file() else None
        if self.panel_model is None:
            self.kind, self.trained_panel = 'pretrained', None
        else:
            self.trained_panel = self.panel_model['panel']
            self.kind = 'trained_full' if self.trained_panel == 'full' else 'trained_panel'
        self.panels = read_json(self.folder / 'panels.json')['panels']
        cells = pd.read_csv(self.folder / 'cells.tsv.gz', sep='\t', keep_default_na=True,
                            dtype={'cell': str, 'section': str, 'sample': str, 'niche': object, 'cell_type': object})
        self.cells = cells
        self.eval_cells = cells.loc[cells['eval'].astype(bool), 'cell'].tolist()
        self.conditions = {}
        for condition in self.cfg['conditions']:
            condition_id = condition['condition_id']
            embedded = (self.folder / 'embeddings' / f'{condition_id}.npz').is_file()
            evaluated = (self.folder / 'evaluation' / condition_id / 'done.json').is_file()
            if embedded and evaluated:
                self.conditions[condition_id] = {**condition, 'n_genes': len(self.panels[condition_id]['genes'])}
            else:
                log(f'{label}: {condition_id} is not embedded and evaluated yet; left out.')
        if self.trained_panel is not None and self.trained_panel not in self.conditions:
            log(f'WARNING: {label} was trained on {self.trained_panel}, which this run has not embedded.')

    def describe(self) -> dict:
        return {'run': self.label, 'kind': self.kind, 'trained_panel': self.trained_panel,
                'seed': (self.panel_model or {}).get('seed'), 'folder': str(self.folder),
                'model_folder': str(self.model_folder), 'n_conditions': len(self.conditions),
                'n_eval_cells': len(self.eval_cells)}


def common_cells(runs: list) -> pd.DataFrame:
    """The cells every run evaluates, in the order of the first run, with section and labels of the first run."""
    shared = set(runs[0].eval_cells)
    for run in runs[1:]:
        shared &= set(run.eval_cells)
    first = runs[0].cells.set_index('cell')
    cells = first.loc[[c for c in runs[0].eval_cells if c in shared]].reset_index()
    for run in runs[1:]:
        other = run.cells.set_index('cell').loc[cells['cell'], 'section'].to_numpy()
        if not np.array_equal(other.astype(str), cells['section'].to_numpy().astype(str)):
            raise ValueError(f'{run.label} assigns cells to other sections than {runs[0].label}.')
    log(f'{len(cells)} cells are evaluated by every run (of {[len(r.eval_cells) for r in runs]}).')
    if len(cells) == 0:
        raise ValueError('No cell is evaluated by every run.')
    return cells


def plan_pairs(runs: list, pair_types: list) -> list:
    """Pairs of (run, condition) to compare; see the module docstring."""
    pairs = []

    def add(pair_type, run_a, condition_a, run_b, condition_b):
        pairs.append({'pair_type': pair_type, 'run_a': run_a.label, 'condition_a': condition_a,
                      'run_b': run_b.label, 'condition_b': condition_b})

    if 'same_model_panel_vs_full' in pair_types:
        for run in runs:
            if 'full' in run.conditions:
                for condition_id in run.conditions:
                    if condition_id != 'full':
                        add('same_model_panel_vs_full', run, condition_id, run, 'full')
    if 'same_panel_across_models' in pair_types:
        for run_a, run_b in combinations(runs, 2):
            for condition_id in run_a.conditions:
                if condition_id in run_b.conditions:
                    add('same_panel_across_models', run_a, condition_id, run_b, condition_id)
    if 'trained_panel_vs_full' in pair_types:
        for run_a in runs:
            if run_a.kind == 'trained_panel' and run_a.trained_panel in run_a.conditions:
                for run_b in runs:
                    if run_b is not run_a and 'full' in run_b.conditions and run_b.kind != 'trained_panel':
                        add('trained_panel_vs_full', run_a, run_a.trained_panel, run_b, 'full')
    if 'panel_vs_other_model_full' in pair_types:
        trained_full = [run for run in runs if run.kind == 'trained_full']
        for run_a in trained_full:
            for run_b in trained_full:
                if run_b is not run_a and 'full' in run_b.conditions:
                    for condition_id in run_a.conditions:
                        if condition_id not in ('full', 'full_repeat'):
                            add('panel_vs_other_model_full', run_a, condition_id, run_b, 'full')
    if 'leiden_seed_ceiling' in pair_types:
        for run in runs:
            if 'full' in run.conditions:
                add('leiden_seed_ceiling', run, 'full', run, 'full')
    for i, pair in enumerate(pairs):
        pair['pair_id'] = i
    return pairs


# ----------------------------------------------------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------------------------------------------------

class Store:
    """Embeddings, kNN indices and partitions of (run, condition, representation) on the common cells."""

    def __init__(self, runs: dict, cells: pd.DataFrame, knn_k: int, n_jobs: int, max_cached: int):
        self.runs = runs
        self.cell_names = cells['cell'].tolist()
        self.sections = cells['section'].to_numpy().astype(str)
        self.knn_k = knn_k
        self.n_jobs = n_jobs
        self.max_cached = max_cached
        self.embeddings = OrderedDict()
        self.knn = {}
        self.partitions = {}
        self.common_hash = hash_list(self.cell_names)
        self.rows = {}

    def _rows(self, names) -> np.ndarray:
        """Positions of the common cells in a list of cell names."""
        key = hash_list(names)
        if key not in self.rows:
            position = {cell: i for i, cell in enumerate(names)}
            missing = [cell for cell in self.cell_names if cell not in position]
            if missing:
                raise ValueError(f'{len(missing)} common cells are missing (e.g. {missing[:3]}).')
            self.rows[key] = np.fromiter((position[c] for c in self.cell_names), dtype=np.int64,
                                         count=len(self.cell_names))
        return self.rows[key]

    def embedding(self, run_label: str, condition_id: str, representation: str) -> np.ndarray:
        key = (run_label, condition_id, representation)
        if key in self.embeddings:
            self.embeddings.move_to_end(key)
            return self.embeddings[key]
        run = self.runs[run_label]
        path = run.folder / 'embeddings' / f'{condition_id}.npz'
        with np.load(path, allow_pickle=False) as data:
            array_key = TERRA_REPRESENTATIONS[representation]
            if array_key not in data:
                raise KeyError(f'{path} has no {array_key}.')
            X = np.asarray(data[array_key][self._rows(list(data['obs_names']))], dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f'{run_label} {condition_id} {representation} has non-finite values.')
        self.embeddings[key] = X
        while len(self.embeddings) > self.max_cached:
            self.embeddings.popitem(last=False)
        return X

    def knn_indices(self, run_label: str, condition_id: str, representation: str) -> np.ndarray:
        key = (run_label, condition_id, representation)
        if key not in self.knn:
            run = self.runs[run_label]
            saved = run.folder / 'evaluation' / condition_id / 'knn.npz'
            if (run.cfg['knn_k'] == self.knn_k and hash_list(run.eval_cells) == self.common_hash
                    and saved.is_file()):
                with np.load(saved) as data:
                    self.knn[key] = data[representation].astype(np.int64)
            else:
                X = self.embedding(run_label, condition_id, representation)
                self.knn[key] = exact_knn_indices(X, self.sections, self.knn_k, n_jobs=self.n_jobs)
        return self.knn[key]

    def kmatched_partitions(self, run_label: str, condition_id: str, representation: str) -> np.ndarray:
        key = (run_label, condition_id, representation)
        if key not in self.partitions:
            run = self.runs[run_label]
            with np.load(run.folder / 'evaluation' / condition_id / 'partitions.npz') as data:
                partitions = data[f'kmatched__{representation}']
            self.partitions[key] = partitions[:, self._rows(run.eval_cells)]
        return self.partitions[key]


# ----------------------------------------------------------------------------------------------------------------------
# Metrics of one pair
# ----------------------------------------------------------------------------------------------------------------------

def per_section(values: np.ndarray, sections: np.ndarray, metric: str, fn=np.mean) -> list:
    rows = [{'metric': metric, 'section': 'all', 'value': float(fn(values))}]
    for section in np.unique(sections):
        rows.append({'metric': metric, 'section': str(section), 'value': float(fn(values[sections == section]))})
    return rows


def compare_pair(store: Store, pair: dict, representation: str, same_space: bool, fit_mask: np.ndarray,
                 seed: int) -> list:
    sections = store.sections
    a = (pair['run_a'], pair['condition_a'], representation)
    b = (pair['run_b'], pair['condition_b'], representation)
    rows = []
    if a == b:
        # Leiden seed-to-seed agreement of one embedding: the ceiling of cluster agreement
        agreement = mean_pairwise_agreement(store.kmatched_partitions(*a), store.kmatched_partitions(*a),
                                            exclude_same_index=True)
        rows.append({'metric': 'cluster_ari_kmatched', 'section': 'all', 'value': agreement['ari']})
        rows.append({'metric': 'cluster_nmi_kmatched', 'section': 'all', 'value': agreement['nmi']})
        for row in rows:
            row['pair_id'] = pair['pair_id']
            row['representation'] = representation
        return rows
    Xa, Xb = store.embedding(*a), store.embedding(*b)

    # Invariant to rotation and isotropic scaling: also across models
    rows += per_section(knn_overlap(store.knn_indices(*a), store.knn_indices(*b)), sections, 'knn_overlap')
    rows.append({'metric': 'linear_cka', 'section': 'all', 'value': linear_cka(Xa, Xb)})
    for section in np.unique(sections):
        mask = sections == section
        rows.append({'metric': 'linear_cka', 'section': str(section), 'value': linear_cka(Xa[mask], Xb[mask])})
    rsa = representational_similarity(Xa, Xb, sections, np.random.default_rng([seed, 1]))
    rows.append({'metric': 'representational_similarity', 'section': 'all', 'value': float(np.mean(list(rsa.values())))})
    rows += [{'metric': 'representational_similarity', 'section': s, 'value': v} for s, v in rsa.items()]
    # Pairs of different Leiden seeds only, as in the summarize step (the same seed on two identical embeddings
    # gives identical partitions)
    agreement = mean_pairwise_agreement(store.kmatched_partitions(*a), store.kmatched_partitions(*b),
                                        exclude_same_index=True)
    rows.append({'metric': 'cluster_ari_kmatched', 'section': 'all', 'value': agreement['ari']})
    rows.append({'metric': 'cluster_nmi_kmatched', 'section': 'all', 'value': agreement['nmi']})

    # Matched cells after an orthogonal alignment fit on the other half of the cells
    test_sections = sections[~fit_mask]
    if Xa.shape[1] == Xb.shape[1]:
        fit = procrustes_fit(Xa[fit_mask], Xb[fit_mask])
        A, B = procrustes_transform(Xa[~fit_mask], Xb[~fit_mask], fit)
        rows += per_section(matched_cosine(A, B), test_sections, 'procrustes_cosine')
        for name, values in reidentification(A, B, test_sections).items():
            rows += per_section(values, test_sections, f'procrustes_{name}')
    else:
        log(f'Different dimensions ({Xa.shape[1]} vs {Xb.shape[1]}): no Procrustes metrics for pair '
            f'{pair["pair_id"]}.')

    # One embedding space: the same cell by its coordinates
    if same_space:
        cosine = matched_cosine(Xa, Xb)
        rows += per_section(cosine, sections, 'cosine')
        rows += per_section(cosine, sections, 'cosine_median', fn=np.median)
        centered = matched_cosine(Xa - Xa.mean(axis=0), Xb - Xb.mean(axis=0))
        rows += per_section(centered, sections, 'cosine_centered')
        null = null_cosine(Xa, Xb, sections, np.random.default_rng([seed, 2]))
        rows += per_section(null, sections, 'cosine_null')
        rows += per_section(cosine - null, sections, 'cosine_gap')
        for name, values in reidentification(Xa[~fit_mask], Xb[~fit_mask], test_sections).items():
            rows += per_section(values, test_sections, name)
    for row in rows:
        row['pair_id'] = pair['pair_id']
        row['representation'] = representation
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------------------------------

def pair_table(pairs: list, runs: dict) -> pd.DataFrame:
    table = pd.DataFrame(pairs)
    for side in ('a', 'b'):
        run_column, condition_column = f'run_{side}', f'condition_{side}'
        table[f'kind_{side}'] = [runs[r].kind for r in table[run_column]]
        table[f'trained_panel_{side}'] = [runs[r].trained_panel for r in table[run_column]]
        table[f'model_{side}'] = [str(runs[r].model_folder) for r in table[run_column]]
        table[f'seed_{side}'] = [(runs[r].panel_model or {}).get('seed') for r in table[run_column]]
        table[f'n_genes_{side}'] = [runs[r].conditions[c]['n_genes'] for r, c in zip(table[run_column],
                                                                                    table[condition_column])]
        table[f'panel_kind_{side}'] = [runs[r].conditions[c]['kind'] for r, c in zip(table[run_column],
                                                                                   table[condition_column])]
        table[f'panel_size_{side}'] = [runs[r].conditions[c]['size'] for r, c in zip(table[run_column],
                                                                                   table[condition_column])]
        table[f'replicate_{side}'] = [runs[r].conditions[c]['replicate'] for r, c in zip(table[run_column],
                                                                                       table[condition_column])]
    table['same_space'] = table['model_a'] == table['model_b']
    # Two models trained here with the same seed share their initialization
    table['same_seed'] = (table['seed_a'].notna() & (table['seed_a'] == table['seed_b'])
                          & (table['model_a'] != table['model_b']))
    return table


def biology_table(runs: dict) -> pd.DataFrame:
    """The evaluate step's annotation metrics (comparison 'intrinsic') of every run and condition."""
    frames = []
    for run in runs.values():
        for condition_id, condition in run.conditions.items():
            records = pd.DataFrame(read_json(run.folder / 'evaluation' / condition_id / 'metrics.json'))
            records = records[records['comparison'] == 'intrinsic']
            records = records.assign(run=run.label, kind=run.kind, trained_panel=run.trained_panel,
                                     n_genes=condition['n_genes'], panel_kind=condition['kind'],
                                     panel_size=condition['size'], replicate=condition['replicate'],
                                     n_eval_cells=len(run.eval_cells))
            frames.append(records)
    return pd.concat(frames, ignore_index=True)


def summary_table(long: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Mean, sd, min, max and count over random panel draws (replicates) of each comparison."""
    overall = long[long['section'] == 'all'].merge(pairs, on='pair_id')
    # The reference side: 'same_panel' when both sides embed the same panel (aggregated over draws), else its condition
    overall['reference_b'] = np.where(overall['condition_b'] == overall['condition_a'], 'same_panel',
                                      overall['condition_b'])
    keys = ['pair_type', 'run_a', 'kind_a', 'run_b', 'kind_b', 'panel_kind_a', 'n_genes_a', 'reference_b',
            'representation', 'metric']
    summary = (overall.groupby(keys, dropna=False)['value'].agg(['mean', 'std', 'min', 'max', 'count'])
               .reset_index())
    return summary.sort_values(['pair_type', 'representation', 'metric', 'run_a', 'run_b', 'n_genes_a'])


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def parse_runs(items: list) -> dict:
    runs = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f"--runs takes LABEL=RUN_FOLDER items, got '{item}'.")
        label, folder = item.split('=', 1)
        if not label or label in runs:
            raise ValueError(f"Run labels must be distinct and not empty: '{item}'.")
        runs[label] = Run(label, folder)
    return runs


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_folder_path', required=True)
    parser.add_argument('--runs', nargs='+', required=True, metavar='LABEL=RUN_FOLDER',
                        help='gene_panel_subsampling.py run folders; the first run gives sections and labels.')
    parser.add_argument('--representations', nargs='+', default=['terra_neighborhood', 'terra_cell'],
                        choices=sorted(TERRA_REPRESENTATIONS))
    parser.add_argument('--pair_types', nargs='+', default=list(PAIR_TYPES), choices=PAIR_TYPES)
    parser.add_argument('--knn_k', type=int, default=None, help='Default: the first run\'s knn_k.')
    parser.add_argument('--seed', type=int, default=0, help='Procrustes halves, null pairs and RSA samples.')
    parser.add_argument('--max_cached_embeddings', type=int, default=48)
    parser.add_argument('--no_figures', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    out = Path(args.output_folder_path)
    out.mkdir(parents=True, exist_ok=True)
    n_jobs = int(os.environ.get('LSB_DJOB_NUMPROC', os.cpu_count() or 1))
    runs = parse_runs(args.runs)
    run_list = list(runs.values())
    pd.DataFrame([run.describe() for run in run_list]).to_csv(out / 'runs.tsv', sep='\t', index=False)
    for run in run_list:
        log(f'{run.label}: {run.kind}{" on " + run.trained_panel if run.trained_panel else ""}, '
            f'{len(run.conditions)} conditions, model {run.model_folder}.')
    cells = common_cells(run_list)
    knn_k = args.knn_k or int(run_list[0].cfg['knn_k'])
    store = Store(runs, cells, knn_k, n_jobs, args.max_cached_embeddings)
    pairs = plan_pairs(run_list, args.pair_types)
    if not pairs:
        raise ValueError('No pairs to compare: check the runs and --pair_types.')
    pairs_df = pair_table(pairs, runs)
    pairs_df.to_csv(out / 'pairs.tsv', sep='\t', index=False)
    log(f'{len(pairs)} pairs x {len(args.representations)} representations; '
        f'{pairs_df["pair_type"].value_counts().to_dict()}.')

    fit_mask = split_halves(store.sections, np.random.default_rng([args.seed, 0]))
    same_space = dict(zip(pairs_df['pair_id'], pairs_df['same_space']))
    records = []
    start = time.time()
    # Pairs sorted by their first embedding so that the cache is reused
    order = sorted(pairs, key=lambda p: (p['run_a'], p['condition_a'], p['run_b'], p['condition_b']))
    for i, pair in enumerate(order, 1):
        for representation in args.representations:
            records += compare_pair(store, pair, representation, bool(same_space[pair['pair_id']]), fit_mask,
                                    args.seed)
        if i % 10 == 0 or i == len(order):
            log(f'{i} of {len(order)} pairs ({(time.time() - start) / 60:.1f} min).')
    long = pd.DataFrame(records)[['pair_id', 'representation', 'metric', 'section', 'value']]
    long.to_csv(out / 'similarity_long.tsv', sep='\t', index=False)
    summary = summary_table(long, pairs_df)
    summary.to_csv(out / 'similarity_summary.tsv', sep='\t', index=False)
    biology = biology_table(runs)
    biology.to_csv(out / 'biology_long.tsv', sep='\t', index=False)
    if not args.no_figures:
        try:
            from plot_panel_comparison import plot_comparison
            plot_comparison(out)
        except ImportError as error:
            log(f'WARNING: no figures ({error}).')
    write_json(out / 'compare_done.json', {
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'compare_version': COMPARE_VERSION,
        'runs': {run.label: str(run.folder) for run in run_list}, 'n_common_cells': len(cells),
        'common_cells_hash': store.common_hash, 'knn_k': knn_k, 'seed': args.seed,
        'representations': args.representations, 'pair_types': args.pair_types, 'n_pairs': len(pairs)})
    log(f'Results in {out}.')


if __name__ == '__main__':
    main()
