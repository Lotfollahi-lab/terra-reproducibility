#!/usr/bin/env python
"""
Split one multi-sample AnnData file (.h5ad) into per-sample silver files for
TERRA tokenization, laid out like the hst_corpus_110m silver files.

One output file per sample: the tokenizer builds the k-NN graph per file
(TERRA cell_tokenizers.py _tokenize_adata), so cells of different samples
(tissue sections) must not share a file.

Output (written to a temporary folder and renamed into place at the end):
    <data_directory_path>/silver/<dataset_name>/adata_batch<k>.h5ad
    <data_directory_path>/silver/<dataset_name>/silver_manifest.tsv
        one row per sample: batch, batch id, sample, cells in / dropped /
        written, vocabulary genes per cell, coordinate checks
    <data_directory_path>/silver/<dataset_name>/silver_prep.json
        settings, input signature, detection results and checks

Each silver file holds
- X: raw counts as CSR int32, canonical (sorted indices, no duplicate or
  explicitly stored zero entries), restricted to the genes of the token
  dictionary (vocabulary genes); columns mapping to the same Ensembl ID are
  summed;
- var: index 'gene_name', column 'ensembl_id';
- obs: 'cell_id' = '<dataset_id>_batch<k>_<i>' (also the index), the input's
  obs columns ('cell_id' and 'filter_pass' of the input are kept as
  'original_cell_id' / 'original_filter_pass': the tokenizer would drop cells
  with filter_pass != 1), 'original_obs_name';
- obsm['spatial']: float32 x/y, times --coord_scale;
- uns: dataset_id, batch, assay, species, tissue, sample, sample_key,
  source_h5ad, counts_source.
Batches are numbered by the natural order of the sample names (NO2 before
NO10). Cells without a sample or finite coordinates, and cells with fewer than
--min_vocab_genes vocabulary genes (the corpus floor, 10) are dropped; samples
with at most --n_neighs cells are skipped (the k-NN graph needs n_neighs + 1
cells). Every drop is reported.

Detection (each can be set explicitly; ambiguity is an error, never a guess):
- sample key: the obs column with exactly --expected_n_samples distinct values
  that contains every --exclude_samples value; several such columns must
  split the cells identically;
- raw counts: the first of layers['counts'], layers['raw_counts'],
  layers['count'], layers['raw'], X, raw.X whose stored values are
  non-negative integers (checked on a sample, then on all values);
- coordinates: obsm['spatial'] (first two columns), then other obsm keys and
  obs column pairs such as x_centroid / y_centroid;
- genes: names are mapped with --gene_name_map (the gene-name -> Ensembl-ID
  dictionary TERRA's harmonize_adata uses; preferred, so that inference
  harmonization maps new data the same way); names it does not map fall back
  to an Ensembl-ID column of var (e.g. 10x 'gene_ids'). Without the map, the
  Ensembl-ID column alone is used. Control probes and codewords map to no
  vocabulary gene and are dropped (reported by name prefix).

Usage (from the terra-reproducibility root, TERRA environment):
    python analysis/slurm_corpus_build/prepare_silver.py \\
        --h5ad /path/to/data.h5ad --dataset_name my_dataset --dataset_id 2001 \\
        --data_directory_path $TERRA_DATA_DIR \\
        --token_dictionary_file_path $TOKEN_DICTIONARY \\
        --gene_name_map $GENE_NAME_MAP \\
        --expected_n_samples 49 --exclude_samples NO4
Add --inspect to print the file structure and the detection results without
writing anything. Exit code 1 on any error.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

try:
    from anndata.io import read_elem
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem

if hasattr(ad.settings, 'allow_write_nullable_strings'):
    # pandas string columns of the input obs (anndata >= 0.11 refuses to
    # write them otherwise)
    ad.settings.allow_write_nullable_strings = True

ENSEMBL_GENE_ID = re.compile(r'(ENS[A-Z]*G\d+)(\.\d+)?')
COUNT_CANDIDATES = ('layers/counts', 'layers/raw_counts', 'layers/count',
                    'layers/raw', 'X', 'raw/X')
OBSM_COORD_KEYS = ('spatial', 'X_spatial', 'spatial_coords', 'spatial_coord',
                   'coords', 'xy')
OBS_COORD_PAIRS = (('x_centroid', 'y_centroid'), ('centroid_x', 'centroid_y'),
                   ('x_location', 'y_location'), ('center_x', 'center_y'),
                   ('cell_centroid_x', 'cell_centroid_y'),
                   ('CenterX_global_px', 'CenterY_global_px'),
                   ('x_global_px', 'y_global_px'), ('x', 'y'), ('X', 'Y'))
SAMPLE_KEY_PREFERENCE = ('sample', 'sample_id', 'Sample', 'sample_name',
                         'SampleID', 'Sample_ID', 'sampleID', 'sample_ID')
SAMPLE_LIKE_NAME = re.compile(r'sample|patient|donor|biopsy|specimen|id',
                              re.IGNORECASE)
# obs columns that may name a slide / section / run inside a sample
SECTION_COLUMN = re.compile(r'slide|section|run|chip|tma|core|fov',
                            re.IGNORECASE)
INTEGER_CHECK_CHUNK = 50_000_000


class PrepError(Exception):
    """An input problem that needs a decision (message says which)."""


def log(message: str = ''):
    print(message, flush=True)


def natural_key(value) -> list:
    """Sort key that orders 'NO2' before 'NO10'."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r'(\d+)', str(value))]


def sha256(path) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def encoding(node) -> str:
    value = node.attrs.get('encoding-type', '')
    return value.decode() if isinstance(value, bytes) else str(value)


def describe_matrix(node) -> str:
    enc = encoding(node)
    if isinstance(node, h5py.Dataset):
        return f'dense {node.shape} {node.dtype}'
    shape = tuple(int(s) for s in node.attrs.get('shape', ()))
    if 'data' in node:
        return (f'{enc or "group"} {shape} data {node["data"].dtype}, '
                f'{node["data"].shape[0]:,} stored values')
    return f'{enc or "group"} {shape}'


def print_structure(f: h5py.File, obs: pd.DataFrame):
    """Print what the file holds (reads only metadata and obs)."""
    log('=' * 78)
    log('FILE STRUCTURE')
    log('=' * 78)
    if 'X' in f:
        log(f'X: {describe_matrix(f["X"])}')
    if 'layers' in f:
        for key in f['layers']:
            log(f'layers[{key!r}]: {describe_matrix(f["layers"][key])}')
    if 'raw' in f:
        raw = f['raw']
        if 'X' in raw:
            log(f'raw.X: {describe_matrix(raw["X"])}')
        if 'var' in raw:
            log(f'raw.var: {len(read_elem(raw["var"]))} genes')
    if 'obsm' in f:
        for key in f['obsm']:
            node = f['obsm'][key]
            shape = (node.shape if isinstance(node, h5py.Dataset)
                     else encoding(node))
            log(f'obsm[{key!r}]: {shape}')
    if 'uns' in f:
        log(f'uns keys: {sorted(f["uns"].keys())}')
    log(f'obs: {obs.shape[0]:,} cells, {obs.shape[1]} columns')
    for column in obs.columns:
        series = obs[column]
        n_unique = series.nunique(dropna=True)
        example = ''
        if n_unique <= 60:
            values = sorted(series.dropna().astype(str).unique(),
                            key=natural_key)
            example = ', '.join(values[:12]) + (' ...' if n_unique > 12
                                                 else '')
        log(f'  {column!r}: {series.dtype}, {n_unique:,} distinct'
            f'{", NaN present" if series.isna().any() else ""}'
            f'{": " + example if example else ""}')


def find_sample_key(obs: pd.DataFrame, sample_key: str,
                    expected_n_samples: int | None,
                    exclude_samples: list[str]) -> str:
    """The obs column that names the sample of each cell (see module doc)."""
    if sample_key != 'auto':
        if sample_key not in obs.columns:
            raise PrepError(f"--sample_key {sample_key!r} is not an obs "
                            f"column. Columns: {list(obs.columns)}")
        return sample_key
    if expected_n_samples is None:
        raise PrepError('--sample_key auto needs --expected_n_samples (the '
                        'number of samples in the file).')
    candidates = []
    for column in obs.columns:
        series = obs[column]
        if pd.api.types.is_float_dtype(series) or pd.api.types.is_bool_dtype(
                series):
            continue
        # Integer columns only when named like a sample (e.g. not a count)
        if (pd.api.types.is_integer_dtype(series)
                and not SAMPLE_LIKE_NAME.search(str(column))):
            continue
        if series.nunique(dropna=True) != expected_n_samples:
            continue
        values = set(series.dropna().astype(str).unique())
        if any(sample not in values for sample in exclude_samples):
            continue
        candidates.append(column)
    if not candidates:
        near = {column: int(obs[column].nunique(dropna=True))
                for column in obs.columns
                if not pd.api.types.is_float_dtype(obs[column])
                and obs[column].nunique(dropna=True) <= 10_000}
        right_size = [c for c, n in near.items() if n == expected_n_samples]
        if right_size and exclude_samples:
            raise PrepError(
                f'Columns with {expected_n_samples} values: {right_size}, but '
                f'none contains every sample of --exclude_samples '
                f'{exclude_samples}. Check the sample names (exact match), '
                'or pass --sample_key to see which names the column has.')
        raise PrepError(
            f'No obs column has exactly {expected_n_samples} distinct values'
            + (f' including {exclude_samples}' if exclude_samples else '')
            + f'. Distinct values per column: {near}. Pass --sample_key.')
    if len(candidates) == 1:
        return candidates[0]
    # Columns that split the cells into the same groups are equivalent
    groups = []
    for column in candidates:
        codes = pd.factorize(obs[column].astype(str))[0]
        for group in groups:
            pairs = pd.DataFrame({'a': group['codes'], 'b': codes})
            if (pairs.groupby('a')['b'].nunique().max() == 1
                    and pairs.groupby('b')['a'].nunique().max() == 1):
                group['columns'].append(column)
                break
        else:
            groups.append({'codes': codes, 'columns': [column]})
    if len(groups) > 1:
        # Different splits: keep the one whose columns are named like a
        # sample (e.g. 'sample' over a cell-type column with as many values)
        named = [g for g in groups
                 if any(SAMPLE_LIKE_NAME.search(str(c)) for c in g['columns'])]
        if len(named) != 1:
            raise PrepError(
                'Several obs columns could be the sample key and they split '
                'the cells differently: '
                f'{[g["columns"] for g in groups]}. Pass --sample_key.')
        others = [c for g in groups if g is not named[0] for c in g['columns']]
        log(f'Sample key: columns {others} also have {expected_n_samples} '
            'values but split the cells differently; they are not named like '
            'a sample and are ignored.')
        groups = named
    columns = groups[0]['columns']
    preferred = [c for c in SAMPLE_KEY_PREFERENCE if c in columns]
    chosen = preferred[0] if preferred else sorted(columns)[0]
    if len(columns) > 1:
        log(f'Sample key candidates {columns} split the cells identically; '
            f'using {chosen!r}.')
    return chosen


def integer_check(node, n_values: int = 2_000_000) -> dict:
    """Check a sample of the stored values of a matrix for non-negative
    integers (first, middle, last and random blocks)."""
    if isinstance(node, h5py.Dataset):
        data, n_rows = node, node.shape[0]
        rows_per_block = max(1, n_values // 3 // max(1, node.shape[1]))
        starts = sorted({0, max(0, n_rows // 2), max(0, n_rows - rows_per_block)})
        blocks = [np.asarray(data[s:s + rows_per_block]).ravel()
                  for s in starts]
    else:
        if 'data' not in node:
            return {'ok': False, 'reason': f'unsupported encoding '
                                            f'{encoding(node)!r}'}
        data = node['data']
        n = data.shape[0]
        if n == 0:
            return {'ok': False, 'reason': 'no stored values'}
        block = max(1, n_values // 8)
        rng = np.random.default_rng(0)
        starts = {0, max(0, n // 2 - block // 2), max(0, n - block)}
        starts |= set(rng.integers(0, max(1, n - block), size=4).tolist())
        blocks = [np.asarray(data[s:s + block]) for s in sorted(starts)]
    values = np.concatenate(blocks) if blocks else np.zeros(0)
    if values.size == 0:
        return {'ok': False, 'reason': 'no values read'}
    values = values.astype(np.float64, copy=False)
    non_integer = np.mod(values, 1) != 0
    result = {
        'dtype': str(data.dtype),
        'n_checked': int(values.size),
        'fraction_non_integer': float(non_integer.mean()),
        'min': float(values.min()),
        'max': float(values.max()),
    }
    result['ok'] = bool(result['fraction_non_integer'] == 0
                        and result['min'] >= 0)
    if not result['ok']:
        result['reason'] = ('non-integer values' if non_integer.any()
                            else 'negative values')
    return result


def choose_counts(f: h5py.File, counts: str, round_counts: bool,
                  n_obs: int) -> tuple[str, dict]:
    """The h5 path of the raw counts and its integer check results."""
    checks = {}
    for path in COUNT_CANDIDATES:
        if path not in f:
            continue
        node = f[path]
        shape = (node.shape if isinstance(node, h5py.Dataset)
                 else tuple(node.attrs.get('shape', ())))
        if not shape or shape[0] != n_obs:
            checks[path] = {'ok': False, 'reason': f'shape {shape}'}
            continue
        checks[path] = integer_check(node)
    log('Raw-count candidates (sampled values):')
    for path, check in checks.items():
        log(f'  {path}: {json.dumps(check)}')
    if counts != 'auto':
        if counts not in f:
            raise PrepError(f'--counts {counts!r} is not in the file '
                            f'(candidates: {list(checks)}).')
        check = checks.get(counts) or integer_check(f[counts])
        if not check['ok'] and not round_counts:
            raise PrepError(
                f'--counts {counts!r} does not hold non-negative integers '
                f'({check}). Pass --round_counts to round to the nearest '
                'integer (e.g. for expected counts), if that is intended.')
        return counts, checks
    passing = [path for path, check in checks.items() if check['ok']]
    if passing:
        return passing[0], checks
    if round_counts:
        usable = [path for path, check in checks.items()
                  if check.get('min', -1) >= 0 and 'n_checked' in check]
        if usable:
            log(f'No candidate holds integers; rounding {usable[0]!r} '
                '(--round_counts).')
            return usable[0], checks
    raise PrepError(
        'No raw-count matrix found: no candidate holds non-negative integers '
        f'({ {p: c.get("reason") for p, c in checks.items()} }). Name the '
        'matrix with --counts (e.g. layers/counts), and add --round_counts '
        'only if rounding non-integer counts is intended.')


def load_counts(f: h5py.File, path: str, round_counts: bool
                ) -> sp.csr_matrix:
    """The full count matrix as CSR with int32 values (checked in full)."""
    matrix = read_elem(f[path])
    if sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    else:
        matrix = sp.csr_matrix(np.asarray(matrix))
    data = matrix.data
    for start in range(0, data.size, INTEGER_CHECK_CHUNK):
        chunk = np.asarray(data[start:start + INTEGER_CHECK_CHUNK],
                           dtype=np.float64)
        if chunk.size and chunk.min() < 0:
            raise PrepError(f'{path} has negative values.')
        if not round_counts and np.any(np.mod(chunk, 1) != 0):
            raise PrepError(f'{path} has non-integer values beyond the '
                            'sampled ones. Pass --round_counts if rounding '
                            'is intended.')
    if data.size and float(np.max(data)) >= np.iinfo(np.int32).max:
        raise PrepError(f'{path} has counts beyond the int32 range.')
    matrix.data = (np.rint(data) if round_counts else data).astype(np.int32)
    matrix.eliminate_zeros()
    return matrix


def ensembl_columns(var: pd.DataFrame) -> tuple[dict, dict]:
    """Number of Ensembl gene IDs per string var column (and the index),
    and the columns as strings."""
    columns = {'<index>': [str(v) for v in var.index]}
    for column in var.columns:
        if (pd.api.types.is_object_dtype(var[column])
                or pd.api.types.is_string_dtype(var[column])
                or isinstance(var[column].dtype, pd.CategoricalDtype)):
            columns[column] = [str(v) for v in var[column]]
    return {column: int(sum(bool(ENSEMBL_GENE_ID.fullmatch(v))
                            for v in values))
            for column, values in columns.items()}, columns


def map_genes(var: pd.DataFrame, name_map: dict | None,
              vocabulary: set[str]) -> tuple[np.ndarray, list, list, dict]:
    """
    Map the genes of `var` to vocabulary Ensembl IDs (see module doc).

    Returns the input column of every mapped gene, its Ensembl ID and gene
    name (duplicates included, in input order) and a report.
    """
    id_counts, string_columns = ensembl_columns(var)
    id_column = max(id_counts, key=id_counts.get) if id_counts else None
    if id_column is not None and id_counts[id_column] < 0.5 * len(var):
        id_column = None
    name_column = None
    if name_map is not None:
        name_hits = {column: int(sum(v in name_map for v in values))
                     for column, values in string_columns.items()}
        name_column = max(name_hits, key=name_hits.get)
        if name_hits[name_column] == 0:
            name_column = None
    if name_column is None and id_column is None:
        raise PrepError(
            'No gene could be mapped: var has no Ensembl-ID column'
            + (' and no column of gene names in --gene_name_map'
               if name_map is not None else
               ' and no --gene_name_map was given')
            + f'. var columns: {list(var.columns)}, first names: '
              f'{list(var.index[:5])}.')
    names = (string_columns[name_column] if name_column is not None
             else string_columns['<index>'])
    ids = (string_columns[id_column] if id_column is not None else None)

    columns, ensembl_ids, gene_names = [], [], []
    by_name = by_id = 0
    disagreements = []
    dropped = collections.Counter()
    for j, name in enumerate(names):
        from_name = name_map.get(name) if name_map is not None else None
        from_id = None
        if ids is not None:
            match = ENSEMBL_GENE_ID.fullmatch(ids[j])
            from_id = match.group(1) if match else None
        if from_name is not None and from_id is not None \
                and from_name != from_id and from_id in vocabulary:
            disagreements.append(f'{name}: name -> {from_name}, '
                                 f'id -> {from_id}')
        if from_name in vocabulary:
            ensembl_id = from_name
            by_name += 1
        elif from_id in vocabulary:
            ensembl_id = from_id
            by_id += 1
        else:
            prefix = re.split(r'[_\-.\d]', name, maxsplit=1)[0] or name
            dropped[prefix] += 1
            continue
        columns.append(j)
        ensembl_ids.append(ensembl_id)
        gene_names.append(name)
    report = {
        'n_input_features': int(len(var)),
        'name_column': name_column,
        'ensembl_id_column': id_column,
        'n_mapped_by_name': by_name,
        'n_mapped_by_ensembl_id': by_id,
        'n_dropped': int(sum(dropped.values())),
        'dropped_by_prefix': dict(dropped.most_common(30)),
        'n_name_id_disagreements': len(disagreements),
        'name_id_disagreements': disagreements[:30],
        'n_duplicated_ensembl_ids': int(
            len(ensembl_ids) - len(set(ensembl_ids))),
    }
    if not columns:
        raise PrepError(f'No gene maps to the token dictionary: {report}')
    return np.asarray(columns), ensembl_ids, gene_names, report


def find_coordinates(f: h5py.File, obs: pd.DataFrame, spatial: str
                     ) -> tuple[np.ndarray, str]:
    """(n_obs, 2) float64 coordinates and where they came from."""
    def from_obsm(key):
        value = read_elem(f['obsm'][key])
        value = (value.to_numpy() if isinstance(value, pd.DataFrame)
                 else np.asarray(value))
        if value.ndim != 2 or value.shape[1] < 2:
            raise PrepError(f"obsm[{key!r}] has shape {value.shape}, not "
                            '(cells, >= 2).')
        return value[:, :2].astype(np.float64)

    def from_obs(x, y):
        return np.column_stack([pd.to_numeric(obs[x]).to_numpy(np.float64),
                                pd.to_numeric(obs[y]).to_numpy(np.float64)])

    if spatial != 'auto':
        kind, _, key = spatial.partition(':')
        if kind == 'obsm' and 'obsm' in f and key in f['obsm']:
            return from_obsm(key), f'obsm[{key!r}]'
        if kind == 'obs' and ',' in key:
            x, y = key.split(',', 1)
            if x in obs.columns and y in obs.columns:
                return from_obs(x, y), f'obs[{x!r}], obs[{y!r}]'
        raise PrepError(f'--spatial {spatial!r} not found (use '
                        'obsm:<key> or obs:<x column>,<y column>).')
    if 'obsm' in f:
        for key in OBSM_COORD_KEYS:
            if key in f['obsm']:
                return from_obsm(key), f'obsm[{key!r}]'
    for x, y in OBS_COORD_PAIRS:
        if x in obs.columns and y in obs.columns:
            return from_obs(x, y), f'obs[{x!r}], obs[{y!r}]'
    raise PrepError(
        'No spatial coordinates found (obsm keys: '
        f'{list(f["obsm"].keys()) if "obsm" in f else []}). Pass --spatial '
        'obsm:<key> or --spatial obs:<x column>,<y column>.')


def coordinate_checks(coords: np.ndarray) -> dict:
    """Nearest-neighbour distance and duplicates of one sample."""
    n_unique = np.unique(coords, axis=0).shape[0]
    distances = cKDTree(coords).query(coords, k=2)[0][:, 1]
    return {
        'median_nn_distance': float(np.median(distances)),
        'fraction_duplicate_coordinates': float(1 - n_unique / len(coords)),
        'x_range': [float(coords[:, 0].min()), float(coords[:, 0].max())],
        'y_range': [float(coords[:, 1].min()), float(coords[:, 1].max())],
    }


def section_columns(obs: pd.DataFrame, sample_key: str) -> dict:
    """Most distinct values per sample of slide/section-like obs columns."""
    result = {}
    for column in obs.columns:
        if column == sample_key or not SECTION_COLUMN.search(str(column)):
            continue
        if pd.api.types.is_float_dtype(obs[column]):
            continue
        n_unique = obs[column].nunique(dropna=True)
        if 2 <= n_unique <= 100_000:
            result[column] = int(obs.groupby(sample_key, observed=True)[
                column].nunique().max())
    return result


def overlapping_sections(obs: pd.DataFrame, sample_key: str,
                         columns: dict, coords: np.ndarray,
                         finite: np.ndarray) -> dict:
    """
    Samples whose slides / sections (values of a slide-like column) overlap
    in the coordinate frame: their cells would become neighbours of each
    other in the k-NN graph of the sample's file (slide-local coordinates).
    """
    overlaps = {}
    for column, most in columns.items():
        if most < 2:
            continue
        frame = pd.DataFrame({'sample': obs[sample_key].astype(str).values,
                              'section': obs[column].astype(str).values,
                              'x': coords[:, 0], 'y': coords[:, 1]})[finite]
        for sample, cells in frame.groupby('sample'):
            boxes = cells.groupby('section').agg(
                x0=('x', 'min'), x1=('x', 'max'),
                y0=('y', 'min'), y1=('y', 'max'))
            if len(boxes) < 2:
                continue
            values = boxes.to_numpy()
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    a, b = values[i], values[j]
                    if a[0] < b[1] and b[0] < a[1] and a[2] < b[3] \
                            and b[2] < a[3]:
                        overlaps.setdefault(column, []).append(
                            f'{sample}: {boxes.index[i]} / {boxes.index[j]}')
    return overlaps


def input_signature(args) -> dict:
    """What determines the output: the input file and the settings."""
    stat = os.stat(args.h5ad)
    settings = {k: v for k, v in sorted(vars(args).items())
                if k not in ('overwrite', 'inspect', 'data_directory_path',
                             'h5ad', 'token_dictionary_file_path',
                             'gene_name_map')}
    return {
        'h5ad': str(Path(args.h5ad).resolve()),
        'h5ad_size': stat.st_size,
        'h5ad_mtime_ns': stat.st_mtime_ns,
        'token_dictionary_sha256': sha256(args.token_dictionary_file_path),
        'gene_name_map_sha256': sha256(args.gene_name_map),
        'settings': settings,
    }


def parse_arguments(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5ad', required=True, help='Input multi-sample .h5ad.')
    ap.add_argument('--dataset_name', required=True,
                    help='Silver folder name (silver/<dataset_name>).')
    ap.add_argument('--dataset_id', required=True, type=int,
                    help="Integer id; cell ids are '<id>_batch<k>_<i>'. "
                         'Unique per dataset of the corpus.')
    ap.add_argument('--data_directory_path', required=True,
                    help='Data root holding silver/.')
    ap.add_argument('--token_dictionary_file_path', required=True,
                    help='Token dictionary (.pkl); its Ensembl IDs are the '
                         'vocabulary.')
    ap.add_argument('--gene_name_map', default=None,
                    help='Pickled dict gene name -> Ensembl ID (the corpus '
                         'harmonization map). Optional, recommended.')
    ap.add_argument('--sample_key', default='auto',
                    help="obs column naming the sample of each cell, or "
                         "'auto' (default).")
    ap.add_argument('--expected_n_samples', type=int, default=None,
                    help='Number of samples in the file (before exclusion); '
                         'checked, and needed by --sample_key auto.')
    ap.add_argument('--exclude_samples', nargs='*', default=[],
                    help='Samples to leave out (exact values of the sample '
                         'key).')
    ap.add_argument('--counts', default='auto',
                    help="h5 path of the raw counts, e.g. 'layers/counts', "
                         "'X' or 'raw/X'; 'auto' (default) detects it.")
    ap.add_argument('--round_counts', action='store_true',
                    help='Round non-integer counts (e.g. expected counts) '
                         'to the nearest integer instead of failing.')
    ap.add_argument('--spatial', default='auto',
                    help="'obsm:<key>', 'obs:<x>,<y>' or 'auto' (default).")
    ap.add_argument('--coord_scale', type=float, default=1.0,
                    help='Factor applied to the coordinates (e.g. 0.2125 for '
                         'Xenium pixels -> micrometres). The corpus uses '
                         'micrometres.')
    ap.add_argument('--assay', default='xenium')
    ap.add_argument('--species', default='homo_sapiens')
    ap.add_argument('--tissue', default='liver')
    ap.add_argument('--min_vocab_genes', type=int, default=10,
                    help='Drop cells with fewer vocabulary genes (corpus: 10).')
    ap.add_argument('--n_neighs', type=int, default=16,
                    help='Neighbours of the tokenization; samples with at '
                         'most this many cells are skipped.')
    ap.add_argument('--max_fraction_dropped', type=float, default=0.05,
                    help='Fail if more than this fraction of the cells lack '
                         'a sample or coordinates.')
    ap.add_argument('--inspect', action='store_true',
                    help='Print the structure and detection results only.')
    ap.add_argument('--overwrite', action='store_true',
                    help='Replace an existing silver folder.')
    return ap.parse_args(argv)


def main(args) -> int:
    t0 = time.time()
    exclude = [str(s) for s in args.exclude_samples]
    silver_root = Path(args.data_directory_path) / 'silver'
    output_dir = silver_root / args.dataset_name
    log(f'Input: {args.h5ad}')
    log(f'Dataset: {args.dataset_name} (id {args.dataset_id}) -> '
        f'{output_dir}')

    signature = input_signature(args)
    prep_json = output_dir / 'silver_prep.json'
    if not args.inspect and output_dir.exists():
        previous = (json.loads(prep_json.read_text())
                    if prep_json.exists() else None)
        if (previous is not None and not args.overwrite
                and previous.get('signature') == signature):
            log(f'{output_dir} was prepared from the same input and '
                'settings; nothing to do.')
            return 0
        if not args.overwrite:
            raise PrepError(
                f'{output_dir} exists and was not prepared from this input '
                'and these settings. Pass --overwrite to replace it.')

    with open(args.token_dictionary_file_path, 'rb') as file:
        vocabulary = {key for key in pickle.load(file) if 'ENS' in str(key)}
    log(f'Vocabulary genes (token dictionary): {len(vocabulary):,}')
    name_map = None
    if args.gene_name_map:
        with open(args.gene_name_map, 'rb') as file:
            name_map = pickle.load(file)
        log(f'Gene-name map: {len(name_map):,} names')

    with h5py.File(args.h5ad, 'r') as f:
        obs = read_elem(f['obs'])
        print_structure(f, obs)
        log('=' * 78)
        log('DETECTION')
        log('=' * 78)

        # Sample of each cell
        sample_key = find_sample_key(obs, args.sample_key,
                                     args.expected_n_samples, exclude)
        samples_all = obs[sample_key]
        n_samples = samples_all.nunique(dropna=True)
        log(f'Sample key: {sample_key!r} ({n_samples} samples)')
        if (args.expected_n_samples is not None
                and n_samples != args.expected_n_samples):
            raise PrepError(f'{sample_key!r} has {n_samples} samples, '
                            f'expected {args.expected_n_samples}.')
        sample_values = samples_all.astype(str).where(samples_all.notna())
        present = set(sample_values.dropna().unique())
        missing = [s for s in exclude if s not in present]
        if missing:
            close = {s: [v for v in present
                         if v.strip().lower() == s.strip().lower()]
                     for s in missing}
            raise PrepError(f'--exclude_samples {missing} not found in '
                            f'{sample_key!r} (case-insensitive matches: '
                            f'{close}).')
        sections = section_columns(obs, sample_key)
        if sections:
            log('Slide/section-like obs columns, most distinct values in '
                f'one sample: {sections}')

        # Raw counts and genes
        counts_path, counts_checks = choose_counts(
            f, args.counts, args.round_counts, len(obs))
        log(f'Raw counts: {counts_path}')
        var = read_elem(f['raw/var'] if counts_path == 'raw/X' else f['var'])
        gene_columns, ensembl_ids, gene_names, gene_report = map_genes(
            var, name_map, vocabulary)
        log(f'Genes: {json.dumps(gene_report, indent=2)}')
        unique_ids = list(dict.fromkeys(ensembl_ids))
        if gene_report['n_mapped_by_ensembl_id'] and name_map is not None:
            log('NOTE: some genes were mapped by their Ensembl-ID column '
                'because --gene_name_map does not know their names.')

        # Coordinates
        coords, coords_source = find_coordinates(f, obs, args.spatial)
        coords = coords * args.coord_scale
        log(f'Coordinates: {coords_source} x {args.coord_scale}')

        finite = np.isfinite(coords).all(axis=1)
        has_sample = sample_values.notna().to_numpy()
        n_no_coords = int((~finite).sum())
        n_no_sample = int((~has_sample).sum())
        dropped_fraction = (n_no_coords + n_no_sample) / max(1, len(obs))
        log(f'Cells without finite coordinates: {n_no_coords:,}; without a '
            f'sample: {n_no_sample:,}')
        if dropped_fraction > args.max_fraction_dropped:
            raise PrepError(
                f'{dropped_fraction:.1%} of the cells lack coordinates or a '
                f'sample (limit {args.max_fraction_dropped:.0%}).')

        kept_samples = sorted((s for s in present if s not in exclude),
                              key=natural_key)
        log(f'Samples kept: {len(kept_samples)}; excluded: {exclude or "-"}')

        # Per-sample coordinate checks (cheap: no counts needed)
        sample_array = sample_values.to_numpy()
        rows_by_sample = {s: np.flatnonzero((sample_array == s) & finite)
                          for s in kept_samples}
        coordinate_report = {}
        for s in kept_samples:
            rows = rows_by_sample[s]
            if len(rows) > args.n_neighs:
                coordinate_report[s] = coordinate_checks(coords[rows])
        nn = np.array([c['median_nn_distance']
                       for c in coordinate_report.values()])
        if len(nn):
            log(f'Median nearest-neighbour distance per sample: min '
                f'{nn.min():.2f}, median {np.median(nn):.2f}, max '
                f'{nn.max():.2f} (cells in micrometres: about 5-20)')
            if not 2 <= np.median(nn) <= 40:
                log('WARNING: the nearest-neighbour distances do not look '
                    'like micrometres. The relative coordinates the model '
                    'reads (cell_pos_enc: coord) are in the units of the '
                    'input; set --coord_scale (e.g. 0.2125 for Xenium '
                    'pixels).')
        overlaps = overlapping_sections(obs, sample_key, sections, coords,
                                        finite)
        for column, pairs in overlaps.items():
            log(f'WARNING: within a sample, values of {column!r} overlap in '
                f'the coordinate frame ({len(pairs)} pairs, e.g. '
                f'{pairs[:5]}). If these are different sections with '
                'section-local coordinates, their cells become neighbours '
                'in the k-NN graph: split the samples further (e.g. a '
                'combined sample column) before preparing.')
        many_duplicates = {s: round(c['fraction_duplicate_coordinates'], 4)
                           for s, c in coordinate_report.items()
                           if c['fraction_duplicate_coordinates'] > 0.01}
        if many_duplicates:
            log('WARNING: samples with more than 1% duplicated coordinates '
                f'(overlapping sections in one coordinate frame?): '
                f'{many_duplicates}')

        if args.inspect:
            log('=' * 78)
            log('Cells per kept sample: ' + ', '.join(
                f'{s}: {len(rows_by_sample[s]):,}' for s in kept_samples))
            log(f'--inspect: nothing written ({time.time() - t0:.0f} s).')
            return 0

        log('Reading the count matrix...')
        counts = load_counts(f, counts_path, args.round_counts)
    log(f'Counts: {counts.shape}, {counts.nnz:,} stored values')

    # Vocabulary columns; columns of the same Ensembl ID are summed
    position = {gene_id: k for k, gene_id in enumerate(unique_ids)}
    aggregate = sp.csr_matrix(
        (np.ones(len(gene_columns), dtype=np.int32),
         (gene_columns, [position[g] for g in ensembl_ids])),
        shape=(counts.shape[1], len(unique_ids)))
    counts = (counts @ aggregate).tocsr()
    counts.eliminate_zeros()
    counts.sort_indices()
    counts.sum_duplicates()
    first_name = {}
    for gene_id, name in zip(ensembl_ids, gene_names):
        first_name.setdefault(gene_id, name)
    var_out = pd.DataFrame(
        {'ensembl_id': unique_ids},
        index=pd.Index([first_name[g] for g in unique_ids], name='gene_name'))
    if not var_out.index.is_unique:
        var_out.index = pd.Index(
            [f'{name}_{gene_id}' if dup else name for name, gene_id, dup in
             zip(var_out.index, unique_ids, var_out.index.duplicated(False))],
            name='gene_name')
    genes_per_cell = np.diff(counts.indptr)

    # obs columns kept in every silver file
    obs_out = obs.copy()
    obs_out = obs_out.rename(columns={
        c: f'original_{c}' for c in ('cell_id', 'filter_pass')
        if c in obs_out.columns})
    obs_out['original_obs_name'] = obs_out.index.astype(str)

    tmp_dir = silver_root / (f'.{args.dataset_name}.partial-'
                             f'{os.environ.get("SLURM_JOB_ID", os.getpid())}')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    manifest = []
    skipped = {}
    batch = 0
    for s in kept_samples:
        rows = rows_by_sample[s]
        n_in = int(((sample_array == s)).sum())
        rows = rows[genes_per_cell[rows] >= args.min_vocab_genes]
        if len(rows) <= args.n_neighs:
            skipped[s] = int(len(rows))
            log(f'WARNING: sample {s!r} has {len(rows)} cells with >= '
                f'{args.min_vocab_genes} vocabulary genes; skipped (the '
                f'{args.n_neighs}-NN graph needs {args.n_neighs + 1}).')
            continue
        batch_name = f'batch{batch}'
        cell_ids = [f'{args.dataset_id}_{batch_name}_{i}'
                    for i in range(len(rows))]
        obs_s = obs_out.iloc[rows].copy()
        for column in obs_s.columns:
            if isinstance(obs_s[column].dtype, pd.CategoricalDtype):
                obs_s[column] = obs_s[column].cat.remove_unused_categories()
        obs_s.insert(0, 'cell_id', cell_ids)
        obs_s.index = pd.Index(cell_ids)
        adata = ad.AnnData(X=counts[rows], obs=obs_s, var=var_out.copy())
        adata.obsm['spatial'] = coords[rows].astype(np.float32)
        adata.uns['dataset_id'] = int(args.dataset_id)
        adata.uns['batch'] = batch_name
        adata.uns['assay'] = args.assay
        adata.uns['species'] = args.species
        adata.uns['tissue'] = args.tissue
        adata.uns['sample'] = s
        adata.uns['sample_key'] = sample_key
        adata.uns['source_h5ad'] = str(Path(args.h5ad).resolve())
        adata.uns['counts_source'] = counts_path
        file_name = f'adata_{batch_name}.h5ad'
        try:
            adata.write_h5ad(tmp_dir / file_name)
        except Exception as error:  # an obs column anndata cannot write
            log(f'Writing {file_name} failed ({error}); writing the '
                'non-numeric, non-categorical obs columns as strings.')
            for column in adata.obs.columns:
                dtype = adata.obs[column].dtype
                if not (isinstance(dtype, pd.CategoricalDtype)
                        or pd.api.types.is_numeric_dtype(dtype)
                        or pd.api.types.is_bool_dtype(dtype)):
                    adata.obs[column] = adata.obs[column].astype(
                        str).astype(object)
            adata.write_h5ad(tmp_dir / file_name)
        gpc = genes_per_cell[rows]
        checks = coordinate_checks(adata.obsm['spatial'].astype(np.float64))
        manifest.append({
            'batch': batch_name,
            'batch_id': f'{args.dataset_id}_{batch_name}',
            'sample': s,
            'file': file_name,
            'n_cells_input': n_in,
            'n_cells_no_coordinates': int(n_in - len(rows_by_sample[s])),
            'n_cells_low_genes': int(len(rows_by_sample[s]) - len(rows)),
            'n_cells': int(len(rows)),
            'mean_vocab_genes': round(float(gpc.mean()), 2),
            'mean_filled_512': round(float(np.minimum(gpc, 512).mean()), 2),
            'median_nn_distance': round(checks['median_nn_distance'], 3),
            'fraction_duplicate_coordinates': round(
                checks['fraction_duplicate_coordinates'], 5),
        })
        log(f'  {file_name}: sample {s!r}, {len(rows):,} cells '
            f'({n_in - len(rows):,} dropped), '
            f'{gpc.mean():.0f} vocabulary genes per cell')
        batch += 1
    if not manifest:
        raise PrepError('No sample has enough cells; nothing written.')

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(tmp_dir / 'silver_manifest.tsv', sep='\t',
                       index=False)
    summary = {
        'dataset_name': args.dataset_name,
        'dataset_id': args.dataset_id,
        'signature': signature,
        'sample_key': sample_key,
        'n_samples_in_file': int(n_samples),
        'excluded_samples': exclude,
        'skipped_samples': skipped,
        'n_files': len(manifest),
        'n_cells_input': int(len(obs)),
        'n_cells_written': int(manifest_df['n_cells'].sum()),
        'n_cells_without_coordinates': n_no_coords,
        'n_cells_without_sample': n_no_sample,
        'counts_source': counts_path,
        'counts_checks': counts_checks,
        'round_counts': args.round_counts,
        'coordinates_source': coords_source,
        'coord_scale': args.coord_scale,
        'genes': gene_report,
        'n_vocabulary_genes_written': len(unique_ids),
        'section_columns_max_per_sample': sections,
        'overlapping_sections': overlaps,
        'seconds': round(time.time() - t0, 1),
    }
    (tmp_dir / 'silver_prep.json').write_text(json.dumps(summary, indent=2))

    # Replace the output folder only now, so it is either complete or absent
    replaced = None
    if output_dir.exists():
        replaced = silver_root / (f'.{args.dataset_name}.replaced-'
                                  f'{os.environ.get("SLURM_JOB_ID", os.getpid())}')
        os.rename(output_dir, replaced)
    os.rename(tmp_dir, output_dir)
    if replaced is not None:
        shutil.rmtree(replaced, ignore_errors=True)

    log('=' * 78)
    log(f'Wrote {len(manifest)} files, {summary["n_cells_written"]:,} of '
        f'{len(obs):,} cells, {len(unique_ids)} vocabulary genes, to '
        f'{output_dir} ({summary["seconds"]:.0f} s).')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(parse_arguments()))
    except PrepError as error:
        log(f'ERROR: {error}')
        sys.exit(1)
