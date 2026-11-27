"""
Gene panel subsampling: does TERRA embed the same tissue consistently when it is profiled with different gene panels?

TERRA's cell and neighborhood embeddings are means over the genes that a section's panel contains (reviewer comment
R3C2.1; rebuttal plan E12). This experiment profiles the same held-out tissue in silico with smaller panels: it embeds
the held-out Xenium Prime 5K skin benchmark sections (xhs1000-39b_1p batches 11, 19 and 32; 5,000 genes, of which
about 4,900 map to the TERRA vocabulary) with the full panel and with random subsets of 1,000 to 4,000 of its genes
(5 random draws per size, nested within a draw), plus panels of 1,000 and 2,000 genes that share no gene with the
random panel of the same size and draw ("disjoint" panels). It measures

- biology recovery per panel: NMI and ARI of Leiden clusters against the niche and cell-type annotations (resolution
  matched to the number of annotated classes; and a fixed resolution grid), and leave-one-section-out kNN label
  transfer;
- consistency with the full panel: overlap of the k nearest neighbors of every cell, agreement of Leiden clusters
  (against a seed-to-seed noise ceiling), linear CKA, paired cosine similarity, and label transfer across panels
  (kNN classifier trained on full-panel embeddings of two sections, applied to subset-panel embeddings of the third);
- consistency of two panels that share no gene (disjoint vs random panel of the same size);
- the same for non-pretrained baselines: PCA of log-normalized expression (cells) and of expression averaged over each
  cell and its spatial neighbors (neighborhoods).

Only the genes are subset; counts of the kept genes, the cells, their coordinates and the spatial graph stay the same.
TERRA tokenizes only detected genes, so a smaller panel removes tokens; with TERRA-96M's shifted-log values and
raw-count ranking, the values and the order of the kept genes do not change. Cells that detect no gene of a panel keep
their place in their neighbors' sequences as an all-padding segment (`data.pad_empty_neighbors`; on by default, in a
copy of the model folder), so the neighborhood structure is the same in every condition; the metrics use the cells
that detect at least one gene in every panel.

Steps (submit_gene_panel_subsampling.sh runs them on the farm):
    configure  (login node)  write run_config.json with all parameters and the conditions; prints their number
    prepare    (CPU job)     panels, evaluation cells, panel statistics, model folder copy
    embed      (GPU array)   one condition per element: harmonize, tokenize and embed with TERRA
    evaluate   (CPU array)   one condition per element: PCA baselines, kNN, Leiden, label transfer
    summarize  (CPU job)     comparisons with the full panel and between disjoint panels, tables, figures
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_VERSION = 1

DEFAULT_SILVER_FOLDER_PATH = '/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver'
DEFAULT_MODEL_FOLDER_PATH = '/nfs/team361/sb75/terra-reproducibility/artifacts/models/01112025_081731_614'

# Representations evaluated per condition: source ('terra': embeddings of the embed step; 'pca': baselines computed
# in the evaluate step), key of the array, and the annotation they are scored against
REPRESENTATIONS = {
    'terra_neighborhood': {'source': 'terra', 'key': 'neighborhood_emb', 'label': 'niche'},
    'terra_cell': {'source': 'terra', 'key': 'cell_emb', 'label': 'cell_type'},
    'pca_neighborhood': {'source': 'pca', 'key': 'neighborhood', 'label': 'niche'},
    'pca_cell': {'source': 'pca', 'key': 'cell', 'label': 'cell_type'},
}
SPATIAL_CELL_REPRESENTATION = {'terra_spatial_cell': {'source': 'terra', 'key': 'spatial_cell_emb',
                                                      'label': 'cell_type'}}

# Model-folder files that map gene names to Ensembl IDs (current name, then the name in bundles published before
# June 2026) and that count pretraining occurrences; the same files harmonize_tokenize_embed_pipeline uses by default
GENE_MAPPING_FILE_NAMES = ('ensembl_dictionary.pkl', 'ensembl_dic.pkl')
GENE_OCCURRENCE_FILE_NAME = 'gene_count_dictionary.pkl'
SILVER_GENE_MAPPING_FILE_NAME = 'gene_mapping_from_silver.pkl'


def log(message: str):
    """Print a time-stamped message to stderr (stdout of `configure` carries only the number of conditions)."""
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}', file=sys.stderr, flush=True)


def write_json(path: Path, obj):
    """Write JSON atomically (temporary file in the same folder, then rename)."""
    path = Path(path)
    tmp_path = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)
    os.replace(tmp_path, path)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_npz(path: Path, **arrays):
    """Save arrays to an .npz file atomically."""
    path = Path(path)
    tmp_path = path.with_name(f'.{path.stem}.tmp-{os.getpid()}.npz')
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)


# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------

def make_conditions(panel_sizes: list, n_replicates: int, disjoint_sizes: list, full_repeat: bool) -> list:
    """
    Conditions in job-array order (element i embeds and evaluates condition i - 1):
    - full: every panel gene of the model vocabulary;
    - full_repeat: the full panel again (run-to-run variation of embedding, e.g. GPU non-determinism);
    - random_<size>_r<r>: the first <size> genes of random permutation r of the panel genes (nested within r);
    - disjoint_<size>_r<r>: genes <size> to 2 * <size> - 1 of permutation r, which share no gene with random_<size>_r<r>.
    """
    conditions = [{'condition_id': 'full', 'kind': 'full', 'size': None, 'replicate': None, 'partner': None}]
    if full_repeat:
        conditions.append({'condition_id': 'full_repeat', 'kind': 'full_repeat', 'size': None, 'replicate': None,
                           'partner': 'full'})
    for size in panel_sizes:
        for replicate in range(n_replicates):
            conditions.append({'condition_id': f'random_{size}_r{replicate}', 'kind': 'random', 'size': size,
                               'replicate': replicate, 'partner': None})
    for size in disjoint_sizes:
        for replicate in range(n_replicates):
            conditions.append({'condition_id': f'disjoint_{size}_r{replicate}', 'kind': 'disjoint', 'size': size,
                               'replicate': replicate, 'partner': f'random_{size}_r{replicate}'})
    return conditions


def config_path(output_folder_path) -> Path:
    return Path(output_folder_path) / 'run_config.json'


def load_config(output_folder_path) -> dict:
    path = config_path(output_folder_path)
    if not path.is_file():
        raise FileNotFoundError(f'{path} does not exist; run the configure step first.')
    return read_json(path)


def condition_by_index(cfg: dict, condition_index: int) -> dict:
    """Condition of a 1-based job array index."""
    n = len(cfg['conditions'])
    if not 1 <= condition_index <= n:
        raise ValueError(f'Condition index {condition_index} is outside 1..{n}.')
    return cfg['conditions'][condition_index - 1]


def cmd_configure(args):
    if args.disjoint_sizes and not set(args.disjoint_sizes) <= set(args.panel_sizes):
        raise ValueError('Every disjoint size must also be a panel size (its random panel is the partner).')
    if len(set(args.panel_sizes)) != len(args.panel_sizes) or any(size < 1 for size in args.panel_sizes):
        raise ValueError(f'Panel sizes must be distinct positive integers: {args.panel_sizes}.')
    if args.n_replicates < 1:
        raise ValueError('--n_replicates must be at least 1.')
    cfg = {
        'script_version': SCRIPT_VERSION,
        'output_folder_path': str(Path(args.output_folder_path).resolve()),
        'silver_folder_path': args.silver_folder_path,
        'dataset_name': args.dataset_name,
        'batches': args.batches,
        'sample_key': args.sample_key,
        'niche_key': args.niche_key,
        'cell_type_key': args.cell_type_key,
        'model_folder_path': args.model_folder_path,
        'gene_mapping_dict_file_path': args.gene_mapping_dict_file_path,
        'gene_occurrence_count_file_path': args.gene_occurrence_count_file_path,
        'gene_occurrence_count_filter_value': args.gene_occurrence_count_filter_value,
        'panel_sizes': sorted(args.panel_sizes),
        'n_replicates': args.n_replicates,
        'disjoint_sizes': sorted(args.disjoint_sizes),
        'seed': args.seed,
        'full_repeat': args.full_repeat,
        'pad_empty_neighbors': args.pad_empty_neighbors,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'nproc': args.nproc,
        'cache_folder_path': args.cache_folder_path or str(Path(args.output_folder_path).resolve() / 'cache'),
        'knn_k': args.knn_k,
        'n_leiden_seeds': args.n_leiden_seeds,
        'leiden_resolutions': args.leiden_resolutions,
        'min_cells_per_class': args.min_cells_per_class,
        'pca_n_comps': args.pca_n_comps,
        'evaluate_spatial_cell_emb': args.evaluate_spatial_cell_emb,
        'conditions': make_conditions(sorted(args.panel_sizes), args.n_replicates, sorted(args.disjoint_sizes),
                                      args.full_repeat),
    }
    # Keys only written when used, so that the configuration of existing runs stays the same
    if args.panels_from:
        source = Path(args.panels_from).resolve()
        source_cfg = read_json(config_path(source))
        if not (source / 'prepare_done.json').is_file():
            raise FileNotFoundError(f'{source} has not finished its prepare step; its panels.json is needed.')
        for key in ('silver_folder_path', 'dataset_name', 'batches', 'sample_key', 'niche_key', 'cell_type_key'):
            if source_cfg[key] != cfg[key]:
                raise ValueError(f'--panels_from {source} has {key} {source_cfg[key]!r}, this run {cfg[key]!r}: the '
                                 'panels of a run can only be reused on the same cells.')
        for key in ('panel_sizes', 'n_replicates', 'disjoint_sizes', 'seed', 'full_repeat'):
            cfg[key] = source_cfg[key]
        cfg['conditions'] = source_cfg['conditions']
        cfg['panels_from'] = str(source)
    if args.conditions:
        known = [c['condition_id'] for c in cfg['conditions']]
        unknown = [c for c in args.conditions if c not in known]
        if unknown:
            raise ValueError(f'Unknown conditions {unknown}; the conditions are {known}.')
        cfg['conditions'] = [c for c in cfg['conditions'] if c['condition_id'] in set(args.conditions)]
        cfg['condition_filter'] = list(args.conditions)
    path = config_path(cfg['output_folder_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = read_json(path)
        if existing != cfg and not args.overwrite:
            changed = sorted(k for k in set(existing) | set(cfg) if existing.get(k) != cfg.get(k))
            raise ValueError(f'{path} exists with other parameters ({", ".join(changed)}). Use another output folder, '
                             'or --overwrite to replace it (the results of the old parameters are then stale).')
        if existing != cfg:
            log(f'Overwriting {path} (changed parameters; delete stale results).')
    write_json(path, cfg)
    log(f'Wrote {path} with {len(cfg["conditions"])} conditions.')
    print(len(cfg['conditions']))


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------

def section_files(cfg: dict) -> list:
    """Silver file of every section (adata_<batch>.h5ad or <batch>.h5ad in the dataset folder or below)."""
    folder = Path(cfg['silver_folder_path']) / cfg['dataset_name']
    files = sorted(folder.rglob('*.h5ad'))
    if not files:
        raise FileNotFoundError(f'No .h5ad files in {folder}.')
    paths = []
    for batch in cfg['batches']:
        matches = [f for f in files if f.stem == batch or f.stem.endswith(f'_{batch}')]
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one silver file for {batch} in {folder}, found {len(matches)}: '
                                    f'{[str(m) for m in matches]}.')
        paths.append(matches[0])
    return paths


def load_sections(cfg: dict):
    """
    Raw-count AnnData of the benchmark sections: the cells with a niche annotation (as in the TERRA benchmarking
    notebook, which removes the others before embedding), with obs['section'] = batch name.
    """
    import anndata as ad
    import pandas as pd

    adatas = []
    for batch, path in zip(cfg['batches'], section_files(cfg)):
        adata = ad.read_h5ad(path)
        # Cells are matched across panels by name, so names must be unique. The names in silver files can repeat
        # (xhs1000: Xenium cell id + sample, e.g. in adata_batch32); the silver cell_id (<dataset>_<batch>_<row>) is
        # unique by construction, else the row in the file is used. The silver name is kept in obs['silver_name'].
        adata.obs['silver_name'] = adata.obs_names.astype(str)
        n_repeated = int(adata.obs_names.duplicated().sum())
        if 'cell_id' in adata.obs and adata.obs['cell_id'].astype(str).is_unique:
            names, source = adata.obs['cell_id'].astype(str).to_numpy(), 'cell_id'
        else:
            names, source = np.array([f'row{i}' for i in range(adata.n_obs)]), 'row'
        adata.obs_names = pd.Index([f'{batch}:{name}' for name in names])
        if n_repeated:
            xy = pd.DataFrame(np.asarray(adata.obsm['spatial']), columns=['x', 'y'])
            xy['name'] = adata.obs['silver_name'].to_numpy()
            n_identical = int(xy.duplicated().sum())
            log(f'{path}: {n_repeated} repeated cell names ({n_identical} of them also with identical coordinates); '
                f'cells are named <batch>:<{source}> instead.')
        labelled = ~adata.obs[cfg['niche_key']].isna().to_numpy()
        adata = adata[labelled].copy()
        adata.obs['section'] = batch
        adata.uns = {}
        adatas.append(adata)
        log(f'Loaded {path}: {adata.n_obs} cells with a niche label, {adata.n_vars} genes.')
    adata = ad.concat(adatas, join='outer', index_unique=None)
    if not adata.obs_names.is_unique:
        raise ValueError('Cell names are not unique across the sections.')
    if not adata.var_names.is_unique:
        raise ValueError('Gene names are not unique.')
    if cfg['sample_key'] not in adata.obs:
        raise KeyError(f"Sample key '{cfg['sample_key']}' is not in adata.obs.")
    n_samples = adata.obs.groupby('section', observed=True)[cfg['sample_key']].nunique()
    if (n_samples != 1).any():
        raise ValueError(f"Every section must have one value of '{cfg['sample_key']}': {n_samples.to_dict()}.")
    return adata


def gene_reference_files(cfg: dict) -> tuple:
    """
    Gene mapping and occurrence-count files: explicit paths, else the model folder's files; for the mapping, else the
    mapping of the silver files written by the prepare step (no Ensembl download on compute nodes), else None.
    """
    model_folder = Path(cfg['model_folder_path'])
    mapping = cfg.get('gene_mapping_dict_file_path')
    if not mapping:
        mapping = next((str(model_folder / name) for name in GENE_MAPPING_FILE_NAMES
                        if (model_folder / name).is_file()), None)
    if not mapping and (Path(cfg['output_folder_path']) / SILVER_GENE_MAPPING_FILE_NAME).is_file():
        mapping = str(Path(cfg['output_folder_path']) / SILVER_GENE_MAPPING_FILE_NAME)
    occurrence = cfg.get('gene_occurrence_count_file_path')
    if not occurrence and (model_folder / GENE_OCCURRENCE_FILE_NAME).is_file():
        occurrence = str(model_folder / GENE_OCCURRENCE_FILE_NAME)
    return mapping, occurrence


def write_silver_gene_mapping(cfg: dict) -> bool:
    """
    Write {gene name: Ensembl ID} from var['ensembl_id'] of the silver files (the mapping with which they were
    tokenized for pretraining) for a model folder without a gene mapping. Returns whether it was written.
    """
    import pickle

    import anndata as ad

    mapping = {}
    for path in section_files(cfg):
        adata = ad.read_h5ad(path, backed='r')
        var = adata.var.copy()
        adata.file.close()
        if 'ensembl_id' not in var:
            log(f'WARNING: {path} has no var["ensembl_id"]; gene names are mapped with an Ensembl release lookup.')
            return False
        for gene, ensembl_id in zip(var.index, var['ensembl_id']):
            if not (isinstance(ensembl_id, str) and ensembl_id.startswith('ENS')):
                continue
            if mapping.setdefault(gene, ensembl_id) != ensembl_id:
                raise ValueError(f'The silver files map {gene} to different Ensembl IDs.')
    path = Path(cfg['output_folder_path']) / SILVER_GENE_MAPPING_FILE_NAME
    tmp_path = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with open(tmp_path, 'wb') as f:
        pickle.dump(mapping, f)
    os.replace(tmp_path, path)
    log(f'The model folder has no gene mapping; using the silver files\' mapping ({len(mapping)} genes): {path}.')
    return True


def embedding_model_folder(cfg: dict) -> Path:
    """Model folder used for embedding: the copy with pad_empty_neighbors, or the original folder."""
    if cfg['pad_empty_neighbors']:
        return Path(cfg['output_folder_path']) / 'model_folder'
    return Path(cfg['model_folder_path'])


def make_model_folder_copy(cfg: dict):
    """
    Folder whose files link to the model folder, except model_config.yaml, which sets data.pad_empty_neighbors so that
    a neighbor without panel genes keeps its segment (as padding) instead of shifting the later segments.
    """
    import yaml

    source = Path(cfg['model_folder_path'])
    target = embedding_model_folder(cfg)
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name == 'model_config.yaml' or not path.is_file():
            continue
        link = target / path.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.resolve())
    with open(source / 'model_config.yaml') as f:
        model_config = yaml.safe_load(f)
    model_config['data']['pad_empty_neighbors'] = True
    tmp_path = target / f'.model_config.yaml.tmp-{os.getpid()}'
    with open(tmp_path, 'w') as f:
        yaml.safe_dump(model_config, f, sort_keys=False)
    os.replace(tmp_path, target / 'model_config.yaml')
    log(f'Model folder copy with data.pad_empty_neighbors = true: {target}.')


def read_model_config(cfg: dict) -> dict:
    import yaml

    with open(Path(cfg['model_folder_path']) / 'model_config.yaml') as f:
        return yaml.safe_load(f)


def label_column(cfg: dict, label: str) -> str:
    return {'niche': cfg['niche_key'], 'cell_type': cfg['cell_type_key']}[label]


def representations(cfg: dict) -> dict:
    reps = dict(REPRESENTATIONS)
    if cfg.get('evaluate_spatial_cell_emb'):
        reps.update(SPATIAL_CELL_REPRESENTATION)
    return reps


# ----------------------------------------------------------------------------------------------------------------------
# Prepare
# ----------------------------------------------------------------------------------------------------------------------

def cmd_prepare(args):
    import pickle

    import pandas as pd
    import scipy.sparse as sp

    from terra.inference import harmonize_adata

    cfg = load_config(args.output_folder_path)
    out = Path(cfg['output_folder_path'])
    model_config = read_model_config(cfg)
    seq_len_cell = int(model_config['data']['seq_len_cell'])

    adata = load_sections(cfg)
    n_genes_silver = adata.n_vars

    # Panel genes: the genes that the embedding pipeline keeps (same harmonization and files) and that are in the
    # model vocabulary; they are sampled by gene name
    if gene_reference_files(cfg)[0] is None:
        write_silver_gene_mapping(cfg)
    mapping_path, occurrence_path = gene_reference_files(cfg)
    log(f'Gene mapping: {mapping_path or "Ensembl release lookup"}; occurrence counts: {occurrence_path or "none"}.')
    harmonized = harmonize_adata(
        adata, gene_mapping_dict_file_path=mapping_path, gene_occurrence_count_file_path=occurrence_path,
        gene_occurrence_count_filter_value=cfg['gene_occurrence_count_filter_value'],
        min_genes_per_cell=0, min_cells_per_gene=0)
    with open(Path(cfg['model_folder_path']) / 'token_dictionary.pkl', 'rb') as f:
        vocabulary = {key for key in pickle.load(f) if 'ENS' in key}
    in_vocabulary = harmonized.var['ensembl_id'].isin(vocabulary).to_numpy()
    panel_genes = np.array(sorted(harmonized.var_names[in_vocabulary]))
    n_panel = len(panel_genes)
    ensembl_ids = harmonized.var['ensembl_id'].to_dict()
    n_duplicated_ids = int(pd.Series([ensembl_ids[g] for g in panel_genes]).duplicated().sum())
    log(f'{n_genes_silver} genes in the silver data, {harmonized.n_vars} after harmonization, {n_panel} in the '
        f'model vocabulary (panel genes); {n_duplicated_ids} share an Ensembl ID with another panel gene.')

    for condition in cfg['conditions']:
        if condition['kind'] == 'random' and condition['size'] >= n_panel:
            raise ValueError(f'Panel size {condition["size"]} is not smaller than the {n_panel} panel genes.')
        if condition['kind'] == 'disjoint' and 2 * condition['size'] > n_panel:
            raise ValueError(f'Disjoint panels of {condition["size"]} genes need {2 * condition["size"]} panel genes; '
                             f'there are {n_panel}.')

    panels = {}
    if cfg.get('panels_from'):
        # The panels of another run (e.g. the same panels embedded with another model); every gene must be a panel
        # gene of this run as well
        source_panels = read_json(Path(cfg['panels_from']) / 'panels.json')['panels']
        available = set(panel_genes)
        for condition in cfg['conditions']:
            genes = source_panels[condition['condition_id']]['genes']
            missing = sorted(set(genes) - available)
            if missing:
                raise ValueError(f'{len(missing)} genes of {condition["condition_id"]} in {cfg["panels_from"]} are '
                                 f'not panel genes here (not kept by harmonization or not in this model\'s '
                                 f'vocabulary), e.g. {missing[:5]}.')
            panels[condition['condition_id']] = {'genes': list(genes), 'ensembl_ids': [ensembl_ids[g] for g in genes]}
        log(f'Panels of {len(panels)} conditions taken from {cfg["panels_from"]}.')
    else:
        # Panels: permutation r of the sorted panel genes (seed sequence [seed, r])
        permutations = {r: np.random.default_rng([cfg['seed'], r]).permutation(n_panel)
                        for r in range(cfg['n_replicates'])}
        for condition in cfg['conditions']:
            kind, size, replicate = condition['kind'], condition['size'], condition['replicate']
            if kind in ('full', 'full_repeat'):
                genes = panel_genes
            elif kind == 'random':
                genes = panel_genes[np.sort(permutations[replicate][:size])]
            else:
                genes = panel_genes[np.sort(permutations[replicate][size:2 * size])]
            panels[condition['condition_id']] = {'genes': genes.tolist(),
                                                 'ensembl_ids': [ensembl_ids[g] for g in genes]}
    for condition in cfg['conditions']:
        if condition['kind'] == 'disjoint':
            shared = set(panels[condition['condition_id']]['genes']) & set(panels[condition['partner']]['genes'])
            assert not shared, f'{condition["condition_id"]} shares genes with its partner.'

    # Detected panel genes per cell and condition (nonzero counts), on the harmonized counts
    X = sp.csr_matrix(harmonized[:, panel_genes].X)
    detected = (X > 0).astype(np.int32).tocsc()
    gene_position = {gene: i for i, gene in enumerate(panel_genes)}
    obs = harmonized.obs
    cells = pd.DataFrame({
        'cell': harmonized.obs_names,
        'section': obs['section'].astype(str).to_numpy(),
        'sample': obs[cfg['sample_key']].astype(str).to_numpy(),
        'niche': obs[cfg['niche_key']].astype(object).to_numpy(),
        'cell_type': (obs[cfg['cell_type_key']].astype(object).to_numpy()
                      if cfg['cell_type_key'] in obs else np.full(harmonized.n_obs, None)),
        'total_counts_full': np.asarray(X.sum(axis=1)).ravel(),
    })
    n_genes_detected = {}
    for condition in cfg['conditions']:
        columns = [gene_position[g] for g in panels[condition['condition_id']]['genes']]
        n_genes_detected[condition['condition_id']] = np.asarray(detected[:, columns].sum(axis=1)).ravel()
    detected_everywhere = np.all(np.stack([n > 0 for n in n_genes_detected.values()]), axis=0)
    passes_qc = (obs['filter_pass'].to_numpy() == 1) if 'filter_pass' in obs else np.ones(harmonized.n_obs, bool)
    # Detected genes of all panel genes (the full panel), also in runs without the 'full' condition
    cells['n_genes_full'] = np.asarray(detected.sum(axis=1)).ravel()
    cells['eval'] = detected_everywhere & passes_qc
    if cfg.get('panels_from'):
        # The evaluation cells of the source run, so that every run with these panels is scored on the same cells
        source_cells = pd.read_csv(Path(cfg['panels_from']) / 'cells.tsv.gz', sep='\t', usecols=['cell', 'eval'],
                                   dtype={'cell': str})
        source_eval = dict(zip(source_cells['cell'], source_cells['eval'].astype(bool)))
        cells['eval'] = cells['eval'].to_numpy() & cells['cell'].map(source_eval).fillna(False).to_numpy(bool)
    log(f'Evaluation cells (at least one detected gene in every panel): {int(cells["eval"].sum())} of {len(cells)}.')

    # Panel statistics per condition and section
    rows = []
    for condition in cfg['conditions']:
        n = n_genes_detected[condition['condition_id']]
        for section in ['all'] + list(cfg['batches']):
            mask = np.ones(len(cells), bool) if section == 'all' else (cells['section'] == section).to_numpy()
            rows.append({
                'condition': condition['condition_id'], 'kind': condition['kind'],
                'size': condition['size'] if condition['size'] is not None else n_panel,
                'replicate': condition['replicate'], 'section': section,
                'n_panel_genes': len(panels[condition['condition_id']]['genes']), 'n_cells': int(mask.sum()),
                'median_detected_genes': float(np.median(n[mask])),
                'mean_detected_genes': float(np.mean(n[mask])),
                'pct_cells_no_gene': float(100 * np.mean(n[mask] == 0)),
                'pct_cells_below_10_genes': float(100 * np.mean(n[mask] < 10)),
                f'pct_cells_above_{seq_len_cell}_genes': float(100 * np.mean(n[mask] > seq_len_cell)),
                'pct_eval_cells': float(100 * np.mean(cells['eval'].to_numpy()[mask])),
            })
    panel_statistics = pd.DataFrame(rows)

    # Write outputs
    write_json(out / 'panels.json', {'n_panel_genes': n_panel, 'n_genes_silver': int(n_genes_silver),
                                     'n_genes_harmonized': int(harmonized.n_vars),
                                     'gene_mapping_file_path': mapping_path,
                                     'gene_occurrence_count_file_path': occurrence_path,
                                     'panels': panels})
    detected_table = pd.DataFrame(n_genes_detected)
    detected_table.insert(0, 'cell', harmonized.obs_names)
    detected_table.to_csv(out / 'detected_genes_per_cell.tsv.gz', sep='\t', index=False)
    cells.to_csv(out / 'cells.tsv.gz', sep='\t', index=False)
    panel_statistics.to_csv(out / 'panel_statistics.tsv', sep='\t', index=False)
    if cfg['pad_empty_neighbors']:
        make_model_folder_copy(cfg)
    write_json(out / 'prepare_done.json', {'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                                           'n_cells': int(len(cells)), 'n_eval_cells': int(cells['eval'].sum()),
                                           'n_panel_genes': n_panel, 'seq_len_cell': seq_len_cell})
    print(panel_statistics[panel_statistics['section'] == 'all'].drop(columns=['section']).to_string(index=False),
          file=sys.stderr)
    log('Prepare finished.')


def load_panels(cfg: dict) -> dict:
    path = Path(cfg['output_folder_path']) / 'panels.json'
    if not (Path(cfg['output_folder_path']) / 'prepare_done.json').is_file():
        raise FileNotFoundError(f'The prepare step has not finished for {cfg["output_folder_path"]}.')
    return read_json(path)


# ----------------------------------------------------------------------------------------------------------------------
# Embed
# ----------------------------------------------------------------------------------------------------------------------

def terra_provenance() -> dict:
    """Location, version and git commit of the TERRA package used."""
    import terra

    info = {'terra_file': terra.__file__, 'terra_version': getattr(terra, '__version__', None)}
    source = Path(terra.__file__).resolve().parent
    try:
        def git(*git_args):
            return subprocess.run(['git', '-C', str(source), *git_args], capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        info['terra_git_commit'] = git('rev-parse', 'HEAD') or None
        info['terra_git_branch'] = git('rev-parse', '--abbrev-ref', 'HEAD') or None
        info['terra_git_dirty'] = bool(git('status', '--porcelain', '--untracked-files=no'))
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def embedding_path(cfg: dict, condition_id: str) -> Path:
    return Path(cfg['output_folder_path']) / 'embeddings' / f'{condition_id}.npz'


def cmd_embed(args):
    cfg = load_config(args.output_folder_path)
    condition = condition_by_index(cfg, args.condition_index)
    condition_id = condition['condition_id']
    path = embedding_path(cfg, condition_id)
    if path.is_file() and not args.force:
        log(f'{path} exists; skipping (use --force to embed again).')
        return
    panels = load_panels(cfg)
    genes = panels['panels'][condition_id]['genes']
    log(f'Condition {condition_id}: {len(genes)} panel genes.')

    from terra.inference import harmonize_tokenize_embed_pipeline

    adata = load_sections(cfg)
    adata = adata[:, genes].copy()
    mapping_path, occurrence_path = gene_reference_files(cfg)
    cache = Path(cfg['cache_folder_path']) / f'{condition_id}-{os.environ.get("LSB_JOBID", os.getpid())}'
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    start = time.time()
    try:
        embedded = harmonize_tokenize_embed_pipeline(
            adata=adata,
            sample_key=cfg['sample_key'],
            batch_key='batch',
            model_folder_path=str(embedding_model_folder(cfg)),
            cache_directory_path=str(cache),
            gene_mapping_dict_file_path=mapping_path,
            gene_occurrence_count_file_path=occurrence_path,
            gene_occurrence_count_filter_value=cfg['gene_occurrence_count_filter_value'],
            min_cells_per_gene=0,
            min_genes_per_cell=0,
            nproc=cfg['nproc'],
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'])
    finally:
        # The tokenized dataset is only needed for embedding (several GB per condition)
        shutil.rmtree(cache, ignore_errors=True)
    runtime = time.time() - start
    if embedded.n_vars != len(genes):
        raise RuntimeError(f'The pipeline kept {embedded.n_vars} of the {len(genes)} panel genes.')
    # The tokenizer drops cells with obs['filter_pass'] != 1, if the column exists; no other cell may be lost
    expected = (adata.obs_names[adata.obs['filter_pass'].to_numpy() == 1] if 'filter_pass' in adata.obs
                else adata.obs_names)
    if embedded.n_obs != len(expected) or set(embedded.obs_names) != set(expected):
        raise RuntimeError(f'The pipeline returned {embedded.n_obs} cells for {len(expected)} input cells.')
    arrays = {'obs_names': np.asarray(embedded.obs_names, dtype=str)}
    for key in ('cell_emb', 'neighborhood_emb', 'spatial_cell_emb'):
        if key in embedded.obsm:
            arrays[key] = np.asarray(embedded.obsm[key], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    metadata = {'condition': condition, 'n_genes': len(genes), 'n_cells': int(embedded.n_obs),
                'embedding_dims': {k: list(v.shape) for k, v in arrays.items() if k != 'obs_names'},
                'model_folder_path': str(embedding_model_folder(cfg)), 'gene_mapping_file_path': mapping_path,
                'gene_occurrence_count_file_path': occurrence_path, 'runtime_seconds': runtime,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'torch_version': torch.__version__, 'lsf_job_id': os.environ.get('LSB_JOBID'),
                'lsf_job_index': os.environ.get('LSB_JOBINDEX'), 'host': os.uname().nodename,
                'finished': time.strftime('%Y-%m-%d %H:%M:%S'), **terra_provenance()}
    write_json(path.with_suffix('.json'), metadata)
    # Embeddings last: their file marks the condition as embedded
    save_npz(path, **arrays)
    log(f'Saved {path} ({runtime / 60:.1f} min).')


def load_terra_embeddings(cfg: dict, condition_id: str, eval_cells, keys) -> dict:
    """TERRA embeddings of a condition for the evaluation cells, in the order of eval_cells."""
    path = embedding_path(cfg, condition_id)
    if not path.is_file():
        raise FileNotFoundError(f'No embeddings for {condition_id} ({path}); run the embed step.')
    with np.load(path, allow_pickle=False) as data:
        position = {cell: i for i, cell in enumerate(data['obs_names'])}
        missing = [cell for cell in eval_cells if cell not in position]
        if missing:
            raise ValueError(f'{len(missing)} evaluation cells are not in {path} (e.g. {missing[:3]}).')
        rows = np.fromiter((position[cell] for cell in eval_cells), dtype=np.int64, count=len(eval_cells))
        embeddings = {key: data[key][rows] for key in keys}
    for key, X in embeddings.items():
        if not np.isfinite(X).all():
            raise ValueError(f'{condition_id} {key} has non-finite values for evaluation cells.')
    return embeddings


# ----------------------------------------------------------------------------------------------------------------------
# Evaluate
# ----------------------------------------------------------------------------------------------------------------------

def load_cells(cfg: dict):
    import pandas as pd

    cells = pd.read_csv(Path(cfg['output_folder_path']) / 'cells.tsv.gz', sep='\t', keep_default_na=True,
                        dtype={'cell': str, 'section': str, 'sample': str, 'niche': object, 'cell_type': object})
    return cells


def pca_baselines(cfg: dict, genes: list, eval_cells, n_neighs: int) -> dict:
    """
    PCA of log-normalized expression (cells) and of log-normalized expression averaged over each cell and its
    n_neighs nearest spatial neighbors in its section (neighborhoods), on the panel genes; fit on all cells, returned
    for the evaluation cells.
    """
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    adata = load_sections(cfg)
    adata = adata[:, genes].copy()
    adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X.tocsr()
    sections = adata.obs['section'].astype(str).to_numpy()
    coordinates = np.asarray(adata.obsm['spatial'], dtype=np.float64)

    rows, columns = [], []
    for section in np.unique(sections):
        index = np.flatnonzero(sections == section)
        nn = NearestNeighbors(n_neighbors=n_neighs).fit(coordinates[index])
        _, neighbors = nn.kneighbors()
        rows.append(np.repeat(index, n_neighs + 1))
        columns.append(np.concatenate([index[:, None], index[neighbors]], axis=1).ravel())
    averaging = sp.csr_matrix((np.full(sum(len(r) for r in rows), 1.0 / (n_neighs + 1), dtype=np.float32),
                               (np.concatenate(rows), np.concatenate(columns))), shape=(adata.n_obs, adata.n_obs))
    X_neighborhood = averaging @ X

    n_comps = min(cfg['pca_n_comps'], len(genes) - 1, adata.n_obs - 1)
    position = {cell: i for i, cell in enumerate(adata.obs_names)}
    eval_rows = np.fromiter((position[cell] for cell in eval_cells), dtype=np.int64, count=len(eval_cells))
    result = {}
    for key, matrix in (('cell', X), ('neighborhood', X_neighborhood)):
        pca = PCA(n_components=n_comps, svd_solver='randomized', random_state=0)
        result[key] = pca.fit_transform(np.asarray(matrix.toarray(), dtype=np.float32))[eval_rows].astype(np.float32)
    return result


def leiden_flavor() -> str:
    import inspect

    import scanpy as sc

    return 'igraph' if 'flavor' in inspect.signature(sc.tl.leiden).parameters else 'leidenalg'


def run_leiden(adata, resolution: float, seed: int, flavor: str) -> np.ndarray:
    import scanpy as sc

    kwargs = {'resolution': resolution, 'random_state': seed, 'neighbors_key': 'rep', 'key_added': '_leiden'}
    if flavor == 'igraph':
        sc.tl.leiden(adata, flavor='igraph', n_iterations=2, directed=False, **kwargs)
    else:
        sc.tl.leiden(adata, **kwargs)
    return adata.obs['_leiden'].astype(int).to_numpy()


def leiden_matching_k(adata, target_k: int, seed: int, flavor: str, max_iterations: int = 25) -> tuple:
    """
    Leiden clusters whose number is target_k (or the closest reached): geometric bisection of the resolution in
    [1e-3, 10] from 1. Returns resolution, number of clusters, clusters and number of Leiden runs.
    """
    tried = {}
    low, high, resolution = 1e-3, 10.0, 1.0
    for _ in range(max_iterations):
        clusters = run_leiden(adata, resolution, seed, flavor)
        n_clusters = int(clusters.max()) + 1
        tried[resolution] = (n_clusters, clusters)
        if n_clusters == target_k:
            break
        if n_clusters < target_k:
            low = resolution
        else:
            high = resolution
        resolution = float(np.sqrt(low * high))
    best = min(tried, key=lambda r: (abs(tried[r][0] - target_k), r))
    return best, tried[best][0], tried[best][1], len(tried)


def evaluation_folder(cfg: dict, condition_id: str) -> Path:
    return Path(cfg['output_folder_path']) / 'evaluation' / condition_id


def cmd_evaluate(args):
    import anndata as ad
    import scanpy as sc

    from panel_metrics import (exact_knn_indices, label_scores, leave_one_section_out_transfer,
                               n_classes_with_min_cells)

    cfg = load_config(args.output_folder_path)
    condition = condition_by_index(cfg, args.condition_index)
    condition_id = condition['condition_id']
    folder = evaluation_folder(cfg, condition_id)
    if (folder / 'done.json').is_file() and not args.force:
        log(f'{folder} is complete; skipping (use --force to evaluate again).')
        return
    folder.mkdir(parents=True, exist_ok=True)
    panels = load_panels(cfg)
    genes = panels['panels'][condition_id]['genes']
    cells = load_cells(cfg)
    cells = cells[cells['eval'].astype(bool)].reset_index(drop=True)
    eval_cells = cells['cell'].tolist()
    sections = cells['section'].to_numpy()
    labels = {'niche': cells['niche'].to_numpy(dtype=object), 'cell_type': cells['cell_type'].to_numpy(dtype=object)}
    n_jobs = int(os.environ.get('LSB_DJOB_NUMPROC', os.cpu_count() or 1))
    flavor = leiden_flavor()
    reps = representations(cfg)
    log(f'Evaluating {condition_id}: {len(eval_cells)} cells, {len(genes)} genes, Leiden flavor {flavor}.')

    terra_keys = sorted({spec['key'] for spec in reps.values() if spec['source'] == 'terra'})
    data = {('terra', key): X for key, X in load_terra_embeddings(cfg, condition_id, eval_cells, terra_keys).items()}
    n_neighs = int(read_model_config(cfg)['data']['n_neighs'])
    pca = pca_baselines(cfg, genes, eval_cells, n_neighs)
    data.update({('pca', key): X for key, X in pca.items()})
    save_npz(folder / 'pca.npz', **pca)

    records, partitions, knn = [], {}, {}

    def record(representation, label, metric, value, section='all'):
        records.append({'condition': condition_id, 'comparison': 'intrinsic', 'reference': '',
                        'representation': representation, 'label': label, 'metric': metric, 'section': section,
                        'value': float(value)})

    for name, spec in reps.items():
        X = data[(spec['source'], spec['key'])]
        label = spec['label']
        y = labels[label]
        start = time.time()

        knn[name] = exact_knn_indices(X, sections, cfg['knn_k'], n_jobs=n_jobs).astype(np.int32)

        adata = ad.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
        adata.obsm['X_rep'] = X
        sc.pp.neighbors(adata, n_neighbors=cfg['knn_k'], use_rep='X_rep', key_added='rep', random_state=0)

        fixed = []
        for resolution in cfg['leiden_resolutions']:
            clusters = run_leiden(adata, resolution, 0, flavor)
            fixed.append(clusters)
            scores = label_scores(y, clusters)
            record(name, label, f'nmi_res{resolution:g}', scores['nmi'])
            record(name, label, f'ari_res{resolution:g}', scores['ari'])
            record(name, label, f'n_clusters_res{resolution:g}', int(clusters.max()) + 1)
        partitions[f'fixed__{name}'] = np.stack(fixed).astype(np.int32)

        target_k = n_classes_with_min_cells(y, cfg['min_cells_per_class'])
        matched, nmis, aris = [], [], []
        for seed in range(cfg['n_leiden_seeds']):
            resolution, n_clusters, clusters, n_runs = leiden_matching_k(adata, target_k, seed, flavor)
            matched.append(clusters)
            scores = label_scores(y, clusters)
            nmis.append(scores['nmi'])
            aris.append(scores['ari'])
            record(name, label, 'kmatched_resolution', resolution, section=f'seed{seed}')
            record(name, label, 'kmatched_n_clusters', n_clusters, section=f'seed{seed}')
            record(name, label, 'nmi_kmatched', scores['nmi'], section=f'seed{seed}')
            record(name, label, 'ari_kmatched', scores['ari'], section=f'seed{seed}')
        record(name, label, 'target_k', target_k)
        record(name, label, 'nmi_kmatched', np.mean(nmis))
        record(name, label, 'ari_kmatched', np.mean(aris))
        partitions[f'kmatched__{name}'] = np.stack(matched).astype(np.int32)

        transfer = leave_one_section_out_transfer(X, X, y, sections, cfg['knn_k'], n_jobs=n_jobs)
        for result in transfer:
            record(name, label, 'transfer_accuracy', result['accuracy'], section=result['section'])
            record(name, label, 'transfer_macro_f1', result['macro_f1'], section=result['section'])
        record(name, label, 'transfer_accuracy', np.mean([r['accuracy'] for r in transfer]))
        record(name, label, 'transfer_macro_f1', np.mean([r['macro_f1'] for r in transfer]))
        log(f'{name}: K = {target_k}, ARI {np.mean(aris):.3f}, NMI {np.mean(nmis):.3f}, transfer macro F1 '
            f'{np.mean([r["macro_f1"] for r in transfer]):.3f} ({time.time() - start:.0f} s).')

    save_npz(folder / 'partitions.npz', **partitions)
    save_npz(folder / 'knn.npz', **knn)
    write_json(folder / 'metrics.json', records)
    write_json(folder / 'done.json', {'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'leiden_flavor': flavor,
                                      'scanpy_version': sc.__version__, 'n_eval_cells': len(eval_cells),
                                      'eval_cells_hash': hash_cells(eval_cells)})
    log(f'Evaluation of {condition_id} finished.')


def hash_cells(cells) -> str:
    import hashlib

    return hashlib.sha256('\n'.join(cells).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------------------------
# Summarize
# ----------------------------------------------------------------------------------------------------------------------

def cmd_summarize(args):
    import pandas as pd

    from panel_metrics import (knn_overlap, leave_one_section_out_transfer, linear_cka, mean_pairwise_agreement,
                               paired_cosine_similarity, partition_agreement)

    cfg = load_config(args.output_folder_path)
    out = Path(cfg['output_folder_path'])
    results = out / 'results'
    results.mkdir(parents=True, exist_ok=True)
    reps = representations(cfg)
    cells = load_cells(cfg)
    cells = cells[cells['eval'].astype(bool)].reset_index(drop=True)
    eval_cells = cells['cell'].tolist()
    sections = cells['section'].to_numpy()
    labels = {'niche': cells['niche'].to_numpy(dtype=object), 'cell_type': cells['cell_type'].to_numpy(dtype=object)}
    n_jobs = int(os.environ.get('LSB_DJOB_NUMPROC', os.cpu_count() or 1))
    expected_hash = hash_cells(eval_cells)

    conditions = {c['condition_id']: c for c in cfg['conditions']}
    complete, missing = [], []
    for condition_id in conditions:
        done = evaluation_folder(cfg, condition_id) / 'done.json'
        if done.is_file() and embedding_path(cfg, condition_id).is_file():
            if read_json(done).get('eval_cells_hash') != expected_hash:
                raise ValueError(f'{condition_id} was evaluated on other cells; evaluate it again (--force).')
            complete.append(condition_id)
        else:
            missing.append(condition_id)
    has_full = 'full' in complete
    if not has_full:
        if 'full' in conditions:
            raise FileNotFoundError('The full panel has not been embedded and evaluated; nothing to compare with.')
        log('This run has no full panel (--conditions): only the annotation metrics are tabulated; compare it with '
            'other runs with compare_panel_runs.py.')
    if not complete:
        raise FileNotFoundError('No condition has been embedded and evaluated.')
    if missing:
        log(f'WARNING: {len(missing)} conditions are not complete and are left out: {", ".join(missing)}.')

    records = []
    for condition_id in complete:
        records.extend(read_json(evaluation_folder(cfg, condition_id) / 'metrics.json'))

    def load_evaluation(condition_id):
        folder = evaluation_folder(cfg, condition_id)
        with np.load(folder / 'partitions.npz') as p, np.load(folder / 'knn.npz') as k, \
                np.load(folder / 'pca.npz') as pc:
            partitions = {key: p[key] for key in p.files}
            knn = {key: k[key] for key in k.files}
            pca = {key: pc[key] for key in pc.files}
        terra_keys = sorted({spec['key'] for spec in reps.values() if spec['source'] == 'terra'})
        embeddings = load_terra_embeddings(cfg, condition_id, eval_cells, terra_keys)
        embeddings = {**{('terra', key): X for key, X in embeddings.items()},
                      **{('pca', key): X for key, X in pca.items()}}
        return {'partitions': partitions, 'knn': knn, 'embeddings': embeddings}

    reference = load_evaluation('full') if has_full else None

    def compare(condition_id, evaluation, reference_id, other):
        """Consistency metrics of evaluation (condition_id) against other (reference_id), same cells."""
        rows = []

        def add(representation, label, metric, value, section='all'):
            rows.append({'condition': condition_id, 'comparison': f'vs_{"full" if reference_id == "full" else "partner"}',
                         'reference': reference_id, 'representation': representation, 'label': label,
                         'metric': metric, 'section': section, 'value': float(value)})

        for name, spec in reps.items():
            label = spec['label']
            overlap = knn_overlap(evaluation['knn'][name], other['knn'][name])
            for section in np.unique(sections):
                add(name, label, 'knn_overlap', overlap[sections == section].mean(), section=section)
            add(name, label, 'knn_overlap', np.mean([overlap[sections == s].mean() for s in np.unique(sections)]))
            # Pairs of different Leiden seeds only: the same seed on (nearly) the same graph repeats its partition,
            # which would inflate the agreement of near-identical embeddings above the seed-to-seed ceiling
            agreement = mean_pairwise_agreement(evaluation['partitions'][f'kmatched__{name}'],
                                                other['partitions'][f'kmatched__{name}'], exclude_same_index=True)
            add(name, label, 'cluster_ari_kmatched', agreement['ari'])
            add(name, label, 'cluster_nmi_kmatched', agreement['nmi'])
            for i, resolution in enumerate(cfg['leiden_resolutions']):
                scores = partition_agreement(evaluation['partitions'][f'fixed__{name}'][i],
                                             other['partitions'][f'fixed__{name}'][i])
                add(name, label, f'cluster_ari_res{resolution:g}', scores['ari'])
            X = evaluation['embeddings'][(spec['source'], spec['key'])]
            Y = other['embeddings'][(spec['source'], spec['key'])]
            add(name, label, 'linear_cka', linear_cka(X, Y))
            if spec['source'] == 'terra':
                # Same model, same space: compare each cell's embeddings directly and transfer labels across panels
                add(name, label, 'paired_cosine_median', np.median(paired_cosine_similarity(X, Y)))
                y = labels[label]
                for centered in (False, True):
                    X_query = X - X.mean(axis=0) if centered else X
                    Y_reference = Y - Y.mean(axis=0) if centered else Y
                    transfer = leave_one_section_out_transfer(Y_reference, X_query, y, sections, cfg['knn_k'],
                                                              n_jobs=n_jobs)
                    suffix = '_centered' if centered else ''
                    for result in transfer:
                        add(name, label, f'cross_panel_transfer_accuracy{suffix}', result['accuracy'],
                            section=result['section'])
                        add(name, label, f'cross_panel_transfer_macro_f1{suffix}', result['macro_f1'],
                            section=result['section'])
                    add(name, label, f'cross_panel_transfer_accuracy{suffix}',
                        np.mean([r['accuracy'] for r in transfer]))
                    add(name, label, f'cross_panel_transfer_macro_f1{suffix}',
                        np.mean([r['macro_f1'] for r in transfer]))
        return rows

    # Seed-to-seed noise ceiling of the K-matched clusters of the full panel
    for name, spec in (reps.items() if has_full else []):
        ceiling = mean_pairwise_agreement(reference['partitions'][f'kmatched__{name}'],
                                          reference['partitions'][f'kmatched__{name}'], exclude_same_index=True)
        for metric in ('ari', 'nmi'):
            records.append({'condition': 'full', 'comparison': 'vs_full', 'reference': 'full',
                            'representation': name, 'label': spec['label'],
                            'metric': f'cluster_{metric}_kmatched', 'section': 'all', 'value': ceiling[metric]})

    for condition_id in (complete if has_full else []):
        if condition_id == 'full':
            continue
        log(f'Comparing {condition_id} with the full panel.')
        evaluation = load_evaluation(condition_id)
        records.extend(compare(condition_id, evaluation, 'full', reference))
        partner = conditions[condition_id]['partner']
        if conditions[condition_id]['kind'] == 'disjoint':
            if partner in complete:
                log(f'Comparing {condition_id} with {partner}.')
                records.extend(compare(condition_id, evaluation, partner, load_evaluation(partner)))
            else:
                log(f'WARNING: the partner {partner} of {condition_id} is not complete.')

    metrics = pd.DataFrame(records)
    condition_table = pd.DataFrame(cfg['conditions']).rename(columns={'condition_id': 'condition'})
    panels = load_panels(cfg)
    condition_table['n_genes'] = [len(panels['panels'][c]['genes']) for c in condition_table['condition']]
    metrics = metrics.merge(condition_table[['condition', 'kind', 'size', 'replicate', 'n_genes']], on='condition',
                            how='left')
    metrics.to_csv(results / 'metrics_long.tsv', sep='\t', index=False)

    overall = metrics[metrics['section'] == 'all']
    summary = (overall.groupby(['kind', 'n_genes', 'comparison', 'representation', 'label', 'metric'], dropna=False)
               ['value'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index())
    summary = summary.sort_values(['comparison', 'representation', 'metric', 'kind', 'n_genes'])
    summary.to_csv(results / 'summary.tsv', sep='\t', index=False)
    shutil.copy(out / 'panel_statistics.tsv', results / 'panel_statistics.tsv')

    headline = headline_table(summary)
    headline.to_csv(results / 'headline.tsv', sep='\t', index=False)
    print(headline.to_string(index=False), file=sys.stderr)

    try:
        from plot_gene_panel_subsampling import plot_results
        plot_results(metrics, pd.read_csv(out / 'panel_statistics.tsv', sep='\t'), results / 'figures',
                     n_panel_genes=panels['n_panel_genes'])
    except ImportError as error:
        log(f'WARNING: no figures ({error}).')
    write_json(results / 'summary_done.json', {'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                                               'complete_conditions': complete, 'missing_conditions': missing})
    log(f'Results in {results}.')


HEADLINE_METRICS = [
    ('intrinsic', 'terra_neighborhood', 'ari_kmatched', 'Niche ARI (TERRA)'),
    ('intrinsic', 'pca_neighborhood', 'ari_kmatched', 'Niche ARI (nbhd PCA)'),
    ('intrinsic', 'terra_cell', 'ari_kmatched', 'Cell type ARI (TERRA)'),
    ('intrinsic', 'pca_cell', 'ari_kmatched', 'Cell type ARI (PCA)'),
    ('vs_full', 'terra_neighborhood', 'knn_overlap', 'Niche kNN overlap vs full (TERRA)'),
    ('vs_full', 'pca_neighborhood', 'knn_overlap', 'Niche kNN overlap vs full (nbhd PCA)'),
    ('vs_full', 'terra_cell', 'knn_overlap', 'Cell kNN overlap vs full (TERRA)'),
    ('vs_full', 'pca_cell', 'knn_overlap', 'Cell kNN overlap vs full (PCA)'),
    ('vs_full', 'terra_neighborhood', 'cluster_ari_kmatched', 'Niche clusters ARI vs full (TERRA)'),
    ('vs_full', 'pca_neighborhood', 'cluster_ari_kmatched', 'Niche clusters ARI vs full (nbhd PCA)'),
    ('vs_full', 'terra_cell', 'cluster_ari_kmatched', 'Cell clusters ARI vs full (TERRA)'),
    ('vs_full', 'pca_cell', 'cluster_ari_kmatched', 'Cell clusters ARI vs full (PCA)'),
    ('vs_full', 'terra_neighborhood', 'cross_panel_transfer_macro_f1', 'Niche transfer full->panel F1'),
    ('intrinsic', 'terra_neighborhood', 'transfer_macro_f1', 'Niche transfer within panel F1'),
    ('vs_full', 'terra_cell', 'cross_panel_transfer_macro_f1', 'Cell type transfer full->panel F1'),
    ('intrinsic', 'terra_cell', 'transfer_macro_f1', 'Cell type transfer within panel F1'),
]


def headline_table(summary):
    """One row per panel (kind and size), one column per headline metric, as 'mean ± sd' over replicates."""
    import pandas as pd

    rows = {}
    for (kind, n_genes), group in summary.groupby(['kind', 'n_genes'], dropna=False):
        row = rows.setdefault((kind, n_genes), {'panel': kind, 'n_genes': n_genes})
        for comparison, representation, metric, column in HEADLINE_METRICS:
            match = group[(group['comparison'] == comparison) & (group['representation'] == representation)
                          & (group['metric'] == metric)]
            if len(match) == 1:
                mean, std, count = match.iloc[0][['mean', 'std', 'count']]
                row[column] = f'{mean:.3f}' if count < 2 else f'{mean:.3f} ± {std:.3f}'
    order = {'random': 0, 'disjoint': 1, 'full': 2, 'full_repeat': 3}
    table = pd.DataFrame(rows.values())
    columns = [column for *_, column in HEADLINE_METRICS if column in table]
    table[columns] = table[columns].fillna('NA')  # e.g. no comparison of the full panel with itself
    return table.sort_values(['n_genes', 'panel'], key=lambda s: s.map(order) if s.name == 'panel' else s)


# ----------------------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    configure = subparsers.add_parser('configure', help='Write run_config.json and print the number of conditions.')
    configure.add_argument('--output_folder_path', required=True)
    configure.add_argument('--silver_folder_path', default=DEFAULT_SILVER_FOLDER_PATH)
    configure.add_argument('--dataset_name', default='xhs1000-39b_1p')
    configure.add_argument('--batches', nargs='+', default=['batch11', 'batch19', 'batch32'],
                           help='Sections (silver files adata_<batch>.h5ad) embedded together.')
    configure.add_argument('--sample_key', default='sample')
    configure.add_argument('--niche_key', default='niche_type')
    configure.add_argument('--cell_type_key', default='new_annotation')
    configure.add_argument('--model_folder_path', default=DEFAULT_MODEL_FOLDER_PATH,
                           help='TERRA model folder (model_config.yaml, token_dictionary.pkl, model_checkpoint.pt). '
                                'Default: the released TERRA-96M (benchmarking checkpoint).')
    configure.add_argument('--gene_mapping_dict_file_path', default=None,
                           help='Gene name to Ensembl ID mapping; default: the one in the model folder.')
    configure.add_argument('--gene_occurrence_count_file_path', default=None,
                           help='Pretraining occurrence counts; default: the one in the model folder, if any.')
    configure.add_argument('--gene_occurrence_count_filter_value', type=int, default=10)
    configure.add_argument('--panel_sizes', nargs='+', type=int, default=[1000, 2000, 3000, 4000])
    configure.add_argument('--n_replicates', type=int, default=5, help='Random panels per size.')
    configure.add_argument('--disjoint_sizes', nargs='*', type=int, default=[1000, 2000],
                           help='Sizes with a second random panel that shares no gene with the first.')
    configure.add_argument('--seed', type=int, default=0)
    configure.add_argument('--full_repeat', action=argparse.BooleanOptionalAction, default=True,
                           help='Embed the full panel twice (run-to-run variation).')
    configure.add_argument('--pad_empty_neighbors', action=argparse.BooleanOptionalAction, default=True,
                           help='Keep the segment of a neighbor without panel genes as padding.')
    configure.add_argument('--batch_size', type=int, default=128)
    configure.add_argument('--num_workers', type=int, default=8)
    configure.add_argument('--nproc', type=int, default=4, help='Processes for tokenization.')
    configure.add_argument('--cache_folder_path', default=None,
                           help='Hugging Face dataset cache (one subfolder per job, removed after it); '
                                'default: <output_folder_path>/cache.')
    configure.add_argument('--knn_k', type=int, default=15,
                           help='Neighbours for the clustering graph, kNN overlap and kNN label transfer.')
    configure.add_argument('--n_leiden_seeds', type=int, default=5)
    configure.add_argument('--leiden_resolutions', nargs='+', type=float, default=[0.1, 0.25, 0.5, 1.0])
    configure.add_argument('--min_cells_per_class', type=int, default=100,
                           help='K-matched Leiden targets the number of annotated classes with this many cells.')
    configure.add_argument('--pca_n_comps', type=int, default=50)
    configure.add_argument('--evaluate_spatial_cell_emb', action='store_true',
                           help='Also evaluate the spatially contextualized cell embedding.')
    configure.add_argument('--panels_from', default=None, metavar='RUN_FOLDER',
                           help='Reuse the panels (panels.json), conditions and evaluation cells of another run on the '
                                'same cells, e.g. to embed the same panels with another model. The panel options of '
                                'this run are then taken from that run.')
    configure.add_argument('--conditions', nargs='+', default=None, metavar='CONDITION',
                           help='Embed and evaluate only these conditions (e.g. random_1000_r0 for a model trained on '
                                'that panel). Without full, the summarize step only tabulates the annotation metrics.')
    configure.add_argument('--overwrite', action='store_true')

    for name, help_text in (('prepare', 'Panels, evaluation cells and panel statistics.'),
                            ('summarize', 'Comparisons, tables and figures.')):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument('--output_folder_path', required=True)
    for name, help_text in (('embed', 'Embed one condition with TERRA.'),
                            ('evaluate', 'Evaluate one condition.')):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument('--output_folder_path', required=True)
        sub.add_argument('--condition_index', type=int,
                         default=int(os.environ['LSB_JOBINDEX']) if os.environ.get('LSB_JOBINDEX') else None,
                         help='1-based condition index (default: $LSB_JOBINDEX).')
        sub.add_argument('--force', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    if getattr(args, 'condition_index', 0) is None:
        raise SystemExit('--condition_index is required outside an LSF job array.')
    os.environ.setdefault('NUMBA_CPU_NAME', 'generic')  # reproducible UMAP connectivities and Leiden across nodes
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    {'configure': cmd_configure, 'prepare': cmd_prepare, 'embed': cmd_embed, 'evaluate': cmd_evaluate,
     'summarize': cmd_summarize}[args.command](args)


if __name__ == '__main__':
    main()
