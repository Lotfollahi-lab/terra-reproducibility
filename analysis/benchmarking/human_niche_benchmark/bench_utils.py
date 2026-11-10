"""Shared definitions for the human niche-identification benchmark.

Paths, protocol constants, the datasets and the method registry, plus the two
functions every method script uses: ``load_benchmark_dataset`` (identical cells
for every method) and ``save_embedding`` (hands the embedding over to scoring,
score.py). Data loading mirrors
``analysis/benchmarking/human_benchmarking/terra_benchmarking.ipynb``; method
settings and metrics mirror ``analysis/benchmarking/mouse_benchmarking``.
"""
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

# ---------------------------------------------------------------------------
# Paths (adapt here; the .lsf scripts and job_wrapper.sh hold the matching
# job-level paths in a block at their top).
#
# Storage layout: jobs read inputs from lustre and write into a job-local lustre
# working folder (BENCH_OUTPUT_DIR / BENCH_CACHE_DIR, set by job_wrapper.sh); on
# success the wrapper copies results to FINAL_OUTPUT_FOLDER_PATH on NFS (backed
# up) and deletes the lustre folder. Without BENCH_OUTPUT_DIR, outputs go
# straight to the NFS folder (e.g. when results are read in notebooks).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
SILVER_FOLDER_PATH = os.environ.get(
    'BENCH_SILVER_DIR', '/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver')
FINAL_OUTPUT_FOLDER_PATH = str(REPO_ROOT / 'artifacts' / 'benchmarking' / 'human')
OUTPUT_FOLDER_PATH = os.environ.get('BENCH_OUTPUT_DIR', FINAL_OUTPUT_FOLDER_PATH)
CACHE_FOLDER_PATH = os.environ.get(
    'BENCH_CACHE_DIR',
    f'/lustre/scratch126/cellgen/lotfollahi/{os.environ.get("USER", "user")}/'
    'human_niche_benchmark/cache')

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
MIN_CELLS_PER_NICHE = 100   # common cluster number = GT niches with > 100 cells
N_NEIGHS = 10               # spatial neighbours (mouse benchmark)
N_LATENT_NEIGHBORS = 15     # kNN graph on the embedding for Leiden
SECTION_KEY = 'section'

# ---------------------------------------------------------------------------
# Datasets: sections, labels and preprocessing copied from
# human_benchmarking/terra_benchmarking.ipynb (Section 2).
# ---------------------------------------------------------------------------
DATASETS = {
    'xhs1000-39b_1p': dict(
        label='Xenium skin (eczema)',
        file_patterns=['adata_batch11.h5ad', 'adata_batch19.h5ad',
                       'adata_batch32.h5ad'],
        niche_key='niche_type',
        drop_unlabeled=True,
        coord_scale=1.0,
        spot_size=20,
        # Cell-type annotations (cell-type composition baseline): finest level
        # first; 'coarse' = key of the paper's cell-type benchmark.
        cell_type_levels=dict(fine='cell_type', coarse='new_annotation')),
    'xhs49-1b_1p': dict(
        label='Xenium skin (melanoma)',
        file_patterns=['Xenium_Prime_Human_Skin_FFPE_outs'],
        niche_key='niche_name',
        drop_unlabeled=True,
        coord_scale=1.0,
        spot_size=20,
        # Not in silver: read from the annotated source file, matched on
        # obs['original_cell_id'] == source obs['cell_id'] (all 98,749 cells).
        cell_type_levels=dict(fine='fine_celltypes', broad='broad_celltypes',
                              coarse='celltype'),
        cell_type_source=('/nfs/team361/sb75/DATASETS/bronze/xhs49-1b_1p/'
                          'adata_xenium5k_melanoma_11jan.h5ad')),
    'chr78-11b_1p': dict(
        label='ISS reproductive tract',
        file_patterns=['CM007', 'CM008', 'CM009', 'CM011', 'CM013', 'CM015',
                       'CM016', 'CM017', 'CM018', 'CM019', 'CM020'],
        niche_key='broad_anatomy',
        drop_unlabeled=False,
        coord_scale=0.093,
        label_replace={'Urogenital Sinus': 'UrogenitalSinus'},
        spot_size=240 * 0.093),
    'chl59-8b_1p': dict(
        label='CosMx lung (NSCLC)',
        file_patterns=['Lung9_Rep1', 'Lung9_Rep2', 'Lung5_Rep1',
                       'Lung5_Rep2', 'Lung6', 'Lung12'],
        niche_key='niche',
        drop_unlabeled=True,
        coord_scale=0.18,   # NanoString px -> um
        spot_size=80 * 0.18,
        cell_type_levels=dict(fine='cell_type')),
    'vhb3003-3b_1p': dict(
        label='Visium HD breast (8 um bins)',
        # Built by analysis/data_preparation/process_visium_hd_breast.py
        # (samples P26, P29, P32; 8-um bins with >= 50 counts; niche =
        # pathologist annotation of the bin).
        silver_folder='/nfs/team361/me13/DATASETS/silver',
        file_patterns=[f'adata_batch{k}.h5ad' for k in range(3)],
        niche_key='niche',
        drop_unlabeled=True,
        coord_scale=1.0,
        spot_size=8),
}
DATASET_LABELS = {name: cfg['label'] for name, cfg in DATASETS.items()}
# Datasets of the main figure (plot_results.py); the others get their own
# figures (plot_results.py --dataset).
MAIN_DATASETS = ['xhs1000-39b_1p', 'xhs49-1b_1p', 'chr78-11b_1p', 'chl59-8b_1p']

# ---------------------------------------------------------------------------
# Methods: order, labels and colours as in the paper's mouse figure
# (mouse_benchmarking/benchmarking_comparison.ipynb, PALETTE / METHOD_LABELS /
# method_emb_pairs). The paper assigns colours by position in its order, which
# includes GraphST (#78c7d2) between Banksy and Novae; colours are fixed per
# method here so dropping GraphST does not shift the others.
#
# A method's key is its result folder (artifacts/benchmarking/human/<key>/) and
# the ``method`` argument of save_embedding. New methods get a unique key and
# colour; the order here is the order in the figures. The run scripts of the
# methods below are on the branch human-niche-benchmark.
# ---------------------------------------------------------------------------
METHODS = {
    'terra': dict(label='TERRA', color='#73b7a2', note=''),
    'terra_hvg': dict(label='TERRA + HVG', color='#3d8c74',
                      note='2000 HVGs (seurat_v3) before tokenization'),
    'terra_112m': dict(label='TERRA-112M', color='#2b7a8c',
                       note='trained on the full corpus (no held-out split)'),
    'terra_random': dict(label='TERRA (random init)', color='#c9c9c9',
                         note='same architecture, untrained (capacity control)'),
    'terra_noneigh': dict(label='TERRA (no-neigh. pretraining)', color='#5e8f84',
                          note='TERRA-96M retrained with neighbour cells masked; '
                               'neighborhood_emb'),
    'terra_noneigh_cell_emb_nbavg': dict(label='TERRA (no-neigh. pretraining, cell emb., '
                                               'neigh. avg.)', color='#82a59d',
                                         note='no-neighbour model, cell_emb averaged over '
                                              '10 spatial neighbours'),
    'neigh_expr_pca': dict(label='Neigh. Expr. PCA', color='#f4a6a6', note=''),
    # Left out of the figures (results kept on disk): scVI latent averaged over 10
    # spatial neighbours ('scvi_nbavg'), TERRA-96M cell embedding with and without
    # neighbour averaging ('terra_cell_emb', 'terra_cell_emb_nbavg'), the
    # no-neighbour model's cell embedding ('terra_noneigh_cell_emb') and the
    # cell-type composition baseline clustered with Leiden ('cell_type_composition').
    'cell_type_composition_kmeans': dict(label='Neigh. Cell-Type Comp. (k-means)',
                                         color='#8fa35a',
                                         note='annotated cell types (finest level), '
                                              'k-means at the common cluster number '
                                              '(10 seeds); n/a for ISS'),
    'nichecompass': dict(label='NicheCompass', color='#d6b36a', note=''),
    'cellcharter': dict(label='CellCharter', color='#6fb6d8', note=''),
    'banksy': dict(label='Banksy', color='#b99ad6',
                   note='forced integration: + Harmony'),
    'novae': dict(label='Novae', color='#ffde17',
                  note='Leiden on Novae batch-corrected embedding'),
    # Earlier embeddings (method 'nicheformer', results kept on disk) replaced by
    # the rerun with the authors' recipe (run_nicheformer.py).
    'nicheformer_rerun': dict(label='NicheFormer', color='#95a3c3',
                              note='authors\' recipe: NaN technology mean -> 1, context tokens'),
    'cellplm': dict(label='CellPLM', color='#ea9776', note=''),
    'scgpt-spatial': dict(label='scGPT-spatial', color='#db96c0', note=''),
}

# (method, dataset) pairs left out (n/a) although results exist on disk.
NOT_APPLICABLE = {
    # Nicheformer has no ISS technology token, and its ISS technology mean covers
    # only 27 of the 169 panel genes.
    ('nicheformer_rerun', 'chr78-11b_1p'): 'ISS not supported by Nicheformer',
    # Visium HD: no technology token and no technology mean in Nicheformer.
    ('nicheformer_rerun', 'vhb3003-3b_1p'): 'Visium HD not supported by Nicheformer',
}


def method_output_dir(method: str) -> Path:
    path = Path(OUTPUT_FOLDER_PATH) / method
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_benchmark_dataset(dataset_name: str,
                           subsample: int | None = None,
                           seed: int = 0) -> ad.AnnData:
    """Load the benchmark sections of a dataset as one AnnData.

    Raw counts stay in ``X``; the ground-truth niche is copied to
    ``obs['niche_gt']`` and the section (``<dataset_id>_batch<k>``) to
    ``obs['section']``. ``obs_names`` are the unique harmonized cell IDs.
    """
    cfg = DATASETS[dataset_name]
    niche_key = cfg['niche_key']
    folder = Path(cfg.get('silver_folder', SILVER_FOLDER_PATH)) / dataset_name

    files = sorted(f for f in os.listdir(folder)
                   if f.endswith('.h5ad')
                   and any(p in f for p in cfg['file_patterns']))
    if len(files) != len(cfg['file_patterns']):
        raise ValueError(f'Expected {len(cfg["file_patterns"])} files for '
                         f'{dataset_name}, found {files}.')

    adata_list = []
    for file in files:
        adata = sc.read_h5ad(folder / file)
        if cfg['drop_unlabeled']:
            adata = adata[~adata.obs[niche_key].isna()].copy()
        adata.obs[SECTION_KEY] = adata.obs['cell_id'].apply(
            lambda x: '_'.join(x.split('_')[:2]))
        # Original silver obs_names, used to match embeddings that were saved
        # without cell_id (e.g. some scGPT-spatial outputs).
        adata.obs['orig_obs_name'] = adata.obs_names.astype(str)
        adata_list.append(adata)

    adata = ad.concat(adata_list, join='outer', index_unique=None)
    adata.obs_names = adata.obs['cell_id'].astype(str).values
    if not adata.obs_names.is_unique:
        raise ValueError('cell_id is not unique across sections.')
    adata.obsm['spatial'] = (np.asarray(adata.obsm['spatial'], dtype=float)
                             * cfg['coord_scale'])

    labels = adata.obs[niche_key].astype(str)
    for old, new in cfg.get('label_replace', {}).items():
        labels = labels.str.replace(old, new, regex=False)
    adata.obs['niche_gt'] = pd.Categorical(labels)
    adata.obs[SECTION_KEY] = pd.Categorical(adata.obs[SECTION_KEY])

    if subsample is not None and subsample < adata.n_obs:
        # Smoke tests only: keep whole spatial structure of a random
        # contiguous window would be nicer, but a random subset is enough to
        # exercise the code paths.
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(adata.n_obs, subsample, replace=False))
        adata = adata[idx].copy()

    adata.uns['benchmark'] = dict(dataset_name=dataset_name,
                                  niche_key=niche_key,
                                  n_target_clusters=n_target_clusters(adata),
                                  spot_size=cfg['spot_size'])
    return adata


def n_target_clusters(adata: ad.AnnData) -> int:
    """Common cluster number: ground-truth niches with > 100 cells."""
    return int((adata.obs['niche_gt'].value_counts()
                > MIN_CELLS_PER_NICHE).sum())


def avg_cosine_similarity(X: np.ndarray) -> float:
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    S = X.sum(axis=0)
    return float((S @ S - n) / (n * (n - 1)))


def save_embedding(adata: ad.AnnData, method: str, emb_key: str,
                   info: dict) -> None:
    """Write a method's embedding for scoring; the last step of every method script.

    Writes ``<method>/<dataset>_<method>_embedded.h5ad`` (slim AnnData) and
    ``<dataset>_<method>_run_info.json`` (``info`` plus the embedding key, cell
    and gene counts, the common cluster number and the embedding's average
    cosine similarity). job_wrapper.sh then clusters and scores it with
    ``score.py --pending`` in the reference environment.
    """
    out_dir = method_output_dir(method)
    dataset = adata.uns['benchmark']['dataset_name']
    write_run_info(out_dir, dataset, method, dict(
        emb_key=emb_key, n_cells=adata.n_obs, n_genes=adata.n_vars,
        n_sections=adata.obs[SECTION_KEY].nunique(),
        n_target_clusters=adata.uns['benchmark']['n_target_clusters'],
        avg_cosine_similarity=avg_cosine_similarity(
            np.asarray(adata.obsm[emb_key])),
        **info))
    save_results(adata, out_dir / f'{dataset}_{method}_embedded.h5ad')
    print(f'Wrote embedding {emb_key} to {out_dir}; scoring follows (score.py).')


def write_run_info(out_dir: Path, dataset_name: str, method: str,
                   info: dict) -> None:
    with open(out_dir / f'{dataset_name}_{method}_run_info.json', 'w') as f:
        json.dump(info, f, indent=2, default=str)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.seconds = time.perf_counter() - self.start


def _drop_none(d: dict) -> dict:
    return {k: (_drop_none(v) if isinstance(v, dict) else v)
            for k, v in d.items() if v is not None}


def save_results(adata: ad.AnnData, path: Path) -> None:
    """Write a slim AnnData (no expression; embeddings, clusters, graphs).

    ``None`` values in ``uns`` (e.g. ``uns['log1p']['base']``) are dropped so
    the file also opens with the older anndata in the terra environment.
    """
    adata_out = adata[:, []].copy()
    for key in list(adata_out.layers):
        del adata_out.layers[key]
    adata_out.uns = _drop_none(dict(adata_out.uns))
    adata_out.write_h5ad(path)


def git_commit(repo_path) -> str:
    """Short commit of a git checkout, with '+dirty' if it has local changes.

    Recorded in run_info so results can be traced to the exact code version
    (e.g. the terra / nichecompass checkouts the environments load from).
    """
    import subprocess
    git = ['git', '-c', f'safe.directory={repo_path}', '-C', str(repo_path)]
    try:
        commit = subprocess.run(git + ['rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(git + ['status', '--porcelain'],
                               capture_output=True, text=True,
                               check=True).stdout.strip()
        return commit + ('+dirty' if dirty else '')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'
