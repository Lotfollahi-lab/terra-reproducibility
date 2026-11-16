"""HEIST zero-shot cell embeddings for the human niche benchmark.

HEIST (Madhu et al., ICLR 2026, arXiv 2506.11152; code
https://github.com/Graph-and-Geometric-Learning/HEIST, weights
https://huggingface.co/HirenMadhu/HEIST) is a hierarchical graph transformer: a
spatial cell graph whose cells each carry a gene co-expression graph. This
script runs the authors' inference pipeline (``utils/inference.embed_adata``)
unchanged, with the pretrained ``nonmarker_all`` checkpoint (the model card's
default) and no fine-tuning:

- genes outside the model's 6,495-symbol vocabulary are dropped (upper-case
  symbol match). The checkpoint has no gene-identity embedding, so the panel
  need not match the pretraining panels beyond that;
- a section with more than 30,000 cells is split into spatial quadrant tiles of
  < 35,000 cells. Per tile: raw counts -> CP10k + log1p, top 200 HVGs, Leiden
  clusters (auto resolution, target 7), MAGIC, a Delaunay cell graph, and one
  mutual-information gene network per Leiden cluster (GPU);
- the model encodes METIS blocks of 384 cells. The embedding is
  [cell-graph embedding | mean-pooled gene embedding] (2 x 128).

``embed_adata`` preprocesses like the authors' v2 pretraining data
(data_prep/build_chunks.py), except that it keeps every cell, as the benchmark
needs. Pretraining dropped cells with fewer than 10 transcripts and cells with no
counts in their tile's 200 HVGs. Here such cells still enter their tile's Leiden
clusters, MAGIC and cell graph; run_info.json records how many there are
(``frac_cells_below_pretraining_min_counts``, ``tiles[].frac_zero_hvg_cells``).

Benchmark choices:
- one ``embed_adata`` call per section, so graphs never cross sections;
- no cell-type annotations (``cell_type=None``): the gene networks are built per
  Leiden cluster, as for the authors' Xenium pretraining data. The Leiden target
  is the authors' constant, not the benchmark's cluster number;
- python, numpy and torch are seeded before each section (MAGIC's PCA and the MI
  estimator draw random numbers);
- reproducible gene networks (``reproducible_gene_networks``). The authors' KSG
  MI estimator adds random noise to break ties, which changes which edges pass
  the threshold. It sizes its batches from the free GPU memory, so the noise a
  gene pair receives depends on the memory state and on all earlier batches.
  Two runs on the same data then build different networks, and the embeddings
  differ by up to 1.2. Here the batch size comes from a fixed memory budget
  (``KSG_MEMORY_BUDGET``; a GPU with >= 35 GB free) and torch is reseeded before
  each network. The estimator itself is unchanged;
- no Voronoi polygons (``no_voronoi_polygons``). The authors' preprocess only
  stores them as node attributes of the cell graph, whose edges come from a
  Delaunay triangulation of the same coordinates, and the model never reads
  them. geovoronoi fails on the regular bin grid of Visium HD ("generated
  polygon is not valid or is empty");
- a fresh graph cache per run, because ``preprocess`` silently reuses cached
  graphs of the same name. ``--cache_dir`` resumes an interrupted run of this
  script from its cache instead; every section must be cached completely or not
  at all, because a section's tiles share one random-number stream
  (``check_graph_cache``).

checks/test_heist.py checks the input path against the authors' code on a
benchmark section (identical graphs and embeddings), the gene networks against
the authors' unpatched estimator, and that seeded runs are reproducible.
obs['heist_tile'] and obs['heist_cluster'] keep HEIST's tiles and internal
Leiden clusters (no labels used), for interpreting the embedding.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the heist environment (envs/heist.sh) on a GPU:
    SCRIPT=run_heist.py PYTHON_ENV=PYTHON_HEIST DATASET=xhs49-1b_1p \\
        bsub -G s10396 -q inference -n 8 -M 64GB \\
        -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \\
        -gpu "num=1:mode=exclusive_process" -W 24:00 < run_method.lsf
(README: settings per dataset.)
"""
import argparse
import importlib.metadata
import json
import os
import random
import shutil
import sys
import tempfile
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')   # compute nodes have no internet
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # deterministic cuBLAS
# The HEIST checkout tracks a .pyc file: writing bytecode would make it dirty
# (code_revision).
sys.dont_write_bytecode = True

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

# Local copies (envs/heist.sh): the authors' code (GitHub) and the weights (HF).
HEIST_CODE_PATH = Path(os.environ.get(
    'HEIST_CODE_PATH', '/lustre/scratch126/cellgen/lotfollahi/jr43/models/HEIST_code'))
HEIST_MODEL_PATH = Path(os.environ.get(
    'HEIST_MODEL_PATH', '/lustre/scratch126/cellgen/lotfollahi/jr43/models/HEIST'))
CODE_REVISION = '3679a981d8edebc3d0cc25f20ea929e8a1d5b933'
MODEL_REVISION = 'b3cfb2a82f22fe051dd5b7cabdbf13a39bebf621'
CHECKPOINT = 'nonmarker_all'
sys.path.insert(0, str(HEIST_CODE_PATH))
import utils.inference as heist_inference  # noqa: E402
import utils.preprocess as heist_preprocess  # noqa: E402
from data_prep.build_chunks import MIN_COUNTS, MIN_KEPT_COUNTS  # noqa: E402
from utils.inference import BATCH_SIZE, V2_PREPROCESS, embed_adata, load_heist  # noqa: E402
from utils.preprocess import filter_genes_to_vocab, spatial_chunks  # noqa: E402

sys.path.append(str(Path(__file__).parent))
from bench_utils import (CACHE_FOLDER_PATH, SECTION_KEY, Timer,  # noqa: E402
                         frac_duplicate_rows, git_commit, hf_snapshot_revision,
                         load_benchmark_dataset, row_groups, save_embedding)

METHOD = 'heist'
EMB_KEY = 'X_heist'
SEED = 0
# GPU memory per KSG batch (bytes; the authors' estimate: 5 float32 N x N tensors per
# gene pair). The authors use 70% of the free memory, ~31 GB on an L40S. The batch
# size changes the noise each gene pair gets, so the networks depend on this
# constant like on a seed: keep it fixed across datasets.
KSG_MEMORY_BUDGET = 24e9
PACKAGES = ['torch', 'torch_geometric', 'numpy', 'scipy', 'scikit-learn', 'anndata',
            'scanpy', 'igraph', 'pynndescent', 'umap-learn', 'numba', 'magic-impute',
            'graphtools', 'scprep', 'pygsp', 'pymetis', 'geovoronoi', 'shapely',
            'networkx']


def snapshot_revision(model_path=HEIST_MODEL_PATH):
    """Commit and sha256 of the checkpoint's weights, from huggingface_hub's download metadata."""
    return hf_snapshot_revision(model_path, f'{CHECKPOINT}/model.safetensors',
                                MODEL_REVISION, 'envs/heist.sh')


def code_revision(code_path=HEIST_CODE_PATH):
    """The HEIST checkout must be the pinned commit without local changes."""
    commit = git_commit(code_path)
    if commit.endswith('+dirty') or not CODE_REVISION.startswith(commit):
        raise ValueError(f'HEIST code at {code_path} is {commit}, expected a clean '
                         f'checkout of {CODE_REVISION} (envs/heist.sh).')
    return CODE_REVISION


def check_gpu_memory():
    """Before anything is on the GPU: the fixed KSG budget must not exceed what the
    authors' rule (70% of the free memory) would use, or the gene networks run out of
    memory where the authors' code would shrink its batches."""
    free, _ = torch.cuda.mem_get_info()
    if 0.7 * free < KSG_MEMORY_BUDGET:
        raise RuntimeError(
            f'{torch.cuda.get_device_name()} has {free / 1e9:.0f} GB free; HEIST\'s gene '
            f'networks need {KSG_MEMORY_BUDGET / 0.7 / 1e9:.0f} GB (KSG_MEMORY_BUDGET). Use '
            'a larger GPU: changing the budget changes the networks.')


def load_model(device, model_path=HEIST_MODEL_PATH):
    """Pretrained HEIST encoder, its gene vocabulary and its config."""
    model, vocab = load_heist(str(model_path / CHECKPOINT), device=device)
    config = json.loads((model_path / CHECKPOINT / 'config.json').read_text())
    return model, vocab, config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # deterministic GPU kernels where torch has them (e.g. scatter sums in PyG)
    torch.use_deterministic_algorithms(True, warn_only=True)


def section_adata(adata, rows):
    """One section as the authors' input: raw counts, gene symbols, coordinates (um)."""
    return ad.AnnData(X=sp.csr_matrix(adata.X[rows], dtype=np.float32),
                      obs=pd.DataFrame(index=adata.obs_names[rows]),
                      var=pd.DataFrame(index=adata.var_names.astype(str)),
                      obsm={'spatial': np.asarray(adata.obsm['spatial'][rows],
                                                  dtype=np.float64)})


def ksg_batch(n_cells):
    """Gene pairs per KSG batch for a cluster of n_cells, from the fixed memory budget."""
    by_memory = int(KSG_MEMORY_BUDGET / (5 * 4 * n_cells ** 2))
    # the authors' cap: torch.topk fails on tensors with >= 2**31 elements
    return max(1, min(by_memory, (2**31 - 1) // n_cells ** 2))


@contextmanager
def patched(module, name, replacement):
    """Inside: module.<name> is replacement."""
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)


def reproducible_gene_networks():
    """Inside: the authors' build_gene_network_gpu with a fixed KSG batch size, and torch
    reseeded (with the seed last set by seed_everything) before every network."""
    original = heist_preprocess.build_gene_network_gpu

    def build(cell_data, **kwargs):
        kwargs.setdefault('ksg_batch', ksg_batch(cell_data.shape[0]))
        torch.manual_seed(torch.initial_seed())
        return original(cell_data, **kwargs)

    return patched(heist_preprocess, 'build_gene_network_gpu', build)


@contextmanager
def no_voronoi_polygons():
    """Inside: the authors' preprocess builds no Voronoi polygons. Its
    build_graph_from_cell_coords then stores None instead of each cell's polygon and
    takes the edges from a Delaunay triangulation, as always."""
    with warnings.catch_warnings(), \
            patched(heist_preprocess, 'calcualte_voronoi_from_coords', lambda *a, **k: []):
        warnings.filterwarnings('ignore', 'Number of cells does not match number of voronoi')
        yield


def on_each_tile(callback):
    """Inside: callback(obs_names, graph name, graphs) runs after embed_adata's preprocess
    has built (or loaded from the cache) the graphs of each tile."""
    original = heist_inference.preprocess

    def preprocess(adata, save_root, name, **kwargs):
        obs_names = adata.obs_names.copy()   # preprocess changes adata in place
        graphs = original(adata, save_root, name, **kwargs)
        callback(obs_names, name, graphs)
        return graphs

    return patched(heist_inference, 'preprocess', preprocess)


def hvg_expression(counts, hvgs):
    """A tile's log-normalized expression of its HVGs as preprocess computes it before
    MAGIC (genes in >= 3 cells, CP10k, log1p), from the tile's raw counts."""
    a = counts.copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    X = a[:, list(hvgs)].X
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def tile_diagnostics(graphs, counts):
    """Sizes, Leiden clusters, graph degrees and gene inputs of one tile, from its graphs
    and its cells' raw counts (vocabulary genes). Also per cell: Delaunay degree and
    HEIST's internal Leiden cluster."""
    cell, genes = graphs[0], graphs[1:]
    n = cell.num_nodes
    src, dst = cell.edge_index
    degree = torch.bincount(src[src != dst], minlength=n).numpy()   # without self loops
    cluster = cell.cell_type.numpy()
    first = {int(c): int(np.where(cluster == c)[0][0]) for c in np.unique(cluster)}
    hvg = hvg_expression(counts, cell.gene_names)
    hvg_counts = np.asarray(counts[:, list(cell.gene_names)].X.sum(axis=1)).ravel()
    inputs = row_groups(torch.stack([g.X.view(-1) for g in genes]).numpy())
    # Cells whose gene input a cell with other HVG expression shares (the authors' fixed
    # bug made every cell of a cluster share one). Cells with identical HVG expression
    # (e.g. no counts in the HVGs) can get identical inputs from MAGIC.
    pairs = np.unique(np.c_[inputs, row_groups(hvg)], axis=0)
    shared = (np.bincount(pairs[:, 0])[inputs] > 1)
    resolution = getattr(cell, 'leiden_resolution', None)   # PyG drops None attributes
    return dict(
        n_cells=int(n), n_genes=len(cell.gene_names),
        leiden_resolution=None if resolution is None else float(resolution),
        cluster_sizes=np.bincount(cluster).tolist(),
        # undirected edges of each cluster's gene network (shared by its cells)
        gene_network_edges=[int(genes[k].edge_index.shape[1]) // 2 for k in first.values()],
        median_degree=float(np.median(degree)), n_isolated=int((degree == 0).sum()),
        # pretraining dropped these cells (build_chunks.MIN_KEPT_COUNTS)
        frac_zero_hvg_cells=float((hvg_counts < MIN_KEPT_COUNTS).mean()),
        frac_duplicate_gene_inputs=float((np.bincount(inputs)[inputs] > 1).mean()),
        frac_gene_inputs_shared_across_expression=float(shared.mean()),
    ), degree, np.asarray(cell.cell_types)[cluster]


def embed_section(model, vocab, a, cache_dir, name, seed=SEED):
    """The authors' embed_adata on one section: seeded, with reproducible gene networks
    and no Voronoi polygons (graphs cached as <name>[_<tile>].pt). Returns the
    embedding and, per tile: (tile label, positions of its cells in a, tile_diagnostics,
    per-cell Delaunay degree, per-cell Leiden cluster)."""
    counts = filter_genes_to_vocab(a, vocab, verbose=False)[0]
    tiles = []

    def diagnose(obs_names, tile_name, graphs):
        idx = a.obs_names.get_indexer(obs_names)
        if (idx < 0).any():
            raise RuntimeError(f'{tile_name}: cells not in the section')
        tiles.append((tile_name[len(name) + 1:], idx, *tile_diagnostics(graphs, counts[idx])))

    seed_everything(seed)
    with reproducible_gene_networks(), no_voronoi_polygons(), on_each_tile(diagnose):
        emb = embed_adata(a, model, vocab, str(cache_dir), name)
    return emb, tiles


def tile_file(name, tile):
    """The authors' cache file of a tile of the section embedded as `name`."""
    return f'{name}_{tile}.pt' if tile else f'{name}.pt'


def check_graph_cache(cache_dir, sections):
    """The cached graphs a resumed run reuses, checked: every section ((name,
    coordinates) in `sections`) is cached completely or not at all, and every file is a
    complete torch.save archive. Returns the reused file names."""
    found = {p.name for p in Path(cache_dir).glob('*.pt')}
    reused, known = [], set()
    for name, xy in sections:
        expected = {tile_file(name, tile) for tile in set(spatial_chunks(np.asarray(xy)))}
        known |= expected
        cached = found & expected
        if cached and cached != expected:
            raise ValueError(
                f'{cache_dir}: {len(cached)} of the {len(expected)} tiles of {name} are '
                f'cached. Its tiles share one random-number stream: delete {name}[_*].pt '
                'so that the whole section is rebuilt.')
        reused += sorted(cached)
    if found - known:
        raise ValueError(f'{cache_dir} holds graphs of no tile of this dataset: '
                         f'{sorted(found - known)}')
    broken = []
    for f in reused:
        try:
            with zipfile.ZipFile(Path(cache_dir) / f) as z:
                if not any(n.endswith('/data.pkl') for n in z.namelist()):
                    broken.append(f)
        except zipfile.BadZipFile:
            broken.append(f)
    if broken:
        raise ValueError(f'{cache_dir}: incomplete graph files (written by a killed job?) '
                         f'{broken}; delete their sections\' files to rebuild them.')
    return reused


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    if not torch.cuda.is_available():
        raise RuntimeError('HEIST needs a CUDA GPU (its gene networks are built on the GPU).')
    device = 'cuda'
    check_gpu_memory()
    commit, weights_sha256 = snapshot_revision()
    code_commit = code_revision()
    xy_all = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    sections = [(f'section{i}', section, np.where((adata.obs[SECTION_KEY] == section).values)[0])
                for i, section in enumerate(adata.obs[SECTION_KEY].cat.categories)]
    sections = [s for s in sections if len(s[2])]
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        reused = check_graph_cache(cache_dir, [(name, xy_all[rows])
                                               for name, _, rows in sections])
        print(f'Graph cache: {cache_dir}, reusing {len(reused)} tile graphs')
    else:
        cache_root = Path(CACHE_FOLDER_PATH) / METHOD
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(tempfile.mkdtemp(prefix=f'{dataset}_', dir=cache_root))
        reused = []
        print(f'Graph cache: {cache_dir}')

    with Timer() as t_load:
        model, vocab, config = load_model(device)
    seconds = t_load.seconds
    genes = adata.var_names.astype(str)
    ignored = filter_genes_to_vocab(adata[:0], vocab, verbose=False)[1]
    in_vocab = ~genes.isin(ignored)
    print(f'{in_vocab.sum()} of {len(genes)} panel genes in the HEIST vocabulary; '
          f'ignored: {ignored}')
    n_zero = int((np.asarray(adata.X[:, in_vocab].sum(axis=1)).ravel() == 0).sum())
    n_counts = np.asarray(adata.X.sum(axis=1)).ravel()

    emb = np.full((adata.n_obs, 2 * config['output_dim']), np.nan, dtype=np.float32)
    tiles, n_dup_xy = [], 0.0
    degrees = np.zeros(adata.n_obs, dtype=np.int64)
    heist_tile = np.empty(adata.n_obs, dtype=object)
    heist_cluster = np.empty(adata.n_obs, dtype=object)
    for name, section, rows in sections:
        n_dup_xy += frac_duplicate_rows(xy_all[rows]) * len(rows)
        with Timer() as t:   # includes the tile diagnostics (seconds)
            emb[rows], section_tiles = embed_section(model, vocab, section_adata(adata, rows),
                                                     cache_dir, name)
        seconds += t.seconds
        for tile, idx, diag, degree, cluster in section_tiles:
            tiles.append(dict(section=section, tile=tile, **diag))
            label = f'{section}:{tile or "0"}'
            degrees[rows[idx]] = degree
            heist_tile[rows[idx]] = label
            heist_cluster[rows[idx]] = [f'{label}:{c}' for c in cluster]
        diags = [d for _, _, d, _, _ in section_tiles]
        print(f'{section}: {len(rows)} cells, {len(diags)} tiles, {t.seconds:.0f} s; '
              f'Leiden clusters per tile {[len(d["cluster_sizes"]) for d in diags]}, '
              f'median degree {np.median(degrees[rows]):.0f}', flush=True)
    adata.obsm[EMB_KEY] = emb
    # HEIST's own tiles and internal Leiden clusters (built without labels; for
    # interpreting the embedding, not scored)
    adata.obs['heist_tile'] = pd.Categorical(heist_tile)
    adata.obs['heist_cluster'] = pd.Categorical(heist_cluster)

    if not np.isfinite(emb).all():
        raise ValueError('Missing or non-finite HEIST embeddings.')
    shared = max(d['frac_gene_inputs_shared_across_expression'] for d in tiles)
    if shared:
        print(f'WARNING: up to {shared:.4f} of a tile\'s cells share their gene input with '
              'cells of other HVG expression (the authors\' fixed bug?)', flush=True)
    if args.keep_cache or args.cache_dir:
        print(f'Graph cache kept at {cache_dir}')
    else:
        shutil.rmtree(cache_dir)
    save_embedding(adata, METHOD, EMB_KEY, dict(
        embedding_seconds=seconds,
        model=dict(repo='HirenMadhu/HEIST', checkpoint=CHECKPOINT, revision=commit,
                   weights_sha256=weights_sha256, path=str(HEIST_MODEL_PATH / CHECKPOINT),
                   n_parameters=sum(p.numel() for p in model.parameters()), **config),
        code=dict(repo='Graph-and-Geometric-Learning/HEIST', commit=code_commit,
                  path=str(HEIST_CODE_PATH)),
        n_genes_panel=int(adata.n_vars), n_genes_in_vocab=int(in_vocab.sum()),
        ignored_genes=ignored, n_zero_count_cells=n_zero,
        # cells pretraining would have dropped (build_chunks.py); all are embedded here
        frac_cells_below_pretraining_min_counts=float((n_counts < MIN_COUNTS).mean()),
        max_frac_zero_hvg_cells=max(d['frac_zero_hvg_cells'] for d in tiles),
        n_tiles=len(tiles),
        # seconds above exclude the preprocessing of reused tiles
        graph_cache=dict(path=str(cache_dir), n_reused_tiles=len(reused),
                         reused_tiles=reused),
        max_frac_duplicate_gene_inputs=max(d['frac_duplicate_gene_inputs'] for d in tiles),
        max_frac_gene_inputs_shared_across_expression=shared,
        frac_isolated_cells=float((degrees == 0).mean()),
        median_degree=float(np.median(degrees)),
        frac_duplicate_coordinates=n_dup_xy / adata.n_obs,   # within a section
        frac_duplicate_embeddings=frac_duplicate_rows(emb),
        device=torch.cuda.get_device_name(),
        peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
        package_versions={p: importlib.metadata.version(p) for p in PACKAGES},
        params=dict(**V2_PREPROCESS, batch_size=BATCH_SIZE, seed=SEED,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    gene_networks='authors\' KSG MI; batch size from a fixed memory '
                                  'budget, torch reseeded per network (reproducible)',
                    ksg_memory_budget_gb=KSG_MEMORY_BUDGET / 1e9,
                    gene_matching='upper-case symbol in the model vocabulary; others dropped',
                    cell_filter='none (pretraining dropped cells with < 10 counts or no '
                                'counts in the HVGs)',
                    cell_type='none: gene networks per Leiden cluster (authors\' auto resolution)',
                    spatial_context='Delaunay cell graph per tile (< 35k cells) within a '
                                    'section; attention within METIS blocks of 384 cells',
                    voronoi_polygons='not built (unused by the model)',
                    embedding='[cell-graph | mean-pooled gene] embedding',
                    integration='none'),
        tiles=tiles))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    parser.add_argument('--keep_cache', action='store_true',
                        help='Keep the cached graphs (debugging).')
    parser.add_argument('--cache_dir', default=None,
                        help='Resume: graph cache of an interrupted run of this script on '
                             'the same dataset. Each section must be cached completely or '
                             'not at all (delete the files of an unfinished section). Kept '
                             'afterwards.')
    main(parser.parse_args())
