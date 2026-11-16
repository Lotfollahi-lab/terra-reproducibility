# Human Niche-Identification Benchmark

Zero-shot niche identification on human spatial datasets, with the protocol of
the TERRA paper's mouse-brain benchmark (`../mouse_benchmarking/`). Every method
only produces an embedding per cell; clustering, scoring and figures are shared,
so all methods are compared on identical cells with an identical protocol.

## Protocol

- **Cells**: `bench_utils.load_benchmark_dataset` gives every method the same
  cells (sections, label keys, NaN filtering, coordinate scaling).
- **Spatial graphs** are built per tissue section (`obs['section']`); no
  cross-section neighbours.
- **Target number of clusters**: ground-truth niches with > 100 cells.
- **Clustering** (`scoring.py`, every method): 15-NN graph on the embedding,
  Leiden (igraph, 2 iterations, seed 0). One upward scan on a 0.01 resolution
  grid, every run scored, until the cluster count has been above the target for
  5 consecutive steps. From that table:
  - **headline** = lowest grid resolution giving exactly the target;
  - **matched** = all runs giving exactly the target;
  - **sweep** = ±4 grid steps around the median matching resolution (the bars in
    the main figure).

  If no grid point hits the target, the first crossing is refined on a 0.001
  grid; if still unreachable, the best reachable number is used and flagged.
  Methods whose own clustering is k-means can use k-means at the target
  instead (10 seeds).
- **Metrics**: NMI and ARI vs. the ground truth; for multi-section datasets also
  iLISI, batch ASW and MMD (mean over section pairs).
- All clustering and scoring runs in one reference environment (banksy), so
  every method uses the same igraph / leidenalg / scanpy versions;
  `scoring.py` refuses to run anywhere else.

| Dataset | Technology | Sections | Ground truth | Target |
|---|---|---|---|---|
| `xhs1000-39b_1p` | Xenium skin (eczema) | 3 | `niche_type` | 11 |
| `xhs49-1b_1p` | Xenium skin (melanoma) | 1 | `niche_name` | 7 |
| `chr78-11b_1p` | ISS reproductive tract | 11 | `broad_anatomy` | 19 |
| `chl59-8b_1p` | CosMx lung (NSCLC) | 6 | `niche` | 12 |
| `vhb3003-3b_1p` | Visium HD breast, 8 µm bins (separate figure) | 3 | pathology annotation | 8 |

## Files

| File | Role |
|---|---|
| `bench_utils.py` | Paths, protocol constants, `DATASETS`, method registry `METHODS`, `NOT_APPLICABLE`, `load_benchmark_dataset`, `save_embedding` |
| `run_neigh_expr_pca.py` | Example method (neighbourhood-expression PCA); template for new methods |
| `scoring.py`, `score.py` | The only clustering and scoring code (reference environment) |
| `job_wrapper.sh`, `config.sh` | Job steps (method → scoring → verified copy to the result folder); environments and job paths |
| `run_method.lsf`, `score.lsf` | LSF jobs: one method on one dataset; re-scoring a finished result |
| `collect_results.py`, `plot_results.py` | Result tables; figures (main figure, one dataset, spatial maps) |
| `checks/test_resolution_scan.py` | Check of the resolution scan against brute force |
| `run_cifm.py`, `checks/test_cifm.py` | CIFM zero-shot embeddings; check against the authors' code |
| `run_heist.py`, `checks/test_heist.py` | HEIST zero-shot embeddings; check against the authors' code |

Results: `<repo>/artifacts/benchmarking/human/<method>/`, one folder per method:

| File | Content |
|---|---|
| `<dataset>_<method>_metric_scores.csv` | every Leiden run: resolution, number of clusters, NMI, ARI, flags `headline`, `matched`, `in_sweep` |
| `<dataset>_<method>_embedded_clustered.h5ad` | embedding, headline clustering (`obs['predicted_niche']`), ground truth, coordinates |
| `<dataset>_<method>_run_info.json` | parameters, timings, environments |
| `<dataset>_<method>_batch_integration_metrics.csv` | iLISI, ASW, MMD (multi-section datasets) |

## Adding a method

1. **Script** `run_<method>.py` (copy `run_neigh_expr_pca.py`):
   - load the cells with `load_benchmark_dataset(dataset, subsample=...)`:
     raw counts in `X`, `obs['section']`, `obsm['spatial']` (µm),
     `obs_names` = unique `cell_id`;
   - compute one embedding for every cell, in the same order, into
     `adata.obsm[<key>]`;
   - finish with `save_embedding(adata, '<method>', '<key>', info)`; `info`
     (parameters, `embedding_seconds`, ...) goes into `run_info.json`.

   Rules:
   - no clustering or scoring in the script; `score.py` runs afterwards in the
     same job;
   - never use the ground truth (`obs['niche_gt']`, `obs[niche_key]`); the only
     allowed input from it is `uns['benchmark']['n_target_clusters']`, and only
     for a method's own clusters (e.g. a native domain assignment, saved
     separately);
   - spatial context only within a section (`SECTION_KEY`); `N_NEIGHS` (10)
     spatial neighbours if the method has no default of its own; if the method
     corrects for batches, use the section as the batch and record it in `info`;
   - keep the `--dataset` and `--subsample` arguments (smoke tests);
   - for k-means clustering at the target instead of the Leiden scan, add
     `clustering='kmeans'` to `info`.

   Embeddings computed elsewhere (another cluster, a notebook): load them in the
   script, match them to the benchmark cells by `cell_id` (`obs_names`), and
   call `save_embedding` as above.
2. **Register** the method in `bench_utils.METHODS`: unique key (= the
   `method` argument of `save_embedding` and the result folder name), figure
   label, unique colour, note. Add `(method, dataset)` to `NOT_APPLICABLE` for
   datasets the method cannot handle (e.g. an unsupported technology), with the
   reason.
3. **Environment**: add a line `PYTHON_<NAME>=/path/to/env/bin/python` to
   `config.sh`. The method environment never needs the scoring packages.
4. **Smoke test**, then the full runs (below). Add a row to the method table in
   the README of the branch you work on.

Only add lines to the shared files (`bench_utils.py`, `config.sh`, README);
do not change existing entries, the protocol constants or `scoring.py` (a
change there invalidates every existing result).

## Running

From this folder; create the log folder once:
`mkdir -p ../../../artifacts/benchmarking/human/logs`.

```bash
cd analysis/benchmarking/human_niche_benchmark
# Smoke test (3000 random cells; result folder <method>/<dataset>_sub3000_*, delete afterwards)
SCRIPT=run_neigh_expr_pca.py PYTHON_ENV=PYTHON_BANKSY DATASET=xhs49-1b_1p EXTRA_ARGS="--subsample 3000" \
    bsub -q normal -n 8 -M 16GB -R "select[mem>16GB] rusage[mem=16GB] span[hosts=1]" -W 1:00 < run_method.lsf
# One method on one dataset (repeat per dataset)
SCRIPT=run_neigh_expr_pca.py PYTHON_ENV=PYTHON_BANKSY DATASET=xhs1000-39b_1p \
    bsub -q normal -n 8 -M 64GB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" -W 12:00 < run_method.lsf
# GPU method: add the GPU queue and request, e.g.
#   -q gpu -gpu "num=1:mode=exclusive_process"          (farm)
#   -q inference -G <group> -gpu "num=1"                 (tiger)
# Re-score a finished result (e.g. after a protocol change), no re-embedding
METHOD=neigh_expr_pca DATASET=xhs1000-39b_1p \
    bsub -q normal -n 8 -M 48GB -R "select[mem>48GB] rusage[mem=48GB] span[hosts=1]" -W 12:00 < score.lsf
```

What a job does (`job_wrapper.sh`): the method script runs in a job-local
lustre folder; `score.py --pending` clusters and scores every embedding it
wrote, in the reference environment (cores = resolution-scan workers); the
results are copied to the result folder and the copy is verified by checksum,
then the lustre folder is removed. On any failure nothing is copied and the
lustre folder is kept for debugging (path in the log).

Memory: the largest datasets (CosMx lung, 534k cells; Visium HD breast, 1.5M
bins) need 64–128 GB for CPU methods and up to 256 GB host memory for GPU
methods; scoring itself fits in 48–96 GB.

## Results and figures

```bash
python collect_results.py   # tables only: summary/headline.csv, scan.csv, sweep.csv, matched.csv, batch.csv
python plot_results.py      # collects, then main figure: summary/figures/niche_{NMI,ARI}_comparison_human[_matched].*
python plot_results.py --dataset vhb3003-3b_1p          # bars for one dataset: summary/figures/<dataset>/
python plot_results.py --dataset vhb3003-3b_1p --maps   # + spatial maps (high memory, run via LSF)
```

Figures include every method in `METHODS` with results, in `METHODS` order.
They run in the terra or banksy environment. The bars take seconds; the
spatial maps read every method's clustered result (up to ~200 GB for Visium HD),
e.g.:
```bash
bsub -G team361 -q normal -n 4 -M 200GB -R "select[mem>200GB] rusage[mem=200GB] span[hosts=1]" -W 4:00 \
    -o ../../../artifacts/benchmarking/human/logs/maps_%J.out \
    "$PYTHON_BANKSY plot_results.py --dataset vhb3003-3b_1p --maps"   # after: source config.sh
```

## Environments

The paths in `config.sh` point to existing environments (read-only). The
reference environment's package versions are pinned in
`scoring.REFERENCE_VERSIONS`; every result records the versions of its
scoring environment in `run_info.json`.

`BENCH_FINAL_OUTPUT_DIR` (optional, read by `config.sh` and `bench_utils.py`)
sends results to another folder than `<repo>/artifacts/benchmarking/human`,
e.g. to keep a method's results apart from the shared ones. Set it for the
jobs and for `collect_results.py` / `plot_results.py`, and pass `-o`/`-e` to
`bsub` so the logs go there too. It must be an absolute path. Set it per command
(`BENCH_FINAL_OUTPUT_DIR=... bsub ...`) rather than `export`ing it, so that later
commands in the same shell still use the shared folder.

## Checks

```bash
python checks/test_resolution_scan.py   # banksy env: scan vs. brute force on toy data
python checks/test_cifm.py              # cifm env, GPU: run_cifm.py vs. the authors' CIFM code
python checks/test_heist.py             # heist env, GPU: run_heist.py vs. the authors' HEIST code
```

## Methods on this branch

| Method key | Script | `PYTHON_ENV` | Notes |
|---|---|---|---|
| `neigh_expr_pca` | `run_neigh_expr_pca.py` | `PYTHON_BANKSY` | summed 10-NN counts, CP10k + log1p, PCA 20 |
| `cifm` | `run_cifm.py` | `PYTHON_CIFM` (GPU; peak 15 GB) | `ynyou/CIFM` rev. `19c507f`, zero-shot; 20 µm radius graph per section, tiled with a 45 µm halo (exact) |
| `heist` | `run_heist.py` | `PYTHON_HEIST` (GPU) | `HirenMadhu/HEIST` `nonmarker_all` rev. `b3cfb2a`, code `3679a98`, zero-shot; authors' `embed_adata` per section |

### CIFM

- Environment and model snapshot: `envs/cifm.sh` (login node; compute nodes
  have no internet). The model loads from the local snapshot
  (`CIFM_MODEL_PATH`); `run_info.json` records its revision and weights sha256.
- Input as in the authors' tutorial and preprint: raw counts without
  mitochondrial genes, CP10k + log1p, genes matched to CIFM's 18,289 input
  channels by Ensembl ID (`var['ensembl_id']` of the silver files; at least
  97.7% of every panel). `checks/test_cifm.py` shows that the tiled embedding with
  this gene matching equals the authors' `channel_matching` + `embed()`.
- Caveats: CIFM was pretrained on ~100 human slides from the 10x Genomics
  public database (Visium, Visium HD, Xenium V1 and Prime). `xhs49-1b_1p` is
  the 10x Xenium Prime Human Skin FFPE dataset, so it is probably in CIFM's
  pretraining data (no labels were used). CosMx (`chl59`) and ISS (`chr78`)
  were never seen in pretraining. ISS has a sparse 20 µm graph: 6% of its
  cells have no neighbour within 20 µm (`frac_isolated_cells` in
  `run_info.json`; 0.1–3.5% on the other datasets).

Runs (on L40S GPUs): embedding takes 7 s to 3.5 min per dataset and peaks at
15 GB of GPU memory; Leiden scoring takes most of the job time (up to 1.5 h on
Visium HD). Peak host memory was 3–26 GB, and 68 GB on `vhb3003-3b_1p`.

```bash
source config.sh
OUT=/nfs/team361/jr43/terra/benchmarking/human; LOGS=$OUT/logs; mkdir -p $LOGS
BENCH_FINAL_OUTPUT_DIR=$OUT SCRIPT=run_cifm.py PYTHON_ENV=PYTHON_CIFM DATASET=xhs49-1b_1p \
    bsub -G s10396 -q inference -n 8 -M 64GB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 12:00 \
    -o $LOGS/method_%J.out -e $LOGS/method_%J.err < run_method.lsf
# vhb3003-3b_1p: -M 128GB and EXTRA_ARGS="--max_tile_cells 15000"
```

### HEIST

- Environment, code and model snapshot: `envs/heist.sh` (login node). The
  authors' code is a clone of `Graph-and-Geometric-Learning/HEIST` at
  `3679a98` (`HEIST_CODE_PATH`), and the weights are the `nonmarker_all`
  checkpoint of `HirenMadhu/HEIST` at `b3cfb2a` (`HEIST_MODEL_PATH`).
  `run_info.json` records both revisions and the weights' sha256.
- Checkpoint: `nonmarker_all` is the model card's default. It has no gene-identity
  embedding, so it works for any panel. The paper's checkpoint
  (`nonmarker_all_iclr`) is not used: it was pretrained with a preprocessing bug
  in which all cells of a cell type shared one expression vector.
- Recipe: the authors' `utils/inference.embed_adata`, unchanged, called once
  per section:
  - genes outside the 6,495-symbol vocabulary are dropped;
  - sections with more than 30k cells are split into quadrant tiles of < 35k cells;
  - per tile: CP10k + log1p, 200 HVGs, Leiden (auto resolution, target 7),
    MAGIC, a Delaunay cell graph, and GPU mutual-information gene networks per
    Leiden cluster;
  - METIS blocks of 384 cells;
  - the embedding is [cell-graph | mean-pooled gene] (256-d).
- Benchmark choices:
  - no cell-type annotations: the gene networks use the authors' internal Leiden
    clusters, as for their Xenium pretraining data;
  - seeds and torch's deterministic algorithms set before each section;
  - every cell is embedded. `embed_adata` does not apply the cell filters of the
    authors' pretraining data (see Caveats);
  - a fresh graph cache per run, because the authors' `preprocess` silently
    reuses cached graphs of the same name. `--cache_dir` resumes an interrupted
    run from its cache instead. A section's tiles share one random-number stream,
    so `check_graph_cache` stops the run at the start unless every section is
    cached completely or not at all, and every cached file is complete.
- Reproducibility (`reproducible_gene_networks` in `run_heist.py`): the authors'
  KSG mutual-information estimator adds random noise to break ties, and the
  noise changes which gene-network edges pass the threshold. Its batches are
  sized from the free GPU memory, so the noise each gene pair gets depends on the
  memory state. Two runs on the same section built different networks, and the
  embeddings differed by up to 1.2. Here the batch size comes from a fixed 24 GB
  budget (`KSG_MEMORY_BUDGET`) and torch is reseeded before each network. The
  estimator is unchanged. The networks depend on the budget as on a seed, so it is
  the same for every dataset; the run stops at the start on a GPU with less than
  35 GB free, where the authors' code would shrink its batches instead.
- No Voronoi polygons (`no_voronoi_polygons` in `run_heist.py`). The authors'
  `preprocess` only stores them as node attributes of the cell graph, whose edges
  come from a Delaunay triangulation, and the model never reads them. geovoronoi
  fails on the regular 8 µm bin grid of Visium HD ("generated polygon is not valid
  or is empty"), so `run_heist.py` builds them on no dataset. The results so far
  (xhs1000, xhs49, chr78, chl59, and the vhb3003 run of job 230959) come from an
  earlier version that built them wherever geovoronoi succeeded; their model
  input is the same.
- `checks/test_heist.py`:
  - the benchmark's input path equals the authors' `embed_adata` on a silver
    section: identical graphs, and embeddings identical to the bit;
  - a different seed moves the embedding by mean cosine 0.9965 (min 0.88);
  - added on 2026-11-16, not yet run on a GPU: in (b), the authors' path builds
    the polygons and run_heist's does not; (c) only cells with identical HVG
    expression share a gene input; (d) the gene networks differ from the authors'
    unpatched estimator no more than two of its runs differ from each other. A
    CPU run on 707 cells passed (c), and its embedding without polygons was
    identical to the authors' path with them.
- `obs['heist_tile']` and `obs['heist_cluster']` in the result h5ad hold HEIST's
  tiles and internal Leiden clusters, which use no labels. They help interpret
  the embedding; they are not scored.
- Caveats:
  - `xhs49-1b_1p` (`Xenium_Prime_Human_Skin_FFPE`) is in HEIST's pretraining
    set (`data_prep/xenium_samples.txt` in the HEIST repo). CosMx, ISS and
    Visium HD were not.
  - Vocabulary coverage: Xenium 100%, ISS 126/173, CosMx 802/946, Visium HD
    6,228/18,085 (only 200 HVGs per tile are used).
  - Cell filters: the authors' pretraining data (`data_prep/build_chunks.py`)
    dropped cells with fewer than 10 transcripts and cells with no counts in their
    tile's 200 HVGs. `embed_adata` keeps them, and they still enter their tile's
    Leiden clusters, MAGIC and cell graph. No benchmark cell has fewer than 10
    transcripts. 3.5% of the Xenium cells (`xhs1000`, `xhs49`; up to 7.8% of a
    tile) have no counts in their tile's HVGs; ISS and CosMx have none. Tile by
    tile, their share matches the share of cells that share their gene input with
    another cell (up to 8% of a Xenium tile, `max_frac_duplicate_gene_inputs`).
    `run_info.json` records both shares for new runs
    (`frac_cells_below_pretraining_min_counts`, `tiles[].frac_zero_hvg_cells`).
  - Visium HD bins are not cells.
  - Unlike CIFM's halo tiling, the embedding depends on the tiles and METIS
    blocks: there is no attention across a block. Each tile's standardized
    coordinates are the model's cell-level input.
  - Results: on every dataset, the embedding's 15-NN graph is one giant
    component plus small islands. The target cluster count is never reached
    exactly (`below_grid`), and HEIST scores lowest of all methods (headline NMI
    0.006–0.108). Even the best run anywhere in the scan reaches only NMI
    0.24–0.27. HEIST's own internal Leiden clusters (`obs['heist_cluster']`)
    agree better with the ground truth (NMI 0.23–0.49) than its embedding does.

Runs (on L40S GPUs): embedding takes 25 min (`xhs1000`), 34 min (`chr78`),
60 min (`xhs49`) and 2.7 h (`chl59`), and ~9 h per section of `vhb3003`. Most
of that is the per-cluster mutual-information networks. Peak GPU memory is 25.5
GB (the fixed KSG budget plus the model). Peak host memory is 6–11 GB, and 32 GB
on `vhb3003`.

```bash
source config.sh
OUT=/nfs/team361/jr43/terra/benchmarking/human; LOGS=$OUT/logs; mkdir -p $LOGS
BENCH_FINAL_OUTPUT_DIR=$OUT SCRIPT=run_heist.py PYTHON_ENV=PYTHON_HEIST DATASET=xhs49-1b_1p \
    bsub -G s10396 -q inference -n 8 -M 64GB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 24:00 \
    -o $LOGS/method_%J.out -e $LOGS/method_%J.err < run_method.lsf
# chl59-8b_1p: -M 96GB; vhb3003-3b_1p: -M 128GB -W 48:00 (~28 h embedding + 1.5 h scoring)
# Needs >= 35 GB of free GPU memory (L40S, A100); smaller GPUs stop at the start.
# Resume an interrupted run: delete its unfinished section's section<i>[_*].pt files, then
#   EXTRA_ARGS="--cache_dir <its graph cache>"   (checked at the start; kept afterwards)
```
