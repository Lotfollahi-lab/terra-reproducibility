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
| `run_stofm.py`, `checks/test_stofm.py` | SToFM zero-shot embeddings; check against the authors' code and demo |
| `run_aido_tissue.py`, `checks/test_aido_tissue.py` | AIDO.Tissue zero-shot embeddings; check against the authors' code and demo |

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

`embedding_minutes` in `headline.csv` is the method's own `embedding_seconds` and is
not comparable across the zero-shot methods: AIDO.Tissue's includes its unscored
cell-only pass, and its `chr78`, `chl59` and `xhs49` runs predate its speed fix
(~4× slower); HEIST's `vhb3003` time covers only the section recomputed after the
resume (the full embedding took ~28 h).
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
python checks/test_stofm.py             # stofm env, GPU: run_stofm.py vs. the authors' SToFM code and demo
python checks/test_aido_tissue.py       # aido_tissue env, GPU: run_aido_tissue.py vs. the authors' mgen predict
```

## Methods on this branch

| Method key | Script | `PYTHON_ENV` | Notes |
|---|---|---|---|
| `neigh_expr_pca` | `run_neigh_expr_pca.py` | `PYTHON_BANKSY` | summed 10-NN counts, CP10k + log1p, PCA 20 |
| `cifm` | `run_cifm.py` | `PYTHON_CIFM` (GPU; peak 15 GB) | `ynyou/CIFM` rev. `19c507f`, zero-shot; 20 µm radius graph per section, tiled with a 45 µm halo (exact) |
| `heist` | `run_heist.py` | `PYTHON_HEIST` (GPU) | `HirenMadhu/HEIST` `nonmarker_all` rev. `b3cfb2a`, code `3679a98`, zero-shot; authors' `embed_adata` per section |
| `stofm` | `run_stofm.py` | `PYTHON_STOFM` (GPU) | `PharMolix/SToFM` code `2354d57`, checkpoint pinned by sha256, zero-shot; authors' `load_data` per section |
| `aido_tissue` | `run_aido_tissue.py` | `PYTHON_AIDO_TISSUE` (GPU, flash-attn) | `genbio-ai/GB.Tissue-60M` rev. `e414516`, ModelGenerator `bda6f2d`, zero-shot; cell + 10 nearest cells per section, max-pool |

### CIFM

- Environment and model snapshot: `envs/cifm.sh` (login node; compute nodes
  have no internet). The model loads from the local snapshot
  (`CIFM_MODEL_PATH`); `run_info.json` records its revision and weights sha256.
- Input as in the authors' tutorial and preprint: raw counts without
  mitochondrial genes, CP10k + log1p, genes matched to CIFM's 18,289 input
  channels by Ensembl ID (`var['ensembl_id']` of the silver files; at least
  97.7% of every panel). `checks/test_cifm.py` shows that the tiled embedding with
  this gene matching equals the authors' `channel_matching` + `embed()`. A few
  genes that CIFM has under an older Ensembl ID are lost because the silver ID is
  newer: `CD99`, `CSF2RA`, `IL3RA`, `IL9R` and `SHOX` (pseudo-autosomal; `xhs1000`,
  `xhs49`) and `CSF2RA`, `IL3RA` and `SOD2` (`chl59`); each holds at most 0.13% of a
  dataset's counts. They are still counted in the CP10k total.
- Caveats: CIFM was pretrained on ~100 human slides from the 10x Genomics
  public database (Visium, Visium HD, Xenium V1 and Prime). `xhs49-1b_1p` is
  the 10x Xenium Prime Human Skin FFPE dataset, so it is probably in CIFM's
  pretraining data (no labels were used). CosMx (`chl59`) and ISS (`chr78`)
  were never seen in pretraining. ISS has a sparse 20 µm graph: 6% of its
  cells have no neighbour within 20 µm (`frac_isolated_cells` in
  `run_info.json`; 0.06–3.5% on the other datasets).

Runs (on L40S GPUs): embedding takes 7 s to 3.5 min per dataset and peaks at
15 GB of GPU memory; Leiden scoring takes most of the job time (up to 1.5 h on
Visium HD). Peak host memory of the whole job (embedding and scoring) was 3–26 GB,
and 68 GB on `vhb3003-3b_1p`.

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
  - in (b) the authors' path builds the polygons and run_heist's does not; the
    graphs and embeddings are still identical to the bit;
  - (c) only cells with identical HVG expression share a gene input (1.9% of the
    section's cells share one at all; 1.9% have no counts in the HVGs);
  - (d) the gene networks differ from the authors' unpatched estimator no more than
    two of its runs differ from each other (mean fraction of differing edges 0.024
    vs. 0.025; max 0.108 vs. 0.111; 8 clusters);
  - all passed on an L40S (job 231063). A 3,000-cell `xhs1000` run resumed from
    its graph cache gave the same embedding to the bit.
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
  - Results: headline NMI 0.004–0.108, the lowest of the four zero-shot models on
    every dataset except `chr78`, where AIDO.Tissue (0.070) is below HEIST's 0.088.
    Of the other baselines only the TERRA SSL ablations (`terra_ssl`,
    `terra_ssl_context`) score lower, on `xhs49` and `chr78`. The embedding's 15-NN graph is so
    coarse that even resolution 0.01 gives more clusters than the target, so every
    headline comes from the `below_grid` resolutions (1e-3 to 1e-6):
    - `xhs1000`: the target (11) is reached exactly at 1e-3. The graph's second
      component holds 18% of the cells.
    - `xhs49` and `vhb3003`: the target is unreachable. The graph has 15 and 38
      connected components, against targets of 7 and 8.
    - `chr78` and `chl59`: the target lies between two of the decade steps (13 and
      20 clusters for 19 on `chr78`; 8 and 22 for 12 on `chl59`), which the scan
      does not refine. Re-running Leiden on the stored graph reaches 19 clusters on
      `chr78` at 4e-4 (NMI 0.043, against the headline 0.088 at 20 clusters) and 13
      clusters on `chl59` at 3e-4 (NMI 0.018, against 0.006 at 8). The ranking does
      not change.
    - The resolution sweep (the bars of the main figure) is placed at the bottom of
      the grid (0.01–0.05) when the headline is below it. HEIST's sweep therefore has
      23–180 clusters and a mean NMI of 0.06–0.21, well above its headline.
  - What the embedding encodes: HEIST's internal Leiden clusters
    (`obs['heist_cluster']`) explain 67–80% of the embedding's variance (about 9%
    with the labels permuted within each tile), and only 2–8% of 15-NN edges cross a
    tile boundary. The embedding mostly says which tile-specific cluster, and so
    which gene network, a cell belongs to; clusters are not aligned across tiles,
    because each tile picks its own 200 HVGs and `nonmarker_all` has no gene
    identity. The internal clusters reach NMI 0.23–0.49 against the ground truth,
    but at 23–174 clusters; the tile labels alone reach 0.38 (`xhs49`), 0.61
    (`chr78`) and 0.37 (`chl59`), so this is no evidence of better niche signal.
    The best NMI in the upward scan (0.24–0.27) is its last step, cut off by the
    patience rule while NMI was still rising with the cluster count.
  - On `vhb3003`, up to 38% of a tile's bins share their gene input with another
    bin (`max_frac_duplicate_gene_inputs`), against up to 8% on Xenium. In a sampled
    tile (`3003_batch1`, tile 41; 24,359 bins) 28.6% of the bins have no counts in
    the tile's HVGs, and 84 of its 96 Leiden clusters consist only of such bins. The
    authors' Leiden, which aims at 7 clusters, gave 5–96 per tile (median 13, 2,296
    over 93 tiles), each with its own gene network; this is likely why a section
    takes ~9 h.
  - "No benchmark cell has fewer than 10 transcripts" counts all panel genes. In
    HEIST's vocabulary 0.43% of `chr78` cells have fewer than 10
    (`frac_cells_below_pretraining_min_counts` counts all panel genes too).

Runs (on L40S GPUs): embedding takes 25 min (`xhs1000`), 34 min (`chr78`),
60 min (`xhs49`) and 2.7 h (`chl59`), and ~9 h per section of `vhb3003`. Most
of that is the per-cluster mutual-information networks. `vhb3003` failed in its
third section (geovoronoi) and was resumed from the cache of the first two (job
230959: 5.6 h for that section, 34 min scoring). Peak GPU memory is 25.5 GB (the
fixed KSG budget plus the model). Peak host memory is 6–11 GB, and 32 GB on
`vhb3003`.

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

### SToFM

- Environment, code and checkpoint: `envs/stofm.sh` (login node).
  - The authors' code is a clone of `PharMolix/SToFM` at `2354d57` (`STOFM_CODE_PATH`).
    Its bundled Geneformer 0.0.1 is installed from that copy.
  - The checkpoint and the authors' demo come from the Google Drive folder linked in
    their README (`STOFM_MODEL_PATH`). Drive keeps no versions, so `run_stofm.py` pins
    the sha256 of every checkpoint file and stops on any other.
  - `run_info.json` records the code commit and the sha256s.
- Model: a Geneformer-style cell encoder (BERT, 12 layers, width 512; CLS → 256-d,
  L2-normalized) and an SE(2) transformer (12 layers, d = 256). The SE(2) transformer
  runs over sub-slices of ~1,000 cells, plus whole-section "virtual cells". The
  embedding is the last hidden state of each cell token (256-d), as in the authors'
  `get_embeddings.py`.
- Recipe: the authors' `get_embeddings.py`, `model/extraction.py` and
  `preprocessing/preprocess.py`, with their defaults; one `load_data` call per section:
  - cell-encoder input (Geneformer tokenizer): counts / the cell's total over all panel
    genes × 10⁴ / Geneformer's gene medians, then the dictionary genes ranked by that
    value (at most 2,048 tokens, `<cls>` first);
  - coordinates centred and divided by the section's bounding-box diagonal (units cancel);
  - sub-slices: a square grid with one square per ~1,000 cells. A square with ≥ 2,000
    cells is cut into ~1,000-cell chunks **in obs order** (as released, see Caveats).
    Every non-empty square is kept, so a few sub-slices have only 1–50 cells (0.02–0.29%
    of the cells);
  - virtual cells: Leiden (resolution 1.0) on `0.2·xy + 0.8·PCA₂(cell embedding)` of the
    whole section. Their means are added to every sub-slice;
  - attention bias from the 50-NN distances within a sub-slice and the distances to the
    virtual cells;
  - batch sizes 32 (cell encoder) and 4 (sub-slices; padding is masked), seed 1
    (`get_embeddings.sh`) before each section, deterministic torch algorithms. The
    virtual cells are still not exactly reproducible across machines: for sections of
    500–2,560 cells, sklearn's PCA uses its randomized solver, and the BLAS thread count
    changes the Leiden result (e.g. 28 vs. 30 virtual cells on `78_batch0`; ~1% of
    `chr78`'s cells). The embedding moves little (min cosine 0.998 on `78_batch3`).
- Fixes to the released code, each compared with the authors' code in
  `checks/test_stofm.py`:
  - `encode_cell` crashes with `KeyError: 'label'`: it removes every column except
    `input_ids`, and Geneformer's collator then deletes `label`. Here each cell gets a
    dummy label, so the authors' `CellEncoderCollator` runs unchanged.
  - `preprocess.py` densifies the whole cells × genes matrix (~75 GB for a Visium HD
    section). Its tokenizer is re-implemented on row chunks of the sparse matrix (tokens
    identical).
  - `extraction.py` imports `rapids_singlecell` and `cupy`. On a GPU it builds the
    virtual cells' kNN graph exactly with rapids. When that fails it silently falls back to
    scanpy, which is approximate (pynndescent) from 8,192 cells on. rapids is not installed
    here: stubs satisfy the import, and `load_data` runs on its CPU path with scanpy's exact
    brute-force kNN. This is what scanpy does by itself below 8,192 points, so the sub-slice
    graphs are unchanged.
  - The SE(2) pass runs under `torch.no_grad()` (same output, less memory).
  - Embeddings start as NaN instead of 0: a cell left out of every sub-slice stops the run.
    Every cell must be in exactly one sub-slice.
- Benchmark choices:
  - genes are matched to Geneformer's dictionary by the silver Ensembl ID. IDs it lacks
    fall back to Geneformer's own ID for the symbol: the pseudo-autosomal CD99, CSF2RA,
    IL3RA, IL9R and SHOX, whose silver IDs are the chrY copies.
  - Coverage of the panels in Geneformer's dictionary:
    - `xhs1000` 4,938/4,948 and `xhs49` 4,944/4,954 (`ABO`, `ZAN`, … missing);
    - `chr78` 172/173 (`TUBA4B`);
    - `chl59` 944/946 (`SOD2`, `TPSB2`). `TPSB2` is not in the dictionary. `SOD2` is,
      under its old ID `ENSG00000112096`, but the silver file and the symbol fallback
      both give its new ID `ENSG00000291237`, so it is lost (104th of 946 genes by mean
      count, detected in 24% of cells);
    - `vhb3003` 17,914/18,085.

    There are no control probes in the panels. Cells have a median of 11 (ISS),
    121 (CosMx) and 220–270 (Xenium) dictionary genes. One Xenium cell exceeds 2,047
    and loses its lowest-ranked genes.
  - No MT-gene removal (CIFM's preprint asks for it; Geneformer and SToFM do not).
- In the result h5ad, for interpretation (not scored):
  - `obs['stofm_virtual_cell']`: the virtual-cell Leiden clusters, no labels used;
  - `obs['stofm_subslice']`;
  - `obsm['X_stofm_cell']`: the cell-encoder embedding without spatial context.
- Caveats:
  - Pretraining data: no benchmark dataset is in SToCorpus-88M (Hugging Face
    `Toycat/SToCorpus-88M`, which still contains the paper's downstream slides). Its 10x
    Xenium files are colon, kidney, lung, lymph node, pancreas and brain (no skin). Its
    CosMx files are frontal cortex, pancreas and mouse brain (not the NSCLC lung set).
    There is no ISS and no Visium HD. Unlike CIFM and HEIST, SToFM has not seen `xhs49`.
  - The paper evaluates spatial domains only with a head trained on frozen embeddings.
    Its zero-shot Leiden result is for cell types (MERFISH mouse brain). Visium HD bins
    are not cells.
  - Obs-order chunks. A grid square with ≥ 2,000 cells is cut into chunks by its cells'
    order in the file, not by position. Measured on the benchmark sections, this affects:
    - 37–80% of the Xenium cells and 0–84% per ISS section;
    - no CosMx or Visium HD cell.
    The silver obs order is spatial for ISS: the chunks are mostly as compact as their
    square (median 10-NN distance 1.0× the square's; p90 1.0–1.03×, except 1.15× on
    `78_batch7` and 1.37× on `78_batch2`). For Xenium it is partly so: the median distance is 1.0–1.24×
    the square's and the p90 1.4–1.9×, against 1.4–1.7× for a random order. Per section,
    `run_info.json` records `frac_cells_obs_order_split` and
    `obs_order_chunk_knn_ratio_{median,p90}`. `checks/test_stofm.py` (e) measures how
    much the order changes the embedding.
  - The number of virtual cells is not capped: the authors' `< 200` loop is dead code,
    overwritten by the Leiden call after it. It grows with the section size
    (`n_virtual_cells` per section in `run_info.json`). It is 27–69 for sections of
    up to 18k cells, 108–143 for 66k–151k cells, and 207–223 for the 459k–574k
    bins of a `vhb3003` section, so a sub-slice has at most ~2,100 tokens.
  - Results: the headline reaches the target cluster count exactly on every dataset.
    Headline NMI/ARI:

    | Dataset | NMI | ARI |
    |---|---|---|
    | `xhs1000` | .341 | .209 |
    | `xhs49` | .271 | .124 |
    | `chr78` | .246 | .117 |
    | `chl59` | .346 | .208 |
    | `vhb3003` | .225 | .145 |

    That is below CIFM and above HEIST on each dataset. The virtual-cell clusters
    (`obs['stofm_virtual_cell']`; 127–760 per dataset) are small spatial patches
    (ARI vs. the ground truth ≈ 0.01). Unlike HEIST's internal clusters, they say
    little about the niches.

Runs (on L40S GPUs):
- Embedding takes 3.4 min (`xhs1000`), 3.1 min (`chr78`), 9.9 min (`xhs49`), 12 min
  (`chl59`) and 1.4 h (`vhb3003`). Most of that is the cell encoder.
- `vhb3003` needs 22 GB of host memory for the embedding and 18 GB for scoring
  (43 min); no bin reaches the 2,048-token limit (median 183 genes per bin).
- Peak GPU memory is 5–10 GB; peak host memory is 2.5–6 GB.
- Scoring takes 1–18 min.
- `checks/test_stofm.py` (all passed, job 231472, L40S):
  - (a) The authors' demo:
    - the cell encoder on their tokenized cells reproduces their `ce_emb.npy` (max
      |diff| 1.8e-6);
    - their loop (seed 1) gives their `stofm_emb.npy` to a mean cosine of 0.99993 (p1
      0.9995, min 0.983). With the approximate kNN fallback the cosine is the same
      (0.99994), so the demo cannot tell which path made the shipped file. The residual
      is presumably from package versions in the virtual-cell PCA and Leiden.
  - (b) Tokens are identical to the authors' tokenizer on Xenium (17,384 cells), CosMx
    (20,000) and Visium HD (5,000 bins) sections. ISS (all 325,816 `chr78` cells) was
    compared by hand on CPU, not by `test_stofm.py`, and was identical too.
  - (c) `embed_section` equals the authors' loop exactly, on the demo and on an `xhs1000`
    section (17,384 cells, 21 sub-slices, 69 virtual cells). Batch size 1 vs 4 differs by
    at most 2.9e-6. On that section the approximate kNN gives the same embedding as the
    exact one.
  - (d) Seed 1 vs 2 on the smallest ISS section (randomized PCA): mean cosine 1.00000,
    min 0.998.
  - (e) Obs order, on the `xhs1000` section (37% of cells in obs-order chunks; chunk/square
    10-NN ratio median 1.24, p90 1.89):
    - those cells move to a mean cosine of 0.982 (p1 0.92, min 0.80) for a random order
      (ratio 1.73) and 0.983 (p1 0.92) for cells sorted by x (ratio 1.0);
    - the other cells barely move (0.99995).
    The silver order is therefore one of several equally arbitrary choices for these
    cells. The protocol keeps it, as released.

```bash
source config.sh
OUT=/nfs/team361/jr43/terra/benchmarking/human; LOGS=$OUT/logs; mkdir -p $LOGS
BENCH_FINAL_OUTPUT_DIR=$OUT SCRIPT=run_stofm.py PYTHON_ENV=PYTHON_STOFM DATASET=xhs49-1b_1p \
    bsub -G s10396 -q inference -n 8 -M 64GB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 12:00 \
    -o $LOGS/method_%J.out -e $LOGS/method_%J.err < run_method.lsf
# vhb3003-3b_1p: -M 128GB
```

### AIDO.Tissue

- Environment, code, checkpoint and gene map: `envs/aido_tissue.sh` (login node).
  - GenBio AI renamed AIDO.* to GB.* in July 2026. The code is a clone of
    `genbio-ai/ModelGenerator` at `bda6f2d` (`modelgenerator.backbones.gb_tissue_60m`).
    It is installed without its dependencies, plus the packages its backbone, data,
    task and CLI modules import (not openfold, dllogger or tiledbsoma).
  - The weights are `genbio-ai/GB.Tissue-60M` at `e414516`, formerly
    `AIDO.Tissue-60M`: 12 layers, width 512, encoder only.
  - flash-attn 2 is required (no other attention), so the GPU needs compute
    capability ≥ 8.0.
  - `run_info.json` records the code commit, the weights' sha256 and the gene
    map's sha256.
- Checkpoint: 60M, the larger of the two released models. The authors'
  embedding config (`emb.xenium.yaml`) uses the 3M model; `checks/test_aido_tissue.py`
  runs both.
- Model: a transformer over the expressed genes of a cell and its k nearest cells.
  - Each gene token is a gene-identity embedding plus an expression-value MLP.
  - 2D rotary ids encode the gene index and the cell's rank (0 = the cell itself).
  - Coordinates and distances are not inputs; they only choose the neighbours.
  - The output is the tokens of the cell itself: its expressed genes plus 2 depth
    tokens. There is no CLS token.
- Recipe: the authors' `experiments/GB.Tissue/emb.xenium.yaml` and the embedding
  section of its README:
  - input as built by `CellWithNeighborDataModule`: the raw counts of the cell and
    of its 10 nearest cells (cKDTree on x, y; nearest first), each aligned to the
    model's 19,264 genes, then concatenated;
  - the authors' backbone forward, unchanged. Per cell it computes
    `log1p(counts / total × 10⁴)` over the matched genes and appends 2 depth tokens
    (`max(log10 total, 5)`, so 5 for every benchmark cell);
  - batch size 1, bf16 autocast (Lightning `bf16`), seed 42;
  - the embedding is the maximum over the cell's output tokens (512-d).
- Batch size 1 is part of the recipe, not a speed setting. The backbone
  normalizes by the sum over the whole batch (`X.sum()`). Its flash-attention call
  also attends to padding tokens ("assuming NO PADDING"). In batches larger than 1,
  a cell's embedding would therefore depend on its batch-mates.
- The authors' DataModule and prediction writer are replaced by a loop that builds
  the same input rows and pools on the GPU. `checks/test_aido_tissue.py` shows the
  rows are byte-identical. The loop is needed because:
  - `SpatialDataGenerator` builds one kNN over all rows of its file, so neighbours
    would cross sections. Here the kNN is per section;
  - the writer keeps every prediction in memory and writes one file per cell (1.5M
    files on Visium HD), merged in glob order.
- Guards, each of which stops the run:
  - the checkout, the installed `modelgenerator` and the weights are the pinned ones;
  - every encoder tensor equals the checkpoint, because `from_pretrained`
    initializes missing weights at random without stopping. Only the MLM head
    (`cls.*`) is unused;
  - attention and hidden dropout are 0. The flash-attention call passes its dropout
    probability even in eval mode;
  - every cell's output has exactly its expressed genes + 2 tokens.
- Genes: by exact symbol, as `align_adata` and the tissue README do. A panel gene
  whose symbol is not in the list falls back to its silver Ensembl ID. That ID is
  mapped to the list with the authors' bionty map (`build_map`, as their GB.Cell data
  modules align the same gene list), unless another panel gene already has that gene.
  - The model's list uses current HGNC symbols, while the Visium HD panel has
    10x 2020-A names.
  - The fallback recovers 244 Visium HD genes (e.g. `H3F3A` → `H3-3A`, `MARCH7` →
    `MARCHF7`, `AC118549.1` → `ZZZ3`). It recovers one Xenium gene
    (`UGT1A1` → `UGT1A`) and nothing on ISS or CosMx.
  - Coverage:
    - `xhs1000` 4,941/4,948 and `xhs49` 4,947/4,954 (clustered protocadherins,
      `UGT1A4`, `UGT1A9` missing);
    - `chr78` and `chl59` complete;
    - `vhb3003` 17,901/18,085.

    The missing Visium HD genes include the immunoglobulin and TCR constant genes
    (`IGKC`, `IGHG1`, `TRAC`, …), which the model does not have. Like every unmatched
    gene, they are not counted in the total (`frac_counts_unmatched` in
    `run_info.json`).
  - No MT-gene removal (the authors do none).
- In the result h5ad, for interpretation (not scored): `obsm['X_aido_tissue_cell']`,
  the same model and pooling on the cell alone, without spatial context. For a single
  cell the backbone also returns padding outputs; they are cut off before pooling
  (the authors' max-pool drops only NaN rows and would keep them).
- Caveats:
  - Pretraining data: 76 slides from the Vizgen, NanoString and 10x Genomics
    public pages, with no list of datasets.
    - `chl59` (NanoString's public NSCLC CosMx set) is likely in it: the paper's
      NanoString share, 12 slides with 1.7M cells, fits NSCLC plus liver.
    - `xhs49` (10x Xenium Prime skin) possibly.
    - `vhb3003` is unknown.
    - `chr78` (ISS) is not.
  - The paper evaluates only fine-tuned tasks; it has no zero-shot clustering
    result. The run uses 10 neighbours, as every released config does
    (`emb.xenium.yaml` and the 60M fine-tuning configs); the paper says 60M was
    pretrained with 16 or 32, and the 60M model card says K=8.
  - Visium HD bins are not cells. On the 8 µm grid the 10th neighbour is one of 4 bins
    at 16 µm. The silver coordinates are registered to the image, so the grid is
    slightly rotated and scaled, and those bins differ in distance by a few nm: in 87%
    of the bins of a ~3,400-bin window the 10th and 11th neighbours are within 0.1%,
    but none are exactly tied. The neighbour set and order (the rotary cell index) are
    thus fixed by the grid geometry, the same for every run and independent of the
    obs order (check (d)). `run_info.json` records the exact-tie fractions per section.
  - chr78 (ISS, 173 genes over all sections, 169 in some) has a median of 11
    expressed genes per cell, so each cell is only ~13 tokens. AIDO.Tissue scores lowest there of all its datasets.
- Speed: the authors' `gather_data` rebuilds a tensor from a Python list of one entry
  per input position (211,926 for 11 cells) on every call, three times per forward
  pass. That was ~70 ms per cell on the CPU, whatever the number of tokens.
  `run_aido_tissue.py` swaps in a copy that builds the tensor once
  (`cached_gather_data`, inside `embed` only). Values, dtype and operations are the
  same, and check (c) shows identical embeddings. A pass then takes ~23 ms per cell
  with neighbours and ~19 ms alone, instead of ~92 (L40S, up to ~9k tokens).

Results: headline NMI/ARI, and the other baselines' NMI for comparison:

| Dataset | NMI | ARI | CIFM | SToFM | HEIST |
|---|---|---|---|---|---|
| `xhs1000` | .320 | .183 | .477 | .341 | .108 |
| `xhs49` | .273 | .140 | .444 | .271 | .047 |
| `chr78` | .070 | .021 | .382 | .246 | .088 |
| `chl59` | .504 | .369 | .494 | .346 | .006 |
| `vhb3003` | .206 | .124 | .357 | .225 | .004 |

The headline reaches the target cluster count exactly on every dataset. AIDO.Tissue is
level with SToFM on Xenium and slightly below it on Visium HD. It has the highest NMI
of the four zero-shot models on `chl59`, where pretraining exposure is likely (above);
CellCharter, neighbourhood-expression PCA and TERRA score higher there. On `chr78` it is
among the lowest of all methods, level with scGPT-spatial (.071); HEIST's .088 there is
at 20 clusters instead of 19 (see HEIST).

Runs (L40S):
- Each cell is embedded twice: with its neighbours (scored) and alone
  (`X_aido_tissue_cell`), at ~23 and ~19 ms per cell with the cached `gather_data`.
  `embedding_seconds` covers both passes.
  - `xhs1000` took 40 min.
  - `vhb3003` took 18 h to embed and 19 h in all, within the queue's 48 h limit, with
    80 GB peak host memory.
  - `chr78`, `chl59` and `xhs49` ran before the speed fix (~92 ms per cell, up to
    17 h); their output is identical.
- Peak GPU memory is 0.7–2.1 GB.
- No cell has zero matched counts. There are no duplicate embeddings.
  `avg_cosine_similarity` is 0.91–0.95.

`checks/test_aido_tissue.py` (all passed: job 952663, plus (d) in job 923491; L40S):
- (a) On the authors' demo (586 cells), `run_aido_tissue`'s loop equals `mgen predict`
  with `emb.xenium.yaml` plus the README max-pool, for 3M and 60M (max |diff| 0).
- (b) On the smallest `chr78`, `chl59` and `xhs1000` sections:
  - the exact-symbol columns equal `align_adata`. The fallback column
    (`UGT1A1` → `UGT1A`) holds the panel gene's counts, and `align_adata` leaves it
    empty;
  - the neighbour lists equal `SpatialDataGenerator`'s;
  - the input rows are byte-identical (1,776–2,000 cells per section);
  - on `chr78`, the embedding of the whole section equals `mgen predict` (max |diff| 0).
- (c) Two runs are bitwise identical, with and without neighbours. The cached
  `gather_data` equals the authors' bitwise (300 `xhs1000` cells, both paths).
- (d) Shuffled obs order on the `xhs1000` section (17,384 cells) and on the Visium HD
  window (3,381 bins): no exact ties, no neighbour list changes, cosine 1.00000.

```bash
source config.sh
OUT=/nfs/team361/jr43/terra/benchmarking/human; LOGS=$OUT/logs; mkdir -p $LOGS
BENCH_FINAL_OUTPUT_DIR=$OUT SCRIPT=run_aido_tissue.py PYTHON_ENV=PYTHON_AIDO_TISSUE DATASET=xhs49-1b_1p \
    bsub -G s10396 -q tiger22-inference -n 8 -M 64GB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 12:00 \
    -o $LOGS/method_%J.out -e $LOGS/method_%J.err < run_method.lsf
# chl59-8b_1p: -M 96GB -W 24:00; vhb3003-3b_1p: -M 128GB -W 48:00
# Checks: bsub (same queue and GPU request, -W 6:00) \
#   "$PYTHON_AIDO_TISSUE checks/test_aido_tissue.py [--skip-sensitivity]"
```
