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

## Checks

```bash
python checks/test_resolution_scan.py   # banksy env: scan vs. brute force on toy data
```

---

# Methods and analyses on this branch (human-niche-benchmark)

Everything above is the shared framework (branch
`human-niche-benchmark-framework`). This branch adds the implemented methods,
extra analyses and diagnostics. Every script below is a working example of a
method script.

## Methods

| Method key | Script | `PYTHON_ENV` | Notes |
|---|---|---|---|
| `terra` | `run_terra.py` | `PYTHON_TERRA` (GPU ≥ 40 GB) | TERRA-96M (`TERRA_MODELS` in the script), `neighborhood_emb`, sections tokenized separately |
| `terra_112m` | `run_terra.py --model 112m` | `PYTHON_TERRA` | TERRA-112M: trained on the full corpus (no held-out split); `batch` meta token padded |
| `terra_hvg` | `run_terra.py --preprocess hvg` | `PYTHON_TERRA` | 2000 HVGs among the genes TERRA's vocabulary knows (> 10 pretraining occurrences) before tokenization |
| `terra_random` | `run_terra.py --random_init` | `PYTHON_TERRA` | same architecture, untrained (capacity control) |
| `terra_noneigh` | `run_terra.py --model noneigh` | `PYTHON_TERRA` | TERRA-96M retrained with neighbour cells masked |
| `terra_noneigh_cell_emb_nbavg` | `reuse_embedding.py --source terra_noneigh --emb_key cell_emb --neighbour_average` | `PYTHON_BANKSY` | its cell embedding averaged over 10 spatial neighbours |
| `neigh_expr_pca` | `run_neigh_expr_pca.py` | `PYTHON_BANKSY` | summed 10-NN counts, CP10k + log1p, PCA 20 |
| `cell_type_composition_kmeans` | `run_cell_type_composition.py --clustering kmeans` | `PYTHON_BANKSY` | annotated cell types (finest level), k-means at the target (10 seeds); n/a for ISS |
| `banksy` | `run_banksy.py` (`--n_top_genes 2000` for Visium HD) | `PYTHON_BANKSY` | raw counts, 10 NN, λ 0.8, PCA 20, + Harmony (forced integration) |
| `cellcharter` | `run_cellcharter.py` | `PYTHON_CELLCHARTER` (GPU) | scVI (section as batch) + 3-layer aggregation |
| `nichecompass` | `run_nichecompass.py` | `PYTHON_NICHECOMPASS` (GPU ≥ 40 GB) | mouse-notebook hyperparameters; OmniPath / NicheNet / MEBOCOST gene programs from local files |
| `novae` | `run_novae.py` (or `import_fm.py --method novae`) | `PYTHON_NOVAE` (CPU) | `MICS-Lab/novae-human-0`, zero-shot; Leiden on Novae's batch-corrected embedding |
| `nicheformer_rerun` | `run_nicheformer.py` | `PYTHON_NICHEFORMER` (GPU) | authors' recipe; n/a for ISS and Visium HD (no technology token) |
| `cellplm` | `run_cellplm.py` (`--batch_size 50000` for Visium HD) | `PYTHON_CELLPLM` (GPU) | `20231027_85M`, no spatial context |
| `scgpt-spatial` | `run_scgpt_spatial.py` | `PYTHON_SCGPT_SPATIAL` (L40S; fails on H100) | `scGPT_spatial_v1`, per section |

`import_fm.py` (banksy env, `EXTRA_ARGS="--method <novae|scgpt-spatial|nicheformer|cellplm>"`)
imports the earlier zero-shot embeddings of `../human_benchmarking/` instead of
recomputing them. `reuse_embedding.py` registers another embedding of a
finished result as its own method (`--source --emb_key --method`, optional
`--neighbour_average`, `--clustering kmeans`). `run_harmony.py` (post-hoc
Harmony on a finished embedding) was tested and is not in the figures.

Examples:
```bash
SCRIPT=run_terra.py PYTHON_ENV=PYTHON_TERRA DATASET=xhs1000-39b_1p bsub -q gpu -n 12 -M 128GB \
    -R "select[mem>128GB] rusage[mem=128GB] span[hosts=1]" -gpu "num=1:gmem=40G:mode=exclusive_process" \
    -W 12:00 < run_method.lsf
SCRIPT=run_terra.py PYTHON_ENV=PYTHON_TERRA DATASET=xhs1000-39b_1p EXTRA_ARGS="--preprocess hvg" bsub ... < run_method.lsf
SCRIPT=import_fm.py PYTHON_ENV=PYTHON_BANKSY DATASET=xhs1000-39b_1p EXTRA_ARGS="--method novae" bsub ... < run_method.lsf
SCRIPT=reuse_embedding.py PYTHON_ENV=PYTHON_BANKSY DATASET=xhs1000-39b_1p \
    EXTRA_ARGS="--source terra_noneigh --emb_key cell_emb --method terra_noneigh_cell_emb_nbavg --neighbour_average" \
    bsub ... < run_method.lsf
```

## Additional analyses

- **Fixed resolutions** (`score_fixed_resolutions.py` / `.lsf`, banksy env):
  every method clustered at Leiden resolutions 0.1, 0.2, ..., 1.0 instead of
  matching the target; `plot_fixed_resolutions.py` → `summary/figures/fixed_resolution/`.
- **Common cluster numbers** (`score_k_grid.py` / `.lsf`, banksy env): every
  method clustered at k = 5, 10, ..., 30 (lowest resolution giving exactly k);
  `plot_k_grid.py` → `summary/figures/k_grid/` (hollow dots: k not reached
  exactly).

Both read the finished results (`*_embedded_clustered.h5ad`) and pick up any
method in `METHODS` automatically; the array size in the `.lsf` must cover the
number of method × dataset pairs.

## Diagnostics and checks

- `diagnostics/token_diagnostic.py` (terra env): which genes fill TERRA's
  256-token budget per cell (motivates TERRA + HVG).
- `diagnostics/section_depth.py`: per-section sequencing depth vs. niches.
- `checks/test_novae_correction.py` (novae env): equals the paper notebook's
  Novae correction; `checks/compare_silver_copies.py`: lustre silver == NFS silver.
- `method_report.py`: per-method section of `../human_benchmarking/*_benchmarking.ipynb`.

## Known caveats

- The environments are sb75's (`/nfs/team361/sb75/.venvs/*`); `envs/*.txt`
  lists their package versions. TERRA and NicheCompass are imported from sb75's
  checkouts; each run records the checkout's commit in its `run_info.json`.
- NicheFormer on ISS: 27.7% of cells have identical embeddings, so 19 clusters
  are unreachable (n/a in the figures).
- Novae needs a newer NVIDIA driver than the farm and tiger GPU nodes have; it
  runs on the CPU (`--accelerator cpu`, default).
- NicheCompass logs show `omnipathdb.org` connection errors from the omnipath
  package's import-time check (no internet on compute nodes); gene programs are
  loaded from local files.
