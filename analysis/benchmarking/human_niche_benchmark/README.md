# Human Niche-Identification Benchmark

Zero-shot niche identification of TERRA vs. spatial baselines and spatial
foundation models on four human datasets, following the protocol of the TERRA
paper's mouse-brain benchmark (`../mouse_benchmarking/`).

## Protocol
- **Cells**: identical for every method; sections, label keys, NaN filtering,
  coordinate scaling as in `../human_benchmarking/terra_benchmarking.ipynb`.

  | Dataset | Sections | Ground truth | Niches (> 100 cells) |
  |---|---|---|---|
  | xhs1000-39b_1p (Xenium skin, eczema) | batches 11, 19, 32 | `niche_type` | 11 |
  | xhs49-1b_1p (Xenium skin, melanoma) | 1 | `niche_name` | 7 |
  | chr78-11b_1p (ISS reproductive tract) | 11 | `broad_anatomy` | 19 |
  | chl59-8b_1p (CosMx lung, NSCLC) | 6 of 8 | `niche` | 12 |
- **Spatial graphs** are built per tissue section (no cross-section neighbours).
- **Clustering** (all methods, `bench_utils.cluster_and_score`): 15-NN graph on
  the embedding, Leiden (igraph, 2 iterations, seed 0). One upward scan on a
  0.01 resolution grid, every run scored, until the cluster count stays above
  the target for 5 consecutive steps (counts are not monotonic). From that one
  table: **headline** = lowest grid resolution giving exactly the common cluster
  number; **matched** = all runs giving it; **sweep** = ±4 grid steps around the
  median matching resolution. If no grid point hits the target, the first
  crossing is refined on a 0.001 grid; if still unreachable, the best reachable
  number is used (and flagged). All final clusterings were computed in the same
  (banksy) environment via `rescore.py` / `rescore_fm.py`, so every method uses
  the same igraph / leidenalg versions.
- **Metrics**: NMI, ARI vs. ground truth; for multi-section datasets iLISI,
  batch ASW and MMD (mean over section pairs) as in the mouse notebooks.

## Methods

| Method | Script | Environment | Notes |
|---|---|---|---|
| TERRA | `run_terra.py` | terra (GPU) | TERRA-96M, model `01112025_081731_614`, `neighborhood_emb`, sections tokenized separately |
| TERRA-112M | `run_terra.py --model 112m` | terra (GPU) | published TERRA-112M (bundle `11012026_190057_836`): trained on the full corpus without a held-out split, so the benchmark cells may be in its training data; has a `batch` meta token, which the zero-shot pipeline pads |
| TERRA + HVG | `run_terra.py --preprocess hvg` | terra (GPU) | as TERRA, but only the 2000 highly variable genes (`seurat_v3` on raw counts, section as `batch_key`) are tokenized; panels with ≤ 2000 genes unchanged |
| Banksy | `run_banksy.py` | banksy | raw counts, 10 NN, scaled Gaussian, max_m 1, λ 0.8, PCA 20, **+ Harmony (forced integration)** |
| Neigh. Expr. PCA | `run_neigh_expr_pca.py` | banksy | summed 10-NN counts, CP10k + log1p, PCA 20 |
| CellCharter | `run_cellcharter.py` | cellcharter (GPU) | scVI (section as batch) + 3-layer aggregation |
| NicheCompass | `run_nichecompass.py` | nichecompass (GPU ≥ 40 GB) | mouse-notebook hyperparameters; human OmniPath / NicheNet v2 / MEBOCOST gene programs from local files |
| Novae, scGPT-spatial, NicheFormer, CellPLM | `rescore_fm.py` | banksy | existing zero-shot embeddings from `../human_benchmarking/*` re-scored; cells matched by `cell_id` (or section + original obs name). Novae as in the paper's mouse benchmark: native domains at the target level → Novae's batch-effect correction → Leiden on the corrected embedding |

Method order, labels and colours (`bench_utils.METHODS`) are those of the paper
figure (`../mouse_benchmarking/benchmarking_comparison.ipynb`).

## Files

| File | Role |
|---|---|
| `bench_utils.py` | Paths, datasets, methods, data loading, clustering scan, metrics, saving |
| `run_*.py`, `rescore.py`, `rescore_fm.py` | One script per method / re-scoring of saved or existing embeddings |
| `config.sh`, `job_wrapper.sh`, `*.lsf` | LSF jobs: environments and paths (`config.sh`), job-local lustre working folder with verified copy to NFS (`job_wrapper.sh`) |
| `collect_results.py`, `plotting.py`, `benchmark_comparison.ipynb` | Result tables and figures |
| `method_report.py` | Per-method section used by `../human_benchmarking/*_benchmarking.ipynb` |
| `checks/` | Re-runnable correctness checks (see below) |
| `diagnostics/` | `token_diagnostic.py`: which genes fill TERRA's 256-token per-cell budget (motivates TERRA + HVG) |
| `envs/` | Package versions of the environments used (`record_environments.py`) |

## Running

Submit from this folder. Every job runs in a job-local lustre folder via
`job_wrapper.sh` (inputs from lustre silver; results copied to
`<repo>/artifacts/benchmarking/human/<method>/` on success, then the lustre
folder is deleted; on failure it is kept for debugging). Paths and environments
are set in `config.sh` and at the top of `bench_utils.py`.

```bash
cd analysis/benchmarking/human_niche_benchmark
# CPU baselines (array: 1 = BANKSY, 2 = neighbourhood PCA)
DATASET=xhs1000-39b_1p bsub -q normal -n 8 -M 64GB \
    -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" -W 12:00 < run_cpu_methods.lsf
# GPU baselines
METHOD=cellcharter DATASET=xhs1000-39b_1p bsub -q gpu -n 8 -M 64GB \
    -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" -W 12:00 < run_gpu_method.lsf
METHOD=nichecompass DATASET=xhs1000-39b_1p bsub -q gpu -n 8 -M 128GB \
    -R "select[mem>128GB] rusage[mem=128GB] span[hosts=1]" \
    -gpu "num=1:gmem=40G:mode=exclusive_process" -W 48:00 < run_gpu_method.lsf
# TERRA
DATASET=xhs1000-39b_1p bsub -q gpu -n 12 -M 128GB \
    -R "select[mem>128GB] rusage[mem=128GB] span[hosts=1]" \
    -gpu "num=1:gmem=40G:mode=exclusive_process" -W 12:00 < run_terra.lsf
# TERRA + HVG: same job with EXTRA_ARGS="--preprocess hvg"
# Token-budget diagnostic (all datasets, CPU)
bsub -q normal -n 4 -M 96GB -R "select[mem>96GB] rusage[mem=96GB] span[hosts=1]" \
    -W 6:00 < diagnostics/token_diagnostic.lsf
# Foundation-model embeddings (Novae, scGPT-spatial, NicheFormer, CellPLM)
METHOD=novae DATASET=xhs1000-39b_1p bsub -q normal -n 8 -M 64GB \
    -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" -W 12:00 < rescore_fm.lsf
# Re-cluster a saved embedding after a protocol change (no re-embedding)
METHOD=banksy DATASET=xhs1000-39b_1p bsub -q normal -n 8 -M 48GB \
    -R "select[mem>48GB] rusage[mem=48GB] span[hosts=1]" -W 12:00 < rescore.lsf
```
Resources used for the largest dataset (chl59, 534k cells): CPU jobs up to
96–128 GB, GPU jobs 128–256 GB host memory; TERRA and NicheCompass need a ≥ 40 GB GPU
(out of memory on 16 GB cards).

## Results and figures
- `python collect_results.py` → `<repo>/artifacts/benchmarking/human/summary/`:
  `headline.csv`, `scan.csv`, `sweep.csv`, `matched.csv`, `batch.csv`.
- `benchmark_comparison.ipynb` (terra environment; ~96 GB for the spatial maps,
  so run it as an LSF job or on a large interactive node) → figures in
  `summary/figures/`: paper-style `niche_{NMI,ARI}_comparison_human` (sweep) and
  `niche_{NMI,ARI}_comparison_human_matched` (matched runs only), heatmap, sweep
  curves, spatial maps.

## Checks
```bash
python checks/test_resolution_scan.py      # banksy env: scan vs. brute force on toy data
python checks/test_novae_correction.py     # novae env: equals the paper notebook's patched Novae correction
python checks/compare_silver_copies.py     # banksy env (~32 GB): lustre silver == NFS silver
```

## Known caveats
- The environments are sb75's (`/nfs/team361/sb75/.venvs/*`); their package
  versions are recorded in `envs/*.txt`. TERRA and NicheCompass are imported from
  sb75's checkouts (`/nfs/team361/sb75/terra`, `/nfs/team361/sb75/nichecompass`);
  each run records the checkout's commit in its `*_run_info.json`.
- NicheFormer chr78: 27.7% of cells have identical embeddings; the 15-NN graph
  has 211 disconnected components, so 19 clusters are unreachable (best
  reachable reported).
- NicheCompass logs show `omnipathdb.org` connection errors from the omnipath
  package's import-time check (no internet on compute nodes); gene programs are
  loaded from local files, 52 OmniPath rows without gene symbols are dropped.
- TERRA + Harmony (`run_harmony.py`, post-hoc Harmony on the TERRA / TERRA + HVG
  embedding, as for BANKSY) was tested and is not in the figures: it improved
  only Xenium skin (eczema) and lowered NMI/ARI on the ISS reproductive tract
  and CosMx lung datasets. Results stay in `artifacts/benchmarking/human/terra*_harmony/`.
  Per-section sequencing depth (`diagnostics/section_depth.py`) follows niche
  far more than section, so it does not explain TERRA's section-specific clusters.
