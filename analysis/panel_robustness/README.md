# Gene panel robustness

Does TERRA embed the same tissue consistently when it is measured with a different gene panel? (Reviewer comment R3C2.1;
rebuttal plan E12.) TERRA's cell and neighborhood embeddings are means over the gene tokens of the genes a section's
panel contains, so a smaller or different panel changes what is averaged.

## Design (`gene_panel_subsampling.py`)

- **Tissue:** the held-out Xenium Prime 5K skin benchmark sections, `xhs1000-39b_1p` batches 11, 19 and 32 (eczema;
  held out from TERRA-96M, see `analysis/data_preparation/heldout_batch_ids.py`). The panel has 5,000 genes; 4,948 map
  to Ensembl IDs, and the panel genes are those the TERRA pipeline keeps and has in its vocabulary. Cells without a niche
  label are removed before embedding, as in `terra_benchmarking.ipynb`. Cells are named `<batch>:<silver cell_id>`
  (or `<batch>:row<i>`), because the cell names in the silver files can repeat (e.g. `adata_batch32.h5ad`); the silver
  name is kept in `obs['silver_name']`.
- **Panels:** the full panel; random panels of 1,000, 2,000, 3,000 and 4,000 genes (5 random draws per size, nested
  within a draw); for 1,000 and 2,000 genes, a second random panel per draw that shares no gene with the first
  ("disjoint"); and the full panel embedded a second time (run-to-run variation). 32 conditions by default.
- **What changes:** only the genes. Counts of the kept genes, cells, coordinates and spatial neighbors are unchanged.
  TERRA tokenizes detected genes only; with TERRA-96M (raw-count ranking, shifted-log values) the kept genes keep their
  values and order. A cell that detects no gene of a panel keeps its place in its neighbors' sequences as an
  all-padding segment (`data.pad_empty_neighbors`, set in a copy of the model folder), so neighborhoods are identical
  across panels. Metrics use the cells that detect at least one gene in every panel.
- **Model:** the released TERRA-96M benchmarking checkpoint (`artifacts/models/01112025_081731_614`) by default;
  `--model_folder_path` selects another model folder (model_config.yaml, token_dictionary.pkl, model_checkpoint.pt).
  Gene names are mapped with the model folder's `ensembl_dictionary.pkl` (or `ensembl_dic.pkl`); without one, with the
  silver files' `var['ensembl_id']` (no Ensembl download on compute nodes).
- **Baselines:** PCA (50 components) of log-normalized expression (cells) and of log-normalized expression averaged over
  each cell and its 10 nearest spatial neighbors (neighborhoods), fit per panel.

## Readouts

Per panel (`comparison = intrinsic`):
- NMI and ARI of Leiden clusters (k = 15 graph) against the niche (`niche_type`) and cell-type (`new_annotation`)
  annotations: resolution bisected to the number of annotated classes with at least 100 cells (5 seeds;
  `*_kmatched`), and a fixed resolution grid (0.1, 0.25, 0.5, 1; `*_res<r>`);
- leave-one-section-out kNN label transfer (k = 15; accuracy, macro F1).

Against the full panel (`vs_full`; same cells):
- `knn_overlap`: share of each cell's 15 nearest neighbors (within its section) that are also its neighbors with the full
  panel;
- `cluster_ari_kmatched` / `cluster_nmi_kmatched`: agreement of the K-matched Leiden clusters with the full-panel ones,
  averaged over all pairs of different Leiden seeds; the full panel's seed-to-seed agreement (`condition = full`) is
  the noise ceiling (`full_repeat` reaches it when the two embeddings are identical);
- `linear_cka`; for TERRA also `paired_cosine_median` (each cell with itself) and `cross_panel_transfer_*`: a kNN
  classifier trained on full-panel embeddings of two sections labels the subset-panel embeddings of the third (the use
  case of mapping a custom-panel study onto a 5K reference). `_centered` subtracts each panel's mean embedding first.

Between disjoint panels (`vs_partner`): the same metrics for two panels of the same size without a shared gene.
`full_repeat` vs `full` shows how much embeddings vary between two runs of the same input.

## Run on the farm

```bash
cd /nfs/team361/sb75/terra-reproducibility && git pull && cd analysis/panel_robustness && bash submit_gene_panel_subsampling.sh xhs1000_3sections_terra96m
```

Options after the run name go to `gene_panel_subsampling.py configure` (see `--help`), e.g. `--model_folder_path`,
`--batches`, `--panel_sizes`, `--n_replicates`. Steps: `prepare` (CPU, minutes), `embed` (GPU array, about 10-30 min
per element), `evaluate` (CPU array, about 15 min per element), `summarize` (CPU). Resubmitting the same run name skips
finished conditions; `FORCE=1` recomputes them.

Every job uses LSF group `s10396` (`LSF_GROUP`; `team361` does not exist on the GPU cluster). The embed array runs on
`inference` (`GPU_QUEUE`); prepare, evaluate and summarize run on the first of `normal`, `long` and `cpu-interactive`
that the cluster has, otherwise on the GPU queue with one GPU each (`CPU_QUEUE` sets it, e.g.
`CPU_QUEUE=long bash submit_gene_panel_subsampling.sh <run>`). If a submission fails, the jobs of the run that were
already submitted are removed. If the prepare job fails, it kills the jobs that wait for it (otherwise LSF keeps them
pending with "Dependency condition invalid or never satisfied"); its error is in
`logs/panel_robustness/<run>/prepare_<job id>.err`.

## Outputs (`artifacts/panel_robustness/<run>/`)

- `run_config.json`, `panels.json` (genes of every panel), `cells.tsv.gz` (labels, evaluation flag),
  `detected_genes_per_cell.tsv.gz`, `panel_statistics.tsv` (detected genes per cell, cells without genes, cells above
  the 256-token cell segment);
- `embeddings/<condition>.npz|json` (TERRA cell, neighborhood and spatial cell embeddings, provenance),
  `evaluation/<condition>/` (metrics, partitions, kNN indices, PCA embeddings);
- `results/metrics_long.tsv` (every value, per section and overall), `results/summary.tsv` (mean, sd over random
  draws), `results/headline.tsv` (one row per panel size), `results/figures/` (`biology_recovery`,
  `consistency_with_full_panel`, `disjoint_panels`, `panel_statistics`; PDF and PNG). `plot_gene_panel_subsampling.py
  --output_folder_path <run>` redraws the figures.

## Same panels, other models (`--panels_from`, `--conditions`)

`gene_panel_subsampling.py configure --panels_from <run folder>` reuses that run's panels, conditions and evaluation
cells (same silver data and sections), so that another model embeds exactly the same genes of the same cells;
`--conditions <id> ...` embeds and evaluates only some conditions (a run without `full` only tabulates the annotation
metrics). Example, TERRA-112M on the panels of the TERRA-96M run:

```bash
bash submit_gene_panel_subsampling.sh xhs1000_3sections_terra112m \
    --model_folder_path /nfs/team361/sb75/terra-reproducibility/artifacts/models/11012026_190057_836 \
    --panels_from /nfs/team361/sb75/terra-reproducibility/artifacts/panel_robustness/xhs1000_3sections_terra96m
```

## Comparing runs (`compare_panel_runs.py`)

Compares the embeddings of the cells that every given run evaluates, for the TERRA cell and neighborhood embeddings
(`--runs LABEL=RUN_FOLDER ...`; the first run gives sections and labels). Pairs (`pair_type`):
- `same_model_panel_vs_full`: a panel against the full panel of the same run (one model, one embedding space);
- `same_panel_across_models`: the same panel in two runs; two models trained on the full panel with different seeds
  give the training-seed ceiling of every comparison between models;
- `trained_panel_vs_full`: a model trained on a panel (its panel) against a full-panel model (full panel). Panel
  models are trained with seed 0, like the first full-panel model, and share its initialization (column
  `same_seed`); the figure compares them with the seed-1 full-panel model, as the training-seed ceiling does;
- `panel_vs_other_model_full`: a panel of one full-panel training run against the full panel of another (the
  zero-shot counterpart of `trained_panel_vs_full`: also two models with the same training data);
- `leiden_seed_ceiling`: Leiden seed-to-seed agreement of one embedding (ceiling of cluster agreement).

Metrics (`embedding_similarity.py`; per section and over all cells):
- one embedding space only (the same model): `cosine` / `cosine_median` (each cell with itself), `cosine_centered`
  (after subtracting each embedding's mean, which removes the common direction that makes most cells similar),
  `cosine_null` (a random other cell of the section) and `cosine_gap` (matched minus null: how much more similar a
  cell is to itself than to another cell); cross-panel re-identification within the section on a held-out half of
  the cells: `foscttm` (fraction of cells closer than the cell's own counterpart; 0 best, 0.5 random), `top1`,
  `top10`;
- any two runs (invariant to rotation and scale): `knn_overlap` (shared k nearest neighbors within the section),
  `linear_cka`, `representational_similarity` (Spearman correlation of pairwise cosine distances of 1,500 cells per
  section), `cluster_ari_kmatched` / `cluster_nmi_kmatched` (K-matched Leiden partitions, different seeds), and the
  matched-cell metrics after an orthogonal Procrustes alignment fit on one half of the cells and scored on the other:
  `procrustes_cosine`, `procrustes_foscttm`, `procrustes_top1`, `procrustes_top10`.

Raw cosine similarity of the same cell across panels is high for any two cells of these models (`cosine_null` is
high as well), so it is read against its null (`cosine_gap`) or as re-identification; across models only the
invariant and the aligned metrics are meaningful. Outputs: `pairs.tsv`, `similarity_long.tsv`,
`similarity_summary.tsv` (mean and sd over random draws), `biology_long.tsv` (every run's annotation metrics) and
`figures/` (`similarity_to_full_panel`, `biology_by_arm`; `plot_panel_comparison.py` redraws them).

Any runs can be compared, e.g. the TERRA-96M run on its own (output in
`artifacts/panel_training/terra96m_zero_shot/comparison`):

```bash
RUNS="terra96m=/nfs/team361/sb75/terra-reproducibility/artifacts/panel_robustness/xhs1000_3sections_terra96m" bash submit_panel_comparison.sh terra96m_zero_shot
```

## Models trained on a panel (`panel_training.py`)

The zero-shot runs above apply a model trained on all genes to fewer genes. The training arm trains TERRA from
scratch on a panel and compares it with the full-panel model and with zero-shot inference on the same panel:
- **Training data:** the sections of the reference run's dataset that it does not evaluate (xhs1000: 36 of 39; batches
  11, 19 and 32 stay held out), restricted to each panel's genes and tokenized exactly as the embed step tokenizes
  (`harmonize_adata` + `tokenize_adata` with the reference model folder, `data.pad_empty_neighbors`). All cells stay in
  the neighbor graph; rows (center cells) need at least 10 detected panel genes (the corpus rule of 10 vocabulary
  genes per cell). One dataset per panel: `/lustre/.../DATASETS/gold/panel_training/<name>/<panel>/dataset` with
  `data_record.json`.
- **Models:** the recipe of a TERRA training config (default: the job config
  `configs/job/hst_corpus_110m/hst_corpus_110m_terra96m.yaml` of the TERRA checkout and its CONFIG_FILE), trained from
  scratch on the full panel (seeds 0 and 1: the training-seed ceiling) and on random_1000_r0 to random_4000_r0 (nested
  panels of one draw; `--panels` adds e.g. disjoint panels). The data section is the reference model's (the settings
  the data was tokenized with; `make_configs` refuses any other tokenizer setting), with the recipe's data loader keys;
  the run folder is `<training folder>/runs/<model>`.
- **Evaluation:** each packaged model is a `gene_panel_subsampling.py` run `<name>_<model>` on the reference run's
  cells and panels: full-panel models on every panel (zero-shot with the same training data), panel models on their
  own panel. `compare_panel_runs.py` then compares all of them with the reference run (e.g. TERRA-96M).

Steps (from `analysis/panel_robustness`, after `cd /nfs/team361/sb75/terra-reproducibility && git pull`):

```bash
bash submit_panel_training_data.sh xhs1000_skin_panels
```

```bash
source /nfs/team361/sb75/.venvs/terra/bin/activate && python panel_training.py make_configs --training_folder_path /nfs/team361/sb75/terra-reproducibility/artifacts/panel_training/xhs1000_skin_panels --epochs 20
```

`make_configs` writes `configs/model|job/panel_training/xhs1000_skin_panels_<model>.yaml` into the TERRA checkout
(`--terra_dir`, default `$TERRA_REPO_DIR` or /nfs/team361/sb75/terra) and prints the commands that submit the
training jobs with TERRA's launcher (one per model; defaults: 4 GPUs, batch 144 per GPU, 32 cores, 256 GB,
training-parallel, group s10396, a checkpoint every 5 epochs; `--epochs` defaults to the recipe's and must be a
multiple of `--checkpoint_freq`). After training (re-run whenever more models have finished; finished parts are
skipped):

```bash
bash submit_panel_model_runs.sh xhs1000_skin_panels
```

```bash
bash submit_panel_comparison.sh xhs1000_skin_panels
```

The data and comparison jobs only need CPUs: on the main farm (farm22) they run on the normal queue with group team361;
on the GPU cluster (tiger, no CPU queue) on inference with group s10396 and one GPU each (the esubs accept only team
groups on the CPU queues and only cost-code groups on the AI acceleration queues; `LSF_GROUP` and `CPU_QUEUE`
override the choice). Training and `submit_panel_model_runs.sh` (embedding) need tiger's GPUs.
`python panel_training.py status --training_folder_path <training folder>` shows what is done. Results:
`<training folder>/comparison`. To resume a training run that stopped, set `state.load_checkpoint: true` and
`state.read_checkpoint: terra-ep<last finished epoch>.pth.tar` in its model config (state.folder_path is the run
folder already).

## Caveats

- In-silico subsetting removes genes but not the effects of a different chemistry, probe design, or optical crowding on
  detection; detection sensitivity is a separate experiment (rebuttal plan E13).
- Random panels are not designed panels: a custom panel concentrates on informative genes, so random panels of a size
  are a conservative stand-in for real panels of that size.
- The full-panel scores are this experiment's reference (K-matched resolutions, igraph Leiden); they are not the
  numbers of the published benchmark, which used hand-set resolutions.
- The panel-trained models see about a million cells of one dataset, TERRA-96M about 96 million of many: compare
  the panel-trained models with the full-panel model of the same training run, and TERRA-96M only as a reference.
- 23 of the 36 training sections were in TERRA-96M's training data (13 are held out from it), which does not matter
  for the panel-trained models but means TERRA-96M saw that tissue.
- Training rows need 10 detected panel genes, so smaller panels have fewer rows (`data_record.json`) and, with the
  same number of epochs, fewer training steps.
