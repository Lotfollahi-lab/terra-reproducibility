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

## Caveats

- In-silico subsetting removes genes but not the effects of a different chemistry, probe design, or optical crowding on
  detection; detection sensitivity is a separate experiment (rebuttal plan E13).
- Random panels are not designed panels: a custom panel concentrates on informative genes, so random panels of a size
  are a conservative stand-in for real panels of that size.
- The full-panel scores are this experiment's reference (K-matched resolutions, igraph Leiden); they are not the
  numbers of the published benchmark, which used hand-set resolutions.
