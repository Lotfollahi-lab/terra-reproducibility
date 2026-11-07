# Ablation pipeline

The mouse ablations of the TERRA paper (Supp. Figs. 4–6; `sebastianbirk/stjepa-reproducibility`,
`analysis/mouse_ablation`) rerun from scratch on any labelled dataset with the current TERRA code (a pinned commit of
`Lotfollahi-lab/terra` with all fixes of the ablation code reviews switched on), the base config and the launch setup
of the mouse runs. A dataset is defined by one parameter file, `datasets/<dataset>.env`; the scripts have no
dataset-specific defaults. Two datasets are defined:
- `datasets/mouse.env` (runs first; [Mouse rerun](#mouse-rerun)): the mouse sections themselves, MERFISH and STARmap
  PLUS, with their region annotations; two sections, so the evaluation also measures their integration.
- `datasets/ovca.env` ([Steps on the farm (OVCA)](#steps-on-the-farm-ovca)): the SpatialFusion Xenium Prime 5K
  ovarian cancer section with its pathologist niche labels.

## Files

| File | Role |
|---|---|
| `datasets/<dataset>.env` | Parameters of one dataset (bash assignments; lists space-separated): `mouse.env`, `ovca.env`. |
| `ablation_plan.yaml`, `base_config.yaml` | The variants, tokenizations and replicates; the mouse base model config in TERRA's config format. |
| `setup_code.sh` | Pinned model code: TERRA `fefbf01` (`Lotfollahi-lab/terra` dev, 7 Nov 2026) at `CODE_PATH`, with `CODE_VERSION`. |
| `prepare_ovca_data.py` / `.lsf` | OVCA only: silver AnnData, normalization factors, token dictionary checks. |
| `filter_silver_cells.py` | Silver copies without the cells that have no detected gene of the token dictionary (mouse: step 3). |
| `generate_ablation_runs.py` | Run configs, `manifest.tsv` and `dataset.env` in the artifact folder. |
| `submit_tokenization.sh`, `tokenize_dataset.lsf` / `.py` | Tokenized datasets of the plan (job array). |
| `check_setup.py`, `check_mouse_norm_factors.py` | Checks before launch: pinned code, parity with the mouse configs, special values, labels and sections, datasets, factors. |
| `submit_ablation_array.sh`, `run_ablation_array.lsf` | Training, packaging (`package_run.py`) and evaluation (`evaluate_ablation_run.py`), one element per run. |
| `compare_ablations.py` | Results tables and figures (niche identification; batch integration with two or more sections). |
| `norm_factors.py`, `compute_norm_factors.py` | Per-gene normalization factors with the tokenizer's constants. |
| `CODE_REVIEW.md` | Code review of the mouse ablations and where each fix is in TERRA. |

## Dataset parameter file

| Variable | Used by | Mouse | OVCA |
|---|---|---|---|
| `DATASET_NAME` | silver folder and tokenized dataset names, metric file names | `mmb0-1b_smb1-1b_2p_coord_aligned` | `xho50-1b_1p_spatialfusion` |
| `EXPERIMENT_TAG` | wandb run ids `<tag>_<manifest index>_<config hash>`, job names | `mouse` | `ovca` |
| `DATA_DIRECTORY_PATH` | `silver/<DATASET_NAME>/` (tokenizer input), `gold/` (tokenized datasets), `cache/` | `/lustre/.../DATASETS` | `/lustre/.../DATASETS` |
| `SILVER_FILE_PATHS` | all h5ad files of the silver folder | the h5ad files of the folder when sourced (2) | 1 file |
| `LABEL_FILE_PATHS` (optional) | evaluation: files with `obs['cell_id']` and the labels, one per section (the batch of the integration metrics); joined to the tokenized cells by `cell_id` | not set (default: the silver files) | not set |
| `TOKEN_DICTIONARY_FILE_PATH`, `NORM_FACTOR_FILE_PATH` | tokenizer and run folders | mouse dictionary, new factors of the two sections | human dictionary, section factors |
| `LABEL_KEYS`, `UNASSIGNED_LABEL` | evaluation: metrics per key over the cells it labels | `ccf_region_name Sub_molecular_tissue_region`, placeholder (set in step 5) | `path_region`, `Unassigned` |
| `ARTIFACT_FOLDER_PATH` | configs, manifest, runs, evaluations, logs, results, figures of the campaign | `.../artifacts/mouse_ablation` | `.../artifacts/ovca_ablation` |
| `EPOCHS`, `WARMUP`, `N_SPECIAL_VALUES` | run configs | 300, 3, 200 | 82, 1, 1448 |
| `PLAN_FILE_PATH`, `BASE_CONFIG_FILE_PATH` | run configs | the files of this folder | the files of this folder |
| `CODE_PATH`, `ENV_PATH` | pinned TERRA code and the TERRA venv that runs it | as OVCA | `/nfs/team361/sb75/terra_ablation`, `.../.venvs/terra` |
| `MOUSE_ARTIFACT_FOLDER_PATH`, `MOUSE_TOKEN_DICTIONARY_FILE_PATH` | checks against the mouse runs | as OVCA | |
| `BRONZE_FILE_PATH`, `GENE_MAPPING_DICT_FILE_PATH` | OVCA data preparation only | | |

The submit scripts and `setup_code.sh` take the env file as their first argument. The LSF jobs get their values through
`bsub -env`: the tokenization jobs the exported values of the env file (`-env "all"`, frozen at submission), the job
array `ARTIFACT_FOLDER_PATH` and `ENV_PATH`, and `prepare_ovca_data.lsf` the path of the env file. The Python scripts
get every value as a command line argument (`LABEL_FILE_PATHS` as `${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}}`; the env
files set shell variables, so use a fresh shell per dataset). `generate_ablation_runs.py` writes
the values the job array needs into `<ARTIFACT_FOLDER_PATH>/dataset.env` (dataset, silver and label files, label keys,
unassigned label, norm factors, tag, `CODE_PATH` and `CODE_VERSION`), so the artifact folder defines the campaign: every
array element sources it and stops if `${CODE_PATH}/CODE_VERSION` is not the version the configs were generated for,
and `check_setup.py` fails if it differs from the env file (regenerate after every change of the env file).
`CODE_VERSION` is also a top-level key of every run config (the training code ignores it), so the config hashes, and
with them the run folders, change with the code.

## What is run

The variants and replicates are listed in `ablation_plan.yaml`:
- 51 variants, 95 training runs.
- Every variant keeps its mouse method name, and `mouse_runs` lists the mouse run timestamps. One arm is new:
  `metatoken_dropout` (no mouse run).

| Figure | Category | Variants | Replicates |
|---|---|---|---|
| Supp. Fig. 4 | base (shared reference bar) | cell graph, combined, MLP, segment, 64 genes, mask 0.6, no normalization | 3 |
| Supp. Fig. 4 | tokenizer | cell graph rank / counts; cell neighborhood combined / rank / counts | 3 |
| Supp. Fig. 4 | sequence length | 16, 32, 128, 256 genes per cell | 3 |
| Supp. Fig. 4 | expression encoding | 20 / 50 / 100 value bins | 3 |
| Supp. Fig. 4 | spatial encoding | coordinate (`cell_pos_enc: coord`) | 3 |
| Supp. Fig. 4 | masking ratio | 0.3, 0.4, 0.5, 0.7, 0.8, 0.9 | 3 |
| Supp. Fig. 6 | metatoken | no metatoken; metatoken dropout (new: metatoken removed from a random half of the training batches) | 3 |
| Supp. Fig. 5 | normalization | 21 normalization tuples, in the three mouse groups | 1 |
| none | architecture | width 192 / 384 / 960 / 1152; 1, 4 or 5 encoder layers; 3 predictor layers | 1 |

Replicate `r` is trained with seed `1000 * r`. TERRA's opt-in fixes (`tokenizer_options` and `training_options` of
the plan) are on for every run; `CODE_REVIEW.md` lists them.

**Evaluation** follows the mouse protocol:
- Neighborhood embeddings from the target encoder, of the run's own tokenized training dataset.
- kNN graph with k = 15 on all embedded cells (numba compiled for a generic CPU, `NUMBA_CPU_NAME=generic`, so the
  graph does not depend on the node).
- Leiden on the fixed grid of resolutions 0.6–1.4. The grid is identical for all variants, so the protocol is
  label-free.
- NMI and ARI against each label key, over the cells it labels: cells without the key (e.g. a section-specific
  annotation) or with `UNASSIGNED_LABEL` are clustered but not scored. Labels are joined from the label files
  (`LABEL_FILE_PATHS`, default: the silver files) by `cell_id`.
- Batch integration, if the cells come from two or more sections (one label file = one section; the mouse
  MERFISH and STARmap PLUS sections): the metrics of the mouse notebook (`stjepa_v2_ablation.ipynb`, cells 21 and 42),
  seeded, over the labelled cells (a label for at least one key) as in the notebook. iLISI (`scib_metrics.ilisi_knn`
  on a 90-NN `pynndescent` graph; higher is better), ASW (silhouette of the sections on 10,000 cells; 0 is best) and
  MMD (`mmd_comparable`: RBF MMD² of the first two sections on standardized embeddings, median-heuristic width; lower
  is better). Unlike the notebook, which removed unlabelled cells before embedding, the embeddings come from the
  all-cell pass, and the subsamples are seeded.
- Models with a metatoken are evaluated twice, with all metrics: metatoken removed at inference (as in the mouse
  evaluation; the category panels) and kept as in training (`keep_metatoken` outputs; panel `metatoken_inference`).
  For the mouse sections, this tests how the metatoken at inference affects the MERFISH–STARmap integration.
- Also saved: float32 embeddings, cluster assignments (with the section of every cell) and the average cosine
  similarity of the embeddings, raw (as in the mouse runs; it saturates near 0.99 for healthy embeddings) and
  mean-centered.
- `evaluation_complete` records the label keys, label files and unassigned label; an element skips a run only if they
  are the campaign's. A run evaluated without its batch-integration metrics (e.g. by an earlier version) is evaluated
  again, and so is every mode if the label files or the unassigned label changed.

## Mouse rerun

The mouse sections of the original ablations, rerun with this pipeline (`datasets/mouse.env`): MERFISH (batch82) and
STARmap PLUS (batch15), coordinate-aligned, the training data of the mouse base config. Labels: `ccf_region_name`
(MERFISH) and `Sub_molecular_tissue_region` (STARmap PLUS), the region keys of the mouse evaluation. Same plan, base
config, pinned code and launch setup as OVCA, with the mouse schedule (300 epochs, 3 warmup) and `n_special_values`
200. The mouse silver data exists; new are a filtered copy of the two section files (`DATASET_NAME`
`mmb0-1b_smb1-1b_2p_coord_aligned_nonempty`: without cells that have no detected gene of the mouse token dictionary,
which would shift the neighbor segments of the cells around them; the source files stay as they are) and its norm
factors.

Run from this folder in bash (in zsh, `setopt sh_word_split` first), in one shell; in a new shell, first rerun the
`source` lines of step 3. Wait where noted.

```bash
# 1. Code
cd /nfs/team361/sb75/terra-reproducibility && git pull && cd analysis/ablation/ablation_pipeline

# 2. Pinned TERRA code at /nfs/team361/sb75/terra_ablation, shared by all datasets, and the check of the TERRA
#    environment (/nfs/team361/sb75/.venvs/terra: torch, datasets, scanpy, squidpy, scib_metrics, leidenalg, wandb;
#    if an import fails, install it into that venv). If the copy is there, only the environment check runs.
bash setup_code.sh datasets/mouse.env

# 3. Silver files and norm factors (92,520 cells, a few minutes; run it interactively, e.g. in a bsub -Is job).
#    First the two section files without the cells that have no detected gene of the mouse token dictionary, into the
#    dataset's own silver folder (prints how many it removes; the source files are not modified), then the factors of
#    the filtered files (new file, tokenizer constants). compute_norm_factors.py writes <norm factors>.done last; the
#    tokenization waits for it.
source datasets/mouse.env
ls "${DATA_DIRECTORY_PATH}/silver/${SOURCE_DATASET_NAME}"   # the two section files of SOURCE_SILVER_FILE_PATHS
source "${ENV_PATH}/bin/activate" && export PYTHONPATH="${CODE_PATH}/src"
python filter_silver_cells.py --source_silver_file_paths ${SOURCE_SILVER_FILE_PATHS} \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" \
    --output_folder_path "${DATA_DIRECTORY_PATH}/silver/${DATASET_NAME}"
source datasets/mouse.env     # again: SILVER_FILE_PATHS lists the filtered files once they exist
echo "${SILVER_FILE_PATHS}"   # two files, MERFISH and STARmap PLUS
python compute_norm_factors.py --silver_file_paths ${SILVER_FILE_PATHS} --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}"

# 4. Run configs, manifest.tsv and dataset.env (95 runs; 300 epochs, 3 warmup)
python generate_ablation_runs.py --plan_file_path "${PLAN_FILE_PATH}" --base_config_file_path "${BASE_CONFIG_FILE_PATH}" \
    --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}" \
    --dataset_name "${DATASET_NAME}" --silver_file_paths ${SILVER_FILE_PATHS} \
    --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" --experiment_tag "${EXPERIMENT_TAG}" \
    --epochs "${EPOCHS}" --warmup "${WARMUP}" --n_special_values "${N_SPECIAL_VALUES}" --code_path "${CODE_PATH}"

# 5. Labels and sections, before tokenization. Review check 3: per label file the cells, uns dataset id and batch,
#    label keys and their most frequent values. Each key must be in the file of its section; if the silver files lack
#    them, set LABEL_FILE_PATHS in datasets/mouse.env (e.g. the commented evaluation files of the mouse notebook).
#    Set UNASSIGNED_LABEL (a placeholder) to the value of unannotated cells, if there is one. Check 2 must show that
#    the batch special values fit (200 rows). Check 5 reports how many cells the raw-count ranking cuts inside tied
#    counts (ordered by corpus mean). After a change: source datasets/mouse.env again, rerun steps 4 and 5.
python check_setup.py --plan_file_path "${PLAN_FILE_PATH}" --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" \
    --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}" \
    --silver_file_paths ${SILVER_FILE_PATHS} --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" \
    --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --code_path "${CODE_PATH}" \
    --mouse_token_dictionary_file_path "${MOUSE_TOKEN_DICTIONARY_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}" --skip_datasets

# 6. Tokenization: all 27 datasets in the standard class. WAIT until all elements have finished
#    (bjobs -J mouse_tokenize_standard)
bash submit_tokenization.sh datasets/mouse.env standard

# 7. Checks (fails on errors, e.g. tokenized cells that are not in the label files; review the config parity output)
python check_setup.py --plan_file_path "${PLAN_FILE_PATH}" --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" \
    --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}" \
    --silver_file_paths ${SILVER_FILE_PATHS} --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" \
    --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --code_path "${CODE_PATH}" \
    --mouse_token_dictionary_file_path "${MOUSE_TOKEN_DICTIONARY_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}"

# 8. Pilot: base_rep1 (train + evaluate, about 3 h; wandb id mouse_001_<config hash>). WAIT until it has finished and
#    check eval/base/base_rep1_<hash>/: niche and batch-integration metrics of both metatoken modes, evaluation_complete
bash submit_ablation_array.sh datasets/mouse.env 1

# 9. All 95 runs (the pilot is skipped): 2 elements x 4 GPUs = 8 GPUs at a time (the default %2); never two arrays at once
bash submit_ablation_array.sh datasets/mouse.env
#    or 4 GPUs at a time (one run at a time, about twice the wall time; every run keeps its 4 GPUs, so the batch size
#    and schedule are unchanged): bash submit_ablation_array.sh datasets/mouse.env "" 1
#    The limit of a submitted array can be changed at any time: bmod -J "%1" <JOBID> (or "%2")

# 10. Results and figures (also works on a partial campaign; lists the missing runs)
python compare_ablations.py --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" --dataset_name "${DATASET_NAME}" \
    --label_keys ${LABEL_KEYS}
```

- **Runs:** 95 (51 variants), in the order and with the seeds of the OVCA campaign.
- **Runtime:** 722 iterations per epoch on 4 GPUs, 300 epochs (216,600 steps): about 3 h per run for the base config,
  so about 6 days for all 95 runs at 2 at a time (95 × 3 h / 2; about 12 days at 1 at a time), plus evaluation and
  the slower arms (128 and 256
  genes, widths 960 and 1152). Evaluation per run: two embedding passes, then per pass a kNN graph, nine Leiden runs
  and the integration metrics (about 2.5 min per pass at this size, measured on synthetic data: mostly the
  single-threaded 90-NN graph of iLISI; MMD about 25 s, 1.6 GB). Tokenization: 27 small jobs (92,520 cells).
- **Storage:** about 125 GB in the artifact folder (embeddings 284 MB per set, two sets for 92 runs: 53 GB;
  checkpoints 71 GB).
- **Outputs** under `/nfs/team361/sb75/terra-reproducibility/artifacts/mouse_ablation/`: `configs/`, `manifest.tsv`,
  `dataset.env`, `runs/<run>/`, `logs/` and
  - `eval/<variant>/<run>/`, per metatoken mode (prefix `mmb0-1b_smb1-1b_2p_coord_aligned_`, kept metatoken
    `mmb0-1b_smb1-1b_2p_coord_aligned_keep_metatoken_`): `neighborhood_emb_niche_identification_metrics_res_sweep.csv`
    (both label keys), `neighborhood_emb_batch_integration_metrics.csv` (`metric,score`: iLISI, ASW, MMD between
    MERFISH and STARmap PLUS), cosine similarity, clusters (with the section of each cell) and embeddings;
  - `results/`: `niche_identification_ablations{,_summary}.csv`, `batch_integration_ablations{,_summary}.csv`;
  - `figures/`: `ablation_<category>_<label key>_<metric>`, `ablation_<category>_batch_integration_<metric>` and the
    `ablation_metatoken_inference_*` panels (metatoken removed vs kept for the base and metatoken dropout runs, next to
    no metatoken).
- **Differences from the original mouse runs:** those of the OVCA table below that are not specific to OVCA (tokenizer
  fixes, padded masks, seeds 1000/2000/3000, latest checkpoint only, the run's own tokenized dataset with all cells
  embedded, metatoken removed and kept at inference), plus norm factors computed on the two sections with the
  tokenizer's constants instead of the mouse cohort file. Schedule and `n_special_values` are the mouse values.
- **Check on the farm:** that `silver/mmb0-1b_smb1-1b_2p_coord_aligned/` holds exactly the two sections (the base
  config's `raw_data_folder_path` names `silver/mmb0-1b_smb1-1b_2p`, its tokenized data `..._coord_aligned`), that the
  label keys are in them (else `LABEL_FILE_PATHS`), and the unassigned value (step 5).

## Steps on the farm (OVCA)

Run from this folder in bash (the unquoted list variables must split into words; in zsh, `setopt sh_word_split`
first); wait where noted. For another dataset, replace `datasets/ovca.env` (and the OVCA data preparation, see the
last section).

```bash
cd /nfs/team361/sb75/terra-reproducibility && git pull && cd analysis/ablation/ablation_pipeline

# 1. Pinned TERRA code at /nfs/team361/sb75/terra_ablation and environment check (rerun: only the check; stops if
#    another code version is there: move it away first)
bash setup_code.sh datasets/ovca.env

# 2. Data preparation (OVCA only, TERRA env): silver AnnData and norm factors. WAIT until the job has finished
#    (bjobs -J prepare_ovca_data; the log ends with the token dictionary checks)
mkdir -p ../../../logs/ablation/ovca_ablation
bsub -env "all,DATASET_ENV_FILE_PATH=datasets/ovca.env" < prepare_ovca_data.lsf

# 3. Run configs, manifest.tsv and dataset.env (95 runs; 82 epochs, 1 warmup)
source datasets/ovca.env
source "${ENV_PATH}/bin/activate" && export PYTHONPATH="${CODE_PATH}/src"
python generate_ablation_runs.py --plan_file_path "${PLAN_FILE_PATH}" --base_config_file_path "${BASE_CONFIG_FILE_PATH}" \
    --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}" \
    --dataset_name "${DATASET_NAME}" --silver_file_paths ${SILVER_FILE_PATHS} \
    --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" --experiment_tag "${EXPERIMENT_TAG}" \
    --epochs "${EPOCHS}" --warmup "${WARMUP}" --n_special_values "${N_SPECIAL_VALUES}" --code_path "${CODE_PATH}"

# 4. Tokenization: all 27 datasets in the standard class (refuses to submit until step 2 has written <norm factors>.done, its last output). WAIT until
#    all elements have finished (bjobs -J ovca_tokenize_standard)
bash submit_tokenization.sh datasets/ovca.env standard
bash submit_tokenization.sh datasets/ovca.env large    # nothing to submit with the plan's tokenizer options

# 5. Checks (check_setup.py fails on errors; review its config parity output; the second is a report on the mouse
#    gcrd arms)
python check_setup.py --plan_file_path "${PLAN_FILE_PATH}" --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" \
    --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}" \
    --silver_file_paths ${SILVER_FILE_PATHS} --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" \
    --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --code_path "${CODE_PATH}" \
    --mouse_token_dictionary_file_path "${MOUSE_TOKEN_DICTIONARY_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}"
python check_mouse_norm_factors.py --plan_file_path "${PLAN_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}"

# 6. Pilot: base_rep1 (train + evaluate). WAIT until it has finished and check its eval folder
# (submit_ablation_array.sh refuses while an array of the same campaign is pending or running)
bash submit_ablation_array.sh datasets/ovca.env 1

# 7. All 95 runs (finished runs are skipped); never two arrays at once
bash submit_ablation_array.sh datasets/ovca.env

# 8. Results and figures (also works on a partial campaign; lists the missing runs)
python compare_ablations.py --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" --dataset_name "${DATASET_NAME}" \
    --label_keys ${LABEL_KEYS}
```

- **Training and evaluation:** one array element per manifest row, 4 GPUs each, at most 2 running at a time (8 GPUs).
- **Launch order:** all first replicates (base first, figure order), then the second and third replicates.
- **wandb:** run id `<EXPERIMENT_TAG>_<manifest index, 3 digits>_<config hash>` (e.g. `ovca_042_1a2b3c4d`), display
  name the descriptive run id (`<run name>_<config hash>`, also the run folder name). The index follows the plan, so
  regenerating after changing the plan gives the runs new wandb ids.
- **Skips and retries:**
  - An element skips training if its run is finished, and skips everything if its evaluation is complete
    (`evaluation_complete`).
  - It stops with an error, instead of retraining, if the latest checkpoint cannot be read, if the pinned code has
    another `CODE_VERSION` than the configs, or if its manifest row is missing or malformed.
  - Retry a failed element with `brequeue -e "<JOBID>[<INDEX>]"`; it resumes from its latest checkpoint.
  - Change concurrency with `bmod -J "%<N>" <JOBID>`.
- **Outputs** under `ARTIFACT_FOLDER_PATH` (OVCA: `/nfs/team361/sb75/terra-reproducibility/artifacts/ovca_ablation/`):
  `configs/`, `manifest.tsv`, `dataset.env`, `runs/<run>/`, `eval/<variant>/<run>/`, `logs/`, `results/`, `figures/`
  (`ablation_<category>_<label key>_<metric>`; one section, so no batch-integration metrics).

## Runtime and storage (OVCA)

| Item | Mouse | OVCA |
|---|---|---|
| Cells in training data | 92,520 | 338,695 |
| Iterations per epoch on 4 GPUs | 722 | 2,646 (3.66× mouse) |
| Training schedule | 300 epochs (216,600 steps) | 82 epochs (216,972 steps) |
| Time per run | about 3 h | about 3 h for the base config |
| All 95 runs, 2 at a time | — | about 6 days of training (95 × 3 h / 2), plus evaluation |

- **Slower arms:** the 128- and 256-gene arms (2× and 4× the tokens of the base) and the widths 960 and 1152 take
  longer than the base, so plan for more than 6 days.
- **Evaluation** runs in the same element after training: two embedding passes (metatoken removed and kept), a kNN
  graph and nine Leiden runs per pass, on all 338,695 cells.
- **Epoch-matched alternative:** `EPOCHS=300`, `WARMUP=3` takes about 8–11 h per run and 2.5–3 weeks in total.
- **Tokenization:** all 27 datasets in the standard class (200 GB). Measured peak memory: about 83 GB for the base,
  dense-path and 256-gene datasets and about 105 GB for Seurat v3 and Pearson residuals. The large class (1 TB,
  hugemem) is only needed without the plan's ranked-list cut (`stable_rank_ties` / `zero_expr_from_raw_counts`).
- **Storage**, computed with the base width (`enc_emb_dim` 768):
  - Embeddings: 768 × 338,695 cells × 4 bytes = 1.04 GB per set, two sets per run with a metatoken (92 of 95 runs),
    one for the 3 runs without: about 195 GB.
  - Checkpoints: about 0.75 GB per run (only `terra-latest.pth.tar`): about 71 GB.
  - Total about 265 GB in the artifact folder, plus a few GB of cluster tables and logs.
  - Tokenized datasets go to `/lustre/.../DATASETS/gold/*/xho50-1b_1p_spatialfusion/`.

## OVCA: differences from the mouse runs, and why

| Item | Mouse | OVCA | Reason |
|---|---|---|---|
| Code | ST-JEPA v2 (`sebastianbirk/stjepa` `b433d5e`), the farm checkout of the time | copy of TERRA `fefbf01` (`setup_code.sh`), the mouse base config in TERRA's config format, `attention_backend: torch` (PyTorch chooses the SDPA kernel, as in the mouse runs and the released models) | The figure is rerun on the current code. TERRA holds every fix of the reviews (`CODE_REVIEW.md`): opt-in seed, atomic latest checkpoint, metatoken fixes, padded masks, resumption on all ranks and the opt-in tokenizer fixes; the options are switched on in the plan. A separate copy, so changes to the farm checkout (`/nfs/team361/sb75/terra`) cannot change running jobs. |
| Tokenization | ties at the top-64 cut broken by gene order and sort algorithm; Seurat / Pearson ranks include undetected genes; dense-path cells all stored with the full token count, so their batches were not grouped by size; Seurat v3 LOESS span silently 0.75; cell neighborhood aggregated over reverse kNN | `tokenizer_options` of the plan: deterministic ties (rank score, then count value on the dense path, then ascending corpus mean `mean` of the norm factor file, so rarer and lower-expressed genes first, then token id), detected genes only, Seurat v3 LOESS span 0.3 (gene-order independent), each cell's own kNN for the cell neighborhood; TERRA always counts the real gene tokens for the sampler | `CODE_REVIEW.md` issues 3, 5, 7, 9 and 10. TERRA cuts ranked lists at the input size (lossless), so all 27 datasets run in the standard memory class. Dataset names carry the options (`_tiesbymean_detected`, `_span0p3` for Seurat, `_ownknn_cell64`). With raw-count ranks (the base and every arm whose rank is not normalized) the cut at 64 genes falls inside tied counts in many cells; `check_setup.py` (check 5) reports how often. The corpus-mean order also makes the base keep the same genes as the "Mean" arm with unnormalized ranks (counts divided by `mean`), so the two differ only in the values. |
| Context and target masks | trimmed to the shortest in each batch: the sparsest cell sets how many tokens every cell uses | padded to the longest (`training_options`: `mask.pad_masks_to_batch_max`), with padded entries masked in attention and loss | `CODE_REVIEW.md` issue 8: trimming dropped 3.4% of the sampled tokens at 64 genes per cell and 20% at 256 on OVCA, biasing the sequence-length comparison. |
| Metatoken at inference | removed | removed (category panels) and kept (`metatoken_inference` panel) | Training always sees the metatoken; dropping it at inference is out of distribution unless the model was trained with metatoken dropout. |
| Schedule | 300 epochs, 3 warmup (216,600 steps) | 82 epochs, 1 warmup (216,972 steps) | Matched in optimizer steps: OVCA has 3.66× the iterations per epoch. All epoch-based schedules (LR, WD, EMA) scale with it. |
| Replicate seeds | all seed 0 (`_GLOBAL_SEED`, inherited from the I-JEPA training script), so replicates differed only by GPU non-determinism | seeds 1000, 2000, 3000 | Real seed variance (rebuttal E10): the seed sets weight initialization, data order, masks and segment sampling. The spacing keeps the samplers' per-epoch seeds (seed + epoch) disjoint. |
| `n_special_values` | 200 | 1448 | The human batch special value (about 213) exceeds 200 rows and would crash training. |
| Checkpoints | every 5 epochs (about 90 GB per run) | only `terra-latest.pth.tar`, in `runs/<run>_<config hash>`; killed runs resume on all ranks (the launch config sets `state.load_checkpoint` once the file exists) | Storage and walltime safety. A changed config (or code version) gets a new folder instead of resuming an old one. |
| Evaluation input | separate data, re-tokenized with `harmonize_tokenize_embed_pipeline`; unlabelled cells removed first | the run's own tokenized training dataset (`embed_dataset`); all cells, so `Unassigned` stays in the kNN graph | Same section and identical tokens; re-tokenizing broke top-64 ties non-deterministically (and needed about 0.5 TB per dense-path arm without the ranked-list cut). Absolute metric levels are therefore not comparable with the mouse numbers. |
| Norm factors | cohort file of the mouse data | computed on the section (`prepare_ovca_data.py`) with the tokenizer's gene-corrected read depth constants | The factors must cover every gene and match the tokenizer. |
| Dataset metadata | — | dataset id 50, batch0 (same 10x section as `xho50-1b_1p`) | The token dictionary only has special values for existing batches. |

**Inferred from run names; verified by `check_setup.py` against the mouse `params.yaml` before launch:**
- The cell-neighborhood layout: 64 cell + 64 neighborhood tokens (input size 128, `seq_len_cell: 64` in the plan, as in
  later v2 tokenizers). The dataset class needs exactly that many tokens.
- `pred_emb_dim` following `enc_emb_dim` in the width ablation.
- The coordinate arm being `cell_pos_enc: coord`.

**Caveats:**
- **Not held out from TERRA-112M pretraining.** This section is `xho50-1b_1p` in the TERRA-112M corpus. That doesn't
  matter here, because every arm is trained from scratch on this section alone.
- **Known code issues** are listed in `CODE_REVIEW.md` with their resolution. The OVCA tokenizations use the tokenizer
  fixes, which the mouse tokenizations lacked, so arm-level differences between mouse and OVCA can partly reflect them
  (most for the Seurat / Pearson and cell-neighborhood arms).
- **Schedules end at 80%:** `ipe_scale: 1.25` (the V-JEPA value; I-JEPA's configs use 1.0) stretches the LR, weight
  decay and EMA schedules over 1.25 x epochs, and training stops after `epochs`. They end at LR 2.39e-5 (configured
  2e-5), weight decay 0.366 (0.4) and EMA 0.999 (1.0), as in the mouse runs and TERRA-112M/96M. `check_setup.py` prints
  these values for Methods.
- **Gene-corrected read depth on one section:** with a single panel the gene correction is a constant, so the "Gene
  Corr. RD" arms are read-depth arms here (the mean / nonzero-mean ones are exactly read-depth-then-gene-mean; the
  shifted-log one normalizes to about 400 counts instead of 10,000). The panel-size correction needs two or more
  panels.

## Adding a dataset

1. Prepare the silver files into `<DATA_DIRECTORY_PATH>/silver/<DATASET_NAME>/` (raw counts, `var['ensembl_id']`,
   `obs['cell_id']`, the label keys, `uns` dataset id and batch with special values in the token dictionary). The
   tokenizer reads every h5ad file in that folder; `SILVER_FILE_PATHS` must list exactly those files. Labels may
   instead come from separate files (`LABEL_FILE_PATHS`, one per section, with the same `obs['cell_id']`). Every cell
   needs a detected gene of the token dictionary (`check_setup.py`, check 2): a cell without one has empty ranked
   gene lists, and on the sparse ranking path (most arms) the cells around it get shifted segments and lose their
   farthest neighbors. Remove such cells while preparing the data, or write filtered copies with
   `filter_silver_cells.py` as for the mouse sections.
2. Write the normalization factors of the cohort with `compute_norm_factors.py` (tokenizer constants).
3. Copy `datasets/mouse.env` (several sections) or `datasets/ovca.env` to `datasets/<dataset>.env` and set the values:
   a new `ARTIFACT_FOLDER_PATH` and `EXPERIMENT_TAG`, `EPOCHS` / `WARMUP` matched to the mouse runs in optimizer steps,
   `N_SPECIAL_VALUES` covering the batch special values (`check_setup.py` checks it), `LABEL_KEYS` (one or more; a key
   may cover only some files). With two or more label files, the evaluation adds the batch-integration metrics.
4. Follow the mouse steps with the new env file (`setup_code.sh` only reruns the environment check if the pinned code
   is already there).
