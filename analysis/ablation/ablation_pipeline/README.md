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
| `setup_code.sh` | Pinned model code: TERRA `d29e262` (`Lotfollahi-lab/terra` dev, 8 Nov 2026) at `CODE_PATH`, with `CODE_VERSION`. |
| `prepare_ovca_data.py` / `.lsf` | OVCA only: silver AnnData, normalization factors, token dictionary checks. |
| `filter_silver_cells.py` | Silver copies without the cells that have no detected gene of the token dictionary (mouse: step 3). |
| `filter_shared_genes.py` | Evaluation data of a dataset with two or more sections: the silver files restricted to the genes of every section (mouse: step 3). |
| `generate_ablation_runs.py` | Run configs, `manifest.tsv` and `dataset.env` in the artifact folder. |
| `submit_tokenization.sh`, `tokenize_dataset.lsf` / `.py` | Tokenized datasets of the plan (job array). |
| `check_setup.py`, `check_mouse_norm_factors.py` | Checks before launch: pinned code, parity with the mouse configs, special values, labels and sections, datasets, factors, schedule ends, cells with few genes. |
| `check_attention_kernel.py` | 1-GPU check of the attention kernel for a context of padding only (mouse: step 8). |
| `submit_ablation_array.sh`, `run_ablation_array.lsf` | Training, packaging (`package_run.py`) and evaluation (`evaluate_ablation_run.py`), one 4-GPU element per run that needs work. |
| `submit_evaluation_array.sh`, `evaluate_ablation_array.lsf` | Packaging and evaluation of trained runs, one 1-GPU element per run whose evaluation is missing or stale. |
| `list_pending_runs.py` | Run by both submit scripts on the login node: which rows of the range need training or evaluation, and why. |
| `ablation_job_setup.sh` | Setup shared by both arrays (sourced): campaign checks, manifest row, evaluation arguments. |
| `compare_ablations.py` | Results tables and figures: niche identification and granularity, batch integration and cross-section mixing (two or more sections), collapse flags, metatoken contrasts. |
| `norm_factors.py`, `compute_norm_factors.py` | Per-gene normalization factors with the tokenizer's constants. |
| `CODE_REVIEW.md` | Code review of the mouse ablations and where each fix is in TERRA. |
| `tie_order_diagnostic.py` / `.lsf` | Whether a trained model uses the order of tied genes ([Tie-order diagnostic](#tie-order-diagnostic)). |

## Dataset parameter file

| Variable | Used by | Mouse | OVCA |
|---|---|---|---|
| `DATASET_NAME` | silver folder and tokenized dataset names, metric file names | `mmb0-1b_smb1-1b_2p_coord_aligned` | `xho50-1b_1p_spatialfusion` |
| `EXPERIMENT_TAG` | wandb run ids `<tag>_<manifest index>_<config hash>`, job names | `mouse` | `ovca` |
| `DATA_DIRECTORY_PATH` | `silver/<DATASET_NAME>/` (tokenizer input), `gold/` (tokenized datasets), `cache/` | `/lustre/.../DATASETS` | `/lustre/.../DATASETS` |
| `SILVER_FILE_PATHS` | all h5ad files of the silver folder | the h5ad files of the folder when sourced (2) | 1 file |
| `LABEL_FILE_PATHS` (optional) | evaluation: files with `obs['cell_id']` and the labels, one per section (the batch of the integration metrics); joined to the tokenized cells by `cell_id` | not set (default: the silver files) | not set |
| `TOKEN_DICTIONARY_FILE_PATH`, `NORM_FACTOR_FILE_PATH` | tokenizer and run folders | mouse dictionary, new factors of the two sections | human dictionary, section factors |
| `LABEL_KEYS`, `UNASSIGNED_LABEL` | evaluation: metrics per key over the cells it labels; `UNASSIGNED_LABEL` is one value for every key or `<key>=<value>` pairs separated by `;`, one per key (values may contain spaces, not `;`) | `ccf_region_name Sub_molecular_tissue_region`, placeholder (set in step 5) | `path_region`, `Unassigned` |
| `ARTIFACT_FOLDER_PATH` | configs, manifest, runs, evaluations, logs, results, figures of the campaign | `.../artifacts/mouse_ablation` | `.../artifacts/ovca_ablation` |
| `EPOCHS`, `WARMUP`, `N_SPECIAL_VALUES` | run configs | 300, 3, 200 | 82, 1, 1448 |
| `PLAN_FILE_PATH`, `BASE_CONFIG_FILE_PATH` | run configs | the files of this folder | the files of this folder |
| `CODE_PATH`, `ENV_PATH` | pinned TERRA code and the TERRA venv that runs it | as OVCA | `/nfs/team361/sb75/terra_ablation`, `.../.venvs/terra` |
| `MOUSE_ARTIFACT_FOLDER_PATH`, `MOUSE_TOKEN_DICTIONARY_FILE_PATH` | checks against the mouse runs | as OVCA | |
| `EVAL_GENE_SET` | evaluation data: `shared_genes` (the same cells restricted to the genes of every section, as the published mouse evaluation) or `all_genes` (the training data) | `shared_genes` | `all_genes` (one section) |
| `BRONZE_FILE_PATH`, `GENE_MAPPING_DICT_FILE_PATH` | OVCA data preparation only | | |
| `RUN_LIMIT`, `SEPARATE_EVALUATION`, `EVAL_FP32` (optional) | submit scripts: `bsub -W` of the training elements (empty: the queue's RUNLIMIT, `bqueues -l training-parallel \| grep -A 2 RUNLIMIT`); evaluation in a dependent 1-GPU array (1) instead of the training element (0); embeddings in fp32 (1) instead of bf16 autocast (0). The same variable on the command line takes precedence | `""`, 0, 0 | `""`, 0, 0 |

The submit scripts and `setup_code.sh` take the env file as their first argument. The LSF jobs get their values through
`bsub -env`: the tokenization jobs the exported values of the env file (`-env "all"`, frozen at submission), the job
arrays `ARTIFACT_FOLDER_PATH`, `ENV_PATH`, `FORCE_EVAL`, `EVAL_FP32` (and the training array `SEPARATE_EVALUATION`),
and `prepare_ovca_data.lsf` the path of the env file. The Python scripts
get every value as a command line argument (`LABEL_FILE_PATHS` as `${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}}`; the env
files set shell variables, so use a fresh shell per dataset). `generate_ablation_runs.py` writes
the values the job array needs into `<ARTIFACT_FOLDER_PATH>/dataset.env` (dataset, silver and label files, label keys,
unassigned label, norm factors, tag, `CODE_PATH` and `CODE_VERSION`), so the artifact folder defines the campaign: every
array element sources it and stops if `${CODE_PATH}/CODE_VERSION` is not the version the configs were generated for,
and `check_setup.py` fails if it differs from the env file (regenerate after every change of the env file).
`CODE_VERSION` is also a top-level key of every run config (the training code ignores it), so the config hashes, and
with them the run folders, change with the code. So does `eval_tokenized_data_folder_path`: changing `EVAL_GENE_SET`
or the location of the shared-gene data gives every run a new id, i.e. retrains it.

## What is run

The variants and replicates are listed in `ablation_plan.yaml`:
- 51 variants, 139 training runs: 95, plus 42 replicates of the normalization arms and 2 of the predictor-depth arm
  (manifest rows 96-139, appended so that rows 1-95 keep their index and wandb id).
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
| Supp. Fig. 5 | normalization | 21 normalization tuples, in the three mouse groups | 3 (2nd and 3rd: rows 96-139) |
| none | architecture | width 192 / 384 / 960 / 1152; 1, 4 or 5 encoder layers | 1 |
| none | architecture | 3 predictor layers | 3 (2nd and 3rd: rows 96-139) |

Replicate `r` is trained with seed `1000 * r`. TERRA's opt-in fixes (`tokenizer_options` and `training_options` of
the plan) are on for every run; `CODE_REVIEW.md` lists them.

**Evaluation** follows the mouse protocol:
- Neighborhood embeddings from the target encoder, of the run's evaluation dataset (`eval_tokenized_data_folder_path`
  of the run config): with `EVAL_GENE_SET=shared_genes` (mouse), the run's tokenization of the same cells restricted
  to the genes measured in every section (`filter_shared_genes.py`), as the published mouse evaluation embedded
  shared-gene data, so that the integration metrics do not measure the gene panels; with `all_genes` (OVCA, one
  section), the run's own tokenized training dataset.
- kNN graph with k = 15 on all embedded cells (numba compiled for a generic CPU, `NUMBA_CPU_NAME=generic`, so the
  graph does not depend on the node). The embeddings come from the job's GPU in bf16 autocast, as in training
  (`EVAL_FP32=1`: fp32); `evaluation_complete` records the GPU, the torch version and the precision.
- Leiden on the fixed grid of resolutions 0.6–1.4. The grid is identical for all variants, so the protocol is
  label-free.
- NMI, ARI and AMI against each label key, over the cells it labels: cells without the key (e.g. a section-specific
  annotation) or with its `UNASSIGNED_LABEL` value are clustered but not scored. Labels are joined from the label files
  (`LABEL_FILE_PATHS`, default: the silver files) by `cell_id`. Every row also has the granularity: `n_clusters` (all
  cells), `n_labelled_cells`, `n_label_categories` and `n_clusters_labelled`.
- Batch integration, if the cells come from two or more sections (one label file = one section; the mouse
  MERFISH and STARmap PLUS sections): the metrics of the mouse notebook (`stjepa_v2_ablation.ipynb`, cells 21 and 42),
  seeded, over the labelled cells (a label for at least one key) as in the notebook. iLISI (`scib_metrics.ilisi_knn`
  on a 90-NN `pynndescent` graph; higher is better), ASW (silhouette of the sections on 10,000 cells; 0 is best) and
  MMD (`mmd_comparable`: RBF MMD² of the first two sections on standardized embeddings, median-heuristic width; lower
  is better). Unlike the notebook, which removed unlabelled cells before embedding, the embeddings come from the
  all-cell pass, and the subsamples are seeded. These three are label-free: they reward mixing even where the sections
  differ in region composition. Added rows: `iLISI_max` (the asymptotic bound for the section sizes, 0.8 at 2:1, which
  random mixing does not reach), `iLISI_random` (iLISI of the same graph with the sections permuted, mean of 5) and
  `iLISI_relative` = iLISI / iLISI_random (about 1: as mixed as random; comparable between section ratios), and
  `ASW_label_<key>` (section silhouette within labels) for keys that label two or more sections; the mouse keys each
  label one section, so the mouse runs have none. The label-aware readout for the mouse is
  `neighborhood_emb_cross_section_neighbors.csv`: per key, label and section, the share of a cell's kNN neighbors from
  the other section next to the share under random mixing (mixing confined to some regions vs uniform mixing).
- Models with a metatoken are evaluated twice, with all metrics: metatoken removed at inference (as in the mouse
  evaluation) and kept as in training (`keep_metatoken` outputs). The category panels show each run with its training
  input (kept for runs trained with the metatoken, removed for No Metatoken; `--category_mode drop`: removed for all);
  `metatoken_inference` shows both modes and `metatoken_contrasts` the seed-paired differences (Dropout kept - removed
  is clean; Base kept - removed is confounded by the out-of-distribution input; No Metatoken / Dropout - Base give the
  effect of training with the token).
- Also saved: float32 embeddings, cluster assignments (with the section of every cell) and collapse statistics in
  `neighborhood_emb_avg_cosine_similarity.csv`: the average cosine similarity, raw (as in the mouse runs) and
  centered, which do not detect collapse; participation ratio, PCs for 90% of the variance, top PC share, RankMe,
  between-cell variance fraction and per-dimension SD; flags `complete_collapse` (variance fraction below 1e-3) and
  `dimensional_collapse` (top PC above 95% of the variance). Partial collapse is flagged by `compare_ablations.py`
  against base. The evaluation stops before any metric if an embedding is not finite; `package_run.py` refuses
  non-finite weights.
- `evaluation_complete` is removed when an evaluation starts and written last. It records what produced the metrics:
  modes, label keys and files, unassigned label per key, the sha1 of the evaluation code (`evaluate_ablation_run.py`
  and the part of `ablation_utils.py` it uses), TERRA version, checkpoint epoch and md5, evaluation data path and
  provenance md5, precision; and, recorded only, GPU, torch version, cell count and time. `evaluate_ablation_run.py
  --check_only` exits 0 if these match the current inputs and every output file is there, 3 if the run needs its
  evaluation (with the reasons), other codes on errors. Any mismatch evaluates all modes again; `--overwrite` (the
  jobs' `FORCE_EVAL=1`) always does. Editing `evaluate_ablation_run.py`, or a function of `ablation_utils.py` it uses,
  therefore makes every evaluation stale: re-evaluate with `submit_evaluation_array.sh` (1 GPU per run). The label
  files are recorded by path, not content: after editing one in place, use `FORCE_EVAL=1`.

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

# 3. Silver files, norm factors and the shared-gene evaluation data (92,520 cells, a few minutes; run it
#    interactively, e.g. in a bsub -Is job). First the two section files without the cells that have no detected gene
#    of the mouse token dictionary, into the dataset's own silver folder (prints how many it removes; the source files
#    are not modified), then the factors of the filtered files (new file, tokenizer constants; stops on genes without
#    counts and names them). compute_norm_factors.py writes <norm factors>.done last; the tokenization waits for it.
#    Then the evaluation data: the filtered files restricted to the genes of both sections, without cells with fewer
#    than 10 detected shared genes, into silver/${DATASET_NAME}_shared_genes (prints genes and cells kept per section;
#    writes ablation_provenance.json last).
source datasets/mouse.env
ls "${DATA_DIRECTORY_PATH}/silver/${SOURCE_DATASET_NAME}"   # the two section files of SOURCE_SILVER_FILE_PATHS
source "${ENV_PATH}/bin/activate" && export PYTHONPATH="${CODE_PATH}/src"
python filter_silver_cells.py --source_silver_file_paths ${SOURCE_SILVER_FILE_PATHS} \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" \
    --output_folder_path "${DATA_DIRECTORY_PATH}/silver/${DATASET_NAME}"
source datasets/mouse.env     # again: SILVER_FILE_PATHS lists the filtered files once they exist
echo "${SILVER_FILE_PATHS}"   # two files, MERFISH and STARmap PLUS
python compute_norm_factors.py --silver_file_paths ${SILVER_FILE_PATHS} --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}"
python filter_shared_genes.py --silver_file_paths ${SILVER_FILE_PATHS} --data_directory_path "${DATA_DIRECTORY_PATH}" \
    --dataset_name "${DATASET_NAME}" --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}"

# 4. Run configs, manifest.tsv and dataset.env (139 runs; 300 epochs, 3 warmup)
python generate_ablation_runs.py --plan_file_path "${PLAN_FILE_PATH}" --base_config_file_path "${BASE_CONFIG_FILE_PATH}" \
    --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}" \
    --dataset_name "${DATASET_NAME}" --silver_file_paths ${SILVER_FILE_PATHS} \
    --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" --experiment_tag "${EXPERIMENT_TAG}" \
    --epochs "${EPOCHS}" --warmup "${WARMUP}" --n_special_values "${N_SPECIAL_VALUES}" --code_path "${CODE_PATH}" \
    --eval_gene_set "${EVAL_GENE_SET}"

# 5. Labels and sections, before tokenization. Review check 3: per label file the cells, uns dataset id and batch,
#    label keys and their most frequent values. Each key must be in the file of its section; if the silver files lack
#    them, set LABEL_FILE_PATHS in datasets/mouse.env (e.g. the commented evaluation files of the mouse notebook). Set
#    UNASSIGNED_LABEL (a placeholder; check 3 fails until it is set) to the value of unannotated cells, per key if they
#    differ, e.g. "ccf_region_name=unassigned;Sub_molecular_tissue_region=Not assigned" (check 3 counts the unassigned
#    cells per key and file), or to a value that occurs in no label file to score every labelled cell. Check 2 must show
#    that the batch special values fit (200 rows) and differ between the sections, no stored zeros, median
#    nearest-neighbour distances of a few um (about 8-15) within 2x of each other, the batch metatoken id equal to
#    spt_batch and vocab_size equal to the dictionary size. Check 1 fails if the cell-neighborhood layout cannot read
#    the tokens of the mouse runs (for 64/640 it prints the plan change), if the mouse params are missing or if
#    replicates differ in more than seed and folder. Check 5 reports how many cells the raw-count ranking cuts inside
#    tied counts (ordered by corpus mean) and fails on genes without id, factor or counts. Check 7 reports the share of
#    cells with too few genes for a context gene at each mask ratio (for Methods). After a change: source
#    datasets/mouse.env again, rerun steps 4 and 5.
python check_setup.py --plan_file_path "${PLAN_FILE_PATH}" --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" \
    --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}" \
    --silver_file_paths ${SILVER_FILE_PATHS} --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" \
    --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --code_path "${CODE_PATH}" \
    --mouse_token_dictionary_file_path "${MOUSE_TOKEN_DICTIONARY_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}" --eval_gene_set "${EVAL_GENE_SET}" --skip_datasets

# 6. Tokenization: all 27 datasets in the standard class, then the same 27 tokenizations of the shared-gene data (a
#    second array that starts when the first has ended). WAIT until both have finished
#    (bjobs -J mouse_tokenize_standard; bjobs -J mouse_tokenize_shared_genes_standard)
bash submit_tokenization.sh datasets/mouse.env standard

# 7. Checks (fails on errors, e.g. tokenized cells that are not in the label files; check 4 also covers the
#    shared-gene data and its provenance)
python check_setup.py --plan_file_path "${PLAN_FILE_PATH}" --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" \
    --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}" \
    --silver_file_paths ${SILVER_FILE_PATHS} --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" \
    --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --code_path "${CODE_PATH}" \
    --mouse_token_dictionary_file_path "${MOUSE_TOKEN_DICTIONARY_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}" --eval_gene_set "${EVAL_GENE_SET}"

# 8. Attention kernel check on one GPU of the training queue (base_rep1, config from the manifest; a few minutes).
#    The log must end with "OK: ..."; record the torch version, the GPU and the result with the option false.
bsub -G team361 -q training-parallel -n 4 -gpu "num=1:gmem=40000:mode=exclusive_process:block=yes" -M 32G \
    -R "select[mem>32G] rusage[mem=32G] span[hosts=1]" -W 0:30 \
    -o "${ARTIFACT_FOLDER_PATH}/logs/check_attention_kernel_%J.out" \
    -e "${ARTIFACT_FOLDER_PATH}/logs/check_attention_kernel_%J.err" \
    "source ${ENV_PATH}/bin/activate && PYTHONPATH=${CODE_PATH}/src python check_attention_kernel.py \
     --artifact_folder_path ${ARTIFACT_FOLDER_PATH}"

# 9. Pilot: rows 1-4 (base_rep1, cell-graph rank and counts, cell-neighborhood combined; about 3 h each), one at a
#    time (MAX_CONCURRENT 1 = 4 GPUs; arguments: env file, rows, MAX_CONCURRENT, run limit). The script first checks
#    every row of the range on the login node (seconds per run with a checkpoint), prints skip / train / evaluate and
#    the reason per row, and submits only the rows that need work. WAIT until they have finished and check
#    eval/<variant>/<run>/: niche and batch-integration metrics of both metatoken modes, evaluation_complete, the
#    collapse flags in *_avg_cosine_similarity.csv; and the training logs: no FloatingPointError. From the logs, set
#    RUN_LIMIT in datasets/mouse.env above the slowest arm's time (the 128/256-gene and 960/1152-width arms take several
#    times the base's), and decide on SEPARATE_EVALUATION from the time between "Packaged" and "Wrote
#    evaluation_complete" (the 4 GPUs idle during the evaluation).
bash submit_ablation_array.sh datasets/mouse.env 1-4 1

# 10. The other runs in batches with explicit ranges, one array at a time (both submit scripts refuse while an array
#     of the campaign is pending or running; rows that are trained and evaluated are left out). 4 GPUs at a time:
bash submit_ablation_array.sh datasets/mouse.env 5-51 1    # the other first replicates
bash submit_ablation_array.sh datasets/mouse.env 52-95 1   # 2nd and 3rd replicates of the other categories
bash submit_ablation_array.sh datasets/mouse.env 96-139 1  # 2nd and 3rd replicates of normalization, 3 predictor layers
#    or 8 GPUs at a time: MAX_CONCURRENT 2 (the default). The limit of a submitted array can be changed at any time:
#    bmod -J "%1" <JOBID> (or "%2"). Retry failed or killed rows by submitting the same range again (only they are
#    submitted; a run killed at the run limit resumes from its latest checkpoint), or brequeue -e "<JOBID>[<INDEX>]"
#    while LSF still knows the array.

#     Re-evaluation only, when needed (not part of the sequence): runs whose evaluation is missing or stale (evaluation
#     code, TERRA version, checkpoint, evaluation data, labels or precision changed), 1 GPU each (default 4 at a time);
#     rows not trained yet are skipped:   bash submit_evaluation_array.sh datasets/mouse.env 1-51 4
#     FORCE_EVAL=1 re-evaluates every trained row of the range (e.g. after editing a label file in place), EVAL_FP32=1
#     embeds in fp32:   FORCE_EVAL=1 bash submit_evaluation_array.sh datasets/mouse.env 1-4 1
#     SEPARATE_EVALUATION=1 (env file or command line) sends the evaluation of newly trained runs to such 1-GPU
#     elements too: each starts when its training element has ended (GPUs in use: up to 4 N + N at MAX_CONCURRENT N).

# 11. Results and figures (also works on a partial campaign; lists the runs not evaluated, those evaluated on other
#     data than their config and those evaluated by an earlier version). Options: --error_bars sd (default; ci95, or
#     published: the published SD over resolutions x replicates, with --category_mode drop), --category_mode training
#     (default; drop: metatoken removed for all), --collapse_fraction 0.25
python compare_ablations.py --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" --dataset_name "${DATASET_NAME}" \
    --label_keys ${LABEL_KEYS}
```

- **Runs:** 139 (51 variants), in the order and with the seeds of the OVCA campaign.
- **Runtime:** 722 iterations per epoch on 4 GPUs, 300 epochs (216,600 steps): about 3 h per run for the base config,
  so about 9 days for all 139 runs at 2 at a time (139 × 3 h / 2; about 17 days at 1 at a time), plus evaluation and
  the slower arms (128 and 256 genes, widths 960 and 1152). Evaluation per run: two embedding passes, then per pass a
  kNN graph, nine Leiden runs and the integration metrics (about 2.5 min per pass at this size, measured on synthetic
  data: mostly the single-threaded 90-NN graph of iLISI; MMD about 25 s, 1.6 GB). Tokenization: 2 × 27 small jobs (92,520 cells).
- **Storage:** about 180 GB in the artifact folder (embeddings up to 284 MB per set, two sets for 136 runs: 78 GB;
  checkpoints 104 GB).
- **Outputs** under `/nfs/team361/sb75/terra-reproducibility/artifacts/mouse_ablation/`: `configs/`, `manifest.tsv`,
  `dataset.env`, `runs/<run>/`, `logs/` and
  - `eval/<variant>/<run>/`, per metatoken mode (prefix `mmb0-1b_smb1-1b_2p_coord_aligned_`, kept metatoken
    `mmb0-1b_smb1-1b_2p_coord_aligned_keep_metatoken_`): `neighborhood_emb_niche_identification_metrics_res_sweep.csv`
    (both label keys, with AMI and the granularity columns), `neighborhood_emb_batch_integration_metrics.csv`
    (`metric,score`: iLISI, ASW, MMD between MERFISH and STARmap PLUS, then iLISI_max, iLISI_random,
    iLISI_relative), `neighborhood_emb_cross_section_neighbors.csv`, collapse statistics
    (`neighborhood_emb_avg_cosine_similarity.csv`), clusters (with the section of each cell), embeddings, and
    `evaluation_complete`;
  - `results/`: `niche_identification_ablations{,_runs,_summary}.csv`, `batch_integration_ablations{,_summary}.csv`,
    `*_metatoken_mode_differences.csv`, `collapse_diagnostics.csv` and `ablation_summary.csv` (per arm and key);
  - `figures/`: `ablation_<category>_<label key>_{NMI,ARI,AMI,cross_section_mixing}`,
    `ablation_<category>_batch_integration_{iLISI_relative,ASW,MMD}`, `ablation_metatoken_{inference,contrasts}_*`
    and `ablation_summary_<label key>` (NMI against iLISI_relative per arm). Bars are the mean over the replicates of
    each run's mean over the 9 resolutions, error bars the SD over the replicates (ddof 1) with n in each label; panels
    with an arm of one run (architecture widths and encoder depths) show a band of base +/- the pooled seed SD; the
    tables add seed-paired differences to base. Runs flagged for collapse (the evaluation's flags, or a participation
    ratio or variance fraction below 25% of base) are red crosses. Every figure states its bars, mode and data.
- **Differences from the original mouse runs:** those of the OVCA table below that are not specific to OVCA (tokenizer
  fixes, padded masks, `meta.zero_fully_masked_attention_rows`, seeds 1000/2000/3000, latest checkpoint only,
  metatoken removed and kept at inference), plus norm factors computed on the two sections with the tokenizer's
  constants instead of the mouse cohort file. The evaluation embeds shared-gene re-tokenizations of the training
  cells, as published, but tokenized with the run's own tokenizer settings, keeping unlabelled cells (the published
  evaluation removed them before tokenizing, so they were not in its neighborhoods) and dropping cells with fewer
  than 10 detected shared genes (as published). Schedule and `n_special_values` are the mouse values. Excluding a
  real unassigned value (`UNASSIGNED_LABEL`) departs from the mouse evaluation, which excluded only missing labels.
  The panels show the SD over replicates of per-run resolution means, not the published SD over 9 resolutions x
  replicates (`--error_bars published`), the category panels the training input instead of the metatoken removed
  (`--category_mode drop`), and the architecture panel has a base bar.
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

# 3. Run configs, manifest.tsv and dataset.env (139 runs; 82 epochs, 1 warmup)
source datasets/ovca.env
source "${ENV_PATH}/bin/activate" && export PYTHONPATH="${CODE_PATH}/src"
python generate_ablation_runs.py --plan_file_path "${PLAN_FILE_PATH}" --base_config_file_path "${BASE_CONFIG_FILE_PATH}" \
    --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}" \
    --dataset_name "${DATASET_NAME}" --silver_file_paths ${SILVER_FILE_PATHS} \
    --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" --experiment_tag "${EXPERIMENT_TAG}" \
    --epochs "${EPOCHS}" --warmup "${WARMUP}" --n_special_values "${N_SPECIAL_VALUES}" --code_path "${CODE_PATH}" \
    --eval_gene_set "${EVAL_GENE_SET}"

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
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}" --eval_gene_set "${EVAL_GENE_SET}"
python check_mouse_norm_factors.py --plan_file_path "${PLAN_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}"

# 6. Attention kernel check: as mouse step 8, with datasets/ovca.env sourced

# 7. Pilot: rows 1-4 (train + evaluate), 4 GPUs at a time. WAIT until they have finished and check their eval
#    folders and logs, and set RUN_LIMIT and SEPARATE_EVALUATION as in mouse step 9 (submit_ablation_array.sh refuses
#    while an array of the same campaign is pending or running)
bash submit_ablation_array.sh datasets/ovca.env 1-4 1

# 8. The other runs in batches as for mouse (only the rows that need work are submitted); never two arrays at once.
#    Re-evaluation: submit_evaluation_array.sh as in mouse step 10
bash submit_ablation_array.sh datasets/ovca.env 5-51 1
bash submit_ablation_array.sh datasets/ovca.env 52-95 1
bash submit_ablation_array.sh datasets/ovca.env 96-139 1

# 9. Results and figures (also works on a partial campaign; lists the missing runs; options as in mouse step 11)
python compare_ablations.py --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" --dataset_name "${DATASET_NAME}" \
    --label_keys ${LABEL_KEYS}
```

- **Training and evaluation:** one array element per manifest row that needs work, 4 GPUs each, at most
  MAX_CONCURRENT running at a time (default 2, i.e. 8 GPUs; 1 for 4 GPUs). The element evaluates its run after
  training; with `SEPARATE_EVALUATION=1` it stops after packaging and a dependent 1-GPU array
  (`<tag>_ablation_eval`, `bsub -w "ended(<job>[*])"`, same indices) evaluates each run once its training element has
  ended, while rows that only need their evaluation go to their own 1-GPU array.
- **Launch order:** all first replicates (base first, figure order), then the second and third replicates, then the
  appended second and third replicates of the normalization arms and of 3 predictor layers (rows 96-139).
- **wandb:** run id `<EXPERIMENT_TAG>_<manifest index, 3 digits>_<config hash>` (e.g. `ovca_042_1a2b3c4d`), display
  name the descriptive run id (`<run name>_<config hash>`, also the run folder name). The index follows the plan, so
  regenerating after changing the plan gives the runs new wandb ids.
- **Skips and retries:**
  - The submit scripts check every row of the range on the login node first (`list_pending_runs.py`, the jobs' own
    `--check_only` checks; about one checkpoint md5 per trained row) and submit only the rows that need training or
    whose evaluation is missing or stale. An element skips training if its run is finished, and skips everything if
    `evaluate_ablation_run.py --check_only` says its evaluation is up to date (not with `FORCE_EVAL=1`).
  - It stops with an error, instead of retraining, if the latest checkpoint cannot be read, if the pinned code has
    another `CODE_VERSION` than the configs, or if its manifest row is missing or malformed.
  - Training stops with `FloatingPointError` at the first non-finite loss or gradient norm, before the optimizer step
    and on all ranks at once, and never writes a non-finite checkpoint.
  - Retry failed elements by submitting the same range again (only the unfinished rows are submitted), or with
    `brequeue -e "<JOBID>[<INDEX>]"` while LSF knows the array; either resumes from the latest checkpoint, with the
    random number generator states saved in it. An element killed at `RUN_LIMIT` is not requeued. A
    `FloatingPointError` that comes back after a retry is a real divergence.
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
| All 139 runs, 2 at a time | about 9 days | about 9 days of training (139 × 3 h / 2), plus evaluation |

- **Slower arms:** the 128- and 256-gene arms (2× and 4× the tokens of the base) and the widths 960 and 1152 take
  longer than the base, so plan for more than 9 days.
- **Evaluation** runs in the same element after training (or in a 1-GPU element with `SEPARATE_EVALUATION=1`): two
  embedding passes (metatoken removed and kept), a kNN graph and nine Leiden runs per pass, on all 338,695 cells.
- **Epoch-matched alternative:** `EPOCHS=300`, `WARMUP=3` takes about 8–11 h per run and 3–4.5 weeks in total.
- **Tokenization:** all 27 datasets in the standard class (200 GB). Measured peak memory: about 83 GB for the base,
  dense-path and 256-gene datasets and about 105 GB for Seurat v3 and Pearson residuals. The large class (1 TB,
  hugemem) is only needed without the plan's ranked-list cut (`stable_rank_ties` / `zero_expr_from_raw_counts`).
- **Storage**, computed with the base width (`enc_emb_dim` 768):
  - Embeddings: 768 × 338,695 cells × 4 bytes = 1.04 GB per set, two sets per run with a metatoken (136 of 139
    runs), one for the 3 runs without: about 285 GB.
  - Checkpoints: about 0.75 GB per run (only `terra-latest.pth.tar`): about 104 GB.
  - Total about 390 GB in the artifact folder, plus a few GB of cluster tables and logs.
  - Tokenized datasets go to `/lustre/.../DATASETS/gold/*/xho50-1b_1p_spatialfusion/`.

## OVCA: differences from the mouse runs, and why

| Item | Mouse | OVCA | Reason |
|---|---|---|---|
| Code | ST-JEPA v2 (`sebastianbirk/stjepa` `b433d5e`), the farm checkout of the time | copy of TERRA `d29e262` (`setup_code.sh`), the mouse base config in TERRA's config format, `attention_backend: torch` (PyTorch chooses the SDPA kernel, as in the mouse runs and the released models) | The figure is rerun on the current code. TERRA holds every fix of the reviews (`CODE_REVIEW.md`): opt-in seed, atomic latest checkpoint, metatoken fixes, padded masks, resumption on all ranks and the opt-in tokenizer fixes; the options are switched on in the plan. A separate copy, so changes to the farm checkout (`/nfs/team361/sb75/terra`) cannot change running jobs. |
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
  later v2 tokenizers). The dataset class needs exactly that many tokens. Check 1 fails if the mouse runs read other
  tokens (e.g. 64 + 640 with 704 input tokens) and then prints the plan change.
- `pred_emb_dim` following `enc_emb_dim` in the width ablation.
- The coordinate arm being `cell_pos_enc: coord`.

**Caveats:**
- **Not held out from TERRA-112M pretraining.** This section is `xho50-1b_1p` in the TERRA-112M corpus. That doesn't
  matter here, because every arm is trained from scratch on this section alone.
- **Known code issues** are listed in `CODE_REVIEW.md` with their resolution, and what the Methods must state in its
  [Methods notes](CODE_REVIEW.md#methods-notes). The OVCA tokenizations use the tokenizer
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
   `EVAL_GENE_SET`: `shared_genes` for sections with different gene panels (then run `filter_shared_genes.py` after
   the factors, as in mouse step 3), else `all_genes`.
4. Follow the mouse steps with the new env file (`setup_code.sh` only reruns the environment check if the pinned code
   is already there).

## Tie-order diagnostic

With raw-count ranks most genes of a cell are tied (count 1 on imaging data), and the tie rule
(`stable_rank_ties` with `rank_tie_break_norm_factor: mean` here; numpy's unstable sort in the mouse runs and in
TERRA-96M / TERRA-112M) sets their order. With rank-position embeddings (`gt_type` combined or rank) and
`predict_gene: True`, a fixed rule lets the predictor narrow down a masked gene from the visible genes on either side of
it in its tie group. `tie_order_diagnostic.py` measures whether a trained model uses that order: for a sample of cells
it compares the JEPA loss (same masked genes, same context) and the embeddings of the stored order with those of a
random order within each tie group (`tie_shuffle`) and within each segment (`full_shuffle`, a reference).

```bash
cd /nfs/team361/sb75/terra-reproducibility/analysis/ablation/ablation_pipeline
mkdir -p ../../../logs/ablation/tie_order_diagnostic
# A trained ablation run (run_folder_path of manifest.tsv; packaged or not)
MODEL_FOLDER_PATH=<run folder> OUTPUT_FOLDER_PATH=<run folder>/tie_order_diagnostic bsub < tie_order_diagnostic.lsf
# A released model on held-out data tokenized as its training data; the targets of TERRA-96M / TERRA-112M were
# layer-normalized in training
MODEL_FOLDER_PATH=<model folder> DATASET_PATH=<tokenized dataset> TARGET_ENC_LAYER_NORM=true \
    OUTPUT_FOLDER_PATH=<output folder> bsub < tie_order_diagnostic.lsf
```

Reading `tie_order_targets.csv`: a loss increase under `tie_shuffle` on `bracketed` targets (both nearest visible genes
in the target's tie group) that is clearly larger than on `untied` targets means the model uses the order of tied
genes. `gt_type: counts` runs have no rank positions and serve as a negative control (no change beyond numerical
noise). `tie_order_embeddings.csv` shows how much the embeddings move (cosine similarity, 15-nearest-neighbor
overlap). `tie_order_data.json` also gives the share of gene tokens in tie groups and of the masked genes per class.
Tie groups are read from the stored values, so tokenizations whose values are not an increasing function of the
ranking value (e.g. raw-count ranks with gene-normalized values) are refused.
