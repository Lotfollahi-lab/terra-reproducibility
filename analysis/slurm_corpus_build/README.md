# Slurm corpus build from multi-sample .h5ad files

Builds a tokenized TERRA corpus from your own multi-sample AnnData files on a
Slurm cluster. The norm factors come from these datasets only, so they form a
corpus of their own.

The tokenizer settings are those of the re-tokenized hst_corpus_110m:
- 16 neighbours x 512 genes, `model_input_size` 8704;
- shifted_log;
- stable rank ties broken by `tie_break_mean`;
- `pad_empty_neighbors`, `eliminate_stored_zeros`;
- compact rows.

The one difference is `special_tokens: []`. No batch, assay or tissue values are
looked up, so the token dictionary needs no entries for the new batches. The
special-value columns are written as 0, the padding value.

## Stages

Each stage runs as Slurm jobs. They are chained with `afterok`: a failed job
cancels the jobs that depend on it.

| Stage | Jobs | Script | Output |
|---|---|---|---|
| `prep` | one per dataset | `prepare_silver.py` | `silver/<dataset>/adata_batch<k>.h5ad`, one file per sample, plus `silver_manifest.tsv` and `silver_prep.json` |
| `norm` | one, after every `prep` job | `../data_preparation/compute_cohort_norm_factors.py` | `tokenizer/<corpus>/<corpus>_norm_factors.csv` (+ `.json`) |
| `tokenize` | one per dataset, after `norm` | `../tokenization/create_tokenized_dataset.py`, then `../data_preparation/checks/check_tokenized_part.py` | `gold/cell-graph-tokenizer/<dataset>/<dataset>_32_None_None_None_None_None_shifted_log_knn_16_8704.dataset` |

`--stages inspect` runs `prepare_silver.py --inspect` instead. It prints the
file structure and what would be detected, and writes nothing.

### What `prepare_silver.py` does

**One file per sample.** The tokenizer builds the k-NN graph per file, so
cells of different samples must not share a file.

**Sample key.** It is the obs column with exactly `expected_n_samples`
distinct values that contains every excluded sample.
- Columns that split the cells identically are interchangeable.
- Of columns that split them differently, the one named like a sample wins
  (sample, patient, donor, biopsy, id); the log names the others.
- Otherwise the script fails. Set the `sample_key` column of the table.

**Raw counts.** The first matrix holding non-negative integers is used:
`layers['counts']`, `layers['raw_counts']`, `layers['count']`, `layers['raw']`,
`X`, `raw.X`.
- If none of them holds integers, the job fails.
- `--round_counts` rounds instead, e.g. for expected counts from Proseg. Use
  it only if that is intended.

**Genes.** Names are mapped with the gene-name map (`GENE_NAME_MAP`). This is
the same kind of map that TERRA's `harmonize_adata` uses at inference.
- Names the map does not know fall back to an Ensembl-ID column of `var`
  (e.g. 10x `gene_ids`).
- Only genes of the token dictionary are kept. Control probes and codewords
  drop out, and the log lists them by name prefix.
- Columns that map to the same gene are summed.
- Counts are stored as int32 CSR in canonical form.

**Coordinates.** They are taken from `obsm['spatial']` (first two columns),
or from obs pairs such as `x_centroid` / `y_centroid`.
- The log prints the median nearest-neighbour distance per sample, about
  5-20 for cells in micrometres.
- It warns when slides or sections of one sample overlap in the coordinate
  frame (slide-local coordinates). Their cells would become neighbours in the
  k-NN graph; split such samples further first.
- The model's relative coordinates are in the input units, and the corpus
  uses micrometres. Fix other units with `--coord_scale` (0.2125 for Xenium
  pixels).

**Dropped cells, all reported.** Cells without a sample or finite
coordinates; cells with fewer than 10 vocabulary genes (the corpus floor).
Samples with 16 cells or fewer are skipped: the 16-NN graph needs 17.

**Cell ids and batches.** Cell ids are `<dataset_id>_batch<k>_<i>`. Batches
are numbered in natural order of the sample names (NO2 before NO10), and
`silver_manifest.tsv` maps sample to batch.

**Obs columns.** The input's obs columns are kept. `filter_pass` and
`cell_id` are renamed to `original_*`, because the tokenizer drops cells with
`filter_pass != 1`.

## Setup on the cluster

1. Clone **terra-reproducibility** and **TERRA**, both on the branch
   `feat/compact-tokenized-data`.
   - TERRA dev cannot write compact rows. The tokenize stage checks this and
     fails early.
   - Install TERRA with the training extra, which brings wandb:
     `pip install -e ".[train]"`.
2. Copy the **token dictionary** and the **gene-name map** to the cluster.
   - From the farm: `.../DATASETS/tokenizer/token_dictionary_homo_sapiens.pkl`
     and `.../DATASETS/genes/homo_sapiens_gene_name_to_ensembl_id_dict.pkl`.
   - Or from the TERRA-96M bundle: `token_dictionary.pkl`,
     `ensembl_dictionary.pkl`.
   - For a new corpus, any consistent pair works. Train and run inference with
     the same files.
3. Copy `site_config.example.sh` and `datasets.example.tsv` outside the
   repository and edit them.
   - The site config sets the paths, the environment activation, the Slurm
     partition and the resources.
   - The table has one line per dataset. Columns are separated by tabs or
     spaces, so paths must not contain spaces. Use `-` for an empty value.

## Run

From the terra-reproducibility checkout, on a login node:

```bash
# Optional: structure and detection only (minutes; nothing written)
bash analysis/slurm_corpus_build/submit_corpus_build.sh --config my_site.sh --datasets my_datasets.tsv --stages inspect

# Full chain
bash analysis/slurm_corpus_build/submit_corpus_build.sh --config my_site.sh --datasets my_datasets.tsv
```

`--dry_run` prints the `sbatch` commands without submitting them. Logs go to
`$TERRA_DATA_DIR/logs/$CORPUS_NAME/<job name>_<job id>.out`.

**Read the prep logs before trusting the parts.** Check these lines:
- `Sample key:`
- `Raw counts:`
- the gene report (`n_mapped_by_name`, `dropped_by_prefix`)
- `Median nearest-neighbour distance`
- any `WARNING` (coordinate units, duplicated coordinates, overlapping
  sections)
- the cells dropped per sample

Each tokenize log ends with the `check_tokenized_part.py` report and `PASS`.

## Reruns

**Without `--overwrite`, finished steps are kept:**
- a silver folder made from the same input and settings;
- norm factors that count the current silver files;
- parts tokenized with the current norm factors and token dictionary.

A rerun of the whole chain only redoes what is missing. To run only some of
it, use `--stages tokenize` or `--only <dataset>`.

**`--overwrite` redoes the chosen stages.** Parts made with other norm
factors are always redone.

## Resources

Defaults are in `site_config.example.sh`.

**Tokenization processes.** The tokenize job picks as many processes as its
memory allows (`corpus_tools.py nproc`). It uses the memory model of
`checks/tokenization_memory.py`:
- each input file takes about n_cells x (24 x 17 x mean genes per cell,
  capped at 512) bytes;
- up to 3 x processes files are held at once;
- 16 GB base, margin 1.3.

The job log prints the estimate. `TOKENIZE_NPROC` overrides it.

**Disk.**

| What | Size per cell |
|---|---|
| Compact rows | about 10 KB for 480-gene panels, about 30 KB for 5K panels (~100 / ~300 genes per cell) |
| `$TERRA_DATA_DIR/cache` (Hugging Face datasets) | about the same again; delete it once every tokenize job has finished |
| Silver files | 1-3 KB, plus the obs columns |

## Held-out samples

To keep validation or test samples out of the norm factors, list them in a
file with one `<dataset name> <sample>` pair per line and set
`HELDOUT_SAMPLES`.
- The norm stage maps the samples to batch ids and leaves their cells out.
- The output is `<corpus>_train_norm_factors.csv`.
- Decide this before tokenizing: other norm factors mean re-tokenizing every
  part.

## Next steps (not run by these scripts)

**Corpus.** Concatenate the parts:

```bash
python analysis/tokenization/concatenate_datasets.py --data_directory_path $TERRA_DATA_DIR \
    --cohort_name $CORPUS_NAME \
    --cohort_dataset_names_file_path $TERRA_DATA_DIR/tokenizer/$CORPUS_NAME/${CORPUS_NAME}_dataset_names.txt \
    --n_neighs 16 --seq_len 8704 --expect_storage_format compact
```

**Split.** `split_dataset.py` defaults to the held-out batches of
hst_corpus_110m, so pass this corpus' own.

**Training.** Use a TERRA model config such as
`configs/model/hst_corpus_110m/hst_corpus_110m_knn16_seq512.yaml`, with this
corpus' paths and its `norm_factor_file_path`. `train()` checks the file's
sha256 against the record of the corpus.

## Files

| File | Purpose |
|---|---|
| `submit_corpus_build.sh` | Reads the site config and the dataset table, and submits the jobs |
| `corpus_job.sh` | One stage of one job: environment, version printout, stage |
| `prepare_silver.py` | Multi-sample `.h5ad` to per-sample silver files |
| `corpus_tools.py` | Tokenizer config, process count, held-out batch ids, part and norm-factor status |
| `site_config.example.sh`, `datasets.example.tsv` | Templates |
