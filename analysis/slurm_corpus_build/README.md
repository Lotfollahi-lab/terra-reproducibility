# Slurm corpus build from multi-sample .h5ad files

Builds a tokenized TERRA corpus from your own multi-sample AnnData files on a
Slurm cluster, up to the training splits. The norm factors come from these
datasets only, so they form a corpus of their own. Several models can be
trained on the one corpus, e.g. one on every dataset and one per gene panel:
each gets its own split files.

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
| `concat` | one, after every `tokenize` job | `../tokenization/concatenate_datasets.py` | `gold/cell-graph-tokenizer/<corpus>/<corpus>_32_None_None_None_None_None_shifted_log_knn_16_8704.dataset` |
| `split` | one, after `concat` | `make_splits.py` | `tokenizer/<corpus>/splits/<variant>/`: `<corpus>_..._precomputed_split_{train,validation,test}.pkl` and `split_summary.json` |

By default all stages run; `split` only when `SPLIT_SAMPLES` is set (see
[Splits](#splits-one-or-several-models)).

`--stages inspect` runs `prepare_silver.py --inspect` instead. It prints the
file structure and what would be detected, and writes nothing.

`--after <job ids>` makes the first stage of a submission wait for jobs that
are still running or pending, e.g. `--stages concat,split` after the
tokenize jobs of an earlier submission. Give only ids that `squeue` lists:
Slurm refuses dependencies on jobs it has forgotten.

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
are numbered in natural order of the sample names (S2 before S10), and
`silver_manifest.tsv` maps sample to batch.

**Obs columns.** The input's obs columns are kept. `filter_pass` and
`cell_id` are renamed to `original_*`, because the tokenizer drops cells with
`filter_pass != 1`.

**Metadata.** `uns` records the assay (`--assay`, default `xenium`), species
(`--species`, default `homo_sapiens`) and tissue (`--tissue`, default
`unknown`). Set them per dataset in the table's 7th column, e.g.
`--tissue lung`. The tokenizer reads none of them with `special_tokens: []`.

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
- parts tokenized with the current norm factors and token dictionary;
- a corpus newer than its parts, with their norm factors and row count;
- split variants made for the same corpus, datasets and held-out samples.

A rerun of the whole chain only redoes what is missing. To run only some of
it, use `--stages tokenize` or `--only <dataset>`.

**`--overwrite` redoes the chosen stages.** Parts made with other norm
factors are always redone. A corpus or split variant that does not fit its
inputs any more is refused until you pass `--overwrite`. To redo only the
splits (e.g. after editing the held-out samples), use
`--stages split --overwrite`: the corpus is not touched.

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
| Compact rows | about 10 KB for panels of a few hundred genes, about 30 KB for 5K panels (~100 / ~300 genes per cell) |
| `$TERRA_DATA_DIR/cache` (Hugging Face datasets) | about the same again; delete it once every tokenize job has finished |
| The concatenated corpus | a full copy of the parts; the parts can go once the splits are written (training reads the corpus) |
| Split files | about 8 bytes per row and variant |
| Silver files | 1-3 KB, plus the obs columns |

`concat` writes the corpus with `CONCAT_CPUS` processes (memory: the cell
ids and a hash table, about 1.5 GB per 10M cells; 64G by default). `split`
reads only the `cell_id` column (32G by default is ample for 10M cells).

## Held-out samples

To keep validation or test samples out of the norm factors, list them in a
file with one `<dataset name> <sample>` pair per line and set
`HELDOUT_SAMPLES`.
- The norm stage maps the samples to batch ids and leaves their cells out.
- The output is `<corpus>_train_norm_factors.csv`.
- Decide this before tokenizing: other norm factors mean re-tokenizing every
  part.

## Splits: one or several models

The `split` stage writes training / validation / test splits of the corpus
for each **variant**, a set of datasets one model is trained on:

```bash
# in the site config
export SPLIT_SAMPLES=/full/path/split_samples.tsv
export SPLITS="all=ds_a,ds_b,ds_c panel_x=ds_c panel_y=ds_a,ds_b"
```

`SPLITS` defaults to one variant `all` with every dataset of the table.

**Held-out samples.** `SPLIT_SAMPLES` lists the validation and test samples,
one `<dataset name> <sample> <val|test>` per line. A sample is held out from
every variant that holds its dataset, so models trained on different
variants are evaluated on the same samples. To start from a random proposal
(seeded, 5% validation and 10% test samples per dataset, at least one each),
run on the login node after the prep stage, then review or edit the file:

```bash
python analysis/slurm_corpus_build/corpus_tools.py propose-split-samples \
    --names_file $TERRA_DATA_DIR/tokenizer/$CORPUS_NAME/${CORPUS_NAME}_dataset_names.txt \
    --data_directory_path $TERRA_DATA_DIR --out split_samples.tsv
```

`--val_fraction`, `--test_fraction` and `--seed` change the proposal. These
samples are not left out of the norm factors unless `HELDOUT_SAMPLES` names
the same file (it reads the first two columns).

**What a variant holds.** For a variant with datasets D:
- validation: the rows of the validation samples of D;
- test: the rows of the test samples of D;
- train: every other row of D;
- rows of other datasets: none of the three.

The files are written with TERRA's `save_precomputed_split`, so training
refuses them for another corpus. The training split records every held-out
batch as excluded, so training trusts it without scanning the cell ids.

**Checks before a variant is written:**
- the files load through TERRA with the written indices;
- TERRA's training-time check finds no held-out row in the training split;
- the three splits are disjoint and cover exactly the rows of D;
- every held-out sample has rows.

**Using a variant for training.** In `split_summary.json`, `config_data`
holds the `data:` entries of a TERRA training config:
- `tokenized_data_folder_path`
- `precomputed_split`
- `norm_factor_file_path`
- `token_dict_folder_path`
- `val_batch_ids` and `test_batch_ids`

Copy them into a model config such as
`configs/model/hst_corpus_110m/hst_corpus_110m_knn16_seq512.yaml`. Its paths
use `${TERRA_DATA_DIR}`. `train()` checks the norm-factor file's sha256
against the corpus record.

Every variant reads the same tokenized corpus. Its tie-break values come from
the norm factors of all datasets, including for a model of one panel only. A
model that must not see the other panel's data even through the norm factors
needs a corpus of its own.

## Files

| File | Purpose |
|---|---|
| `submit_corpus_build.sh` | Reads the site config and the dataset table, and submits the jobs |
| `corpus_job.sh` | One stage of one job: environment, version printout, stage |
| `prepare_silver.py` | Multi-sample `.h5ad` to per-sample silver files |
| `corpus_tools.py` | Tokenizer config, process count, held-out batch ids, part, norm-factor and corpus status, proposal of split samples |
| `make_splits.py` | Training / validation / test splits of the corpus per variant |
| `site_config.example.sh`, `datasets.example.tsv` | Templates |
