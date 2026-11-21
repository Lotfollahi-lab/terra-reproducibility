# Corpus build checks

Read-only checks and validators for building a tokenized corpus (hst_corpus_110m:
16 neighbours x 512 genes per cell, `model_input_size` 8704, compact rows). Run
them from the terra-reproducibility root in the TERRA venv. Every script prints
its full usage with `--help`. Exit code 0 means pass, and 1 means a check failed.

| When | Script | What it checks |
|---|---|---|
| Before tokenizing | `check_silver_csr.py` | `X` of every silver file is canonical sparse (sorted, no duplicate genes) |
| Before tokenizing | `silver_layout_check.py --n_neighs 16` | Every dataset has files, no nested files, no duplicate or shared cell ids, no file with 16 or fewer cells (the kNN graph needs 17), and every held-out id matches cells |
| Before tokenizing | `count_low_gene_cells.py` | Cells below 10 vocabulary genes (the corpus floor) |
| Before tokenizing | `count_genes_per_cell.py` (+ `count_genes_per_cell.lsf`, one dataset per array element) | Vocabulary genes per cell, per file, per dataset and overall; the mean filled 256 / 512 segment |
| Before tokenizing | `tokenization_memory.py` | Estimated job memory per dataset against the tokenization wrapper's resource table (needs only the silver metadata; more precise with `--gpc_per_file` from `count_genes_per_cell.py`) |
| After the norm factors | `validate_norm_factors.py --full` | `compute_cohort_norm_factors.py` outputs recomputed from silver: held-out batches excluded, a row and a finite `tie_break_mean` for every vocabulary gene |
| Pilot dataset | `verify_compact_corpus.py` | Padded and compact tokenizations of one dataset give bit-identical model inputs (items, collator batches, inference loaders) at the stored and the sliced layouts |
| After split / n_nonzero | `validate_split.py` | Splits load in TERRA, are disjoint, cover every row and hold out exactly the listed batches; the record exists and has the expected settings (`--expect_param storage_format=compact` ...); the rows equal the cells the norm factors counted (`--norm_factor_json`) and, per batch, the cells of `silver_layout_check.py --table` (`--batch_table`), so a dataset missing from the corpus fails; n_nonzero lists, if any (`--no_n_nonzero_lists` for compact corpora) |

Example invocations on the farm (`D=/lustre/scratch126/cellgen/lotfollahi/DATASETS`,
`NAMES=analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt`,
`NF=/nfs/team361/sb75/terra-reproducibility/artifacts/data/hst_corpus_110m/hst_corpus_110m_train_norm_factors.csv`,
`C=$D/gold/cell-graph-tokenizer/hst_corpus_110m/hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_16_8704.dataset`):

```bash
export HDF5_USE_FILE_LOCKING=FALSE
python analysis/data_preparation/checks/silver_layout_check.py --names $NAMES --silver $D/silver \
    --heldout analysis/data_preparation/heldout_batch_ids.py --n_neighs 16 --table batch_table.tsv
python analysis/data_preparation/checks/tokenization_memory.py --names $NAMES --silver $D/silver
python analysis/data_preparation/checks/validate_norm_factors.py --norm_factor_csv $NF \
    --data_directory_path $D --cohort_dataset_names_file_path $NAMES \
    --token_dictionary_file_path $D/tokenizer/token_dictionary_homo_sapiens.pkl --full
python analysis/data_preparation/checks/validate_split.py --tokenized_data_folder_path $C \
    --tokenizer_folder_path $D/tokenizer --splits both --no_n_nonzero_lists \
    --norm_factor_csv $NF --norm_factor_json ${NF%.csv}.json \
    --token_dictionary_file_path $D/tokenizer/token_dictionary_homo_sapiens.pkl \
    --expect_param storage_format=compact --expect_param n_neighs=16 \
    --expect_param model_input_size=8704 --expect_param count_count_norm_method=shifted_log \
    --batch_table batch_table.tsv
```

`validate_split.py` fails when the corpus rows differ from the norm-factor json's `n_cells` /
`n_cells_excluded` (a dataset left out of the concatenation, a silver copy other than the one
the norm factors were computed from, or cells with `obs['filter_pass'] != 1`, which the tokenizer
drops and `compute_cohort_norm_factors.py` counts). `--batch_table` (the `--table` output of
`silver_layout_check.py` on the silver the parts were tokenized from, with `filter_pass` applied)
names the batches and dataset folders without rows. Pass `--allow_fewer_rows_than_json` only
once the missing rows are explained, e.g. by the `filter_pass` cells or by datasets left out on
purpose.

`concatenate_datasets.py` refuses parts without `terra_tokenizer_params.json` (a tokenization job
killed after writing the rows, or a part copied by hand): `rm -rf` such a part and re-tokenize it.

`compute_cohort_norm_factors.py` reads `<data_directory_path>/silver/<dataset>/*.h5ad`. Its
default data root is `/nfs/team361/sb75/DATASETS`, while the tokenizer configs read
`/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver`. Pass the data root that the norm
factors were computed from as `--data_directory_path` to `validate_norm_factors.py`. The
`json` checks there compare the cell counts with that silver.
