# TERRA inference efficiency benchmark

Settles, on one GPU node, the inference-efficiency candidates from the TERRA
code validation of November 2026: how much faster each candidate makes
inference with TERRA-112M and TERRA-96M, and whether it changes the outputs.
The candidates are prototypes, kept as patches against TERRA `dev` d4811cf
(`patches/`); none of them is in TERRA.

## What is compared

Embedding (`embed_dataset`, GPU). Every variant embeds the same tokenized
dataset:

| variant | change |
|---|---|
| `base` | TERRA d4811cf |
| `base_rerun` | `base` again: shows whether the GPU runs are deterministic |
| `p2` | full-context pass with the `[B, 1, 1, L]` key-padding mask instead of a materialized `[B, 1, L, L]` mask |
| `p1` | cell-only context computed on the special-token + index-cell block only (257 of 2817 tokens) |
| `p12` | `p1` + `p2` |
| `base_pin`, `p12_pin` | as `base`, `p12` with `pin_memory=True` |
| `base_nw0` | `base` with `num_workers=0` (how much the DataLoader workers matter) |

Tokenization (`tokenize_adata`, CPU), with `nproc` 1 and 4. Every variant
tokenizes the same harmonized AnnData:

| variant | change |
|---|---|
| `base` | TERRA d4811cf |
| `tok` | CSR row access, one ranking per cell, neighbors read from the CSR arrays |
| `tok_gen` | `tok` + examples formatted inside the dataset generator (no `map` over the stored data) |
| `tok_gen_fp` | `tok_gen` + a random dataset fingerprint instead of hashing the generator and its data |

A candidate preserves the results if its outputs are bit-identical to `base`.
On CPU, `p12` differed from `base` by at most 2.4e-7. On GPU, in bf16, the
differences can be larger. The cell-only pass of `p1` runs on 257 instead of
2817 tokens, so the matrix products and SDPA kernels see other shapes and may
sum in another order. The report therefore gives, for each output
(`cell_emb`, `neighborhood_emb`, `spatial_cell_emb`), the largest absolute and
relative difference and the smallest per-cell cosine similarity. If
`base_rerun` is not bit-identical to `base`, the GPU itself is not
deterministic; judge the candidates against the size of those differences. The
tokenized datasets must be identical (rows, columns, features and every Arrow
column).

## Run

1. On a login node (needs internet for GitHub and Hugging Face):

   ```bash
   cd /nfs/team361/sb75/terra-reproducibility && git pull
   cd analysis/inference_efficiency
   source /nfs/team361/sb75/.venvs/terra/bin/activate
   bash setup_variants.sh
   ```

   This clones TERRA into `work/src/repo` and makes one worktree per variant
   (`base`, `p1`, `p2`, `p12`, `tok`). It also downloads the TERRA-112M and
   TERRA-96M bundles into `work/bundles/`. TERRA is cloned the same way as
   this checkout (HTTPS or SSH); set `TERRA_REPO` to clone it from elsewhere.
   The script is safe to run again: it resets the variant trees and applies
   the patches anew. Set `WORK` to put everything somewhere other than
   `./work`; it needs a few GB.

2. Submit the job. `H5AD` is a spatial AnnData with raw counts in `X` (or
   `layers['counts']`), gene symbols or Ensembl IDs as `var_names`, and
   coordinates in `obsm['spatial']`. A silver dataset works. For a file with
   several sections, pick one with `SAMPLE_KEY` and `SAMPLE`:

   ```bash
   mkdir -p ../../logs/inference_efficiency/run_efficiency
   H5AD=/path/to/data.h5ad SAMPLE_KEY=sample SAMPLE=<section> bsub < run_efficiency.lsf
   ```

   The job takes the 20,000 cells closest to the centre of the section
   (`N_CELLS`). It harmonizes and tokenizes them once per model, then runs the
   tokenization and embedding variants with 2 timed repetitions each (`REPS`),
   after a warm-up on 1,024 cells. `BATCH_SIZE` (128) and `NUM_WORKERS` (12)
   are the `embed_dataset` defaults. `MODELS=TERRA-112M` runs one model.
   `SKIP_TOKENIZATION=1` runs only the embedding variants. LSF passes these
   variables from the submitting shell to the job.

3. Results: `work/results/report.md` (also at the end of the job's `.out`
   log), with the raw numbers in `report.json`. Speed is the embedding loop
   time per 1,000 cells, without model loading. `emb__<model>__<variant>.npz`
   holds the embeddings of each variant. `compare.py work/results` rebuilds
   the report.

## Files

- `setup_variants.sh`: TERRA worktrees of the variants and model bundles
  (login node).
- `run_efficiency.sh`, `run_efficiency.lsf`: the benchmark (GPU node) and its
  LSF wrapper.
- `prepare_data.py`: selects the cells, then harmonizes and tokenizes them
  with `base`.
- `bench_embed.py`, `bench_tokenize.py`: time one variant and save its
  outputs. `bench_tokenize.py compare A B` compares two tokenized datasets.
- `compare.py`: the report.
- `patches/`: the candidates as diffs against TERRA d4811cf.
