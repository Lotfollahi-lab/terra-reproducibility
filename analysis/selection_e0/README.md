# E0: headroom and sampling designs on a corpus-like selection proxy

E0 is the first experiment of the effective-exposure allocation project (data selection for TERRA's pretraining). It
asks whether a small TERRA trained on a corpus-like proxy still improves with compute (headroom), whether unique views
or diversity matter at matched compute, and how two cheap samplers compare, before any allocation method is built.
This folder makes the proxy and the index lists of its runs; the run configs, the launch and the evaluation come next.

It needs TERRA `exp/effective-exposures` (`Lotfollahi-lab/terra`, commit `49b4840`): `add_spatial_index` in the
tokenizer, `data.epoch_index_lists` in the sampler, and the `exposure` config section (`terra.training.exposures`).
Every E0 job runs a pinned copy of it (`setup_code.sh`).

## The proxy

About 48 training batches of hst_corpus_110m (none of the 219 held-out batches), stratified by assay x tissue, each
capped at about 100k cells by a square crop: about 4-5M cells. It is tokenized with the corpus settings (16 neighbours x
512 genes, compact rows, train-only norm factors, tie-break by `tie_break_mean`) plus `add_spatial_index`, so its rows
are the corpus rows of the same cells (except at crop borders) and carry each cell's position in its file, its
coordinates and its neighbours' rows. A model can read fewer neighbours and genes from them (`truncate_neighbors`,
`tokenized_seq_len_cell: 512`), e.g. the ablation size of 10 neighbours x 64 genes.

## The arms (24 runs)

N is the number of proxy rows; the full budget C is 6 passes (6 epochs of N presentations). Every list has 6 epochs of
equal length, so each run's schedules end at its own budget (`optimization.ipe_scale: 1.0`).

| Axis | Arm | Budget | Lists (seeds) |
|---|---|---|---|
| Compute | `uniform_c1`, `uniform_c2`, `uniform_c4`, `uniform_c8`: all rows in passes | C, C/2, C/4, C/8 | 3, 2, 3, 2 |
| Unique views at matched compute | `fixed_2`, `fixed_8`, `fixed_32`: a fixed random 1/k of the centres; all cells stay as context | C | 2 each |
| Diversity at matched size | `sections_8`: the 1/8 from a few whole sections per assay | C | 2 |
| Sampler | `rs2_c4`: a fresh random N/4 each epoch; `poisson_c4`: N/4 Poisson-disk centres re-drawn each epoch | C/4 | 3 each |

`build_index_lists.py` describes each arm. `uniform_c1` lists every row in order, which trains exactly as without lists.
A list's seed is also its run's training seed.

## Files

| File | Role |
|---|---|
| `e0.env` | Parameters (pinned commit and code path, data paths, selection sizes, LSF settings). |
| `setup_code.sh` | Pinned TERRA code at `CODE_PATH` (as `analysis/ablation/ablation_pipeline/setup_code.sh`). |
| `select_proxy.py`, `select_proxy.lsf` | The proxy's batches, crops and silver folder (dry run unless `RUN=1`). |
| `../../config/cell-graph-tokenizer/cell-graph-tokenizer_hst_corpus_110m_e0_proxy.yaml` | The proxy's tokenizer config: the corpus config plus `add_spatial_index` and its own cache folder. |
| `tokenize_proxy.lsf` | The tokenization job: `run_tokenization_locked.sh` with the pinned code on `PYTHONPATH`. |
| `build_index_lists.py`, `build_index_lists.lsf` | The 24 index lists and `lists_summary.tsv`. |
| `submit_data_step.sh` | Submits the `select`, `tokenize` and `lists` jobs. |

`analysis/tokenization/create_tokenized_dataset.py` forwards `add_spatial_index` and reads an optional
`paths.cache_directory_path`; corpus configs have neither, so their jobs are unchanged.

## Steps on the farm

Run these on the CPU farm (as the corpus tokenization), in a separate worktree of this branch, so the corpus
checkout `/nfs/team361/sb75/terra-reproducibility` stays on its branch. Once:

```bash
cd /nfs/team361/sb75/terra-reproducibility && git fetch origin && git worktree add ../terra-reproducibility-e0 exp/effective-exposures
```

```bash
cd /nfs/team361/sb75/terra && git fetch origin exp/effective-exposures && cd /nfs/team361/sb75/terra-reproducibility-e0 && git pull && bash analysis/selection_e0/setup_code.sh analysis/selection_e0/e0.env
```

- `setup_code.sh` copies TERRA `49b4840` to `/nfs/team361/sb75/terra_e0` and checks that the terra venv imports it.
  Fetching does not change the TERRA checkout.

**1. Selection, dry run.** It reads only HDF5 metadata and writes nothing:

```bash
cd /nfs/team361/sb75/terra-reproducibility-e0 && git pull && bash analysis/selection_e0/submit_data_step.sh analysis/selection_e0/e0.env select
```

- Read `logs/selection_e0/select_<jobid>.out`: the strata (assay, tissue) with their units and cells, how many batches
  each gets, the files to crop and the proxy's cells (expect about 4-5M). Change `N_BATCHES`, `MAX_CELLS`,
  `MIN_CELLS` or `SELECTION_SEED` in `e0.env` and rerun if needed.

**2. Selection, written.** Links to the uncropped silver files and copies of the crops in
`$D/silver/hst_corpus_110m_e0_proxy/`, and the manifests in `artifacts/e0/proxy/`:

```bash
cd /nfs/team361/sb75/terra-reproducibility-e0 && git pull && RUN=1 bash analysis/selection_e0/submit_data_step.sh analysis/selection_e0/e0.env select
```

- It refuses an existing, non-empty proxy folder: the tokenized proxy and the index lists belong to its files.

**3. Tokenization.** Check the memory and the quota first. The tokenizer holds about 3 x cores files of up to about
0.21 MB per cell (cropped files: up to about 21 GB each), so `TOKENIZE_CORES=4` needs about 350 GB (`TOKENIZE_RAM`
400G in `e0.env`):

```bash
cd /nfs/team361/sb75/terra-reproducibility-e0 && source /nfs/team361/sb75/.venvs/terra/bin/activate && python analysis/data_preparation/checks/tokenization_memory.py --names <(echo hst_corpus_110m_e0_proxy) --silver /lustre/scratch126/cellgen/lotfollahi/DATASETS/silver
```

```bash
lfs quota -g team361 -h /lustre/scratch126
```

- About 27 KB per cell at 16 x 512: about 125 GB for 4.6M cells, plus the same again in the job's cache folder until
  the save (the job removes it after a run that wrote the record). Leave at least 300 GB free.

```bash
cd /nfs/team361/sb75/terra-reproducibility-e0 && git pull && bash analysis/selection_e0/submit_data_step.sh analysis/selection_e0/e0.env tokenize
```

- The `.out` file in `logs/hst_corpus_110m_e0_proxy/farm-job-outputs/cell-graph-tokenizer/` must show
  `TERRA ... /nfs/team361/sb75/terra_e0/src/terra/__init__.py` and end with `Written dataset: ...` and `add_spatial_index:
  true` among the recorded settings.

**4. Index lists:**

```bash
cd /nfs/team361/sb75/terra-reproducibility-e0 && git pull && bash analysis/selection_e0/submit_data_step.sh analysis/selection_e0/e0.env lists
```

- It first checks every neighbour pair of the proxy (`Neighbour rows checked: ... pairs`), then writes the 24 lists
  and `artifacts/e0/index_lists/lists_summary.tsv`: per list the rows per epoch, budget, unique rows, copies of a row
  per epoch, sections and assay shares, and for `poisson_c4` the j it chose (the sizes of the independent sets for
  j = 1..10 are in the `.out` file; j = 10 is about 13% of the cells).

## Caveats

- **Count encoding.** Use `count_encoding: mlp` (TERRA-96M's) for the E0 runs. With `value_bins` and `nz_spc: false`
  (the round-5 corpus configs), the encoder uses its padding value embedding only in batches that hold a zero count,
  so a batch without padding stops DDP (`static_graph=True`: "Expected to have finished reduction in the prior
  iteration"). This is finding MD-1 of the 22 Nov code review, reproduced on the synthetic proxy here; dense panels
  read without neighbour cut-off make such batches possible. TERRA itself is not changed here.
- Crops cut the neighbour graph at their border, and the selection draws batches, not cells: the proxy's cells per
  stratum follow the batches drawn (`proxy_strata.tsv`).
- Tested locally on a synthetic corpus (held-out batches, a batch over two files, a file with two batches, crops):
  selection, tokenization through `create_tokenized_dataset.py` with the proxy config, the 24 lists, and CPU training on
  `uniform_c8` and `fixed_8` with exposure logging and counters. Nothing has run on the farm.
