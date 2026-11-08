# ST-JEPA v2 code review and mouse-ablation audit (30 Oct 2026)

**Scope:**
- **Model code:** `sebastianbirk/stjepa` branch v2: tokenizers, datasets, masking, model, training, inference.
- **Ablation analysis:** `sebastianbirk/stjepa-reproducibility` `analysis/mouse_ablation`.
- **Coordinate ablation:** the data preparation behind it.

**Method:** the code was read and partly executed on OVCA data. Nothing was run on the farm, so items marked "farm check" are open. The upstream code was not changed; the fixes were first collected in `stjepa_ablation.patch` (20 changes; in git history up to commit 60a77a3 of terra-reproducibility).

**Since 7 Nov 2026 the ablations run on TERRA** (`Lotfollahi-lab/terra` `d29e262`, pinned by `setup_code.sh`), which holds every fix; see [TERRA port](#terra-port-7-nov-2026). Patch numbers below refer to the former patch.

## Bottom line

No issue invalidates the mouse ablation numbers wholesale:
- **Embedding order is preserved:** embeddings are attached to cells by position, and the order holds end to end.
- **Checkpoints did load:**
  - `api_version` v3 at inference vs v4 in training builds identical modules.
  - `use_layer_norm` defaults match.

Several issues change what individual arms compare, and how large the replicate variance looks:
- **Same-seed replicates:** the 3 replicates of every arm used the same seed.
- **Metatoken mismatch:** the metatoken is trained but dropped at inference.
- **Cell-neighborhood split:** the cell/neighborhood split changed on 15 April.

## Findings, most severe first

| # | Severity | Where | Issue | Affects |
|---|---|---|---|---|
| 1 | High | `train.py:55,88-91`, `dataloaders.py:18` | Seed 0 is hard-coded; no config key sets a seed. The 3 replicates per arm differ only by GPU non-determinism, so the error bars understate seed variance. | Mouse Supp. Figs. 4–6. OVCA uses `stjepa_ablation.patch`. |
| 2 | High | `embed.py:193,209,492` vs training `special_token_pad_ratio: 0` | The batch metatoken is always present in training but padded at inference. Models with a metatoken are evaluated out of distribution; the no-metatoken model is not. | The metatoken ablation (Supp. Fig. 6) and every evaluation of a model with a metatoken. |
| 3 | High | `tokenizers/tokenize.py:105` vs `cell_tokenizers.py:761` | Top-64 ties are broken by an unstable `argsort` on the sparse path but a stable `lexsort` on the dense path. With integer counts the cut falls inside a tie group in 99% of cells, so arms on different paths select different genes, not just different values. Results also depend on gene order and possibly CPU type. | Normalization ablation (no-norm and shifted-log vs the other arms); any re-tokenization. |
| 4 | High (OVCA only) | `cell_tokenizers.py:719-724,763-773,838-846` | The dense path, and Seurat/Pearson, keep every gene per cell plus float64 neighborhood arrays: about 1.2 MB per cell, ~0.5 TB for OVCA. | OVCA: 16 normalization tokenizations, which would need hugemem. |
| 5 | Medium-high | `cell_tokenizers.py:768,1485,1508` | With Seurat v3 or Pearson residuals as the rank, zero-count genes enter the top-64 (`include_zero_expr_genes=False` is violated). | 4 normalization arms. |
| 6 | Medium-high | `normalizers.py:94-95` vs `compute_norm_factors.py` / `compute_cohort_norm_factors.py` | Gene-corrected read depth uses 153.4768 + 0.0487·G in the tokenizer but other constants in the factor scripts. Count values in the gcrd-mean arms are rescaled by a dataset-dependent factor (about 1.4× between MERFISH and STARmap). | Mouse gcrd-mean arms (farm check: which factor file). OVCA factors use the tokenizer's constants. |
| 7 | Medium | `cell_tokenizers.py:1062` (`0b2ef1c`, 15 Apr) | The cell-neighborhood split changed from `input/(k+1)` (64/640) to `input/2`. Mouse cell-neighborhood models (trained 11–12 Apr, 64/640) re-tokenized after 15 Apr get index-cell genes 65–128 as "neighborhood" tokens. Also, `graph.py:142` aggregates reverse kNN (`C.T @ X`), not each cell's own 10 neighbors. | Tokenizer ablation, cell-neighborhood arms (farm check: evaluation dates). |
| 8 | Medium | `block_masking.py:279-310` | Context and target lists are trimmed to the batch minimum, so the sparsest cell sets the mask size for the whole batch. On OVCA, 3.4% of tokens are dropped at length 64 and 20% at length 256. | Sequence-length and masking-ratio ablations. |
| 9 | Medium | `tokenize.py:33` | `n_nonzero_tokens` is the full length (704) on the dense path, so the length-grouped sampler degenerates to random order for those arms. | Normalization arms (batch composition differs between arms). |
| 10 | Medium | `normalizers.py:236-238` | `loess(..., options={'span': 0.3, ...})` is probably ignored, so the default span of 0.75 is used. Farm check: `loess(...).model.span`. | Seurat arms. |
| 11 | Medium | `helper.py:367-381`, `train.py:525-527` | Schedules run to `ipe_scale`·epochs, but training stops at epochs. With `ipe_scale: 1.25`, LR ends at 2.39e-5 (not 2e-5), weight decay at 0.366 (not 0.4), EMA at 0.999 (not 1.0). | Every arm equally; Methods should state the actual end values. |
| 12 | Medium | `main.py:138-139`, `train.py:291-318`, `helper.py:144-147` | Resuming with `read_checkpoint: null` resumes rank 0 but silently restarts ranks 1–3 from scratch; DDP then averages gradients of different models. Load errors are swallowed and give random weights. | Any resumed mouse run (farm check: logs). OVCA sets `read_checkpoint` explicitly, and evaluation aborts on load failure. |
| 13 | Low | various | Low-severity items, listed below the table. | Not used in these ablations. |

Low-severity items (finding 13):
- `clip_grad` is not applied during warmup epochs (`train.py:675-687`). Kept, as in the published models; Methods should say that clipping starts after warmup.
- The predictor's segment table adds +1 row instead of +n_special tokens, which crashes with 2 or more special tokens. Fixed (patch 4).
- `use_sampler: False` means no shuffling.
- `cell_masking` is unusable.
- `decode.py` saves the last epoch instead of the best (`best_state` is not copied).
- A debug print in `block_masking.py:206` fires on every batch. Fixed (patch 19).

## Resolution (31 Oct 2026)

Fixes are in `stjepa_ablation.patch` (its header lists all changes). Everything that changes results is opt-in
or limited to code paths that were broken, so mouse configs and the default inference behave as before.

| # | Resolution | OVCA runs | Mouse |
|---|---|---|---|
| 1 | Opt-in top-level `seed`. | Seeds 1000, 2000, 3000. | Reruns use the same seeds. |
| 2 | `keep_special_tokens` at inference (all inference entry points; default: removed); padded-metatoken predictor fixed (see below), so metatoken dropout (`mask.special_token_pad_ratio`) trains correctly. | Every model with a metatoken is evaluated with it removed and kept; new arm `metatoken_dropout` (ratio 0.5). | Metatoken experiment (base, no metatoken, dropout; 3 seeds; removed and kept). |
| 3 | Opt-in `stable_rank_ties`: gene-level order by rank score, count value, reference nonzero mean (ascending), token id, in every path and block. | On. | Datasets unchanged. |
| 4 | With the options of 3 or 5, ranked lists are cut at `model_input_size` (lossless): 0.09-0.21 instead of 1.3 MB per cell. | All 27 datasets in the standard memory class (200 GB; measured peak about 83 GB for base, dense-path and 256 genes, about 105 GB for Seurat and Pearson). | — |
| 5 | Opt-in `zero_expr_from_raw_counts`: candidates are the detected genes (raw count > 0). | On. | — |
| 6 | Shared `norm_factors.py` with the tokenizer's constants; `check_setup.py` compares the factor file with the tokenizer's normalizer. | Consistent. | Farm check `check_mouse_norm_factors.py`; if it reports another fit or a train/eval mismatch, rerun the six gcrd arms with factors from `compute_norm_factors.py`. |
| 7 | `CellNeighborhoodTokenizer(seq_len_cell=...)`, opt-in `aggregate_own_neighbors` (each cell's own kNN, `C @ X`); re-tokenization at inference and the cell-only context take the split from the config. On OVCA, 79% of cells have a reverse in-degree other than 10 and own and reverse neighbour sets agree for 7.7%. | Own kNN, 64/64. | Rerun the three cell-neighborhood arms with the fix (9 runs). |
| 8 | Opt-in `mask.pad_masks_to_batch_max`: masks are padded to the longest in the batch; the context encoder and predictor ignore padded entries as keys and the loss averages over real target tokens. Verified: each cell's predictions in a padded batch equal those of the cell alone (max diff 1e-7), and training with equally long masks is bit-identical to trimming. | On (`training_options` of the plan). | Reruns use it. |
| 9 | Opt-in `exact_n_nonzero_tokens`: `n_nonzero_tokens`, which only the length-grouped sampler reads, is the number of gene tokens; on the dense path it was the full length for every cell, so the 14 dense-path normalization arms trained on batches not grouped by size (more trimming before padding). With the ranked-list options, the dense path also drops trailing pad entries (lossless; 0.10 instead of 0.21 MB per cell). Verified on all 27 plan tokenizations: only `n_nonzero_tokens` of the 14 dense-path datasets changes, and it equals the gene token count. | On. | Reruns re-tokenize with it. |
| 10 | Opt-in `seurat_loess_span` (tokenizer and `normalize_by_seurat`): the LOESS fit uses the given span, on genes ordered by log mean and log variance, so scores no longer depend on the gene order (the old fit did, at about 1e-12). On OVCA (50,000 cells), span 0.3 instead of the effective 0.75 changes the expected SDs by up to 2-5%; the top-64 gene sets are the same in 86% of cells, but their order changes in 87%. | Span 0.3 (Seurat v3 / scanpy). | Reruns re-tokenize with it. |
| 11 | Kept `ipe_scale: 1.25`, the V-JEPA value (I-JEPA's configs use 1.0), as in the mouse runs and TERRA-112M/96M; it is not a code error, but Methods must state the actual end values. `check_setup.py` reports them: LR 2.39e-5 (configured 2e-5), weight decay 0.366 (0.4), EMA 0.999 (1.0). | Same recipe. | Same. |
| 12 | Every rank uses rank 0's run folder (`main.py` passes it to rank 0 only); `load_checkpoint` starts training from scratch only if there is no checkpoint and raises if one exists but cannot be loaded (always at inference); all ranks must resume from the same epoch. Reproduced on the old code: with `read_checkpoint: null`, rank 1 looked for the checkpoint in a folder of its own, started from scratch while rank 0 resumed, and the job hung. Rank 1's load error is invisible because ranks > 0 only log errors. | Fresh runs bit-identical to before; resuming verified on 2 processes (same checkpoint, same epoch, identical weights on both ranks). | Unlikely to be affected: the mouse base config has `load_checkpoint: false`, so killed runs restarted from scratch instead of resuming (farm check below). The reruns resume, so they need the fix. |
| 13 | Predictor segment table fixed; per-batch debug print removed; the other low-severity items are unused code paths or kept by design (listed below the table). | | |

Metatoken review (31 Oct), all fixed in the patch:
- **Predictor, padded special tokens:** in batches without the metatoken (`special_token_pad_ratio` > 0), the
  `new_spc` layout put the first context gene into the first target's slot, so 1/T of the loss of those batches was
  noise. Never triggered by the mouse configs (ratio 0).
- **`restrict_special_attention` with a padded special token:** its query row had no key left (NaN without flash
  attention, spreading to all tokens). Inference read the option from `meta`, training from `mask`.
- **Inference consistency:** `nz_spc` was not passed to the inference dataset (crash for `nz_spc: False` models);
  `gene_embed_dataset` sized the special value embedding from the token dictionary instead of the config, and hard-coded
  metatoken removal and attention settings.
- **Dataset-level padding** with two or more special tokens produced too long sequences, and padded `gt_type: rank`
  special tokens were not 0.
- **`gt_type: rank` metatoken:** token id = special value + an offset hard-coded by vocabulary size; now derived from
  the token dictionary. `check_setup.py` reports whether the mouse rank arms trained with the right token.
- **Cell embeddings of cell-neighborhood models** used `(seq_len - n_special) // n_segments` index-cell tokens (11 for
  a 64/64 model); neighborhood embeddings were correct.
- Not changed (unused option, design question): `restrict_special_attention` applies to the target encoder only; the
  context encoder and predictor get no attention mask.

Tokenizer review (31 Oct), not fixed (not used by the ablations or negligible):
- (Fixed as finding 10: the LOESS span and the gene-order dependence of Seurat v3 scores.)
- The cell-graph tokenizer computes Seurat v3 on raw `X`, ignoring the cell normalization; with `cell_area`, the
  cell-neighborhood tokenizer normalizes `X` instead of `X_neighborhood`.
- Read-depth and gcrd normalizers produce COO layers (anndata 0.11 warns, 0.12 would raise).
- Pearson residuals are NaN for genes without counts (OVCA factors require counts for every gene).
- An empty neighbour gene list removes its segment and shifts the blocks (not possible on OVCA: >= 10 genes per cell);
  neighbour segments are ordered by an unstable argsort on distances (no ties on OVCA).
- Without detected-only candidates, the dense path can place pad tokens mid-sequence for Seurat / Pearson rank streams.

## Second review (31 Oct 2026)

Five reviewers (training, data, inference/evaluation, farm orchestration, end to end) and a final review of the
refactored pipeline found no High issue for the OVCA runs. Resolved:
- Dataset-level padding read the special value columns, which data tokenized without special tokens lacks (a
  regression of patch 8): pads are created without reading them; keeping the metatoken needs data tokenized with
  special tokens (patch 10).
- A checkpoint written after the first step of each epoch made resumed runs repeat that step (patch 7).
- `infer.py` / `train.py` compared the tokenizer type with 'cell_neighborhood' (patch 20).
- Runs record the pinned code version (config key `code_version`, covered by the config hash; the job stops on a
  mismatch with the pinned copy); `setup_code.sh` checks the environment before writing CODE_VERSION, isolates
  `git apply` from an enclosing repository and verifies the patch applied.
- The manifest rejects empty fields and the job reports missing rows; W&B ids are short
  (`<tag>_<index>_<hash>`, display name = run id).
- Evaluation: float32 embeddings, a mean-centred average cosine similarity next to the uncentred one,
  `NUMBA_CPU_NAME=generic`, several silver files and label keys; the completion marker records the label keys.
- Comparison: SD with ddof 0 (as the mouse figures), missing runs reported, readable panels.
- The scripts are dataset-agnostic (all values as parameters; `datasets/<dataset>.env`).
- Tokenization waits for `<norm factors>.done`; two arrays of one campaign cannot run at once.
Open: batch-integration metrics (iLISI, ASW, MMD) for the mouse rerun (added on 1 Nov: seeded, over the labelled
cells as in the notebook). Kept: worker seeds equal across ranks (about 0.6% of steps share their segment count and
metatoken-drop decision across ranks).

## Third review (6 Nov 2026): findings of the TERRA reviews

The TERRA reviews of November checked TERRA, not the pinned code. Of their findings, one also applies to the
ablations; the others are in paths the ablations do not use or are fixed by the patch:
- **Cells without a detected gene of the token dictionary** (TERRA H2, fixed there in 7f6a30c). Such a cell has an
  empty ranked gene list (with `zero_expr_from_raw_counts`: no vocabulary gene with raw count > 0). In the pinned
  `CellGraphTokenizer._format_examples`, the segment count is 1 + the number of distinct neighbor segment ids
  present, so on the sparse ranking path (most arms, including base) an empty neighbor removes its segment:
  - the cells around it get segments of `model_input_size / (n_neighs + 1 - k)` tokens instead of the fixed
    blocks the datasets read;
  - the loop over ids 2..n keeps the empty id (as padding) and drops the farthest neighbor.

  The patched dense path keeps one entry per empty cell, so its segment layout is unaffected (patch 15). Resolved
  at the data level:
  - `check_setup.py` (check 2) fails while a silver cell has no detected vocabulary gene;
  - `filter_silver_cells.py` writes silver copies without these cells (mouse: `..._nonempty`), and
    `prepare_ovca_data.py` removes them. OVCA's harmonization already required 10 detected genes per cell.

  Removing the cells before tokenization also removes them from the neighbor graph (TERRA's default since
  7f6a30c), and the factors, tokenization and evaluation read the same cells.
- Not used by the ablations: ALiBi (not in the pinned code), `agg_type='softmax'` (evaluation pools with `avg`),
  cell-area normalization, cycle consistency, token distance, fine-tuning, perturbation.
- Fixed by the patch: padded-metatoken predictor (3, 4), metatoken at inference (10, 11), cell-only context length
  (11, 17), `nz_spc` at inference (11), stable rank ties (13), dense-path `n_nonzero_tokens` (15), the
  cell-neighborhood split and own-kNN sum (17, 18), mask padding (6), resume (7).
- Pooling over batches with padded special tokens: evaluation never mixes sequences with and without the
  metatoken (`drop` removes it from all, `keep` keeps it in all), so it is not affected.

## Coordinate (spatial-encoding) ablation

- **Data preparation is correct (`mouse_brain_align_coords.ipynb`).**
  - STARmap was scaled ×0.194 to µm and MERFISH was already in µm, so both are in µm. kNN scales agree within about 5%, and orientation is consistent.
  - The aligned silver files still carry stale `obsp['spatial_distances']` in raw units; the tokenizer recomputes them.
- **What "Coordinate" is.** It is the v2 additive 2D sin-cos embedding of raw µm displacements (`cell_pos_enc: coord`; RoPE did not exist before 21 Apr). Its training config was never committed. "Coordinate is worse than rank" therefore holds only for this encoding, not for RoPE.
- **Evaluation data.**
  - Figure versions made before 14 Apr evaluated the coordinate model on unaligned coordinates (STARmap in voxels). Those are invalid (iLISI 2.4e-7).
  - The current figure, from 14 Apr on, uses the aligned data and is valid.
- **Comparability.**
  - The rank reference run `21032026_085513_196` predates the v2 repository, i.e. it comes from a different code snapshot.
  - Model folders were filled by copying `*-latest` checkpoints. Confirm they are final-epoch checkpoints: the 14 Apr runs started at 00:15 and 00:34 and were already in the 11:08 figure.
- **RoPE and no-encoding runs (not in the figure).**
  - Training ignores `rope_freq_scale` and `rope_rotation_augment`.
  - π-scaled frequencies alias on µm coordinates.
  - The rotation is redrawn on every attention call, and the target encoder is never put in eval mode.
  - `none` removes which-cell membership entirely.

## Farm checks

```bash
A=/nfs/team361/sb75/stjepa-reproducibility/artifacts   # adjust if the folder was renamed
# Coordinate and rank runs: encoding, dataset, epoch of the evaluated checkpoint
for ts in 21032026_085513_196 11042026_190115_580 11042026_202814_332 10042026_193042_980 14042026_001543_594 14042026_003429_740; do
  echo "== $ts"; grep -E 'cell_pos_enc|tokenized_data_folder_path|special_tokens|epochs:' $A/models/$ts/model_config.yaml
  python -c "import torch; c = torch.load('$A/models/$ts/model_checkpoint.pt', map_location='cpu', weights_only=False); print('epoch', c['epoch'], 'iter_number', c.get('iter_number'))"
done
# Cell-neighborhood runs: layout and evaluation date (re-tokenization after 15 Apr is misaligned)
for ts in 11042026_110853_342 11042026_142119_393 11042026_142613_749; do
  grep -E 'seq_len_cell|seq_len_neighborhood|model_input_size|n_segments|tokenized_data' $A/models/$ts/model_config.yaml
  ls -l --time-style=long-iso $A/ablation/stjepa_v2_cell_neigh_tokenizer_*/$ts/*res_sweep.csv
done
# Resumed runs (finding 12): ranks > 0 log errors only, so their failed loads are invisible; a resumed run shows
# "Loaded pretrained encoder" in the rank-0 output (with read_checkpoint: null, its ranks 1-3 trained from scratch).
# The mouse base config has load_checkpoint: false, so none is expected.
grep -l "Loaded pretrained encoder" $A/mmb0-1b_smb1-1b_2p*/*_o.* 2>/dev/null
```

`check_setup.py` compares every generated config with the mouse `params.yaml` of its runs. That settles the inferred overrides (cell-neighborhood layout, predictor width, coordinate encoding).

## TERRA port (7 Nov 2026)

The figure is rerun on the current code: the pipeline pins TERRA `d29e262` (dev, 8 Nov 2026) instead of stjepa
`b433d5e` plus the patch. Each change of the former patch, in TERRA:

| Patch | TERRA |
|---|---|
| 1 seed, 2 atomic latest checkpoint, 6 padded masks (`mask.pad_masks_to_batch_max`), 7 resumption from rank 0's folder and raising load errors, 19 no per-batch print, 20 `cell_neigh` comparison | same, under the same names |
| 3 predictor in batches with padded special tokens, 5 padded special token attending to itself, 10 `keep_special_tokens`, 11 inference consistency (restriction from `mask`, `nz_spc`, cell-only context length), 12 options in re-tokenization | same (TERRA commits 10b29d8, 1151e73, fe894f5, 9697067, 1367db2, 87d3262) |
| 4 predictor segment table with two or more special tokens | not needed: the ablations use at most one metatoken |
| 8 dataset-level padding of special tokens | one pad per special token, without reading the special value columns, as in the patch. A padded `gt_type: rank` special token gets the value offset instead of 0 in the dataset, but the collator zeroes padded special tokens before the model sees them |
| 9 `gt_type: rank` offset from the token dictionary | still hard-coded by vocabulary size (1674 for the 2785-token mouse dictionary, else 21957); `check_setup.py` (check 2) fails if it differs from the dictionary's offset while the plan has a rank arm |
| 13 `stable_rank_ties` | same option; ties are ordered by rank score, then count value (dense path), then token id. With `rank_tie_break_norm_factor: mean` (TERRA fefbf01, set in the plan), the corpus mean of the norm factor file (ascending: rarer and lower-expressed genes first) comes before the token id, and since 5d8ecbc it replaces the count value on the dense path, so both paths order ties the same way. The patch used the nonzero mean there, which measures the level where a gene is detected, not how often it is detected |
| 14 `zero_expr_from_raw_counts`, 16 `seurat_loess_span`, 18 `aggregate_own_neighbors` | same options |
| 15 `exact_n_nonzero_tokens` | always on (TERRA a936f2b); the ranked lists are cut at the input size on both paths, without the patch's extra trailing-pad trim |
| 17 cell-neighborhood split | `split_cell_neigh_equally` (default): `seq_len_cell` = input size / 2, as the plan's 64 + 64 |

Pipeline changes for TERRA:
- `base_config.yaml` is TERRA's mouse config for this dataset with the ablation's model and schedule values.
  `attention_backend: torch` lets PyTorch choose the SDPA kernel, as the mouse runs and the released models did; new
  TERRA options are off.
- TERRA raises if `state.load_checkpoint` is set and the checkpoint is missing. The job array therefore writes a launch
  config that resumes only once `terra-latest.pth.tar` exists, and a checkpoint that cannot be loaded still raises.
- The checkpoint name follows `state.write_tag` (`terra-latest.pth.tar`). Training, the launch script and the
  embedding come from the pinned copy (`src/terra/training/main.py`, `scripts/train_model_lsf_mpirun_sanger.sh`,
  `terra.inference.embed.embed_dataset` without the spatial cell embedding).
- Tokenized datasets no longer carry `_nnz`, so they are never mixed up with the datasets of the former pinned code.

Validated on a synthetic two-section mouse-like campaign (2785-token dictionary, 4 cells without counts) with the README
steps:
- setup, filter, factors, generation and both `check_setup.py` passes;
- all 27 tokenizations;
- 2-epoch CPU training through TERRA's `main.py` (gloo; a test-only shim drops the GPU device ids of DDP), packaging
  and evaluation in both metatoken modes for the base, rank, counts, cell-neighborhood, value-bin, coordinate, no
  metatoken and metatoken-dropout arms, and a resumed run.

### Corpus-mean tie-break (7 Nov 2026, TERRA `fefbf01`)

With raw-count ranks (the base and every arm whose rank is not normalized), counts often tie at the per-cell cut, so
the tie order decides which genes are kept. The plan sets `rank_tie_break_norm_factor: mean`: tied genes are ordered by
the corpus mean of the norm factor file (ascending, so rarer and lower-expressed genes come first), then by token id.
Unlike the patch's nonzero mean, the mean also counts the cells in which a gene is not detected (it is the detection
rate times the nonzero mean, over the cells whose panel probes the gene). Arms that divide only the counts by `mean` already ordered ties that way on the dense path, so the base
now keeps and orders the same genes as the "Mean" arm with unnormalized ranks. `check_setup.py` (check 5) verifies the
column and reports how many cells each per-cell length cuts inside tied counts.

Validated:
- TERRA: its tests pass (542). Without the option, the tokenization is bit-identical to `5593017`. With it, the values
  are unchanged and every segment is ordered by count, `mean` and token id.
- Pipeline, on the synthetic campaign: setup, generation, both `check_setup.py` passes and all 27 tokenizations. The
  base and counts-divided-by-`mean` datasets hold identical gene tokens, and the base values equal those of the
  token-id tokenization. The base, cell-neighborhood and "Mean" (unnormalized ranks) arms train, package and evaluate
  (2 epochs on CPU). Re-tokenizing a section from a packaged model folder (TERRA's `tokenize_adata`, which reads the
  option and the bundled `norm_factors.csv`) reproduces the training tokens.

### Audit fixes (8 Nov 2026, TERRA `5d8ecbc`)

From the second audit of the pipeline and TERRA (8 Nov 2026):
- **Dense-path tie order.** With `fefbf01` the dense path (ranking and count values normalized differently) still ordered
  tied genes by their normalized count before the corpus mean. 5 of the 9 arms with unnormalized ranks (gene-corrected
  read depth mean, nonzero mean, gene-corrected read depth nonzero mean, Seurat v3, Pearson residuals) therefore kept
  other genes than the base: 0.4-3.1 of 64 genes in 37-73% of the cells of the synthetic campaign (with token-id ties,
  about 9 genes in 81%). With the option set, TERRA `5d8ecbc` no longer uses the count value to order tied genes, so
  every arm with unnormalized ranks keeps the base's genes and only the values differ.
- **Stored zeros.** `filter_silver_cells.py` removes zeros stored explicitly in a sparse X; the sparse ranking path would
  turn them into zero-count gene tokens. Check 2 fails on them.
- **Dataset provenance.** The dataset name records the tokenization settings, but not the TERRA code or the input files,
  so a re-pin or recomputed factors silently reused stale datasets. `tokenize_dataset.py` now writes
  `ablation_provenance.json` (code version, tokenization, md5 of the token dictionary, the factor file and the silver
  files) and skips an existing dataset only if it matches; check 4 compares it. Datasets tokenized before have no such
  file and must be deleted.
- **Pre-launch checks.** Check 2 fails if two sections share a batch special value; check 3 fails while
  `UNASSIGNED_LABEL` is the placeholder of the env file.

### Audit fixes, round 3 (8 Nov 2026, TERRA `d29e262`)

From the two-round audit of the mouse campaign (7-8 Nov 2026) and the reviews of its fixes. TERRA (dev `d29e262`,
all opt-in changes off by default):
- **Contexts of padding only.** `meta.zero_fully_masked_attention_rows` (set in `base_config.yaml`, required by
  `generate_ablation_runs.py`, checked by `setup_code.sh` and check 0): an attention query without a valid key outputs
  zeros with finite gradients instead of NaN, on every attention path and whatever the SDPA kernel; all other rows are
  bit-identical. This makes the no-metatoken and metatoken-dropout arms safe on any GPU kernel; arms with a metatoken
  are unchanged.
- **Non-finite training state.** Training stops with `FloatingPointError` at the first non-finite loss or gradient
  norm, before the optimizer step and on all ranks at the same step, and never writes a non-finite checkpoint.
- **Resume.** Checkpoints hold the random number generator states of every rank, and a resumed run restores them, so
  it continues the mask and worker-seed stream; all ranks must resume from the same epoch (the check missing from the
  port of patch 7). A run that falls back to a fresh start (`allow_checkpoint_load_failure`) keeps the fresh stream.
- **Launcher and data loading.** `main.py` honours `WANDB_MODE` (e.g. `offline`). DataLoader workers are started with
  `fork` explicitly; they always were on the farm (the `spawn` request never took effect after `load_from_disk`), so
  the re-spawn cost of the training audit does not apply.
- **Equidistant neighbours.** With `stable_rank_ties`, the cell graph tokenizer orders equidistant neighbours with a
  stable sort, so the segments do not depend on the CPU type.
- From the review: the eager attention guard saves no second attention matrix for backward, and the per-step gradient
  norms use the fused kernels of `clip_grad_norm_`.

Pipeline:
- **Replicates.** The 21 normalization arms and 3 predictor layers have 3 replicates; the 44 new runs are manifest rows
  96-139 (seeds 2000 and 3000), so rows 1-95 keep their index (139 runs).
- **Configs.** `data.vocab_size` is the size of the token dictionary (was the stale human 23407).
- **Shared-gene evaluation.** `filter_shared_genes.py` writes the silver files restricted to the genes of every
  section, without cells with fewer than 10 detected shared genes (counted as `sc.pp.filter_cells`), with its
  provenance written last. `submit_tokenization.sh` tokenizes them with every tokenization after the training array
  (`EVAL_GENE_SET=shared_genes`), `tokenize_dataset.py` refuses an incomplete or outdated folder, and
  `evaluate_ablation_run.py` embeds them; OVCA (one section) embeds its training data.
- **Checks.** Check 0: the pin knows the attention-row option. Check 1 fails on a cell-neighborhood layout that cannot
  read the mouse runs' tokens (for 64/640 it prints the plan change), on missing mouse params and on replicates that
  differ in more than seed and folder. Check 2: median nearest-neighbour distance per section (fails above a 2x
  ratio), the batch metatoken id hard-coded in TERRA against `spt_batch`, and `vocab_size`. Check 5: features without
  Ensembl id, factor or counts (`compute_norm_factors.py` names genes without counts, and stops there first). Check 7
  (report): share of cells too sparse for a context gene at each mask ratio.
- **Attention kernel.** `check_attention_kernel.py` runs the context encoder of a campaign config on one farm GPU with
  a sample whose context is padding only, with the option false and true, and fails unless true is finite and leaves
  the other samples bit-identical.

Evaluation, launch and reporting:
- **Evaluation provenance.** `evaluation_complete` is removed when an evaluation starts and written last, and records
  what produced the metrics (modes, labels and unassigned label per key, sha1 of `evaluate_ablation_run.py` and of the
  part of `ablation_utils.py` it uses, TERRA version, checkpoint epoch and md5, evaluation data path and provenance
  md5, precision, GPU, torch). `--check_only` (exit 0 / 3 / other) compares it with the current inputs and requires
  every output file; any mismatch re-evaluates all modes. The evaluation stops if the evaluation data was tokenized
  differently from the training data (provenance) or repeats a cell id.
- **Evaluation data outside the run ids.** The configs no longer hold the evaluation dataset
  (`eval_tokenized_data_folder_path`), which was covered by the config hash: a changed `EVAL_GENE_SET` or data location
  gave all 139 runs new ids (retraining), and a finished run would have kept evaluating the path in its `params.yaml`.
  `evaluate_ablation_run.py` now derives the path when it runs from the run's training dataset and `--eval_gene_set`
  (`get_eval_dataset_path`), which the job arrays (`ablation_job_setup.sh`), `list_pending_runs.py` and the submit
  scripts take from `dataset.env`, so all entry points agree; `generate_ablation_runs.py` checks that the derived path
  is where `tokenize_dataset.py --gene_set` writes. A changed `EVAL_GENE_SET` changes the recorded `eval_data_path`, so
  `--check_only` exits 3 and the trained runs are re-evaluated on the new data without retraining. Run ids differ from
  those of dev 44aefbe.
- **Finiteness.** Non-finite embeddings stop the evaluation before any metric, with counts per mode, section and
  labelled cells; `package_run.py` refuses non-finite target-encoder weights.
- **Collapse.** Participation ratio, PCs for 90% of the variance, top PC share, RankMe, between-cell variance fraction
  and per-dimension SD; flags for complete collapse (variance fraction below 1e-3) and one-direction collapse (top PC
  above 95%; the participation ratio flagged full-rank embeddings with one dominant axis, e.g. two separated
  sections). `compare_ablations.py` adds a flag for a participation ratio or variance fraction below 25% of base.
- **Granularity and integration.** AMI and the cluster and label counts per resolution and key; `iLISI_relative` =
  iLISI / iLISI of the same graph with the sections permuted (random mixing does not reach the analytic bound
  `iLISI_max`: about 0.9 of it at 2:1); `ASW_label_<key>` for keys shared by sections (none for mouse); per-label
  cross-section neighbour shares. `--fp32_embeddings` (`EVAL_FP32=1`) embeds in fp32.
- **Unassigned labels per key.** `UNASSIGNED_LABEL` is one value or `<key>=<value>` pairs (`;`); check 3 counts per
  key and file.
- **Launch.** Jobs skip a run only if `--check_only` says its evaluation is up to date (`FORCE_EVAL=1`: always
  re-evaluate). The submit scripts check every row of the range on the login node (`list_pending_runs.py`, outside the
  hashed code) and submit only rows that need work; `RUN_LIMIT` sets `bsub -W`. `submit_evaluation_array.sh` /
  `evaluate_ablation_array.lsf` evaluate trained runs on 1 GPU each, and `SEPARATE_EVALUATION=1` sends every
  evaluation there (element N after training element N). Options on the command line override the env file.
- **Reporting.** Each run is averaged over the resolutions first; bars are mean +/- SD (ddof 1) or 95% CI over
  replicates with n, single-run arms are compared with base through the pooled seed SD (a band) and seed-paired
  differences, and `--error_bars published` reproduces the published panels (implies `--category_mode drop`). Category
  panels use the training input; `metatoken_contrasts` gives the seed-paired contrasts. Differences to base and the
  relative collapse flags use base in the same metatoken mode (No Metatoken: the panel mode). Only runs with
  `evaluation_complete` and the current evaluation data (`EVAL_GENE_SET`) are used; the gene set is stated in every
  figure.

Verified on CPU only (TERRA 584 tests; the synthetic two-section campaign with the pinned copy: generation, shared-gene
data, tokenization, check 0, the kernel check with the real guard, 2-epoch training of the base and no-metatoken arms,
evaluation on the shared-gene data). Still open, on the farm: the kernel check on a training GPU (step 8), the NCCL path
of the RNG gather and the non-finite reduction (the pilot), the cell-neighborhood layout of the mouse params (check 1)
and the datasets version of the venv (the `fork` claim rests on `load_from_disk` fixing the start method). The
evaluation, launch and reporting changes were run on CPU (evaluation of the tiny runs, every `--check_only` case, both
job scripts locally and both submit scripts with fake `bsub`/`bjobs`, `compare_ablations.py` on a known-answer tree);
open on the farm: real `bsub`/`bjobs` output, the one-to-one `ended(<job id>[*])` dependency of
`SEPARATE_EVALUATION` (test it on a 2-element array first), the 1-GPU evaluation job's queue, memory and 4 h limit,
the login-node time of the row checks (one 750 MB md5 per trained row), bf16 vs fp32 on the GPU, the collapse
thresholds on real embeddings (pilot) and `RUN_LIMIT` (from the pilot's slowest arm).

## Methods notes

What the Methods (or the Supplement) must state about the reruns:
- **Tie order.** Raw-count ties at the per-cell cut are ordered by ascending corpus mean (`mean` of the factor file),
  then token id, the same for every cell. Within a tie, the rank position therefore encodes the gene's corpus mean, a
  positional cue the predictor can use; there is no per-cell random tie order. Check 5 reports how many cells each
  sequence length cuts inside ties.
- **Contexts without a gene.** With `n_targets` 1 the targets are ceil(r K) of a sequence's K gene tokens. In
  index-cell-only batches (one kept segment, probability 1/11) a cell with K < 1/(1-r) genes (K <= 2 at r 0.6, <= 4 at
  0.8, <= 9 at 0.9) has no context gene: its targets are predicted from the metatoken and positions, or, without a
  metatoken (or with it dropped), from an empty context, whose attention rows output zeros
  (`meta.zero_fully_masked_attention_rows`). Report the per-section shares of check 7 at r 0.6, 0.8 and 0.9.
- **Effective schedules and objective.** `ipe_scale` 1.25 stretches the schedules over 1.25 x epochs and training
  stops at 80%: LR 2e-5 -> 6e-5 -> 2.39e-5, weight decay ends at 0.366 (0.4), EMA at 0.999 (1.0) (check 6 prints
  them). Targets come from a segment-subsampled graph (`sample_segments`: k = 1 with probability 1/11, else uniform
  2-11 of the 11 cells, for both encoders), while evaluation embeds all 11 cells. The loss is L1 between the predictor
  outputs and the affine-free layer norm of the target-encoder outputs at target gene tokens (metatoken excluded),
  averaged over the batch's target tokens; gradients are clipped at 2.0 per module from epoch 3.
- **Masking.** A fraction r (rounded up) of all non-padding gene tokens of the sampled sequence, drawn uniformly across
  its cells, forms the targets and the rest (and the metatoken) the context; it is not stratified per cell.
- **Shared-gene evaluation, as published.** With two sections (mouse), embeddings, clusters and integration metrics
  come from re-tokenizations of the training cells restricted to the genes measured in both sections, as the published
  evaluation. Differences: the reruns tokenize them with the training tokenizer settings and factor file, keep
  unlabelled cells (the published evaluation removed them before tokenizing, so its neighborhoods lacked them) and
  drop cells with fewer than 10 detected shared genes (as published). The gene-corrected read depth target uses each
  file's gene count, so on the shared genes the count values of those arms are scaled by T(G_shared)/T(G_panel) per
  section (ranks unchanged). OVCA (one section) embeds its training data.
- **Arms.** `gt_type` changes both the encoder input and the predictor query (rank + segment for rank and combined, gene
  id + segment for counts). The value-bin arms set the size of a learned soft codebook on raw counts, and the MLP
  count encoder acts on raw counts. The cell-neighborhood segment is the raw sum over the index cell and its 10 own
  nearest neighbours, ranked, top 64, and its neighborhood embedding pools that segment only (cell graph: all tokens).
  The layout is 64 + 64 unless check 1 shows otherwise. The coordinate arm computes its sincos phases in bf16, as the
  mouse runs (`meta.fp32_position_encoding` off), and about 43% of its channels are near-constant at the 10-60 um
  scale of the neighborhoods. Architecture arms vary width or depth at fixed LR, heads, EMA
  and weight decay, with the predictor width tied to the encoder width; the base predictor has one full-width layer,
  and 3 predictor layers has 3 replicates. Sequence-length arms also change the targets and the attention cost per
  step.
- **Normalization factors** come from the two sections, pooled per gene over the sections that measure it, with the
  tokenizer's gene-corrected read depth constants.
- **Compute.** With padded masks every attention call carries a mask, so SDPA uses a kernel that supports masks, not
  FlashAttention; embeddings are computed in bf16 autocast on the GPU (of the training job, or of a 1-GPU job of the
  same memory class with `SEPARATE_EVALUATION`), so only the CPU part (kNN graph, Leiden) is independent of the node.
  `evaluation_complete` records the GPU, torch version and precision of every run; fp32 is available
  (`EVAL_FP32=1`).
- **Sampling and seeds.** Length-grouped batches (Geneformer-style mega-batches sorted by length). DataLoader worker
  seeds are equal across ranks (about 0.6% of steps share their segment count and metatoken-drop decision across
  ranks). First replicates with the same model shape share initial weights, data order and worker seeds (a paired
  design). A resumed run continues the random streams and matches an uninterrupted run up to GPU nondeterminism
  (end-of-epoch checkpoints; mid-epoch checkpoints are never written here, as an epoch has 722 (mouse) or 2,646
  (OVCA) iterations, below `checkpoint_freq_iter` 50,000).
- **Metrics.** The integration metrics are label-free and reward mixing even without shared biology; the iLISI ceiling
  depends on the section sizes, so the figures use `iLISI_relative` (iLISI over the iLISI of permuted sections on the
  same graph). For the mouse keys, each labelling one section, the label-aware readout is the per-label share of
  cross-section neighbours; a region-conditioned batch ASW needs a key shared by the sections. Collapse is assessed
  from the spectrum and scale of the embeddings (participation ratio, top PC share, between-cell variance fraction),
  not the average cosine similarity (centred about 0 by construction, raw about 0.99 for healthy and collapsed
  embeddings alike). Cluster counts differ between arms at a fixed resolution, so AMI and a granularity-matched score
  are reported with NMI and ARI.
- **Statistics.** Each run is averaged over the 9 resolutions; error bars are the SD (ddof 1) over the replicates of
  these run means (the published SD over resolutions x replicates mixed the resolution sweep with the seed spread and
  had no seed spread for single-run arms). Single-run architecture arms are compared with base through the pooled
  within-arm seed SD (df from the arms with 2 or more runs; 88 when complete) and seed-paired differences.
- **Metatoken modes.** The category panels show each run with its training input (metatoken kept for runs trained with
  it, removed for No Metatoken); removing it from a model trained with it is out of distribution, so the published
  drop-mode panels are available separately, and the metatoken contrasts are seed-paired.
