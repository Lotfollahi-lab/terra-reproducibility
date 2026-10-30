# ST-JEPA v2 code review and mouse-ablation audit (30 Oct 2026)

**Scope:**
- **Model code:** `sebastianbirk/stjepa` branch v2: tokenizers, datasets, masking, model, training, inference.
- **Ablation analysis:** `sebastianbirk/stjepa-reproducibility` `analysis/mouse_ablation`.
- **Coordinate ablation:** the data preparation behind it.

**Method:** the code was read and partly executed on OVCA data. Nothing was run on the farm, so items marked "farm check" are open. The code itself was not changed, apart from `stjepa_ovca_ablation.patch` used for the OVCA runs (opt-in seed, atomic latest checkpoint).

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
| 1 | High | `train.py:55,88-91`, `dataloaders.py:18` | Seed 0 is hard-coded; no config key sets a seed. The 3 replicates per arm differ only by GPU non-determinism, so the error bars understate seed variance. | Mouse Supp. Figs. 4–6. OVCA uses `stjepa_ovca_ablation.patch`. |
| 2 | High | `embed.py:193,209,492` vs training `special_token_pad_ratio: 0` | The batch metatoken is always present in training but padded at inference. Models with a metatoken are evaluated out of distribution; the no-metatoken model is not. | The metatoken ablation (Supp. Fig. 6) and every evaluation of a model with a metatoken. |
| 3 | High | `tokenizers/tokenize.py:105` vs `cell_tokenizers.py:761` | Top-64 ties are broken by an unstable `argsort` on the sparse path but a stable `lexsort` on the dense path. With integer counts the cut falls inside a tie group in 99% of cells, so arms on different paths select different genes, not just different values. Results also depend on gene order and possibly CPU type. | Normalization ablation (no-norm and shifted-log vs the other arms); any re-tokenization. |
| 4 | High (OVCA only) | `cell_tokenizers.py:719-724,763-773,838-846` | The dense path, and Seurat/Pearson, keep every gene per cell plus float64 neighborhood arrays: about 1.2 MB per cell, ~0.5 TB for OVCA. | OVCA: 16 normalization tokenizations, which run on hugemem. |
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
- `clip_grad` is not applied during warmup epochs (`train.py:675-687`).
- The predictor's segment table adds +1 row instead of +n_special tokens, which crashes with 2 or more special tokens.
- `use_sampler: False` means no shuffling.
- `cell_masking` is unusable.
- `decode.py` saves the last epoch instead of the best (`best_state` is not copied).
- A debug print in `block_masking.py:206` fires on every batch.

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
A=/nfs/team361/sb75/stjepa-reproducibility/artifacts
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
# Resumed runs (finding 12)
grep -l "Loaded pretrained encoder\|Encountered exception when loading" $A/mmb0-1b_smb1-1b_2p*/*_o.* 2>/dev/null
```

`check_setup.py` compares every OVCA config with the mouse `params.yaml` of its runs. That settles the inferred overrides (cell-neighborhood layout, predictor width, coordinate encoding).
