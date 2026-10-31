# OVCA ablations

The mouse ablations of the TERRA paper (Supp. Figs. 4–6; `sebastianbirk/stjepa-reproducibility`,
`analysis/mouse_ablation`) are rerun on the SpatialFusion Xenium Prime 5K ovarian cancer section, with its
pathologist niche labels as ground truth. Models are trained from scratch with the same code, base config and
launch setup as the mouse runs.

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

**Evaluation** follows the mouse protocol:
- Neighborhood embeddings from the target encoder.
- kNN graph with k = 15.
- Leiden on the fixed grid of resolutions 0.6–1.4. The grid is identical for all variants, so the protocol is label-free.
- NMI and ARI against `path_region`. `Unassigned` cells are clustered but not scored.
- Models with a metatoken are evaluated twice: metatoken removed at inference (as in the mouse evaluation; the
  category panels) and kept as in training (`keep_metatoken` outputs; panel `metatoken_inference`).
- Batch-integration metrics do not apply to a single section.

## Differences from the mouse runs, and why

| Item | Mouse | OVCA | Reason |
|---|---|---|---|
| Code | farm checkout of the time | copy of stjepa `b433d5e` + `stjepa_ovca_ablation.patch` (`setup_code.sh`) | Same code as the April runs; v2 later only added debug prints. A separate copy, so changes to the farm checkout (now `/nfs/team361/sb75/terra`) cannot change running jobs. The patch (its header lists all 17 changes) adds an opt-in seed, an atomic latest checkpoint, the metatoken fixes (`CODE_REVIEW.md`), padded masks and opt-in tokenizer fixes; mouse configs train as before. |
| Tokenization | ties at the top-64 cut broken by gene order and sort algorithm; Seurat / Pearson ranks include undetected genes; dense-path cells all stored with the full token count, so their batches were not grouped by size; Seurat v3 LOESS span silently 0.75; cell neighborhood aggregated over reverse kNN | `tokenizer_options` of the plan: deterministic gene-level ties (rank score, count value, reference nonzero mean, token id), detected genes only, the real gene token count, Seurat v3 LOESS span 0.3 (gene-order independent), each cell's own kNN for the cell neighborhood | `CODE_REVIEW.md` issues 3, 5, 7, 9 and 10. The options also cut ranked lists losslessly (0.09-0.10 instead of 1.3 MB per cell), so all but the 256-gene dataset run in the standard memory class. Dataset names carry the options (`_ties_detected_nnz`, `_span0p3` for Seurat, `_ownknn_cell64`). |
| Context and target masks | trimmed to the shortest in each batch: the sparsest cell sets how many tokens every cell uses | padded to the longest (`training_options`: `mask.pad_masks_to_batch_max`), with padded entries masked in attention and loss | `CODE_REVIEW.md` issue 8: trimming dropped 3.4% of the sampled tokens at 64 genes per cell and 20% at 256 on OVCA, biasing the sequence-length comparison. |
| Metatoken at inference | removed | removed (category panels) and kept (`metatoken_inference` panel) | Training always sees the metatoken; dropping it at inference is out of distribution unless the model was trained with metatoken dropout. |
| Schedule | 300 epochs, 3 warmup (216,600 steps) | 82 epochs, 1 warmup (216,972 steps) | Matched in optimizer steps: OVCA has 3.66× the iterations per epoch. All epoch-based schedules (LR, WD, EMA) scale with it. |
| Replicate seeds | all seed 0 (`_GLOBAL_SEED`, inherited from the I-JEPA training script), so replicates differed only by GPU non-determinism | seeds 1000, 2000, 3000 | Real seed variance (rebuttal E10): the seed sets weight initialization, data order, masks and segment sampling. The spacing keeps the samplers' per-epoch seeds (seed + epoch) disjoint. |
| `n_special_values` | 200 | 1448 | The human batch special value (about 213) exceeds 200 rows and would crash training. |
| Checkpoints | every 5 epochs (about 90 GB per run) | only `stjepa-latest.pth.tar`, in `runs/<run>_<config hash>`; killed runs resume on all ranks | Storage and walltime safety. A changed config gets a new folder instead of resuming an old one. |
| Evaluation input | separate data, re-tokenized with `harmonize_tokenize_embed_pipeline`; unlabelled cells removed first | the run's own tokenized training dataset (`embed_dataset`); all cells, so `Unassigned` stays in the kNN graph | Same section and identical tokens. Re-tokenizing needs about 0.5 TB for 16 normalization arms and breaks top-64 ties non-deterministically. Absolute metric levels are therefore not comparable with the mouse numbers. |
| Norm factors | cohort file of the mouse data | computed on the section (`prepare_ovca_data.py`) with the tokenizer's gene-corrected read depth constants | The factors must cover every gene and match the tokenizer. |
| Dataset metadata | — | dataset id 50, batch0 (same 10x section as `xho50-1b_1p`) | The token dictionary only has special values for existing batches. |

**Inferred from run names; verified by `check_setup.py` against the mouse `params.yaml` before launch:**
- The cell-neighborhood layout: 64 cell + 64 neighborhood tokens (input size 128, `seq_len_cell: 64` in the plan, as in later v2 tokenizers). The dataset class needs exactly that many tokens.
- `pred_emb_dim` following `enc_emb_dim` in the width ablation.
- The coordinate arm being `cell_pos_enc: coord`.

**Caveats:**
- **Not held out from TERRA-112M pretraining.** This section is `xho50-1b_1p` in the TERRA-112M corpus. That doesn't matter here, because every arm is trained from scratch on this section alone.
- **Known code issues** are listed in `CODE_REVIEW.md` with their resolution. The OVCA tokenizations use the tokenizer fixes, which the mouse tokenizations lacked, so arm-level differences between mouse and OVCA can partly reflect them (most for the Seurat / Pearson and cell-neighborhood arms).
- **Schedules end at 80%:** `ipe_scale: 1.25` (the V-JEPA value; I-JEPA's configs use 1.0) stretches the LR, weight decay and EMA schedules over 1.25 x epochs, and training stops after `epochs`. They end at LR 2.39e-5 (configured 2e-5), weight decay 0.366 (0.4) and EMA 0.999 (1.0), as in the mouse runs and TERRA-112M/96M. `check_setup.py` prints these values for Methods.
- **Gene-corrected read depth on one section:** with a single panel the gene correction is a constant, so the "Gene Corr. RD" arms are read-depth arms here (the mean / nonzero-mean ones are exactly read-depth-then-gene-mean; the shifted-log one normalizes to about 400 counts instead of 10,000). The panel-size correction needs two or more panels.

## Steps (on the farm)

```bash
cd /nfs/team361/sb75/terra-reproducibility && git pull && cd analysis/ablation/ovca_ablation
bash setup_code.sh                                   # pinned code at /nfs/team361/sb75/stjepa_ovca_ablation + env check (stops if an older patch version is there: move it away first)
mkdir -p ../../../logs/ablation/ovca_ablation && bsub < prepare_ovca_data.lsf   # silver + norm factors (terra env)
source /nfs/team361/sb75/.venvs/stjepa_v2/bin/activate && export PYTHONPATH=/nfs/team361/sb75/stjepa_ovca_ablation/src
python generate_ablation_runs.py                     # 95 configs + manifest.tsv (82 epochs, 1 warmup)
bash submit_tokenization.sh standard                 # 26 datasets, normal queue (ranked lists cut: up to ~70 GB)
bash submit_tokenization.sh large                    # 1 dataset (256 genes), hugemem queue
python check_setup.py                                # pinned code, config parity with mouse runs (review), special values, datasets, norm factors (fails on errors)
python check_mouse_norm_factors.py                   # mouse gcrd arms: factor constants and train/eval consistency (report)
# (if the mouse run folder moved: python check_setup.py --mouse_artifact_folder_path <.../artifacts>)
bash submit_ablation_array.sh 1                      # pilot: base_rep1 (train + evaluate); wait until it finishes
bash submit_ablation_array.sh                        # all 95 runs (finished runs are skipped); never two arrays at once
python compare_ablations.py                          # results/ and figures/ under the artifact folder
```

- **Training and evaluation:** one array element per manifest row, 4 GPUs each, at most 2 running at a time (8 GPUs).
- **Launch order:** all first replicates (base first, figure order), then the second and third replicates.
- **Skips and retries:**
  - An element skips training if its run is finished, and skips everything if its evaluation is complete (`evaluation_complete`).
  - It stops with an error, instead of retraining, if the latest checkpoint cannot be read.
  - Retry a failed element with `brequeue -e "<JOBID>[<INDEX>]"`; it resumes from its latest checkpoint.
  - Change concurrency with `bmod -J "%<N>" <JOBID>`.
- **Outputs:** everything goes to `/nfs/team361/sb75/terra-reproducibility/artifacts/ovca_ablation/`: `configs/`, `manifest.tsv`, `runs/<run>/`, `eval/<variant>/<run>/`, `logs/`, `results/`, `figures/`.

## Runtime and storage

| Item | Mouse | OVCA |
|---|---|---|
| Cells in training data | 92,520 | 338,695 |
| Iterations per epoch on 4 GPUs | 722 | 2,646 (3.66× mouse) |
| Training schedule | 300 epochs (216,600 steps) | 82 epochs (216,972 steps) |
| Time per run | about 3 h | about 2.5–3 h |
| All 95 runs, 2 at a time | — | about 5 days |

- **Epoch-matched alternative:** `--epochs 300 --warmup 3` takes about 8–11 h per run and 2.5–3 weeks in total.
- **Storage:**
  - About 0.75 GB checkpoint plus 0.5 GB embeddings per run, so about 115 GB in total.
  - Tokenized datasets go to `/lustre/.../DATASETS/gold/*/xho50-1b_1p_spatialfusion/`.
