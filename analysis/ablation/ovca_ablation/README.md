# OVCA ablations

The mouse ablations of the TERRA paper (Supp. Figs. 4–6; `sebastianbirk/stjepa-reproducibility`,
`analysis/mouse_ablation`) are rerun on the SpatialFusion Xenium Prime 5K ovarian cancer section, with its
pathologist niche labels as ground truth. Models are trained from scratch with the same code, base config and
launch setup as the mouse runs.

## What is run

The variants and replicates are listed in `ablation_plan.yaml`:
- 50 variants, 92 training runs.
- Every variant keeps its mouse method name, and `mouse_runs` lists the mouse run timestamps.

| Figure | Category | Variants | Replicates |
|---|---|---|---|
| Supp. Fig. 4 | base (shared reference bar) | cell graph, combined, MLP, segment, 64 genes, mask 0.6, no normalization | 3 |
| Supp. Fig. 4 | tokenizer | cell graph rank / counts; cell neighborhood combined / rank / counts | 3 |
| Supp. Fig. 4 | sequence length | 16, 32, 128, 256 genes per cell | 3 |
| Supp. Fig. 4 | expression encoding | 20 / 50 / 100 value bins | 3 |
| Supp. Fig. 4 | spatial encoding | coordinate (`cell_pos_enc: coord`) | 3 |
| Supp. Fig. 4 | masking ratio | 0.3, 0.4, 0.5, 0.7, 0.8, 0.9 | 3 |
| Supp. Fig. 6 | metatoken | no metatoken | 3 |
| Supp. Fig. 5 | normalization | 21 normalization tuples, in the three mouse groups | 1 |
| none | architecture | width 192 / 384 / 960 / 1152; 1, 4 or 5 encoder layers; 3 predictor layers | 1 |

**Evaluation** follows the mouse protocol:
- Neighborhood embeddings from the target encoder.
- kNN graph with k = 15.
- Leiden on the fixed grid of resolutions 0.6–1.4. The grid is identical for all variants, so the protocol is label-free.
- NMI and ARI against `path_region`. `Unassigned` cells are clustered but not scored.
- Batch-integration metrics do not apply to a single section.

## Differences from the mouse runs, and why

| Item | Mouse | OVCA | Reason |
|---|---|---|---|
| Code | farm checkout of the time | copy of stjepa `b433d5e` + `stjepa_ovca_ablation.patch` (`setup_code.sh`) | Same code as the April runs; v2 later only added debug prints. A separate copy, so branch switches of `/nfs/team361/sb75/stjepa` cannot change running jobs. The patch adds an opt-in seed and writes the latest checkpoint atomically. |
| Schedule | 300 epochs, 3 warmup (216,600 steps) | 82 epochs, 1 warmup (216,972 steps) | Matched in optimizer steps: OVCA has 3.66× the iterations per epoch. All epoch-based schedules (LR, WD, EMA) scale with it. |
| Replicate seeds | all seed 0 (`_GLOBAL_SEED`), so replicates differed only by GPU non-determinism | seed = 1000 × (replicate − 1) | Real seed variance (rebuttal E10). Replicate 1 equals the mouse seed; the spacing keeps the samplers' per-epoch seeds (seed + epoch) disjoint. |
| `n_special_values` | 200 | 1448 | The human batch special value (about 213) exceeds 200 rows and would crash training. |
| Checkpoints | every 5 epochs (about 90 GB per run) | only `stjepa-latest.pth.tar`, in `runs/<run>_<config hash>`; killed runs resume on all ranks | Storage and walltime safety. A changed config gets a new folder instead of resuming an old one. |
| Evaluation input | separate data, re-tokenized with `harmonize_tokenize_embed_pipeline`; unlabelled cells removed first | the run's own tokenized training dataset (`embed_dataset`); all cells, so `Unassigned` stays in the kNN graph | Same section and identical tokens. Re-tokenizing needs about 0.5 TB for 16 normalization arms and breaks top-64 ties non-deterministically. Absolute metric levels are therefore not comparable with the mouse numbers. |
| Norm factors | cohort file of the mouse data | computed on the section (`prepare_ovca_data.py`) with the tokenizer's gene-corrected read depth constants | The factors must cover every gene and match the tokenizer. |
| Dataset metadata | — | dataset id 50, batch0 (same 10x section as `xho50-1b_1p`) | The token dictionary only has special values for existing batches. |

**Inferred from run names; verified by `check_setup.py` against the mouse `params.yaml` before launch:**
- The cell-neighborhood layout: 64 cell + 64 neighborhood tokens (input size 128, split equally as in later v2 tokenizers). The dataset class needs exactly that many tokens.
- `pred_emb_dim` following `enc_emb_dim` in the width ablation.
- The coordinate arm being `cell_pos_enc: coord`.

**Caveats:**
- **Not held out from TERRA-112M pretraining.** This section is `xho50-1b_1p` in the TERRA-112M corpus. That doesn't matter here, because every arm is trained from scratch on this section alone.
- **Known code issues** are listed in `CODE_REVIEW.md`. Several bias specific arms identically in mouse and OVCA, e.g. the metatoken is dropped at inference, and top-64 ties are broken by gene order.

## Steps (on the farm)

```bash
cd /nfs/team361/sb75/terra-reproducibility && git pull && cd analysis/ablation/ovca_ablation
bash setup_code.sh                                   # pinned code at /nfs/team361/sb75/stjepa_ovca_ablation
mkdir -p ../../../logs/ablation/ovca_ablation && bsub < prepare_ovca_data.lsf   # silver + norm factors (terra env)
source /nfs/team361/sb75/.venvs/stjepa_v2/bin/activate && export PYTHONPATH=/nfs/team361/sb75/stjepa_ovca_ablation/src
python generate_ablation_runs.py                     # 92 configs + manifest.tsv (82 epochs, 1 warmup)
bash submit_tokenization.sh standard                 # 10 datasets, normal queue, ~35 min each
bash submit_tokenization.sh large                    # 17 datasets, hugemem queue, ~0.6 TB peak each
python check_setup.py                                # config parity with mouse runs (review), special values, datasets (fails on errors)
bash submit_ablation_array.sh 1                      # pilot: base_rep1 (train + evaluate); wait until it finishes
bash submit_ablation_array.sh                        # all 92 runs (finished runs are skipped); never two arrays at once
python compare_ablations.py                          # results/ and figures/ under the artifact folder
```

- **Training and evaluation:** one array element per manifest row, 4 GPUs each, at most 2 running at a time (8 GPUs).
- **Launch order:** all first replicates (base first, figure order), then the second and third replicates.
- **Skips and retries:**
  - An element skips training if its run is finished, and skips everything if its metrics exist.
  - It stops with an error, instead of retraining, if the latest checkpoint cannot be read.
  - Retry a failed element with `brequeue -e "<JOBID>[<INDEX>]"`; it resumes from its latest checkpoint.
  - Change concurrency with `bmod -J "%<N>" <JOBID>`.
- **Outputs:** everything goes to `/nfs/team361/sb75/stjepa-reproducibility/artifacts/ovca_ablation/`: `configs/`, `manifest.tsv`, `runs/<run>/`, `eval/<variant>/<run>/`, `logs/`, `results/`, `figures/`.

## Runtime and storage

| Item | Mouse | OVCA |
|---|---|---|
| Cells in training data | 92,520 | 338,695 |
| Iterations per epoch on 4 GPUs | 722 | 2,646 (3.66× mouse) |
| Training schedule | 300 epochs (216,600 steps) | 82 epochs (216,972 steps) |
| Time per run | about 3 h | about 2.5–3 h |
| All 92 runs, 2 at a time | — | about 5 days |

- **Epoch-matched alternative:** `--epochs 300 --warmup 3` takes about 8–11 h per run and 2.5–3 weeks in total.
- **Storage:**
  - About 0.75 GB checkpoint plus 0.5 GB embeddings per run, so about 115 GB in total.
  - Tokenized datasets go to `/lustre/.../DATASETS/gold/*/xho50-1b_1p_spatialfusion/`.
