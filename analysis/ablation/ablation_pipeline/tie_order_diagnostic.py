"""
This script tests how much a trained TERRA model depends on the order of tied genes, i.e. genes with the same ranking
value in a cell (on imaging data mostly genes with a count of 1). The tokenizer's tie rule sets that order
(`stable_rank_ties` and `rank_tie_break_norm_factor`, or numpy's unstable sort without them), so it carries no expression
information. With rank-position embeddings (`gt_type` combined or rank) a fixed tie rule makes the order of tied genes a
function of gene identity, which the predictor can use as a shortcut: with `predict_gene: True` a masked gene is queried
by its rank position, and the visible genes on either side of it in its tie group bracket its identity.

For a sample of cells of a tokenized dataset, the model's context encoder, predictor and target encoder are run on three
versions of every batch. The context and target masks (the masking of training, padded to the longest mask) are sampled
once and move with the genes, so every version masks the same genes and only their order differs:
- `stored`: the tokens as tokenized.
- `tie_shuffle`: the genes of each tie group (consecutive gene tokens with the same value in the same segment; the
  tokenizer sorts the genes of a segment by their ranking value) permuted at random. Values, positions and segments are
  unchanged, and every cell keeps its genes.
- `full_shuffle` (reference): all genes of each segment permuted at random together with their values, so the positions
  no longer follow the expression.

The masked genes (targets) are classed by their place in the stored order: `bracketed` (the nearest context genes on
both sides are in its tie group), `tied` (in a tie group of two or more genes, not bracketed) and `untied`. For each
condition and class the script reports the JEPA loss (the training loss per target, averaged over the embedding
dimensions) and its paired difference from `stored`. It also embeds every cell with the target encoder as
`embed_dataset` does (cell and neighborhood embeddings of the last layer) and compares each condition's embeddings with
the stored ones.

Reading the results:
- A loss increase under `tie_shuffle` on `bracketed` targets that is much larger than on `untied` targets means that
  the predictor uses the order of tied genes. The reference is a model trained with tied genes in random order, not
  zero: in small test models such a model still changed by up to about 0.3% on `bracketed` or `tied` targets.
- The tie order of TERRA-96M, TERRA-112M and the mouse runs is deterministic too: numpy's unstable sort on sparse
  matrices, a stable sort (column order) on dense ones. In tests with numpy on arm64 (x86 builds may use a SIMD sort;
  not tested), the unstable sort's order of tied genes depended largely on gene identity. These models may therefore
  have learned the order as well and are no negative control. `gt_type: counts` models have no rank positions, so both
  shuffles should change nothing beyond numerical noise (negative control).
- Embeddings: cosine similarity to the stored embedding of the same cell, the share of each cell's 15 nearest neighbors
  (euclidean, as `sc.pp.neighbors`) that are kept, whether a cell's shuffled embedding is nearest to its own stored
  embedding, and its displacement relative to the distance to its nearest other cell.

Tie groups are read from the stored values. This is exact only if the values are a strictly increasing function of
the ranking values within a cell: the same cell and gene normalization for ranking and values, and the same count
normalization or two of none, shifted log and PFlog1pPF. Other tokenizations (e.g. raw-count ranks with gene-normalized
values) stop with an error. For `gt_type: rank` models, which do not read values, they are taken from the same items
built as for a combined model.

Outputs in `--output_folder_path`:
- `tie_order_targets.csv`: condition, target class, number of targets, mean loss and its standard error, paired mean
  difference from `stored` and its standard error, relative difference.
- `tie_order_embeddings.csv`: condition, embedding, cosine similarity (mean, median, 5th percentile), kNN overlap,
  self-retrieval rate and median relative displacement.
- `tie_order_data.json`: the run settings, the checkpoint, and the tie statistics of the data (share of gene tokens in
  tie groups, of full segments and of full segments that end in a tie group; target classes).
- With `--save_embeddings`: `<condition>_<embedding>.npy` (float32) and `cell_ids.csv` (row order).

Usage:
```bash
python tie_order_diagnostic.py --model_folder_path [MODEL_OR_RUN_FOLDER] --output_folder_path [OUTPUT_FOLDER] \
    [--dataset_path TOKENIZED_DATASET] [--n_cells 5000] [--metatoken auto|keep|drop] \
    [--target_enc_layer_norm config|true|false] [--save_embeddings]
```
The model folder is a packaged run or model bundle (`model_config.yaml`, `model_checkpoint.pt`,
`token_dictionary.pkl`) or a training run folder (`params.yaml`, `<write_tag>-latest.pth.tar`, the token dictionary of
the config).
"""

import argparse
import copy
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

CONDITIONS = ("stored", "tie_shuffle", "full_shuffle")
TARGET_CLASSES = ("bracketed", "tied", "untied")
EMB_TYPES = ("cell_emb", "neighborhood_emb")
# Count normalizations that are strictly increasing within a cell
MONOTONE_COUNT_NORMS = (None, "shifted_log", "pflog1ppf")
KNN_K = 15


def values_identify_rank_ties(data_config: dict) -> bool:
    """
    Whether the genes with the same stored value in a segment are exactly the genes with the same ranking value: the
    same cell and gene normalization for ranking and values, and the same count normalization or two that are strictly
    increasing within a cell.
    """
    same_cell_gene = (data_config.get("rank_cell_norm_method") == data_config.get("count_cell_norm_method")
                      and data_config.get("rank_gene_norm_method") == data_config.get("count_gene_norm_method"))
    rank_count = data_config.get("rank_count_norm_method")
    count_count = data_config.get("count_count_norm_method")
    return same_cell_gene and (rank_count == count_count
                               or (rank_count in MONOTONE_COUNT_NORMS and count_count in MONOTONE_COUNT_NORMS))


def tie_structure(tokens: np.ndarray, values: np.ndarray, segments: np.ndarray, n_special_tokens: int):
    """
    Tie groups of the gene tokens of a batch: runs of consecutive gene tokens with the same value in the same segment.

    Returns
    -------
    gene: (B, N) bool, the gene tokens.
    run_id: (B, N) int, the tie group of each gene token (unique within a row, increasing along it; 0 elsewhere).
    run_len: (B, N) int, the size of the tie group of each gene token (0 elsewhere).
    """
    n_rows, n_positions = tokens.shape
    gene = (tokens != 0) & (np.arange(n_positions) >= n_special_tokens)[None, :]
    continues = np.zeros_like(gene)
    continues[:, 1:] = (gene[:, 1:] & gene[:, :-1] & (segments[:, 1:] == segments[:, :-1])
                        & (values[:, 1:] == values[:, :-1]))
    run_id = np.cumsum(gene & ~continues, axis=1) * gene
    flat_id = (run_id + np.arange(n_rows)[:, None] * (n_positions + 1)).ravel()
    sizes = np.bincount(flat_id[gene.ravel()], minlength=n_rows * (n_positions + 1))
    run_len = sizes[flat_id].reshape(n_rows, n_positions) * gene
    return gene, run_id, run_len


def bracketed_positions(gene: np.ndarray, run_id: np.ndarray, is_context: np.ndarray) -> np.ndarray:
    """Gene positions whose nearest context gene tokens on the left and on the right are both in their tie group."""
    n_rows, n_positions = gene.shape
    positions = np.broadcast_to(np.arange(n_positions), (n_rows, n_positions))
    context = is_context & gene
    left = np.maximum.accumulate(np.where(context, positions, -1), axis=1)
    left = np.concatenate([np.full((n_rows, 1), -1), left[:, :-1]], axis=1)
    right = np.minimum.accumulate(np.where(context, positions, n_positions)[:, ::-1], axis=1)[:, ::-1]
    right = np.concatenate([right[:, 1:], np.full((n_rows, 1), n_positions)], axis=1)
    left_run = np.where(left >= 0, np.take_along_axis(run_id, np.clip(left, 0, n_positions - 1), axis=1), -1)
    right_run = np.where(right < n_positions,
                         np.take_along_axis(run_id, np.clip(right, 0, n_positions - 1), axis=1), -1)
    return gene & (left_run == run_id) & (right_run == run_id)


def permute_within_groups(group_id: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Index array (B, N) that permutes the positions of each row at random within each group (`group_id` > 0); positions
    outside groups stay in place.
    """
    n_rows, n_positions = group_id.shape
    permutation = np.tile(np.arange(n_positions), (n_rows, 1))
    for row in range(n_rows):
        positions = np.flatnonzero(group_id[row] > 0)
        if positions.size < 2:
            continue
        groups = group_id[row, positions]
        slots = positions[np.argsort(groups, kind="stable")]
        permutation[row, slots] = positions[np.lexsort((rng.random(positions.size), groups))]
    return permutation


def move_masks(masks: torch.Tensor, permutation: np.ndarray) -> torch.Tensor:
    """
    Masks [n_masks, B, L] for the sequence permuted by `permutation` (position s holds the token of position
    permutation[s]) that select the same tokens as `masks` select in the unpermuted sequence.
    """
    inverse = np.empty_like(permutation)
    np.put_along_axis(inverse, permutation, np.broadcast_to(np.arange(permutation.shape[1]), permutation.shape),
                      axis=1)
    inverse = torch.from_numpy(inverse)
    return torch.stack([torch.gather(inverse, 1, mask) for mask in masks])


def mask_membership(indices: torch.Tensor, valid: torch.Tensor, n_positions: int) -> np.ndarray:
    """(B, N) bool: the positions in the valid entries of a padded mask (B, L)."""
    indices, valid = indices.numpy(), valid.numpy()
    member = np.zeros((indices.shape[0], n_positions), dtype=bool)
    rows = np.broadcast_to(np.arange(indices.shape[0])[:, None], indices.shape)
    member[rows[valid], indices[valid]] = True
    return member


class Accumulator:
    """Running count, sum and sum of squares."""

    def __init__(self):
        self.n, self.sum, self.sumsq = 0, 0.0, 0.0

    def add(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.n += x.size
        self.sum += x.sum()
        self.sumsq += np.square(x).sum()

    def mean(self) -> float:
        return self.sum / self.n if self.n else float("nan")

    def se(self) -> float:
        if self.n < 2:
            return float("nan")
        variance = max(self.sumsq / self.n - self.mean() ** 2, 0.0) * self.n / (self.n - 1)
        return float(np.sqrt(variance / self.n))


class TieValueDataset(torch.utils.data.Dataset):
    """Items of a rank-encoder cell dataset with the values of the same items built as for a combined model."""

    def __init__(self, rank_dataset, value_dataset, n_special_tokens: int):
        self.rank_dataset = rank_dataset
        self.value_dataset = value_dataset
        self.n_special_tokens = n_special_tokens

    def __len__(self):
        return len(self.rank_dataset)

    def __getitem__(self, index):
        item = self.rank_dataset[index]
        value_item = self.value_dataset[index]
        if not torch.equal(item["tokens"][self.n_special_tokens:], value_item["tokens"][self.n_special_tokens:]):
            raise RuntimeError(f"Item {index}: the rank and value items have different gene tokens.")
        item["tie_values"] = value_item["values"]
        return item


def resolve_model_files(model_folder_path: Path, checkpoint_file_path: str | None):
    """Config, checkpoint and token dictionary of a model bundle / packaged run, or of a training run folder."""
    config_file_path = model_folder_path / "model_config.yaml"
    if not config_file_path.exists():
        config_file_path = model_folder_path / "params.yaml"
    with open(config_file_path) as file:
        model_config = yaml.safe_load(file)
    if checkpoint_file_path is None:
        checkpoint_file_path = model_folder_path / "model_checkpoint.pt"
        if not checkpoint_file_path.exists():
            checkpoint_file_path = model_folder_path / f"{model_config['state']['write_tag']}-latest.pth.tar"
    token_dictionary_file_path = model_folder_path / "token_dictionary.pkl"
    if not token_dictionary_file_path.exists():
        token_dictionary_file_path = Path(os.path.expandvars(model_config["data"]["token_dict_folder_path"]))
    for file_path in (config_file_path, Path(checkpoint_file_path), token_dictionary_file_path):
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{file_path} does not exist.")
    return config_file_path, model_config, Path(checkpoint_file_path), token_dictionary_file_path


def load_modules(model_config: dict, token_dict: dict, checkpoint_file_path: Path, device: torch.device):
    """Context encoder, predictor and target encoder of the checkpoint (strict loading)."""
    from terra.utils.helper import init_model_from_config, strip_ddp_prefix

    encoder, predictor = init_model_from_config(model_config, token_dict, device)
    target_encoder = copy.deepcopy(encoder)
    checkpoint = torch.load(checkpoint_file_path, map_location="cpu", weights_only=False)
    for module, key in ((encoder, "encoder"), (predictor, "predictor"), (target_encoder, "target_encoder")):
        if key not in checkpoint:
            raise KeyError(f"{checkpoint_file_path} has no '{key}' state.")
        module.load_state_dict(strip_ddp_prefix(checkpoint[key]))
        module.eval()
    return encoder, predictor, target_encoder, checkpoint.get("epoch")


def build_loader(dataset, model_config: dict, vocab_size: int, pad_special_tokens: bool, batch_size: int,
                 num_workers: int):
    """Cell dataset and training masks (padded to the longest mask) as in training, without neighbor sampling."""
    from terra.datasets.cell_datasets import init_cell_dataset
    from terra.datasets.dataloaders import init_dataloader_and_sampler
    from terra.inference.embed import _restrict_special_attention, _truncation_settings
    from terra.masks.block_masking import BlockMaskCollator

    meta, data, mask = model_config["meta"], model_config["data"], model_config["mask"]
    if not mask.get("block_masking", True) or mask.get("cell_masking", False):
        raise ValueError("Only models trained with block masking (mask.block_masking) are supported.")
    n_special_tokens = len(meta.get("special_tokens") or [])
    truncate_neighbors, tokenized_seq_len_cell = _truncation_settings(dataset, data)
    mask_collator = BlockMaskCollator(
        n_targets=mask["n_targets"],
        n_contexts=mask["n_contexts"],
        n_segments=data["n_segments"],
        seq_len_cell=data["seq_len_cell"],
        seq_len_neighborhood=data["seq_len_neighborhood"],
        n_special_tokens=n_special_tokens,
        per_block_mask_ratio=mask["per_block_mask_ratio"],
        sample_segments=False,
        sample_gene_masks=True,
        restrict_special_attention=_restrict_special_attention(model_config, pad_special_tokens),
        special_token_pad_ratio=1.0 if pad_special_tokens else 0.0,
        pad_masks_to_batch_max=True)

    dataset_kwargs = dict(
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len_cell=data["seq_len_cell"],
        seq_len_neighborhood=data["seq_len_neighborhood"],
        tokenizer_type=data["tokenizer_type"],
        cell_pos_enc=meta["cell_pos_enc"],
        special_tokens=meta.get("special_tokens") or [],
        sampling_strategy=None,
        n_nonzero_tokens_list=[],
        include_cell_id=True,
        sep_gene_tokens_neb=data.get("sep_gene_tokens_neb", False),
        nz_spc=data.get("nz_spc", False),
        pad_special_tokens=pad_special_tokens,
        truncate_neighbors=truncate_neighbors,
        tokenized_seq_len_cell=tokenized_seq_len_cell)
    cell_dataset = init_cell_dataset(gt_type=meta["gt_type"], **dataset_kwargs)
    if meta["gt_type"] == "rank":
        cell_dataset = TieValueDataset(cell_dataset, init_cell_dataset(gt_type="combined", **dataset_kwargs),
                                       n_special_tokens)

    loader, _ = init_dataloader_and_sampler(
        cell_dataset=cell_dataset,
        batch_size=batch_size,
        distributed=False,
        world_size=1,
        rank=0,
        collate_fn=mask_collator,
        pin_memory=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=False,
        mega_batch_mult_max=data.get("mega_batch_mult_max", 1000))
    return loader


def target_losses(modules, batch: dict, masks_attention, masks_enc, masks_pred, masks_enc_valid, masks_pred_valid,
                  pad_special_tokens: bool, settings: dict) -> np.ndarray:
    """JEPA loss of every target mask entry (B, L_tgt), as in the training step with padded masks."""
    from terra.masks.utils import apply_masks, padding_key_masks

    encoder, predictor, target_encoder = modules
    with settings["autocast"]():
        h, _ = target_encoder(batch=batch, masks_attention=masks_attention)
        if settings["target_enc_layer_norm"]:
            h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, masks_pred, concat=False)
        enc_masks_attention, pred_masks_attention = padding_key_masks(
            masks_enc_valid[0], masks_pred_valid[0],
            n_special_tokens=settings["n_special_tokens"],
            new_spc=settings["new_spc"],
            pad_special_tokens=pad_special_tokens)
        z, token_emb = encoder(batch=batch, masks=masks_enc, masks_attention=enc_masks_attention)
        z = predictor(z=z, token_emb=token_emb, batch=batch, masks_enc=masks_enc, masks_pred=masks_pred,
                      masks_attention=pred_masks_attention, pad_special_tokens=pad_special_tokens)
    if isinstance(z, torch.Tensor):
        z = list(torch.split(z, h[0].size(0)))
    prediction, target = z[0].float(), h[0].float()
    if settings["loss_fn_type"] == "smooth_l1":
        loss = F.smooth_l1_loss(prediction, target, reduction="none")
    elif settings["loss_fn_type"] == "l1":
        loss = torch.abs(prediction - target)
    else:
        raise ValueError(f"Unknown loss_fn_type {settings['loss_fn_type']}.")
    return loss.mean(-1).cpu().numpy()


def batch_embeddings(target_encoder, batch: dict, masks_attention, settings: dict):
    """Cell and neighborhood embeddings of a batch as `embed_dataset` computes them (mean over the gene tokens)."""
    from terra.inference.embed import _encoder_backbone
    from terra.utils.embedding import compute_mean_unmasked_emb, create_binary_selection_mask

    emb_layer = settings["emb_layer"]
    with settings["autocast"]():
        full_context, cell_only_context = _encoder_backbone(target_encoder).return_layer_emb(
            layers=[emb_layer], batch=batch, masks_attention=masks_attention, need_cell_only_context=True,
            ignore_spc_tokens=True)
    cell_token_emb = cell_only_context[emb_layer].cpu()
    full_token_emb = full_context[emb_layer].cpu()
    ns_tokens = batch["tokens"][:, settings["n_special_tokens"]:]
    seq_len_cell = settings["seq_len_cell"]
    cell_mask = create_binary_selection_mask(ns_tokens, selection_type="agg_cell", excluded_tokens=None,
                                             seq_len_cell=seq_len_cell, top_k=None).cpu()
    if settings["tokenizer_type"] == "cell_neigh":
        neighborhood_mask = create_binary_selection_mask(ns_tokens, selection_type="agg_neighborhood",
                                                         excluded_tokens=None, seq_len_cell=seq_len_cell,
                                                         top_k=None).cpu()
    else:
        neighborhood_mask = create_binary_selection_mask(ns_tokens, selection_type="agg_graph", excluded_tokens=None,
                                                         seq_len_cell=seq_len_cell, top_k=None,
                                                         n_segments=settings["n_segments"]).cpu()
    return (compute_mean_unmasked_emb(cell_token_emb, cell_mask).float().numpy(),
            compute_mean_unmasked_emb(full_token_emb, neighborhood_mask).float().numpy())


def embedding_metrics(stored: np.ndarray, shuffled: np.ndarray, k: int = KNN_K) -> dict:
    """Agreement of the embeddings of a condition with the stored embeddings of the same cells."""
    from sklearn.neighbors import NearestNeighbors

    norms = np.linalg.norm(stored, axis=1) * np.linalg.norm(shuffled, axis=1)
    cosine = np.sum(stored * shuffled, axis=1) / np.where(norms > 0, norms, np.nan)
    k = min(k, len(stored) - 1)
    stored_index = NearestNeighbors(n_neighbors=k).fit(stored)
    stored_distances, stored_neighbors = stored_index.kneighbors()
    _, shuffled_neighbors = NearestNeighbors(n_neighbors=k).fit(shuffled).kneighbors()
    overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(stored_neighbors, shuffled_neighbors)])
    nearest_stored = stored_index.kneighbors(shuffled, n_neighbors=1, return_distance=False)[:, 0]
    nearest_other = stored_distances[:, 0]
    displacement = np.linalg.norm(shuffled - stored, axis=1) / np.where(nearest_other > 0, nearest_other, np.nan)
    return {"cosine_mean": float(np.nanmean(cosine)),
            "cosine_median": float(np.nanmedian(cosine)),
            "cosine_p05": float(np.nanpercentile(cosine, 5)),
            f"knn{k}_overlap": float(overlap),
            "self_retrieval": float(np.mean(nearest_stored == np.arange(len(stored)))),
            "relative_displacement_median": float(np.nanmedian(displacement))}


def block_statistics(gene: np.ndarray, run_len: np.ndarray, n_special_tokens: int, data_config: dict) -> dict:
    """Full segments (no padding) and full segments whose last gene is in a tie group, per segment block."""
    if data_config["tokenizer_type"] == "cell_neigh":
        bounds = [(n_special_tokens, n_special_tokens + data_config["seq_len_cell"]),
                  (n_special_tokens + data_config["seq_len_cell"], gene.shape[1])]
    else:
        length = data_config["seq_len_cell"]
        bounds = [(start, start + length) for start in range(n_special_tokens, gene.shape[1], length)]
    counts = {"index_blocks": 0, "index_full": 0, "index_full_end_in_tie": 0,
              "all_blocks": 0, "all_full": 0, "all_full_end_in_tie": 0}
    for block, (start, end) in enumerate(bounds):
        present = gene[:, start:end].any(axis=1)
        full = gene[:, start:end].all(axis=1)
        end_in_tie = full & (run_len[:, end - 1] >= 2)
        for prefix in (("index", "all") if block == 0 else ("all",)):
            counts[f"{prefix}_blocks"] += int(present.sum())
            counts[f"{prefix}_full"] += int(full.sum())
            counts[f"{prefix}_full_end_in_tie"] += int(end_in_tie.sum())
    return counts


def main(args):
    import pickle

    from datasets import load_from_disk

    torch.manual_seed(args.seed)
    model_folder_path = Path(args.model_folder_path)
    output_folder_path = Path(args.output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    config_file_path, model_config, checkpoint_file_path, token_dictionary_file_path = resolve_model_files(
        model_folder_path, args.checkpoint_file_path)
    meta, data = model_config["meta"], model_config["data"]
    if meta["gt_type"] not in ("rank", "counts", "combined"):
        raise ValueError(f"Unknown gt_type {meta['gt_type']}.")
    if not values_identify_rank_ties(data):
        raise ValueError(
            "The stored values do not identify the rank ties of this tokenization (ranking "
            f"{[data.get(key) for key in ('rank_cell_norm_method', 'rank_gene_norm_method', 'rank_count_norm_method')]}"
            ", values "
            f"{[data.get(key) for key in ('count_cell_norm_method', 'count_gene_norm_method', 'count_count_norm_method')]}"
            ").")
    with open(token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)

    dataset_path = args.dataset_path or os.path.expandvars(data["tokenized_data_folder_path"])
    if "$" in dataset_path:
        raise ValueError(f"Unresolved variable in the dataset path {dataset_path}; pass --dataset_path.")
    dataset = load_from_disk(dataset_path)
    n_dataset_cells = len(dataset)
    if args.n_cells < n_dataset_cells:
        rng = np.random.default_rng(args.seed)
        dataset = dataset.select(np.sort(rng.choice(n_dataset_cells, args.n_cells, replace=False)).tolist())
    # Datasets saved by the tokenizer have no format, and the dataset classes expect tensors. Cell-id strings stay
    # Python objects, as in TERRA's inference tokenization.
    dataset.set_format(type="torch", columns=[c for c in dataset.column_names if c not in ("cell_id", "cell_ids")],
                       output_all_columns=True)

    special_tokens = meta.get("special_tokens") or []
    n_special_tokens = len(special_tokens)
    has_special_values = all(f"{token}_value" in dataset.column_names for token in special_tokens)
    if args.metatoken == "keep" and not has_special_values:
        raise ValueError(f"--metatoken keep needs the dataset columns {[f'{t}_value' for t in special_tokens]}.")
    pad_special_tokens = (not special_tokens or args.metatoken == "drop"
                          or (args.metatoken == "auto" and not has_special_values))
    if special_tokens and pad_special_tokens:
        print(f"Special tokens {special_tokens} are padded (removed); the model "
              f"{'saw' if model_config['mask'].get('special_token_pad_ratio', 0) else 'never saw'} that in training.")

    if args.target_enc_layer_norm == "config":
        if "target_enc_layer_norm" not in meta:
            print("The config has no meta.target_enc_layer_norm: the targets are not normalized, as in training since "
                  "TERRA 4920a6b (26 Jan 2026). Models trained before (e.g. TERRA-96M and TERRA-112M) normalized "
                  "them: pass --target_enc_layer_norm true for those.")
        target_enc_layer_norm = bool(meta.get("target_enc_layer_norm", False))
    else:
        target_enc_layer_norm = args.target_enc_layer_norm == "true"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_bfloat16 = bool(meta.get("use_bfloat16", False)) and device.type == "cuda"

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16)

    settings = {"autocast": autocast,
                "target_enc_layer_norm": target_enc_layer_norm,
                "loss_fn_type": meta.get("loss_fn_type", "l1"),
                "n_special_tokens": n_special_tokens,
                "new_spc": meta.get("new_spc", False),
                "emb_layer": meta["enc_depth"],
                "seq_len_cell": data["seq_len_cell"],
                "n_segments": data["n_segments"],
                "tokenizer_type": data["tokenizer_type"]}

    from terra.inference.embed import _set_cell_only_seq_len

    encoder, predictor, target_encoder, epoch = load_modules(model_config, token_dict, checkpoint_file_path, device)
    _set_cell_only_seq_len(target_encoder, model_config)
    modules = (encoder, predictor, target_encoder)
    loader = build_loader(dataset, model_config, len(token_dict), pad_special_tokens, args.batch_size,
                          args.num_workers)

    losses = {(condition, cls): Accumulator() for condition in CONDITIONS for cls in TARGET_CLASSES + ("all",)}
    deltas = {(condition, cls): Accumulator() for condition in CONDITIONS[1:] for cls in TARGET_CLASSES + ("all",)}
    embeddings = {(condition, emb_type): [] for condition in CONDITIONS for emb_type in EMB_TYPES}
    cell_ids = []
    stats = {"gene_tokens": 0, "gene_tokens_in_tie_groups": 0}
    block_counts = {}
    target_counts = dict.fromkeys(TARGET_CLASSES, 0)

    with torch.inference_mode():
        for batch_index, (udata, masks_enc, masks_pred, masks_attention, pad_batch) in enumerate(loader):
            masks_enc_valid = udata.pop("masks_enc_valid")
            masks_pred_valid = udata.pop("masks_pred_valid")
            batch_cell_ids = udata.pop("cell_id")
            if isinstance(batch_cell_ids, torch.Tensor):
                batch_cell_ids = batch_cell_ids.tolist()
            cell_ids += [str(cell_id) for cell_id in batch_cell_ids]
            tie_values = udata.pop("tie_values", None)
            tokens = udata["tokens"].numpy()
            n_positions = tokens.shape[1]
            values = (tie_values if tie_values is not None else udata["values"]).numpy()
            gene, run_id, run_len = tie_structure(tokens, values, udata["segments"].numpy(), n_special_tokens)
            stats["gene_tokens"] += int(gene.sum())
            stats["gene_tokens_in_tie_groups"] += int((run_len >= 2).sum())
            for key, count in block_statistics(gene, run_len, n_special_tokens, data).items():
                block_counts[key] = block_counts.get(key, 0) + count

            # Target classes from the stored order (the shuffles keep values, positions and segments)
            is_context = mask_membership(masks_enc[0], masks_enc_valid[0], n_positions)
            bracketed = bracketed_positions(gene, run_id, is_context)
            target_index = masks_pred[0].numpy()
            target_valid = masks_pred_valid[0].numpy() & np.take_along_axis(gene, target_index, axis=1)
            target_class = np.where(np.take_along_axis(bracketed, target_index, axis=1), 0,
                                    np.where(np.take_along_axis(run_len, target_index, axis=1) >= 2, 1, 2))
            for class_index, cls in enumerate(TARGET_CLASSES):
                target_counts[cls] += int((target_valid & (target_class == class_index)).sum())

            rng = np.random.default_rng([args.seed, batch_index])
            permutations = {"stored": None,
                            "tie_shuffle": permute_within_groups(run_id, rng),
                            "full_shuffle": permute_within_groups(np.where(gene, udata["segments"].numpy(), 0), rng)}
            masks_valid_device = ([m.to(device) for m in masks_enc_valid], [m.to(device) for m in masks_pred_valid])
            masks_attention = masks_attention.to(device)
            stored_loss = None
            for condition, permutation in permutations.items():
                batch = dict(udata)
                condition_masks_enc, condition_masks_pred = masks_enc, masks_pred
                if permutation is not None:
                    index = torch.from_numpy(permutation)
                    batch["tokens"] = torch.gather(udata["tokens"], 1, index)
                    if "values" in udata:
                        batch["values"] = torch.gather(udata["values"], 1, index)
                    # The same genes are context and targets, at their new positions
                    condition_masks_enc = move_masks(masks_enc, permutation)
                    condition_masks_pred = move_masks(masks_pred, permutation)
                batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                         for key, value in batch.items()}
                loss = target_losses(modules, batch, masks_attention, [m.to(device) for m in condition_masks_enc],
                                     [m.to(device) for m in condition_masks_pred], *masks_valid_device,
                                     pad_special_tokens=bool(pad_batch), settings=settings)
                if stored_loss is None:
                    stored_loss = loss
                for class_index, cls in enumerate(TARGET_CLASSES + ("all",)):
                    selected = target_valid & ((target_class == class_index) if cls != "all" else True)
                    losses[(condition, cls)].add(loss[selected])
                    if condition != "stored":
                        deltas[(condition, cls)].add(loss[selected] - stored_loss[selected])
                cell_emb, neighborhood_emb = batch_embeddings(target_encoder, batch, masks_attention, settings)
                embeddings[(condition, "cell_emb")].append(cell_emb)
                embeddings[(condition, "neighborhood_emb")].append(neighborhood_emb)
            print(f"Batch {batch_index + 1}/{len(loader)}", flush=True)

    rows = []
    for (condition, cls), accumulator in losses.items():
        stored_mean = losses[("stored", cls)].mean()
        delta = deltas.get((condition, cls))
        rows.append({"condition": condition, "target_class": cls, "n_targets": accumulator.n,
                     "loss_mean": accumulator.mean(), "loss_se": accumulator.se(),
                     "delta_mean": delta.mean() if delta else 0.0, "delta_se": delta.se() if delta else 0.0,
                     "delta_relative": (delta.mean() / stored_mean) if delta else 0.0})
    targets_table = pd.DataFrame(rows)
    targets_table.to_csv(output_folder_path / "tie_order_targets.csv", index=False)

    rows = []
    stacked = {key: np.concatenate(value, axis=0) for key, value in embeddings.items()}
    for condition in CONDITIONS[1:]:
        for emb_type in EMB_TYPES:
            rows.append({"condition": condition, "embedding": emb_type,
                         **embedding_metrics(stacked[("stored", emb_type)], stacked[(condition, emb_type)])})
    embeddings_table = pd.DataFrame(rows)
    embeddings_table.to_csv(output_folder_path / "tie_order_embeddings.csv", index=False)
    if args.save_embeddings:
        for (condition, emb_type), array in stacked.items():
            np.save(output_folder_path / f"{condition}_{emb_type}.npy", array.astype(np.float32))
        pd.DataFrame({"cell_id": cell_ids}).to_csv(output_folder_path / "cell_ids.csv", index=False)

    import terra

    code_version_file_path = Path(terra.__file__).resolve().parents[2] / "CODE_VERSION"
    n_targets = sum(target_counts.values())
    summary = {
        "model_folder_path": str(model_folder_path),
        "config_file_path": str(config_file_path),
        "checkpoint_file_path": str(checkpoint_file_path),
        "checkpoint_epoch": epoch,
        "dataset_path": dataset_path,
        "n_dataset_cells": n_dataset_cells,
        "n_cells": len(cell_ids),
        "seed": args.seed,
        "metatoken": "dropped" if (special_tokens and pad_special_tokens) else ("kept" if special_tokens else "none"),
        "target_enc_layer_norm": target_enc_layer_norm,
        "loss_fn_type": settings["loss_fn_type"],
        "model": {key: meta.get(key) for key in ("gt_type", "predict_gene", "cell_pos_enc", "count_encoding",
                                                 "special_tokens", "enc_depth")},
        "tokenization": {key: data.get(key) for key in (
            "tokenizer_type", "seq_len_cell", "seq_len_neighborhood", "n_segments", "rank_cell_norm_method",
            "rank_gene_norm_method", "rank_count_norm_method", "count_cell_norm_method", "count_gene_norm_method",
            "count_count_norm_method", "stable_rank_ties", "rank_tie_break_norm_factor")},
        "mask": {key: model_config["mask"].get(key) for key in ("n_targets", "per_block_mask_ratio")},
        "gene_tokens": stats["gene_tokens"],
        "share_gene_tokens_in_tie_groups": stats["gene_tokens_in_tie_groups"] / max(stats["gene_tokens"], 1),
        "share_index_segments_full": block_counts["index_full"] / max(block_counts["index_blocks"], 1),
        "share_full_index_segments_ending_in_tie_group": (block_counts["index_full_end_in_tie"]
                                                          / max(block_counts["index_full"], 1)),
        "share_segments_full": block_counts["all_full"] / max(block_counts["all_blocks"], 1),
        "share_full_segments_ending_in_tie_group": block_counts["all_full_end_in_tie"] / max(block_counts["all_full"],
                                                                                             1),
        "targets": n_targets,
        "share_targets": {cls: count / max(n_targets, 1) for cls, count in target_counts.items()},
        "terra_code": (code_version_file_path.read_text().strip() if code_version_file_path.exists()
                       else str(Path(terra.__file__).parent)),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "bfloat16": use_bfloat16,
    }
    with open(output_folder_path / "tie_order_data.json", "w") as file:
        json.dump(summary, file, indent=2)

    pd.set_option("display.width", 200)
    print(json.dumps({key: summary[key] for key in ("checkpoint_epoch", "n_cells", "metatoken",
                                                    "share_gene_tokens_in_tie_groups", "share_index_segments_full",
                                                    "share_full_index_segments_ending_in_tie_group", "share_targets")},
                     indent=2))
    print(targets_table.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print(embeddings_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Results in {output_folder_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--model_folder_path',
                        type=str,
                        required=True,
                        help='Model bundle or packaged run (model_config.yaml, model_checkpoint.pt, '
                             'token_dictionary.pkl), or training run folder (params.yaml, <write_tag>-latest.pth.tar).')
    parser.add_argument('--output_folder_path',
                        type=str,
                        required=True,
                        help='Folder for the results.')
    parser.add_argument('--checkpoint_file_path',
                        type=str,
                        default=None,
                        help='Checkpoint to load instead of the model folder\'s (e.g. terra-ep2.pth.tar).')
    parser.add_argument('--dataset_path',
                        type=str,
                        default=None,
                        help='Tokenized dataset (default: data.tokenized_data_folder_path of the model config). It '
                             'must be tokenized as the model\'s training data.')
    parser.add_argument('--n_cells',
                        type=int,
                        default=5000,
                        help='Number of cells drawn at random from the dataset (all if it has fewer).')
    parser.add_argument('--metatoken',
                        choices=['auto', 'keep', 'drop'],
                        default='auto',
                        help='Special tokens: keep (as in training), drop (padded, as embed_dataset by default), or '
                             'auto (keep if the dataset has their value columns).')
    parser.add_argument('--target_enc_layer_norm',
                        choices=['config', 'true', 'false'],
                        default='config',
                        help='Layer norm of the targets: as meta.target_enc_layer_norm of the config (absent: false, '
                             'as train.py), or forced.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--num_workers',
                        type=int,
                        default=8)
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Seed of the cell sample, the masks and the shuffles.')
    parser.add_argument('--save_embeddings',
                        action='store_true',
                        help='Also save the embeddings of every condition.')
    args = parser.parse_args()

    main(args)
