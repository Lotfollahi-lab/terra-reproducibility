"""
Post-hoc micro-F1, macro-F1, and AUC-ROC for an already-evaluated outer fold.

Model-agnostic: works on zeroshot (KNN), linear-probing, and full TERRA
finetune artifacts already sitting under
  {task_dir}/model_artifacts/{model}/outer-fold_{i}/
without retraining anything -- only forward passes / similarity search over
parameters that are already fit:

  * zeroshot (KNN):    re-runs the KNN similarity search read-only from the
                        saved train/test features, but keeps the per-class
                        vote fraction instead of collapsing to the hard
                        majority vote (knn_classifier() only returns argmax
                        predictions, so soft scores can't be recovered from
                        the saved test_cls_report alone).
  * *-linear-probing:   loads the saved model_lr=*.pt state dict (a single
                        nn.Linear head) and does one forward pass over the
                        test_features.npy from the source embedding dir.
  * *-finetune:         loads the saved checkpoint_best.pt/checkpoint_last.pt
                        into the full ClassificationModel (base encoder +
                        head, rebuilt from the run's saved params.yaml) and
                        does one forward pass over the outer fold's
                        test.h5ad/test.dataset.

The finetune path is the heaviest (full transformer forward pass) and has
not been exercised end-to-end yet since no terra-finetune artifacts exist
on disk at time of writing -- sanity-check it against the first checkpoint
you produce.

Usage:
    python analysis/finetune/compute_extra_metrics_outer-fold.py \
        --task-dir /nfs/team361/am84/TERRA/tasks/xhs_niche-transfer \
        --model terra-linear-probing \
        --outer-fold 1

    python analysis/finetune/compute_extra_metrics_outer-fold.py \
        --task-dir /nfs/team361/am84/TERRA/tasks/xhs_niche-transfer \
        --model terra-zeroshot \
        --outer-fold 1

    python analysis/finetune/compute_extra_metrics_outer-fold.py \
        --task-dir /lustre/.../xhs_niche-transfer \
        --model terra-finetune \
        --outer-fold 1
"""
import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from terra.evaluation.linear import LinearClassifier


def macro_ovr_auc(labels_true, probs, n_classes):
    """Macro one-vs-rest AUC-ROC, skipping classes absent from labels_true.

    sklearn's roc_auc_score(multi_class='ovr', average='macro') returns NaN
    for the *entire* metric if even one class in `labels` has zero positives
    in labels_true (common with many rare classes and a held-out test fold).
    Average over only the classes that are actually scoreable instead.
    """
    aucs = []
    for c in range(n_classes):
        y_bin = (labels_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        aucs.append(roc_auc_score(y_bin, probs[:, c]))
    return float(np.mean(aucs)) if aucs else float('nan')


def metrics_from_probs(labels_true, probs, n_classes):
    """Accuracy, micro/macro F1, and macro one-vs-rest AUC-ROC from per-class probabilities."""
    labels_true = np.asarray(labels_true)
    preds = probs.argmax(axis=1)

    return {
        'accuracy': accuracy_score(labels_true, preds),
        'micro_f1': f1_score(labels_true, preds, average='micro', zero_division=0),
        'macro_f1': f1_score(labels_true, preds, average='macro', zero_division=0),
        'auc_roc': macro_ovr_auc(labels_true, probs, n_classes),
    }


def resolve_feature_filenames(model_fold_dir: Path, selection_type):
    """Same convention as linear-probing_outer-fold.py / zeroshot-transfer_outer-fold.py."""
    has_dual = (model_fold_dir / "test_cell_features.npy").exists() or \
               (model_fold_dir / "test_neighborhood_features.npy").exists()
    if not has_dual:
        return "train_features.npy", "test_features.npy"

    if selection_type == "agg_cell":
        suffix = "cell"
    elif selection_type == "agg_graph":
        suffix = "neighborhood"
    else:
        raise ValueError(
            f"Artifacts at {model_fold_dir} store separate cell/neighborhood embeddings; "
            f"pass --selection-type agg_cell or agg_graph (got: {selection_type!r})"
        )
    return f"train_{suffix}_features.npy", f"test_{suffix}_features.npy"


def detect_kind(model_fold_dir: Path) -> str:
    if list(model_fold_dir.glob("final_model_lr_*/*/checkpoint_best.pt")) or \
       list(model_fold_dir.glob("final_model_lr_*/*/checkpoint_last.pt")):
        return "finetune"
    if list(model_fold_dir.glob("model_lr=*.pt")):
        return "linear_probing"
    if (model_fold_dir / "test_features.npy").exists() or \
       (model_fold_dir / "test_cell_features.npy").exists() or \
       (model_fold_dir / "test_neighborhood_features.npy").exists():
        return "zeroshot_knn"
    raise ValueError(f"Could not detect artifact kind under {model_fold_dir}")


# --------------------------------------------------------------------------
# zeroshot (KNN)
# --------------------------------------------------------------------------

def infer_best_k(model_fold_dir: Path, outer_fold: int, override: int | None):
    if override is not None:
        return override
    cv_path = model_fold_dir / f"cv_results_outer_fold_{outer_fold}.csv"
    if cv_path.exists():
        df = pd.read_csv(cv_path)
        if 'aggregated_best_k' in df.columns:
            return int(df['aggregated_best_k'].iloc[0])
    matches = list(model_fold_dir.glob("test_cls_report_k=*.csv"))
    if matches:
        m = re.search(r'k=(\d+)', matches[0].name)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not infer best k for {model_fold_dir}; pass --k explicitly.")


def eval_zeroshot_knn(model_fold_dir: Path, outer_fold: int, selection_type, batch_size: int, k_override):
    train_fname, test_fname = resolve_feature_filenames(model_fold_dir, selection_type)
    train_features = np.load(model_fold_dir / train_fname)
    train_labels = np.load(model_fold_dir / "train_labels.npy", allow_pickle=True)
    test_features = np.load(model_fold_dir / test_fname)
    test_labels = np.load(model_fold_dir / "test_labels.npy", allow_pickle=True)
    n_classes = len(np.unique(np.concatenate([train_labels, test_labels])))

    best_k = infer_best_k(model_fold_dir, outer_fold, k_override)
    print(f"Using k={best_k}, n_classes={n_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_t = F.normalize(torch.tensor(train_features, dtype=torch.float32, device=device), dim=1)
    test_t = F.normalize(torch.tensor(test_features, dtype=torch.float32, device=device), dim=1)
    train_labels_t = torch.tensor(train_labels, dtype=torch.long, device=device)

    num_test = test_t.size(0)
    probs = np.zeros((num_test, n_classes), dtype=np.float64)
    for start in range(0, num_test, batch_size):
        end = min(start + batch_size, num_test)
        similarity = test_t[start:end] @ train_t.T
        _, indices = similarity.topk(best_k, dim=1, largest=True, sorted=False)
        neighbor_labels = train_labels_t[indices]
        vote_fractions = F.one_hot(neighbor_labels, num_classes=n_classes).float().mean(dim=1)
        probs[start:end] = vote_fractions.cpu().numpy()
        del similarity

    return test_labels, probs, n_classes, {'k': best_k}


# --------------------------------------------------------------------------
# linear-probing
# --------------------------------------------------------------------------

def infer_source_model(model_name: str, override: str | None):
    if override is not None:
        return override
    if model_name.endswith("-linear-probing"):
        return model_name[: -len("-linear-probing")] + "-zeroshot"
    raise ValueError(
        f"Cannot infer source embedding model from {model_name!r}; pass --source-model explicitly."
    )


def infer_best_lr(output_fold_dir: Path, outer_fold: int, override: float | None):
    if override is not None:
        return override
    cv_path = output_fold_dir / f"cv_results_outer_fold_{outer_fold}.csv"
    if cv_path.exists():
        df = pd.read_csv(cv_path)
        if 'aggregated_best_lr' in df.columns:
            return float(df['aggregated_best_lr'].iloc[0])
    matches = list(output_fold_dir.glob("model_lr=*.pt"))
    if len(matches) == 1:
        m = re.search(r'lr=([0-9.eE+-]+)\.pt', matches[0].name)
        if m:
            return float(m.group(1))
    raise ValueError(f"Could not infer best lr for {output_fold_dir}; pass --lr explicitly.")


def eval_linear_probing(task_dir, model_name, output_fold_dir, outer_fold, source_model_override,
                         selection_type, lr_override):
    source_model = infer_source_model(model_name, source_model_override)
    source_fold_dir = task_dir / "model_artifacts" / source_model / f"outer-fold_{outer_fold}"
    print(f"Source embeddings: {source_fold_dir}")

    _, test_fname = resolve_feature_filenames(source_fold_dir, selection_type)
    test_features = np.load(source_fold_dir / test_fname)
    test_labels = np.load(source_fold_dir / "test_labels.npy", allow_pickle=True)

    best_lr = infer_best_lr(output_fold_dir, outer_fold, lr_override)
    ckpt_path = output_fold_dir / f"model_lr={best_lr}.pt"
    if not ckpt_path.exists():
        matches = list(output_fold_dir.glob("model_lr=*.pt"))
        if len(matches) == 1:
            ckpt_path = matches[0]
        else:
            raise FileNotFoundError(f"No checkpoint found for lr={best_lr} under {output_fold_dir}")
    print(f"Using checkpoint: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    num_classes, num_features = state_dict['linear.weight'].shape
    model = LinearClassifier(num_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(test_features, dtype=torch.float32)).numpy()
    probs = softmax(logits, axis=1)

    return test_labels, probs, num_classes, {'lr': best_lr}


# --------------------------------------------------------------------------
# full TERRA finetune
# --------------------------------------------------------------------------

def find_finetune_checkpoint(output_fold_dir: Path, outer_fold: int) -> Path:
    """Latest run's checkpoint for the outer fold's aggregated best LR.

    Each final_model_lr_{lr} dir can hold several timestamped run folders
    (retries after a crash/preemption). Run folder names are
    DDMMYYYY_HHMMSS_mmm -- not lexicographically sortable across month/year
    boundaries, so parse them properly. Within a run folder, prefer
    checkpoint_best.pt (the early-stopped best-val-loss state that
    finetune() itself evaluates on test) over checkpoint_last.pt.
    """
    cv_path = output_fold_dir / f"cv_results_outer_fold_{outer_fold}.csv"
    lr_dirs = None
    if cv_path.exists():
        df = pd.read_csv(cv_path)
        if 'aggregated_best_lr' in df.columns:
            lr = df['aggregated_best_lr'].iloc[0]
            candidate = output_fold_dir / f"final_model_lr_{lr}"
            if candidate.exists():
                lr_dirs = [candidate]
    if lr_dirs is None:
        lr_dirs = sorted(output_fold_dir.glob("final_model_lr_*"))
    if not lr_dirs:
        raise FileNotFoundError(f"No final_model_lr_* dir found under {output_fold_dir}")

    def parse_ts(run_dir):
        try:
            return datetime.strptime(run_dir.name, "%d%m%Y_%H%M%S_%f")
        except ValueError:
            return datetime.min

    run_dirs = [d for lr_dir in lr_dirs for d in lr_dir.iterdir() if d.is_dir()]
    run_dirs.sort(key=parse_ts)
    for run_dir in reversed(run_dirs):
        for fname in ("checkpoint_best.pt", "checkpoint_last.pt"):
            ckpt = run_dir / fname
            if ckpt.exists():
                return ckpt
    raise FileNotFoundError(f"No checkpoint_best.pt/checkpoint_last.pt found under {lr_dirs}")


def eval_finetune(task_dir: Path, output_fold_dir: Path, outer_fold: int):
    # Local imports: only needed (and only importable) on a GPU-provisioned
    # environment with the full terra/app training stack available.
    import scanpy as sc
    from datasets import load_from_disk
    from app.utils import init_model, load_checkpoint
    from app.utils.helper import apply_peft
    from terra.datasets.cell_datasets import init_cell_dataset
    from terra.datasets.dataloaders import init_dataloader_and_sampler
    from terra.masks.block_masking import BlockMaskCollator
    from terra.models.modules import ClassificationModel
    from terra.utils.embedding import create_binary_selection_mask

    ckpt_path = find_finetune_checkpoint(output_fold_dir, outer_fold)
    print(f"Using checkpoint: {ckpt_path}")

    params_path = ckpt_path.parent / "params.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"Expected params.yaml alongside checkpoint at {params_path}")
    with open(params_path) as f:
        train_args = yaml.safe_load(f)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint['model']

    use_mlp = train_args['finetune']['use_mlp']
    num_classes = state_dict['classification_head.2.weight'].shape[0] if use_mlp \
        else state_dict['classification_head.weight'].shape[0]

    pretrained_checkpoint_path = Path(train_args['model']['pretrained_checkpoint_path'])
    with open(pretrained_checkpoint_path / 'model_config.yaml') as f:
        model_config = yaml.safe_load(f)
    with open(pretrained_checkpoint_path / 'token_dictionary.pkl', 'rb') as f:
        token_dict = pickle.load(f)

    n_special_tokens = len(model_config['meta']['special_tokens'])
    seq_len = (model_config['data']['seq_len_cell'] +
               model_config['data']['seq_len_neighborhood'] +
               n_special_tokens)
    vocab_size = len(token_dict)
    n_special_values = sum(1 for key in token_dict if "spv" in key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_encoder, _ = init_model(
        gt_type=model_config['meta']['gt_type'],
        count_encoding=model_config['meta']['count_encoding'],
        n_value_bins=model_config['meta']['n_value_bins'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=model_config['data']['n_segments'],
        n_special_values=n_special_values,
        enc_emb_dim=model_config['meta']['enc_emb_dim'],
        enc_depth=model_config['meta']['enc_depth'],
        pred_emb_dim=model_config['meta']['pred_emb_dim'],
        pred_depth=model_config['meta']['pred_depth'],
        num_heads=model_config['meta']['num_heads'],
        mlp_ratio=model_config['meta']['mlp_ratio'],
        use_flash_attention=model_config['meta']['use_flash_attention'],
        api_version=model_config['meta']['api_version'],
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
        predict_gene=model_config['meta']['predict_gene'],
        pos_learnable=model_config['meta']['pos_learnable'],
    )
    _, _, target_encoder, _, _, _, _ = load_checkpoint(
        device=device,
        r_path=pretrained_checkpoint_path / 'model_checkpoint.pt',
        encoder=None, predictor=None, target_encoder=target_encoder,
        opt=None, scaler=None, is_training=False,
    )

    base_model = target_encoder
    if train_args['finetune']['use_peft']:
        base_model = apply_peft(
            target_encoder=target_encoder,
            peft_method=train_args['finetune']['peft_method'],
            peft_rank=train_args['finetune']['peft_rank'],
            peft_alpha=train_args['finetune']['peft_alpha'],
            peft_dropout=train_args['finetune']['peft_dropout'],
            peft_bias=train_args['finetune']['peft_bias'],
            peft_target_modules=train_args['finetune'].get('peft_target_modules'),
            peft_task_type=train_args['finetune'].get('peft_task_type'),
        )

    model = ClassificationModel(
        base_model=base_model,
        num_classes=num_classes,
        use_mlp=use_mlp,
        hidden_dim=train_args['finetune']['hidden_dim'],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    data_fold_dir = task_dir / "data" / f"outer-fold_{outer_fold}"
    test_adata = sc.read_h5ad(data_fold_dir / "test.h5ad")
    test_dataset = load_from_disk(str(data_fold_dir / "test.dataset"))
    test_dataset.set_format(type="torch", columns=test_dataset.column_names, output_all_columns=True)

    mask_collator = BlockMaskCollator(
        n_targets=model_config['mask']['n_targets'],
        n_contexts=model_config['mask']['n_contexts'],
        n_segments=model_config['data']['n_segments'],
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        n_special_tokens=n_special_tokens,
        per_block_mask_ratio=model_config['mask']['per_block_mask_ratio'],
        sample_segments=False,
        sample_gene_masks=False,
    )
    test_cell_dataset = init_cell_dataset(
        dataset=test_dataset,
        vocab_size=vocab_size,
        seq_len_cell=model_config['data']['seq_len_cell'],
        seq_len_neighborhood=model_config['data']['seq_len_neighborhood'],
        tokenizer_type=model_config['data']['tokenizer_type'],
        gt_type=model_config['meta']['gt_type'],
        cell_pos_enc=model_config['meta']['cell_pos_enc'],
        special_tokens=model_config['meta']['special_tokens'],
        sampling_strategy=None,
        n_nonzero_tokens_list=None,
        include_cell_id=True,
        sep_gene_tokens_neb=model_config['data']['sep_gene_tokens_neb'],
    )
    test_loader, _ = init_dataloader_and_sampler(
        cell_dataset=test_cell_dataset,
        batch_size=train_args['data']['batch_size'],
        distributed=False, world_size=1, rank=0,
        collate_fn=mask_collator,
        pin_memory=train_args['data'].get('pin_memory', False),
        num_workers=0,
        drop_last=False, persistent_workers=False,
        mega_batch_mult_max=train_args['data'].get('mega_batch_mult_max', 1),
    )

    label_name = train_args['data']['label_name']
    label_lookup = test_adata.obs.set_index('cell_id')[label_name]
    selection_type = train_args['finetune']['selection_type']
    excluded_tokens = train_args['finetune'].get('excluded_tokens')
    top_k = train_args['finetune'].get('top_k')

    all_logits, all_targets = [], []
    with torch.no_grad():
        for udata, _, _, masks_attention, _ in test_loader:
            for key in udata.keys():
                if key != 'cell_id':
                    udata[key] = udata[key].to(device, non_blocking=True)
            masks_attention = masks_attention.to(device, non_blocking=True)

            ns_tokens = udata['tokens'][:, n_special_tokens:]
            mask_kwargs = {'n_segments': model_config['data']['n_segments']} if selection_type == 'agg_graph' else {}
            selection_mask = create_binary_selection_mask(
                ns_tokens,
                selection_type=selection_type,
                excluded_tokens=excluded_tokens,
                seq_len_cell=model_config['data']['seq_len_cell'],
                top_k=top_k,
                **mask_kwargs,
            )

            logits = model(udata=udata, masks_attention=masks_attention, selection_mask=selection_mask)
            labels = torch.tensor(label_lookup[udata['cell_id']].values, dtype=torch.long)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    probs = softmax(logits, axis=1)

    return targets, probs, num_classes, {}


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc micro-F1/macro-F1/AUC-ROC for an already-evaluated outer fold "
                    "(zeroshot KNN, linear-probing, or full finetune -- no retraining)."
    )
    parser.add_argument('--task-dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True,
                        help='Model artifact dir name, e.g. terra-zeroshot, terra-linear-probing, terra-finetune')
    parser.add_argument('--outer-fold', type=int, required=True)
    parser.add_argument('--source-model', type=str, default=None,
                        help='linear-probing only: override the inferred source embedding dir name')
    parser.add_argument('--selection-type', type=str, default=None, choices=['agg_cell', 'agg_graph'],
                        help='Only needed if the artifacts store separate cell/neighborhood embeddings')
    parser.add_argument('--knn-batch-size', type=int, default=5000)
    parser.add_argument('--k', type=int, default=None, help='zeroshot KNN only: override inferred best k')
    parser.add_argument('--lr', type=float, default=None, help='linear-probing only: override inferred best lr')
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    output_fold_dir = task_dir / "model_artifacts" / args.model / f"outer-fold_{args.outer_fold}"
    if not output_fold_dir.exists():
        raise FileNotFoundError(output_fold_dir)

    kind = detect_kind(output_fold_dir)
    print(f"Model:      {args.model}")
    print(f"Outer fold: {args.outer_fold}")
    print(f"Kind:       {kind}")

    if kind == "zeroshot_knn":
        labels_true, probs, n_classes, extra = eval_zeroshot_knn(
            output_fold_dir, args.outer_fold, args.selection_type, args.knn_batch_size, args.k
        )
    elif kind == "linear_probing":
        labels_true, probs, n_classes, extra = eval_linear_probing(
            task_dir, args.model, output_fold_dir, args.outer_fold,
            args.source_model, args.selection_type, args.lr
        )
    else:
        labels_true, probs, n_classes, extra = eval_finetune(task_dir, output_fold_dir, args.outer_fold)

    metrics = metrics_from_probs(labels_true, probs, n_classes)
    metrics.update(extra)
    metrics['n_classes'] = n_classes
    metrics['n_test'] = len(labels_true)

    print(json.dumps(metrics, indent=2))

    save_path = output_fold_dir / f"test_extra_metrics_outer_fold_{args.outer_fold}.json"
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
