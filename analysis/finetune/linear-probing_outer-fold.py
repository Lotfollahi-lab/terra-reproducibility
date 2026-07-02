"""
Model-agnostic linear-probing evaluation for a single outer fold.

Same source embeddings as zeroshot-transfer_outer-fold.py (KNN) -- this script
just swaps the KNN classifier for a linear classifier trained on top of the
frozen features.

Expects pre-placed features and labels under:
  {task_dir}/model_artifacts/{model}/outer-fold_{i}/
    train_features.npy, train_labels.npy
    test_features.npy,  test_labels.npy

Some artifact dirs (e.g. terra-zeroshot) store separate cell- and
neighborhood-level embeddings instead of a single generic pair:
    train_cell_features.npy / train_neighborhood_features.npy, etc.
In that case, pick which one to use via embed.selection_type ('agg_cell' or
'agg_graph') in the config.

Inner fold val/train cell_ids are loaded from h5ad (or lightweight cell_ids
npy) to subset outer fold train embeddings (no re-inference). Best LR is
selected on val, then evaluated on the test set.

Results are written to a sibling artifact dir named after the source model
with '-zeroshot' replaced by '-linear-probing' (terra-zeroshot ->
terra-linear-probing, terra-zeroshot -> terra-linear-probing, etc.), so
the source embeddings are never overwritten.

Usage (explicit args):
    python analysis/finetune/linear-probing_outer-fold.py \
        --task-dir /lustre/.../xhs_cell-transfer \
        --model terra-zeroshot \
        --outer-fold 1 \
        --lr-values 0.001 0.01 0.1

Usage (config file):
    python analysis/finetune/linear-probing_outer-fold.py \
        --config config/finetuning/terra-linear-probing_xhs_niche-transfer.yaml \
        --model terra-zeroshot \
        --outer-fold 1
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

import argparse
import json
import yaml
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from terra.evaluation import linear_classifier


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


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


def compute_metrics(labels_true, preds, logits, n_classes):
    """Accuracy, micro/macro F1, and macro one-vs-rest AUC-ROC from a classifier's outputs.

    micro-F1 is numerically identical to accuracy for single-label multiclass
    problems, but is still reported explicitly for clarity/downstream tooling.
    """
    labels_true = np.asarray(labels_true)
    preds = np.asarray(preds)
    probs = softmax(logits)

    return {
        'accuracy': accuracy_score(labels_true, preds),
        'micro_f1': f1_score(labels_true, preds, average='micro', zero_division=0),
        'macro_f1': f1_score(labels_true, preds, average='macro', zero_division=0),
        'auc_roc': macro_ovr_auc(labels_true, probs, n_classes),
    }


def save_cls_report_csv(
        labels_true,
        labels_pred,
        save_path,
        lr
    ):
    report = classification_report(
        labels_true,
        labels_pred,
        output_dict=True,
        zero_division=0,
        digits=4
    )
    df = pd.DataFrame(report).transpose()
    df['lr'] = lr
    df.index.name = 'class'
    df.to_csv(save_path)
    print(f"  Saved: {save_path}")


def get_output_model_name(model: str) -> str:
    """Derive the linear-probing output artifact dir name from the source embedding model."""
    base = model[: -len("-zeroshot")] if model.endswith("-zeroshot") else model
    return f"{base}-linear-probing"


def resolve_feature_filenames(model_fold_dir: Path, selection_type):
    """Return (train_filename, test_filename) for the source embeddings.

    Some artifact dirs store both cell- and neighborhood-level embeddings side
    by side; pick one via embed.selection_type. Others store a single generic
    pair (train_features.npy / test_features.npy).
    """
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
            f"set embed.selection_type to 'agg_cell' or 'agg_graph' in the config "
            f"(got: {selection_type!r})"
        )
    return f"train_{suffix}_features.npy", f"test_{suffix}_features.npy"


def main():
    parser = argparse.ArgumentParser(
        description="Model-agnostic linear-probing eval for a single outer fold"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file. Provides defaults for task-dir, lr-values, linear_config.'
    )
    parser.add_argument(
        '--task-dir',
        type=str,
        default=None,
        help='Base task directory (e.g. .../xhs_niche-transfer). Overrides config.'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Source embedding artifact dir name (e.g. terra-zeroshot, terra-zeroshot)'
    )
    parser.add_argument(
        '--outer-fold',
        type=int,
        required=True
    )
    parser.add_argument(
        '--lr-values',
        type=float,
        nargs='+',
        default=None,
        help='LR values to evaluate. Overrides config.'
    )
    args = parser.parse_args()

    # Load config file defaults, then let explicit CLI args override
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    task_dir_str = args.task_dir or cfg.get('task_dir') or cfg.get('data_root')
    if task_dir_str is None:
        raise ValueError("task_dir must be specified via --task-dir or config file (task_dir/data_root)")
    task_dir = Path(task_dir_str)

    outer_fold = args.outer_fold
    linear_cfg = cfg.get('linear_config', {})
    lr_values  = args.lr_values or linear_cfg.get('lr_values', [0.001, 0.01, 0.1])
    n_epochs   = linear_cfg.get('n_epochs', 500)
    batch_size = linear_cfg.get('batch_size', 100)
    patience   = linear_cfg.get('patience', 10)
    selection_type = cfg.get('embed', {}).get('selection_type')

    output_model    = get_output_model_name(args.model)
    model_fold_dir  = task_dir / "model_artifacts" / args.model / f"outer-fold_{outer_fold}"
    output_fold_dir = task_dir / "model_artifacts" / output_model / f"outer-fold_{outer_fold}"
    output_fold_dir.mkdir(parents=True, exist_ok=True)

    # Detect lightweight format (cell_ids.npy directly under outer-fold_N/)
    lightweight_fold_dir = task_dir / f"outer-fold_{outer_fold}"
    lightweight = (lightweight_fold_dir / "train_cell_ids.npy").exists()
    data_fold_dir = lightweight_fold_dir if lightweight else task_dir / "data" / f"outer-fold_{outer_fold}"

    print(f"Task dir:     {task_dir}")
    print(f"Model:        {args.model}")
    print(f"Output model: {output_model}")
    print(f"Outer fold:   {outer_fold}")
    print(f"LR values:    {lr_values}")
    print(f"n_epochs={n_epochs}  batch_size={batch_size}  patience={patience}")
    print(f"Format:       {'lightweight' if lightweight else 'full'}")

    # -------------------------------------------------------------------------
    # Load outer fold features and labels
    train_filename, test_filename = resolve_feature_filenames(model_fold_dir, selection_type)
    train_features = np.load(model_fold_dir / train_filename)
    train_labels   = np.load(model_fold_dir / "train_labels.npy", allow_pickle=True)
    test_features  = np.load(model_fold_dir / test_filename)
    test_labels    = np.load(model_fold_dir / "test_labels.npy", allow_pickle=True)
    print(f"Train: {train_features.shape}, Test: {test_features.shape}")

    n_classes = len(np.unique(np.concatenate([train_labels, test_labels])))
    print(f"n_classes (inferred): {n_classes}")

    # Build cell_id -> index map for inner fold subsetting
    if lightweight:
        train_cell_ids = np.load(data_fold_dir / "train_cell_ids.npy", allow_pickle=True).tolist()
    else:
        train_adata = sc.read_h5ad(data_fold_dir / "train.h5ad", backed='r')
        train_cell_ids = train_adata.obs['cell_id'].tolist()
        del train_adata
    train_cell_id_to_idx = {cid: i for i, cid in enumerate(train_cell_ids)}

    # -------------------------------------------------------------------------
    # Inner fold loop -- find best lr on val
    inner_fold_best_lrs = []
    inner_fold_results = []

    for inner_fold_dir in sorted(data_fold_dir.glob("inner-fold_*")):
        inner_fold = int(inner_fold_dir.name.split("_")[-1])
        output_inner_dir = output_fold_dir / f"inner-fold_{inner_fold}"
        output_inner_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Inner fold {inner_fold}")

        if lightweight:
            val_ids_path = inner_fold_dir / "val_cell_ids.npy"
            train_inner_ids_path = inner_fold_dir / "train_cell_ids.npy"
            if not val_ids_path.exists() or not train_inner_ids_path.exists():
                print(f"    Missing cell_ids npy — skipping")
                continue
            val_ids = np.load(val_ids_path, allow_pickle=True).tolist()
            train_inner_ids = np.load(train_inner_ids_path, allow_pickle=True).tolist()
        else:
            val_h5ad = inner_fold_dir / "val.h5ad"
            train_inner_h5ad = inner_fold_dir / "train.h5ad"
            if not val_h5ad.exists() or not train_inner_h5ad.exists():
                print(f"    Missing h5ad — skipping")
                continue
            val_adata = sc.read_h5ad(val_h5ad, backed='r')
            val_ids = val_adata.obs['cell_id'].tolist()
            del val_adata
            train_inner_adata = sc.read_h5ad(train_inner_h5ad, backed='r')
            train_inner_ids = train_inner_adata.obs['cell_id'].tolist()
            del train_inner_adata

        val_idx = [train_cell_id_to_idx[c] for c in val_ids]
        val_features = train_features[val_idx]
        val_labels   = train_labels[val_idx]

        train_inner_idx = [train_cell_id_to_idx[c] for c in train_inner_ids]
        train_inner_features = train_features[train_inner_idx]
        train_inner_labels   = train_labels[train_inner_idx]

        # Save inner fold features
        np.save(output_inner_dir / "val_features.npy", val_features)
        np.save(output_inner_dir / "val_labels.npy", val_labels)
        np.save(output_inner_dir / "train_features.npy", train_inner_features)
        np.save(output_inner_dir / "train_labels.npy", train_inner_labels)

        # Evaluate lr values on val set
        val_accuracies = {}
        val_metrics_by_lr = {}
        for lr in lr_values:
            val_preds, _, val_logits, _ = linear_classifier(
                features_train=train_inner_features,
                labels_train=train_inner_labels,
                features_val=val_features,
                labels_val=val_labels,
                features_test=None,
                labels_test=None,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                n_classes=n_classes,
                results_save_path=None,
            )
            metrics = compute_metrics(val_labels, val_preds, val_logits, n_classes)
            val_accuracies[lr] = metrics['accuracy']
            val_metrics_by_lr[lr] = metrics
            print(f"    lr={lr}: val_acc={metrics['accuracy']:.4f} "
                  f"micro_f1={metrics['micro_f1']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                  f"auc_roc={metrics['auc_roc']:.4f}")

        best_lr = max(val_accuracies, key=val_accuracies.get)
        best_val_acc = val_accuracies[best_lr]
        best_val_metrics = val_metrics_by_lr[best_lr]
        print(f"    Best lr={best_lr} (val_acc={best_val_acc:.4f})")

        best_preds, _, best_logits, best_model = linear_classifier(
            features_train=train_inner_features,
            labels_train=train_inner_labels,
            features_val=val_features,
            labels_val=val_labels,
            features_test=None,
            labels_test=None,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=best_lr,
            patience=patience,
            n_classes=n_classes,
            results_save_path=str(output_inner_dir / "val_cls_report.txt"),
        )
        save_cls_report_csv(val_labels, best_preds, output_inner_dir / f"val_cls_report_lr={best_lr}.csv", best_lr)
        torch.save(best_model.state_dict(), output_inner_dir / f"model_lr={best_lr}.pt")

        inner_fold_best_lrs.append(best_lr)
        inner_fold_results.append({
            'outer_fold': outer_fold,
            'inner_fold': inner_fold,
            'best_lr': best_lr,
            'val_accuracy': best_val_acc,
            'val_micro_f1': best_val_metrics['micro_f1'],
            'val_macro_f1': best_val_metrics['macro_f1'],
            'val_auc_roc': best_val_metrics['auc_roc'],
            **{f'lr_{lr}_val_acc': val_accuracies.get(lr, np.nan) for lr in lr_values},
        })

    # -------------------------------------------------------------------------
    # Evaluate on outer fold test set using aggregated best lr
    if not inner_fold_best_lrs:
        print("\nNo inner folds processed — cannot evaluate on test set.")
        return

    best_lr_counts = Counter(inner_fold_best_lrs)
    aggregated_best_lr = best_lr_counts.most_common(1)[0][0]
    avg_val_acc = np.mean([r['val_accuracy'] for r in inner_fold_results])
    print(f"\nAggregated best lr={aggregated_best_lr}, avg val_acc={avg_val_acc:.4f}")

    test_preds, _, test_logits, test_model = linear_classifier(
        features_train=train_features,
        labels_train=train_labels,
        features_val=None,
        labels_val=None,
        features_test=test_features,
        labels_test=test_labels,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=aggregated_best_lr,
        patience=patience,
        n_classes=n_classes,
        results_save_path=str(output_fold_dir / "test_cls_report.txt"),
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_logits, n_classes)
    test_acc = test_metrics['accuracy']
    print(f"Test accuracy (lr={aggregated_best_lr}): {test_acc:.4f}")
    print(f"Test micro_f1={test_metrics['micro_f1']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}  "
          f"auc_roc={test_metrics['auc_roc']:.4f}")

    save_cls_report_csv(test_labels, test_preds, output_fold_dir / f"test_cls_report_lr={aggregated_best_lr}.csv", aggregated_best_lr)
    torch.save(test_model.state_dict(), output_fold_dir / f"model_lr={aggregated_best_lr}.pt")

    with open(output_fold_dir / f"test_extra_metrics_lr={aggregated_best_lr}.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    for r in inner_fold_results:
        r['aggregated_best_lr'] = aggregated_best_lr
        r['test_accuracy'] = test_acc
        r['test_micro_f1'] = test_metrics['micro_f1']
        r['test_macro_f1'] = test_metrics['macro_f1']
        r['test_auc_roc'] = test_metrics['auc_roc']
        r['avg_val_accuracy'] = avg_val_acc

    summary_path = output_fold_dir / f"cv_results_outer_fold_{outer_fold}.csv"
    pd.DataFrame(inner_fold_results).to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
