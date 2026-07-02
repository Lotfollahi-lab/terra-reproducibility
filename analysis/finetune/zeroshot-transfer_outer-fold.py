"""
Model-agnostic KNN evaluation for a single outer fold.

Expects pre-placed features and labels under:
  {task_dir}/model_artifacts/{model}/outer-fold_{i}/
    train_features.npy, train_labels.npy
    test_features.npy,  test_labels.npy

Inner fold val/train cell_ids are loaded from h5ad to subset outer fold train embeddings
(no re-inference). Best k is selected on val, then evaluated on the test set.

Usage (explicit args):
    python analysis/finetune/zeroshot-transfer_outer-fold.py \
        --task-dir /lustre/.../xhs_cell-transfer \
        --model novae-zeroshot \
        --outer-fold 1 \
        --k-values 4 8 12 16 20 \
        --knn-batch-size 10000

Usage (config file):
    python analysis/finetune/zeroshot-transfer_outer-fold.py \
        --config config/finetuning/novae-zeroshot_xhs_niche-transfer.yaml \
        --model novae-zeroshot \
        --outer-fold 1
"""
import argparse
import yaml
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, classification_report

from terra.evaluation import knn_classifier


def save_cls_report_csv(
        labels_true,
        labels_pred,
        save_path,
        k
    ):
    report = classification_report(
        labels_true,
        labels_pred,
        output_dict=True,
        zero_division=0,
        digits=4
    )
    df = pd.DataFrame(report).transpose()
    df['k'] = k
    df.index.name = 'class'
    df.to_csv(save_path)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Model-agnostic KNN eval for a single outer fold"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file. Provides defaults for task-dir, k-values, knn-batch-size.'
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
        help='Model artifact directory name (e.g. novae-zeroshot)'
    )
    parser.add_argument(
        '--outer-fold',
        type=int,
        required=True
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=None,
        help='k values to evaluate. Overrides config.'
    )
    parser.add_argument(
        '--knn-batch-size',
        type=int,
        default=None,
        help='KNN batch size. Overrides config.'
    )
    args = parser.parse_args()

    # Load config file defaults, then let explicit CLI args override
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    task_dir_str = args.task_dir or cfg.get('task_dir')
    if task_dir_str is None:
        raise ValueError("task_dir must be specified via --task-dir or config file")
    task_dir = Path(task_dir_str)

    outer_fold = args.outer_fold
    k_values   = args.k_values or cfg.get('evaluation', {}).get('k_values', [4, 8, 12, 16, 20])
    knn_batch_size = args.knn_batch_size if args.knn_batch_size is not None else cfg.get('evaluation', {}).get('knn_batch_size', None)

    model_fold_dir = task_dir / "model_artifacts" / args.model / f"outer-fold_{outer_fold}"

    # Detect lightweight format (cell_ids.npy directly under outer-fold_N/)
    lightweight_fold_dir = task_dir / f"outer-fold_{outer_fold}"
    lightweight = (lightweight_fold_dir / "train_cell_ids.npy").exists()
    data_fold_dir = lightweight_fold_dir if lightweight else task_dir / "data" / f"outer-fold_{outer_fold}"

    print(f"Task dir:   {task_dir}")
    print(f"Model:      {args.model}")
    print(f"Outer fold: {outer_fold}")
    print(f"k values:   {k_values}")
    print(f"KNN batch size: {knn_batch_size}")
    print(f"Format:     {'lightweight' if lightweight else 'full'}")

    # -------------------------------------------------------------------------
    # Load outer fold features and labels
    train_features = np.load(model_fold_dir / "train_features.npy")
    train_labels   = np.load(model_fold_dir / "train_labels.npy", allow_pickle=True)
    test_features  = np.load(model_fold_dir / "test_features.npy")
    test_labels    = np.load(model_fold_dir / "test_labels.npy", allow_pickle=True)
    print(f"Train: {train_features.shape}, Test: {test_features.shape}")

    # Build cell_id -> index map for inner fold subsetting
    if lightweight:
        train_cell_ids = np.load(data_fold_dir / "train_cell_ids.npy", allow_pickle=True).tolist()
    else:
        train_adata = sc.read_h5ad(data_fold_dir / "train.h5ad", backed='r')
        train_cell_ids = train_adata.obs['cell_id'].tolist()
        del train_adata
    train_cell_id_to_idx = {cid: i for i, cid in enumerate(train_cell_ids)}

    # -------------------------------------------------------------------------
    # Inner fold loop — find best k on val
    inner_fold_best_ks = []
    inner_fold_results = []

    for inner_fold_dir in sorted(data_fold_dir.glob("inner-fold_*")):
        inner_fold = int(inner_fold_dir.name.split("_")[-1])
        output_inner_dir = model_fold_dir / f"inner-fold_{inner_fold}"
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

        # Evaluate k values on val set
        val_accuracies = {}
        for k in k_values:
            preds = knn_classifier(
                features_train=train_inner_features,
                labels_train=train_inner_labels,
                features_test=val_features,
                labels_test=val_labels,
                k=k,
                batch_size=knn_batch_size,
            )
            acc = accuracy_score(val_labels, preds)
            val_accuracies[k] = acc
            print(f"    k={k}: val_acc={acc:.4f}")

        best_k = max(val_accuracies, key=val_accuracies.get)
        best_val_acc = val_accuracies[best_k]
        print(f"    Best k={best_k} (val_acc={best_val_acc:.4f})")

        best_preds = knn_classifier(
            features_train=train_inner_features,
            labels_train=train_inner_labels,
            features_test=val_features,
            labels_test=val_labels,
            k=best_k,
            batch_size=knn_batch_size,
        )
        save_cls_report_csv(val_labels, best_preds, output_inner_dir / f"val_cls_report_k={best_k}.csv", best_k)

        inner_fold_best_ks.append(best_k)
        inner_fold_results.append({
            'outer_fold': outer_fold,
            'inner_fold': inner_fold,
            'best_k': best_k,
            'val_accuracy': best_val_acc,
            **{f'k_{k}_val_acc': val_accuracies.get(k, np.nan) for k in k_values},
        })

    # -------------------------------------------------------------------------
    # Evaluate on outer fold test set using aggregated best k
    if not inner_fold_best_ks:
        print("\nNo inner folds processed — cannot evaluate on test set.")
        return

    best_k_counts = Counter(inner_fold_best_ks)
    aggregated_best_k = best_k_counts.most_common(1)[0][0]
    avg_val_acc = np.mean([r['val_accuracy'] for r in inner_fold_results])
    print(f"\nAggregated best k={aggregated_best_k}, avg val_acc={avg_val_acc:.4f}")

    test_preds = knn_classifier(
        features_train=train_features,
        labels_train=train_labels,
        features_test=test_features,
        labels_test=test_labels,
        k=aggregated_best_k,
        batch_size=knn_batch_size,
    )
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test accuracy (k={aggregated_best_k}): {test_acc:.4f}")

    save_cls_report_csv(test_labels, test_preds, model_fold_dir / f"test_cls_report_k={aggregated_best_k}.csv", aggregated_best_k)

    for r in inner_fold_results:
        r['aggregated_best_k'] = aggregated_best_k
        r['test_accuracy'] = test_acc
        r['avg_val_accuracy'] = avg_val_acc

    summary_path = model_fold_dir / f"cv_results_outer_fold_{outer_fold}.csv"
    pd.DataFrame(inner_fold_results).to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
