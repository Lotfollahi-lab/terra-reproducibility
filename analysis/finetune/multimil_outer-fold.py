"""
MIL classifier hyperparameter tuning for a single outer fold.

Model-agnostic, like zeroshot-transfer_outer-fold.py / linear-probing_outer-fold.py:
--model selects the source embedding artifacts under
  {task_dir}/model_artifacts/{model}/outer-fold_N/
    train_features.npy, train_labels.npy
    test_features.npy,  test_labels.npy
and the corresponding full data split under {task_dir}/data/outer-fold_N/
  {train,test}.h5ad, inner-fold_M/{train,val}.h5ad

Results are written to a sibling artifact dir named after the source model
with '-zeroshot' stripped and a 'multimil-' prefix added (e.g.
scgpt-spatial-zeroshot -> multimil-scgpt-spatial), so source embeddings are
never overwritten. If the config sets `output_suffix`, it's appended to that
dir name (e.g. multimil-terra-nbr-bag-patient) -- needed whenever two configs
share the same task_dir and --model but differ in `sample_key` (bagging
granularity), since otherwise they'd collide on the same output directory.

MIL bagging needs sample_key (e.g. Patient ID) per cell from obs metadata.
Both the lightweight (cell_ids.npy-only) and full (h5ad) split formats are
supported: in lightweight mode, per-cell metadata is looked up from
`source_adata` in the config via load_source_metadata_lookup().

Nested CV structure:
- Inner loop: grid-searches `hyperparameters` from the config, selecting the
  best combination by validation accuracy.
- Aggregates the per-inner-fold best hyperparameters (median for numeric,
  mode for categorical), retrains on the full outer-fold trainval set, and
  evaluates on the held-out outer-fold test set. The trained final model is
  checkpointed to {output_outer_dir}/model/{timestamp}/.

Metric granularity: MILClassifier.get_model_output() predicts one label per
bag and broadcasts it to every cell in that bag. Both bag-level (deduplicated
by sample_key, one vote per bag) and cell-count-weighted (every cell's
broadcast prediction scored individually) metrics are computed for the test
set and reported side by side -- the latter mirrors what a naive per-cell
accuracy_score/f1_score on `.obs` would give, and over-weights bags with more
cells relative to the former.

Usage:
    python analysis/finetune/multimil_outer-fold.py \
        --config config/finetuning/multimil_scgpt-spatial-zeroshot_xhs_disease-classification-patient.yaml \
        --model scgpt-spatial-zeroshot \
        --outer-fold 1

Inner-fold embeddings are not read from disk (transfer_embeddings_between_splits.py only ever
writes outer-fold embeddings) -- they are derived by slicing the already-loaded outer-fold
trainval embeddings/metadata by cell_id, using inner-fold cell_id lists that are always present
(lightweight npy under data/outer-fold_N/inner-fold_M/, or read from full inner-fold h5ad if the
lightweight npy isn't there).
"""
import argparse
import json
import pickle
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from pytorch_lightning.callbacks import EarlyStopping

import multimil as mtm


# ============================================================================
# Shared eval helpers (ported from linear-probing_outer-fold.py so multimil
# reports the same accuracy/micro_f1/macro_f1/auc_roc set as the other
# evaluation scripts)
# ============================================================================

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def macro_ovr_auc(labels_true, probs, n_classes):
    """Macro one-vs-rest AUC-ROC, skipping classes absent from labels_true.

    sklearn's roc_auc_score(multi_class='ovr', average='macro') returns NaN
    for the *entire* metric if even one class has zero positives in
    labels_true (common with rare classes and a small held-out patient set).
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
    """Accuracy, micro/macro F1, and macro one-vs-rest AUC-ROC."""
    labels_true = np.asarray(labels_true)
    preds = np.asarray(preds)
    probs = softmax(logits)

    return {
        'accuracy': accuracy_score(labels_true, preds),
        'micro_f1': f1_score(labels_true, preds, average='micro', zero_division=0),
        'macro_f1': f1_score(labels_true, preds, average='macro', zero_division=0),
        'auc_roc': macro_ovr_auc(labels_true, probs, n_classes),
    }


def save_cls_report_csv(labels_true, labels_pred, save_path, hyperparams):
    report = classification_report(
        labels_true,
        labels_pred,
        output_dict=True,
        zero_division=0,
        digits=4
    )
    df = pd.DataFrame(report).transpose()
    df['hyperparams'] = json.dumps(hyperparams)
    df.index.name = 'class'
    df.to_csv(save_path)
    print(f"    Saved: {save_path}")


def get_output_model_name(model: str) -> str:
    """Derive the multimil output artifact dir name from the source embedding model."""
    base = model[: -len("-zeroshot")] if model.endswith("-zeroshot") else model
    return f"multimil-{base}"


def bag_level_predictions(
    adata_with_predictions: ad.AnnData,
    classification_key: str,
    sample_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Reduce cell-level (bag-broadcast) predictions to one row per bag (patient).

    get_model_output() writes the same bag-level prediction and per-class
    logits to every cell in a bag, so dropping duplicates on sample_key
    recovers one row per patient without depending on multimil's internal
    per-bag counters.

    Returns (true_idx, pred_idx, logits, n_classes), all bag/patient-level,
    with true/pred labels mapped into a common 0..n_classes-1 index space
    (the obsm logits column order, i.e. the class_names the model trained on).
    """
    pred_col = f"predicted_{classification_key}"
    logits_df = adata_with_predictions.obsm[f"full_predictions_{classification_key}"]

    frame = pd.DataFrame({
        'sample': adata_with_predictions.obs[sample_key].values,
        'true': adata_with_predictions.obs[classification_key].values,
        'pred': adata_with_predictions.obs[pred_col].values,
    })
    frame = pd.concat([frame.reset_index(drop=True), logits_df.reset_index(drop=True)], axis=1)
    frame = frame.drop_duplicates(subset='sample')

    class_names = list(logits_df.columns)
    label_mapping = {name: i for i, name in enumerate(class_names)}

    def to_idx(values):
        idx = np.empty(len(values), dtype=np.int64)
        for i, v in enumerate(values):
            if v in label_mapping:
                idx[i] = label_mapping[v]
            elif str(v) in label_mapping:
                idx[i] = label_mapping[str(v)]
            else:
                raise KeyError(f"Value {v!r} not found among class_names {class_names}")
        return idx

    true_idx = to_idx(frame['true'].tolist())
    pred_idx = to_idx(frame['pred'].tolist())
    logits = frame[class_names].to_numpy()

    return true_idx, pred_idx, logits, len(class_names)


def cell_level_predictions(
    adata_with_predictions: ad.AnnData,
    classification_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Same as bag_level_predictions but WITHOUT deduplicating by sample_key.

    Every cell keeps its bag's broadcast prediction/logits, so a patient's contribution
    to the resulting metrics is weighted by their cell count rather than counted once --
    this reproduces the cell-count-weighted accuracy/F1 the notebook reports (each cell
    scored individually), for direct comparison against the bag/patient-level metrics.
    """
    pred_col = f"predicted_{classification_key}"
    logits_df = adata_with_predictions.obsm[f"full_predictions_{classification_key}"]

    frame = pd.DataFrame({
        'true': adata_with_predictions.obs[classification_key].values,
        'pred': adata_with_predictions.obs[pred_col].values,
    })
    frame = pd.concat([frame.reset_index(drop=True), logits_df.reset_index(drop=True)], axis=1)

    class_names = list(logits_df.columns)
    label_mapping = {name: i for i, name in enumerate(class_names)}

    def to_idx(values):
        idx = np.empty(len(values), dtype=np.int64)
        for i, v in enumerate(values):
            if v in label_mapping:
                idx[i] = label_mapping[v]
            elif str(v) in label_mapping:
                idx[i] = label_mapping[str(v)]
            else:
                raise KeyError(f"Value {v!r} not found among class_names {class_names}")
        return idx

    true_idx = to_idx(frame['true'].tolist())
    pred_idx = to_idx(frame['pred'].tolist())
    logits = frame[class_names].to_numpy()

    return true_idx, pred_idx, logits, len(class_names)


# ============================================================================
# Helper Functions
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_source_metadata_lookup(
    source_adata_path: str,
    sample_key: str,
    label_column: str
) -> Tuple[pd.DataFrame, str]:
    """Build a cheap cell_id -> {sample_key, classification_key} lookup from the source h5ad.

    Reads the source h5ad in backed mode (obs only, never loads .X) so this works against a
    lightweight task_dir (cell_ids + labels only, no per-fold h5ad) without needing a full
    per-fold h5ad rebuild: MIL only ever needs these two obs columns per cell, never the
    expression matrix or the tokenized dataset that a --full split additionally materializes.

    The label -> index mapping matches create_nested_stratified_group_splits.py's
    build_split_data exactly (sorted unique label values), so classification_key values line up
    with any full-format split of the same source data.
    """
    print(f"  Loading source metadata (obs only, backed) from {source_adata_path}...")
    src = sc.read_h5ad(source_adata_path, backed='r')
    label_cats = sorted(src.obs[label_column].unique().tolist())
    label2idx = {lbl: i for i, lbl in enumerate(label_cats)}
    classification_key = f"{label_column}_cls_idx"

    lookup = src.obs[['cell_id', sample_key, label_column]].copy()
    lookup[classification_key] = lookup[label_column].map(label2idx)
    lookup = lookup.set_index('cell_id', drop=False)
    return lookup, classification_key


def metadata_for_cell_ids(lookup: pd.DataFrame, cell_ids: List[str]) -> ad.AnnData:
    """Slice the source metadata lookup for a list of cell_ids, wrapped as adata_metadata
    (create_anndata_from_embeddings only ever reads .obs, never .X)."""
    obs = lookup.loc[cell_ids].reset_index(drop=True)
    return ad.AnnData(obs=obs)


def create_anndata_from_embeddings(
    embeddings: np.ndarray,
    adata_metadata: ad.AnnData,
    sample_key: str,
    classification_key: str
) -> ad.AnnData:
    """
    Create AnnData object with embeddings in X and metadata in obs.

    Parameters:
    -----------
    embeddings: np.ndarray
        Embeddings array of shape (n_cells, embedding_dim).
    adata_metadata: ad.AnnData
        AnnData object containing metadata (must have cell_id in obs).
    sample_key: str
        Key in adata_metadata.obs for sample grouping.
    classification_key: str
        Key in adata_metadata.obs for classification labels.

    Returns:
    -------
    adata: ad.AnnData
        AnnData object with embeddings in X and metadata in obs.
    """
    cell_ids = adata_metadata.obs['cell_id'].tolist()

    assert len(embeddings) == len(cell_ids), \
        f"Mismatch: {len(embeddings)} embeddings but {len(cell_ids)} cells"

    adata = ad.AnnData(X=embeddings.astype(np.float32))

    adata.obs['cell_id'] = cell_ids
    adata.obs[sample_key] = adata_metadata.obs[sample_key].values
    adata.obs[classification_key] = adata_metadata.obs[classification_key].values

    for col in adata_metadata.obs.columns:
        if col not in adata.obs.columns:
            adata.obs[col] = adata_metadata.obs[col].values

    return adata


def hyperparameter_tuning_inner_fold(
    train_adata: ad.AnnData,
    val_adata: ad.AnnData,
    sample_key: str,
    classification_key: str,
    hyperparameter_grid: dict,
    max_epochs: int = 100,
    early_stopping_patience: int = 20,
    batch_size: int = 256,
    validation_size: float = None,
) -> Tuple[dict, float, list]:
    """
    Tune hyperparameters on inner fold train/val.

    Parameters:
    -----------
    train_adata: ad.AnnData
        Training AnnData with embeddings in X.
    val_adata: ad.AnnData
        Validation AnnData with embeddings in X.
    sample_key: str
        Key for sample grouping.
    classification_key: str
        Key for classification labels.
    hyperparameter_grid: dict
        Dictionary of hyperparameter lists to search over.
    max_epochs: int
        Maximum number of epochs for training.
    early_stopping_patience: int
        Patience for early stopping.
    batch_size: int
        Batch size for training.

    Returns:
    -------
    best_hyperparameters: dict
        Best hyperparameters found (by val accuracy), or None if none succeeded.
    best_val_accuracy: float
        Best validation accuracy achieved (bag/patient level).
    all_results: list
        Per-combination results (hyperparams + bag-level val metrics).
    """
    # Ensure categories are consistent between train and val
    if hasattr(train_adata.obs[classification_key], 'cat'):
        train_cats = train_adata.obs[classification_key].cat.categories
    else:
        train_cats = train_adata.obs[classification_key].unique()

    if hasattr(val_adata.obs[classification_key], 'cat'):
        val_cats = val_adata.obs[classification_key].cat.categories
    else:
        val_cats = val_adata.obs[classification_key].unique()

    all_cats = sorted(set(list(train_cats) + list(val_cats)))
    train_adata.obs[classification_key] = train_adata.obs[classification_key].astype('category')
    train_adata.obs[classification_key] = train_adata.obs[classification_key].cat.set_categories(all_cats)
    val_adata.obs[classification_key] = val_adata.obs[classification_key].astype('category')
    val_adata.obs[classification_key] = val_adata.obs[classification_key].cat.set_categories(all_cats)

    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    param_combinations = list(product(*param_values))

    print(f"    Testing {len(param_combinations)} hyperparameter combinations...")

    best_val_accuracy = -1.0
    best_hyperparameters = None
    all_results = []

    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))

        if 'lr' in params and isinstance(params['lr'], str):
            params['lr'] = float(params['lr'])

        print(f"      [{i+1}/{len(param_combinations)}] Testing: {params}")

        try:
            train_adata_iter = train_adata.copy()
            val_adata_iter = val_adata.copy()

            mtm.model.MILClassifier.setup_anndata(
                train_adata_iter,
                categorical_covariate_keys=[sample_key, classification_key],
                continuous_covariate_keys=None,
            )

            model = mtm.model.MILClassifier(
                adata=train_adata_iter,
                sample_key=sample_key,
                classification=[classification_key],
                dropout=params.get('dropout', 0.2),
                n_layers_classifier=params.get('n_layers_classifier', 2),
                n_hidden_classifier=params.get('n_hidden_classifier', 128),
                n_layers_cell_aggregator=params.get('n_layers_cell_aggregator', 1),
                n_hidden_cell_aggregator=params.get('n_hidden_cell_aggregator', 128),
                attn_dim=params.get('attn_dim', 16),
                scoring=params.get('scoring', 'gated_attn'),
            )

            # MILClassifier.train() hardcodes early_stopping_patience=50 as a literal
            # passed to TrainRunner, so it can't be overridden via that kwarg directly
            # (passing it here would collide and raise a TypeError). Instead we add our
            # own EarlyStopping callback with the real config patience: model.train()'s
            # save_best handling *appends* to whatever `callbacks` list we pass rather
            # than replacing it, and Lightning stops as soon as ANY EarlyStopping-family
            # callback fires -- so ours (shorter patience) wins the race against the
            # library's hardcoded one, without touching library internals.
            inner_val_size = validation_size if validation_size is not None else 0.1
            model.train(
                max_epochs=max_epochs,
                lr=params.get('lr', 5e-4),
                train_size=1.0 - inner_val_size,
                validation_size=inner_val_size,
                batch_size=batch_size,
                early_stopping=True,
                early_stopping_monitor="accuracy_validation",
                early_stopping_mode="max",
                callbacks=[EarlyStopping(
                    monitor="accuracy_validation",
                    mode="max",
                    patience=early_stopping_patience,
                )],
            )

            # load_query_data properly extends categories/setup for the val set
            val_model = mtm.model.MILClassifier.load_query_data(val_adata_iter, model)
            val_model.get_model_output(batch_size=batch_size)

            val_true_idx, val_pred_idx, val_logits, val_n_classes = bag_level_predictions(
                val_model.adata, classification_key, sample_key
            )
            val_metrics = compute_metrics(val_true_idx, val_pred_idx, val_logits, val_n_classes)
            val_accuracy = val_metrics['accuracy']

            print(f"        val_acc={val_metrics['accuracy']:.4f} "
                  f"micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f} "
                  f"auc_roc={val_metrics['auc_roc']:.4f}")

            all_results.append({
                **params,
                'val_accuracy': val_metrics['accuracy'],
                'val_micro_f1': val_metrics['micro_f1'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_auc_roc': val_metrics['auc_roc'],
            })

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_hyperparameters = params.copy()
                print(f"        New best! Accuracy: {best_val_accuracy:.4f}")

        except Exception as e:
            print(f"        Error during training: {e}")
            continue

    if best_hyperparameters is None:
        print(f"    Warning: No successful hyperparameter combinations found!")
        return None, -1.0, all_results

    print(f"    Best hyperparameters: {best_hyperparameters}")
    print(f"    Best validation accuracy: {best_val_accuracy:.4f}")

    return best_hyperparameters, best_val_accuracy, all_results


def aggregate_hyperparameters(inner_fold_results: List[dict]) -> dict:
    """
    Aggregate best hyperparameters across inner folds (median for numeric,
    mode for categorical).
    """
    if not inner_fold_results:
        raise ValueError("No inner fold results to aggregate")

    aggregated = {}

    param_keys = set()
    for result in inner_fold_results:
        param_keys.update([k for k in result.keys() if k != 'val_accuracy' and k != 'inner_fold'])

    for key in param_keys:
        values = [r[key] for r in inner_fold_results if key in r]

        if not values:
            continue

        converted_values = []
        for v in values:
            if isinstance(v, str):
                try:
                    converted_v = float(v)
                    if converted_v.is_integer():
                        converted_v = int(converted_v)
                    converted_values.append(converted_v)
                except ValueError:
                    converted_values.append(v)
            else:
                converted_values.append(v)

        if isinstance(converted_values[0], (int, float)):
            median_val = float(np.median(converted_values))
            if any(keyword in key.lower() for keyword in ['layer', 'hidden', 'dim', 'attn_dim']):
                aggregated[key] = int(round(median_val))
            else:
                aggregated[key] = median_val
        else:
            value_counts = Counter(converted_values)
            aggregated[key] = value_counts.most_common(1)[0][0]

    return aggregated


# ============================================================================
# Main Single Outer Fold Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MIL hyperparameter tuning for a single outer fold"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--task-dir',
        type=str,
        default=None,
        help='Base task directory (e.g. .../xhs_disease-classification-patient). Overrides config.'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Source embedding artifact dir name (e.g. scgpt-spatial-zeroshot, novae-zeroshot)'
    )
    parser.add_argument(
        '--outer-fold',
        type=int,
        required=True,
        help='Outer fold index to process (1-indexed)'
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    task_dir_str = args.task_dir or config.get('task_dir')
    if task_dir_str is None:
        raise ValueError("task_dir must be specified via --task-dir or config file")
    task_dir = Path(task_dir_str)
    data_dir = task_dir / "data"

    output_model = get_output_model_name(args.model)
    output_suffix = config.get('output_suffix')
    if output_suffix:
        output_model = f"{output_model}-{output_suffix}"
    embeddings_root = task_dir / "model_artifacts" / args.model
    output_root = task_dir / "model_artifacts" / output_model
    output_root.mkdir(parents=True, exist_ok=True)

    outer_fold_to_run = args.outer_fold

    print(f"Task dir:     {task_dir}")
    print(f"Model:        {args.model}")
    print(f"Output model: {output_model}")
    print(f"Outer fold:   {outer_fold_to_run}")

    sample_key = config.get('sample_key', 'batch')

    if 'hyperparameters' not in config:
        raise ValueError(
            "hyperparameters must be specified in the config file. "
            "Please add a 'hyperparameters' section to your YAML config.\n"
            f"Config file: {args.config}\n"
            f"Available keys in config: {list(config.keys())}"
        )
    hyperparameter_grid = config['hyperparameters']

    max_epochs = config.get('training', {}).get('max_epochs', 200)
    early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 50)
    batch_size = config.get('training', {}).get('batch_size', 256)
    validation_size = config.get('training', {}).get('validation_size')

    print(f"Sample key: {sample_key}")
    print(f"Hyperparameter grid: {hyperparameter_grid}")

    # ---------------------------------------------------------------------------------
    # Load split metadata
    split_metadata_path = task_dir / "split_metadata.pkl"
    print("Loading split metadata...")
    with open(split_metadata_path, 'rb') as f:
        split_info = pickle.load(f)
    print(f"Found {split_info['K_outer']} outer folds and {split_info['K_inner']} inner folds")

    if outer_fold_to_run < 1 or outer_fold_to_run > split_info['K_outer']:
        raise ValueError(
            f"Invalid outer fold {outer_fold_to_run}. "
            f"Must be between 1 and {split_info['K_outer']}"
        )

    label_column = split_info['label_column']

    # Lightweight task_dir (cell_ids + labels only, no per-fold h5ad): derive sample_key/
    # classification_key via a cheap lookup against the source h5ad's obs instead of requiring
    # a --full split rebuild (MIL never needs the expression matrix or tokenized dataset that
    # --full additionally materializes -- see load_source_metadata_lookup).
    lightweight = (task_dir / f"outer-fold_{outer_fold_to_run}" / "train_cell_ids.npy").exists()
    print(f"Format: {'lightweight' if lightweight else 'full'}")

    metadata_lookup = None
    if lightweight:
        source_adata_path = config.get('source_adata')
        if source_adata_path is None:
            raise ValueError(
                "config must set 'source_adata' (path to the full h5ad with obs metadata) "
                "when task_dir is a lightweight split (no per-fold h5ad found)."
            )
        metadata_lookup, classification_key = load_source_metadata_lookup(
            source_adata_path, sample_key, label_column
        )
    else:
        classification_key = f"{label_column}_cls_idx"
    print(f"Classification key: {classification_key}")

    # Note: split_info['outer_folds'] is only populated by build_split_data (--full mode) --
    # save_lightweight_splits() never fills it in, so it's always [] for a lightweight task_dir.
    # Don't rely on it; outer_fold_to_run is already bounds-checked against K_outer above, and
    # inner folds are discovered by globbing the fold directory (matching the convention used by
    # linear-probing_outer-fold.py / zeroshot-transfer_outer-fold.py).
    outer_fold = outer_fold_to_run
    lightweight_fold_dir = task_dir / f"outer-fold_{outer_fold}"
    outer_fold_dir = lightweight_fold_dir if lightweight else data_dir / f"outer-fold_{outer_fold}"
    embeddings_outer_dir = embeddings_root / f"outer-fold_{outer_fold}"
    output_outer_dir = output_root / f"outer-fold_{outer_fold}"
    output_outer_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing Outer Fold {outer_fold}/{split_info['K_outer']}")
    print(f"{'='*60}")

    # ---------------------------------------------------------------------------------
    # Load test data (outer fold)
    print(f"\nLoading test data for outer fold {outer_fold}...")
    test_features_path = embeddings_outer_dir / "test_features.npy"
    test_labels_path = embeddings_outer_dir / "test_labels.npy"

    if not test_features_path.exists() or not test_labels_path.exists():
        raise FileNotFoundError(f"Test embeddings not found: {test_features_path}")

    if lightweight:
        test_cell_ids = np.load(lightweight_fold_dir / "test_cell_ids.npy", allow_pickle=True).tolist()
        test_adata_metadata = metadata_for_cell_ids(metadata_lookup, test_cell_ids)
    else:
        test_adata_path = outer_fold_dir / "test.h5ad"
        if not test_adata_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_adata_path}")
        test_adata_metadata = sc.read_h5ad(test_adata_path)
    test_features = np.load(test_features_path)

    print(f"  Test features shape: {test_features.shape}")

    # ---------------------------------------------------------------------------------
    # Load trainval data (outer fold)
    print(f"\nLoading trainval data for outer fold {outer_fold}...")
    trainval_features_path = embeddings_outer_dir / "train_features.npy"
    trainval_labels_path = embeddings_outer_dir / "train_labels.npy"

    if not trainval_features_path.exists() or not trainval_labels_path.exists():
        raise FileNotFoundError(f"Trainval embeddings not found: {trainval_features_path}")

    if lightweight:
        trainval_cell_ids_raw = np.load(lightweight_fold_dir / "train_cell_ids.npy", allow_pickle=True).tolist()
        trainval_adata_metadata = metadata_for_cell_ids(metadata_lookup, trainval_cell_ids_raw)
    else:
        trainval_adata_path = outer_fold_dir / "train.h5ad"
        if not trainval_adata_path.exists():
            raise FileNotFoundError(f"Trainval data not found: {trainval_adata_path}")
        trainval_adata_metadata = sc.read_h5ad(trainval_adata_path)
    trainval_features = np.load(trainval_features_path)

    print(f"  Trainval features shape: {trainval_features.shape}")

    if test_features.shape[1] != trainval_features.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: test={test_features.shape[1]}, "
            f"trainval={trainval_features.shape[1]}"
        )
    embedding_dim = test_features.shape[1]
    print(f"  Embedding dimension: {embedding_dim}")

    # ---------------------------------------------------------------------------------
    # Inner fold hyperparameter tuning
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning on Inner Folds")
    print(f"{'='*60}")

    param_combinations = list(product(*hyperparameter_grid.values()))

    if len(param_combinations) == 1:
        # Nothing to select between -- skip the inner-fold tuning loop entirely (it would
        # just be a redundant extra training) and use the single combo directly.
        print("Only one hyperparameter combination in the grid -- skipping inner-fold "
              "tuning and using it directly.")
        single_combo = dict(zip(hyperparameter_grid.keys(), param_combinations[0]))
        if 'lr' in single_combo and isinstance(single_combo['lr'], str):
            single_combo['lr'] = float(single_combo['lr'])
        aggregated_hyperparameters = single_combo
        avg_val_accuracy = float('nan')
        inner_fold_results = [{
            'outer_fold': outer_fold,
            'inner_fold': 0,
            **single_combo,
            'val_accuracy': None,
        }]
    else:
        inner_fold_results = []
        inner_fold_best_hyperparameters = []

        # Inner-fold embeddings/metadata are derived by slicing the outer fold's already-loaded
        # trainval features/metadata by cell_id -- transfer_embeddings_between_splits.py never
        # writes separate inner-fold embedding files, so there's nothing to load from disk here.
        trainval_cell_ids = trainval_adata_metadata.obs['cell_id'].tolist()
        trainval_cell_id_to_idx = {cid: i for i, cid in enumerate(trainval_cell_ids)}

        for data_inner_dir in sorted(outer_fold_dir.glob("inner-fold_*")):
            inner_fold = int(data_inner_dir.name.split("_")[-1])
            output_inner_dir = output_outer_dir / f"inner-fold_{inner_fold}"
            output_inner_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Processing Inner Fold {inner_fold}/{split_info['K_inner']}...")

            train_inner_ids_path = data_inner_dir / "train_cell_ids.npy"
            val_ids_path = data_inner_dir / "val_cell_ids.npy"

            if train_inner_ids_path.exists() and val_ids_path.exists():
                train_inner_ids = np.load(train_inner_ids_path, allow_pickle=True).tolist()
                val_ids = np.load(val_ids_path, allow_pickle=True).tolist()
            else:
                train_inner_h5ad_path = data_inner_dir / "train.h5ad"
                val_h5ad_path = data_inner_dir / "val.h5ad"
                if not train_inner_h5ad_path.exists() or not val_h5ad_path.exists():
                    print(f"    Warning: Inner fold {inner_fold} cell_ids not found. Skipping...")
                    continue
                train_inner_ids = sc.read_h5ad(train_inner_h5ad_path, backed='r').obs['cell_id'].tolist()
                val_ids = sc.read_h5ad(val_h5ad_path, backed='r').obs['cell_id'].tolist()

            train_inner_idx = [trainval_cell_id_to_idx[c] for c in train_inner_ids]
            val_idx = [trainval_cell_id_to_idx[c] for c in val_ids]

            train_inner_features = trainval_features[train_inner_idx]
            val_inner_features = trainval_features[val_idx]
            train_inner_adata_metadata = trainval_adata_metadata[train_inner_idx].copy()
            val_inner_adata_metadata = trainval_adata_metadata[val_idx].copy()

            train_inner_adata = create_anndata_from_embeddings(
                train_inner_features, train_inner_adata_metadata, sample_key, classification_key
            )
            val_inner_adata = create_anndata_from_embeddings(
                val_inner_features, val_inner_adata_metadata, sample_key, classification_key
            )

            best_params, best_val_acc, all_results = hyperparameter_tuning_inner_fold(
                train_inner_adata,
                val_inner_adata,
                sample_key,
                classification_key,
                hyperparameter_grid,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                batch_size=batch_size,
                validation_size=validation_size,
            )

            if best_params is None:
                print(f"    Warning: No successful hyperparameter combinations for inner fold {inner_fold}. Skipping...")
                continue

            best_params['inner_fold'] = inner_fold
            best_params['val_accuracy'] = best_val_acc
            inner_fold_best_hyperparameters.append(best_params)

            with open(output_inner_dir / "best_hyperparameters.json", 'w') as f:
                json.dump(best_params, f, indent=2)

            with open(output_inner_dir / "val_results.json", 'w') as f:
                json.dump(all_results, f, indent=2)

            inner_fold_df = pd.DataFrame(all_results)
            inner_fold_df['inner_fold'] = inner_fold
            inner_fold_df['outer_fold'] = outer_fold
            inner_fold_df.to_csv(output_inner_dir / "hyperparameter_results.csv", index=False)

            print(f"\n    Inner fold {inner_fold} hyperparameter results:")
            print(inner_fold_df.to_string(index=False))
            print()

            inner_fold_results.append({
                'outer_fold': outer_fold,
                'inner_fold': inner_fold,
                **best_params,
            })

        if not inner_fold_best_hyperparameters:
            raise ValueError(
                "No inner fold produced a successful hyperparameter combination — "
                "cannot proceed with final training."
            )

        aggregated_hyperparameters = aggregate_hyperparameters(inner_fold_best_hyperparameters)
        avg_val_accuracy = float(np.mean([r['val_accuracy'] for r in inner_fold_best_hyperparameters]))

    # ---------------------------------------------------------------------------------
    # Aggregate hyperparameters
    print(f"\n{'='*60}")
    print(f"Aggregating Hyperparameters Across Inner Folds")
    print(f"{'='*60}")

    print(f"Aggregated hyperparameters:")
    for key, value in aggregated_hyperparameters.items():
        print(f"  {key}: {value}")

    with open(output_outer_dir / "aggregated_hyperparameters.json", 'w') as f:
        json.dump(aggregated_hyperparameters, f, indent=2)

    # ---------------------------------------------------------------------------------
    # Final training on trainval
    print(f"\n{'='*60}")
    print(f"Training Final Model on Trainval")
    print(f"{'='*60}")

    trainval_adata = create_anndata_from_embeddings(
        trainval_features, trainval_adata_metadata, sample_key, classification_key
    )

    mtm.model.MILClassifier.setup_anndata(
        trainval_adata,
        categorical_covariate_keys=[sample_key, classification_key],
        continuous_covariate_keys=None,
    )

    print(f"Creating model with aggregated hyperparameters...")
    final_model = mtm.model.MILClassifier(
        adata=trainval_adata,
        sample_key=sample_key,
        classification=[classification_key],
        dropout=float(aggregated_hyperparameters.get('dropout', 0.2)),
        n_layers_classifier=int(aggregated_hyperparameters.get('n_layers_classifier', 2)),
        n_hidden_classifier=int(aggregated_hyperparameters.get('n_hidden_classifier', 128)),
        n_layers_cell_aggregator=int(aggregated_hyperparameters.get('n_layers_cell_aggregator', 1)),
        n_hidden_cell_aggregator=int(aggregated_hyperparameters.get('n_hidden_cell_aggregator', 128)),
        attn_dim=int(aggregated_hyperparameters.get('attn_dim', 16)),
        scoring=aggregated_hyperparameters.get('scoring', 'gated_attn'),
    )

    print(f"Training final model...")
    final_val_size = validation_size if validation_size is not None else 0.05
    final_model.train(
        max_epochs=max_epochs,
        lr=aggregated_hyperparameters.get('lr', 5e-4),
        train_size=1.0 - final_val_size,
        validation_size=final_val_size,
        batch_size=batch_size,
        early_stopping=True,
        early_stopping_monitor="accuracy_validation",
        early_stopping_mode="max",
        callbacks=[EarlyStopping(
            monitor="accuracy_validation",
            mode="max",
            patience=early_stopping_patience,
        )],
    )
    print(f"Final model training complete!")

    current_timestamp = (
        datetime.now().strftime("%d%m%Y_%H%M%S") +
        f"_{datetime.now().microsecond // 1000:03d}"
    )
    model_save_dir = output_outer_dir / "model" / current_timestamp
    final_model.save(model_save_dir, overwrite=True)
    print(f"Final model checkpoint saved to: {model_save_dir}")

    # ---------------------------------------------------------------------------------
    # Test evaluation (bag/patient level)
    print(f"\n{'='*60}")
    print(f"Evaluating on Test Set")
    print(f"{'='*60}")

    test_adata = create_anndata_from_embeddings(
        test_features, test_adata_metadata, sample_key, classification_key
    )

    test_model = mtm.model.MILClassifier.load_query_data(test_adata, final_model)
    test_model.get_model_output(batch_size=batch_size)

    test_true_idx, test_pred_idx, test_logits, n_classes = bag_level_predictions(
        test_model.adata, classification_key, sample_key
    )
    test_metrics = compute_metrics(test_true_idx, test_pred_idx, test_logits, n_classes)

    # Cell-count-weighted (non-deduplicated) counterpart, for comparison against bag-level --
    # every cell keeps its bag's broadcast prediction, so a bag's contribution to these
    # metrics is weighted by its cell count rather than counted once. Not what we report by
    # default (see module docstring), but useful to see side by side.
    cell_true_idx, cell_pred_idx, cell_logits, _ = cell_level_predictions(
        test_model.adata, classification_key
    )
    test_metrics_cell_weighted = compute_metrics(cell_true_idx, cell_pred_idx, cell_logits, n_classes)

    print(f"Test accuracy (bag-level, n={len(test_true_idx)}): {test_metrics['accuracy']:.4f}")
    print(f"Test micro_f1={test_metrics['micro_f1']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}  "
          f"auc_roc={test_metrics['auc_roc']:.4f}")
    print(f"Test accuracy (cell-count-weighted, n={len(cell_true_idx)}): {test_metrics_cell_weighted['accuracy']:.4f}")
    print(f"Test micro_f1={test_metrics_cell_weighted['micro_f1']:.4f}  macro_f1={test_metrics_cell_weighted['macro_f1']:.4f}  "
          f"auc_roc={test_metrics_cell_weighted['auc_roc']:.4f}")
    print(f"n_test (bags): {len(test_true_idx)}, n_classes: {n_classes}")

    save_cls_report_csv(
        test_true_idx, test_pred_idx,
        output_outer_dir / "test_cls_report.csv",
        aggregated_hyperparameters
    )

    with open(output_outer_dir / f"test_extra_metrics_outer_fold_{outer_fold}.json", "w") as f:
        json.dump({
            'bag_level': {
                **test_metrics,
                'n_classes': n_classes,
                'n_test': int(len(test_true_idx)),
            },
            'cell_count_weighted': {
                **test_metrics_cell_weighted,
                'n_classes': n_classes,
                'n_test': int(len(cell_true_idx)),
            },
        }, f, indent=2)

    for r in inner_fold_results:
        r['aggregated_best_hyperparameters'] = json.dumps(aggregated_hyperparameters)
        r['test_accuracy'] = test_metrics['accuracy']
        r['test_micro_f1'] = test_metrics['micro_f1']
        r['test_macro_f1'] = test_metrics['macro_f1']
        r['test_auc_roc'] = test_metrics['auc_roc']
        r['avg_val_accuracy'] = avg_val_accuracy

    summary_path = output_outer_dir / f"cv_results_outer_fold_{outer_fold}.csv"
    pd.DataFrame(inner_fold_results).to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    print(f"\n{'='*60}")
    print(f"Outer fold {outer_fold} complete!")
    print(f"Results saved to: {output_outer_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
