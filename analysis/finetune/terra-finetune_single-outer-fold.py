"""
TERRA Fine-tuning for a Single Outer Fold.

This script performs nested cross-validation for TERRA finetuning on a SINGLE outer fold:
- Processes one specified outer fold (for parallelization across jobs)
- Inner loop: searches over learning rate values to find the best LR
- Trains on the full training set (train+val) with the best LR
- Evaluates on the held-out test set for the specified outer fold

Usage:
    python terra-finetune_single-outer-fold.py --config config.yaml --outer-fold 1
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from datasets import load_from_disk
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.training.finetune import finetune


def save_classification_report_csv(
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
        save_path: Path,
        method: str = "terra-finetune"
    ):
    """
    Save classification report as CSV.
    
    Parameters:
    ----------
    - labels_true: np.ndarray
        True labels
    - labels_pred: np.ndarray
        Predicted labels
    - save_path: Path
        Path to save the CSV file
    - method: str, optional
        Method name for the report (default: "terra-finetune")
    """
    report = classification_report(
        labels_true, labels_pred,
        output_dict=True,
        zero_division=0,
        digits=4
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    df['method'] = method
    df.index.name = 'class'
    
    # Save to CSV
    df.to_csv(save_path)
    print(f"    Saved classification report to: {save_path}")


def build_finetune_config(
        pretrained_checkpoint_path: str,
        label_name: str,
        lr: float,
        num_epochs: int,
        data_config: dict,
        finetune_config: dict,
    ) -> dict:
    """
    Build config dictionary for TERRA finetuning.
    
    Parameters:
    ----------
    - pretrained_checkpoint_path: str
        Path to pretrained model checkpoint
    - label_name: str
        Name of the label column
    - lr: float
        Learning rate for this run
    - num_epochs: int
        Number of epochs for this run
    - data_config: dict
        Data configuration from YAML
    - finetune_config: dict
        Finetune configuration from YAML
        
    Returns:
    -------
    - config: dict
        Configuration dictionary for TERRA finetuning
    """
    return {
        'model': {
            'pretrained_checkpoint_path': pretrained_checkpoint_path,
        },
        'data': {
            'batch_size': data_config.get('batch_size', 100),
            'num_workers': int(os.environ.get("LSB_DJOB_NUMPROC", data_config.get('num_workers', 4))),
            'pin_memory': data_config.get('pin_memory', False),
            'label_name': label_name,
            'mega_batch_mult_max': data_config.get('mega_batch_mult_max', 1),
        },
        'finetune': {
            'use_peft': finetune_config.get('use_peft', False),
            'peft_method': finetune_config.get('peft_method', 'lora'),
            'peft_rank': finetune_config.get('peft_rank', 8),
            'peft_alpha': finetune_config.get('peft_alpha', 16),
            'peft_dropout': finetune_config.get('peft_dropout', 0.0),
            'peft_bias': finetune_config.get('peft_bias', 'none'),
            'peft_target_modules': finetune_config.get('peft_target_modules', [
                '11.attn.qkv',
                '11.attn.proj',
                '11.mlp.fc1',
                '11.mlp.fc2',
            ]),
            'peft_task_type': finetune_config.get('peft_task_type', None),
            'use_mlp': finetune_config.get('use_mlp', False),
            'hidden_dim': finetune_config.get('hidden_dim', 128),
            'lr': lr,
            'num_epochs': num_epochs,
            'patience': finetune_config.get('patience', 10),
            'selection_type': finetune_config.get('selection_type', 'agg_cell'),
            'excluded_tokens': finetune_config.get('excluded_tokens', None),
            'top_k': finetune_config.get('top_k', None),
        },
    }


# ============================================================================
# Main Single Outer Fold Execution
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="TERRA finetuning for a single outer fold"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to config YAML file for nested cross-validation of TERRA finetuning'
    )
    parser.add_argument(
        '--outer-fold', 
        type=int, 
        required=True, 
        help='Outer fold index to process (1-indexed)'
    )
    args = parser.parse_args()
    
    # Load configuration from YAML file
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    outer_fold_to_run = args.outer_fold
    
    # Extract paths from config
    data_root = Path(config['data_root'])
    split_metadata_path = data_root / "split_metadata.pkl"
    data_dir = data_root / "data"
    
    # Model checkpoint path
    pretrained_checkpoint_path = config['pretrained_checkpoint_path']
    
    # Output directory for TERRA finetuning results
    output_root = Path(config.get('output_root', str(data_root / "model_artifacts" / "terra-finetuning")))
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Load split metadata
    print("Loading split metadata...")
    with open(split_metadata_path, 'rb') as f:
        split_info = pickle.load(f)
    
    print(f"Found {split_info['K_outer']} outer folds and {split_info['K_inner']} inner folds")
    print(f"Label column: {split_info['label_column']}")
    
    # Validate outer fold index
    if outer_fold_to_run < 1 or outer_fold_to_run > split_info['K_outer']:
        raise ValueError(
            f"Invalid outer fold {outer_fold_to_run}. "
            f"Must be between 1 and {split_info['K_outer']}"
        )
    
    # Determine label name (config can override split metadata's label_column)
    label_column = config.get('label_column', split_info['label_column'])
    label_name = f"{label_column}_cls_idx"
    
    # Load data config from YAML (with defaults)
    data_config = config.get('data_config', {})
    
    # Load finetune config from YAML (with defaults)
    finetune_config = config.get('finetune_config', {})
    
    # Extract key training parameters
    num_epochs_inner = finetune_config.get('num_epochs_inner', 2)
    num_epochs_outer = finetune_config.get('num_epochs_outer', 20)
    lr_values = finetune_config.get('lr_values', [0.0001, 0.001])
    
    # Check for resume checkpoint path (optional)
    resume_checkpoint_path = config.get('resume_checkpoint_path', None)
    if resume_checkpoint_path:
        print(f"Resume checkpoint path: {resume_checkpoint_path}")
    
    print(f"Data config: {data_config}")
    print(f"Finetune config: {finetune_config}")
    
    # List to collect all results for summary CSV
    all_results = []
    
    # Find the specific outer fold to process
    outer_fold_info = None
    for fold_info in split_info['outer_folds']:
        if fold_info['outer_fold'] == outer_fold_to_run:
            outer_fold_info = fold_info
            break
    
    if outer_fold_info is None:
        raise ValueError(
            f"Outer fold {outer_fold_to_run} not found in split metadata"
        )
    
    # Process the single outer fold
    outer_fold = outer_fold_info['outer_fold']
    outer_fold_dir = data_dir / f"outer-fold_{outer_fold}"
    output_outer_dir = output_root / f"outer-fold_{outer_fold}"
    output_outer_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing Outer Fold {outer_fold}/{split_info['K_outer']}")
    print(f"{'='*60}")
    
    # Load test data (outer fold)
    print(f"\nLoading test data for outer fold {outer_fold}...")
    test_adata_path = outer_fold_dir / "test.h5ad"
    test_dataset_path = outer_fold_dir / "test.dataset"
    
    if not test_adata_path.exists() or not test_dataset_path.exists():
        raise FileNotFoundError(
            f"Test data not found for outer fold {outer_fold}. "
            f"Expected: {test_adata_path} and {test_dataset_path}"
        )
    
    test_adata = sc.read_h5ad(test_adata_path)
    test_dataset = load_from_disk(str(test_dataset_path))
    test_dataset.set_format(
        type="torch",
        columns=[c for c in test_dataset.column_names],
        output_all_columns=True
    )
    print(f"  Loaded test data: {test_adata.n_obs} samples")
    
    # Load train data (outer fold - this is train+val combined)
    print(f"\nLoading train data for outer fold {outer_fold}...")
    train_adata_path = outer_fold_dir / "train.h5ad"
    train_dataset_path = outer_fold_dir / "train.dataset"
    
    if not train_adata_path.exists() or not train_dataset_path.exists():
        raise FileNotFoundError(
            f"Train data not found for outer fold {outer_fold}. "
            f"Expected: {train_adata_path} and {train_dataset_path}"
        )
    
    train_adata = sc.read_h5ad(train_adata_path)
    train_dataset = load_from_disk(str(train_dataset_path))
    train_dataset.set_format(
        type="torch",
        columns=[c for c in train_dataset.column_names],
        output_all_columns=True
    )
    print(f"  Loaded train data: {train_adata.n_obs} samples")
    
    # Store best LR values from inner folds
    inner_fold_best_lrs = []
    inner_fold_results = []
    
    # Iterate over inner folds
    for inner_fold_info in outer_fold_info['inner_folds']:
        inner_fold = inner_fold_info['inner_fold']
        inner_fold_dir = outer_fold_dir / f"inner-fold_{inner_fold}"
        output_inner_dir = output_outer_dir / f"inner-fold_{inner_fold}"
        output_inner_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Processing Inner Fold {inner_fold}/{split_info['K_inner']}...")
        
        # Load val data (inner fold)
        print(f"    Loading val data for inner fold {inner_fold}...")
        val_adata_path = inner_fold_dir / "val.h5ad"
        val_dataset_path = inner_fold_dir / "val.dataset"
        
        if not val_adata_path.exists() or not val_dataset_path.exists():
            print(f"      Warning: Val data not found for inner fold {inner_fold}. Skipping...")
            continue
        
        val_adata = sc.read_h5ad(val_adata_path)
        val_dataset = load_from_disk(str(val_dataset_path))
        val_dataset.set_format(
            type="torch",
            columns=[c for c in val_dataset.column_names],
            output_all_columns=True
        )
        print(f"      Loaded val data: {val_adata.n_obs} samples")
        
        # Load train data (inner fold)
        print(f"    Loading train data for inner fold {inner_fold}...")
        train_inner_adata_path = inner_fold_dir / "train.h5ad"
        train_inner_dataset_path = inner_fold_dir / "train.dataset"
        
        if not train_inner_adata_path.exists() or not train_inner_dataset_path.exists():
            print(f"      Warning: Train data not found for inner fold {inner_fold}. Skipping...")
            continue
        
        train_inner_adata = sc.read_h5ad(train_inner_adata_path)
        train_inner_dataset = load_from_disk(str(train_inner_dataset_path))
        train_inner_dataset.set_format(
            type="torch",
            columns=[c for c in train_inner_dataset.column_names],
            output_all_columns=True
        )
        print(f"      Loaded train data: {train_inner_adata.n_obs} samples")
        
        # Evaluate different LR values on val set
        print(f"    Evaluating LR values {lr_values} on val set...")
        val_accuracies = {}
        best_lr_predictions = None
        
        for lr in lr_values:
            print(f"      Training with lr={lr}...")
            
            # Build config for this LR (inner fold - fewer epochs)
            inner_finetune_args = build_finetune_config(
                pretrained_checkpoint_path=pretrained_checkpoint_path,
                label_name=label_name,
                lr=lr,
                num_epochs=num_epochs_inner,
                data_config=data_config,
                finetune_config=finetune_config,
            )
            
            # Determine resume checkpoint for this specific inner fold and LR
            inner_resume_path = None
            if resume_checkpoint_path:
                # Look for existing checkpoint in the inner fold lr directory
                inner_checkpoint_dir = output_inner_dir / f"lr_{lr}"
                if inner_checkpoint_dir.exists():
                    # Find the latest checkpoint
                    checkpoints = sorted(inner_checkpoint_dir.glob("*/checkpoint_epoch_*.pt"))
                    if checkpoints:
                        inner_resume_path = str(checkpoints[-1])
                        print(f"      Found resume checkpoint: {inner_resume_path}")
            
            # Train and select hyperparameters based on val
            val_predictions, val_targets, val_acc = finetune(
                args=inner_finetune_args,
                train_adata=train_inner_adata,
                train_dataset=train_inner_dataset,
                val_adata=val_adata,
                val_dataset=val_dataset,
                test_adata=None,
                test_dataset=None,
                save_folder_path=str(output_inner_dir / f"lr_{lr}"),
                resume_checkpoint_path=inner_resume_path,
            )
            
            val_accuracies[lr] = val_acc
            print(f"      lr={lr}: val_accuracy={val_acc:.4f}")
            
            # Store predictions for best LR
            if best_lr_predictions is None or val_acc > max(val_accuracies.values()):
                best_lr_predictions = val_predictions
                best_lr_targets = val_targets
        
        # Pick best LR based on val loss for this inner fold
        best_lr = max(val_accuracies, key=val_accuracies.get)
        best_val_acc = val_accuracies[best_lr]
        print(f"    Best lr={best_lr} with val_accuracy={best_val_acc:.4f}")
        
        # Save classification report for best LR on val set
        val_report_path = output_inner_dir / f"val_cls_report_lr={best_lr}.csv"
        save_classification_report_csv(
            labels_true=np.array(best_lr_targets),
            labels_pred=np.array(best_lr_predictions),
            save_path=val_report_path,
            method="terra_finetuning"
        )
        
        # Store best LR and results for this inner fold
        inner_fold_best_lrs.append(best_lr)
        inner_fold_results.append({
            'outer_fold': outer_fold,
            'inner_fold': inner_fold,
            'best_lr': best_lr,
            'val_accuracy': best_val_acc,
            **{f'lr_{lr}_val_acc': val_accuracies.get(lr, np.nan) for lr in lr_values},
        })
    
    # After all inner folds, aggregate best LR (use most common, or mode)
    if len(inner_fold_best_lrs) > 0:
        best_lr_counts = Counter(inner_fold_best_lrs)
        aggregated_best_lr = best_lr_counts.most_common(1)[0][0]
        print(f"\n  Aggregated best LR across all inner folds: {aggregated_best_lr} (chosen {best_lr_counts[aggregated_best_lr]} times)")
        
        avg_val_accuracy = np.mean([r['val_accuracy'] for r in inner_fold_results])
        print(f"  Average validation accuracy across inner folds: {avg_val_accuracy:.4f}")
        
        # Train on full train+val set (outer fold train) with aggregated best LR
        print(f"\n  Training on full train set with lr={aggregated_best_lr} and evaluating on test set...")

        # Create a small validation set from training data for early stopping
        val_fraction = 0.05  # 5% of training data for validation
        train_indices, val_indices = train_test_split(
            np.arange(train_adata.n_obs),
            test_size=val_fraction,
            random_state=42,
            stratify=train_adata.obs[label_name]  # Stratified split
        )

        # Split adata
        outer_train_adata = train_adata[train_indices].copy()
        outer_val_adata = train_adata[val_indices].copy()

        # Split dataset by cell ID (adata and dataset may have different row orderings)
        outer_train_cell_ids = set(outer_train_adata.obs['cell_id'])
        outer_val_cell_ids = set(outer_val_adata.obs['cell_id'])
        ds_cell_ids = train_dataset['cell_id']
        outer_train_ds_indices = [i for i, cid in enumerate(ds_cell_ids) if cid in outer_train_cell_ids]
        outer_val_ds_indices = [i for i, cid in enumerate(ds_cell_ids) if cid in outer_val_cell_ids]
        outer_train_dataset = train_dataset.select(outer_train_ds_indices)
        outer_val_dataset = train_dataset.select(outer_val_ds_indices)

        # Set format for both datasets
        for ds in [outer_train_dataset, outer_val_dataset]:
            ds.set_format(
                type="torch",
                columns=[c for c in ds.column_names],
                output_all_columns=True,
            )

        # Create save folder for this final model run
        final_model_save_path = output_outer_dir / f"final_model_lr_{aggregated_best_lr}"
        final_model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Build config for outer fold (more epochs)
        outer_finetune_args = build_finetune_config(
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            label_name=label_name,
            lr=aggregated_best_lr,
            num_epochs=num_epochs_outer,
            data_config=data_config,
            finetune_config=finetune_config,
        )
        
        # Determine resume checkpoint for outer fold training
        outer_resume_path = None
        if resume_checkpoint_path:
            # Look for existing checkpoint in the final model directory
            if final_model_save_path.exists():
                checkpoints = sorted(final_model_save_path.glob("*/checkpoint_epoch_*.pt"))
                if checkpoints:
                    outer_resume_path = str(checkpoints[-1])
                    print(f"  Found resume checkpoint for outer fold: {outer_resume_path}")
        
        # Train with early stopping on held-out val, eval on test
        test_predictions, test_targets, test_accuracy = finetune(
            args=outer_finetune_args,
            train_adata=outer_train_adata,
            train_dataset=outer_train_dataset,
            val_adata=outer_val_adata,
            val_dataset=outer_val_dataset,
            test_adata=test_adata,
            test_dataset=test_dataset,
            save_folder_path=str(final_model_save_path),
            resume_checkpoint_path=outer_resume_path,
        )
        
        print(f"  Test accuracy with lr={aggregated_best_lr}: {test_accuracy:.4f}")
        
        # Save classification report for test set inside the model artifacts folder
        test_report_path = final_model_save_path / "test_cls_report.csv"
        save_classification_report_csv(
            labels_true=np.array(test_targets),
            labels_pred=np.array(test_predictions),
            save_path=test_report_path,
            method="terra-finetune"
        )
        
        # Update all inner fold results with test accuracy and aggregated best LR
        for result in inner_fold_results:
            result['aggregated_best_lr'] = aggregated_best_lr
            result['test_accuracy'] = test_accuracy
            result['avg_val_accuracy'] = avg_val_accuracy
        
        all_results.extend(inner_fold_results)
    
    # Save summary CSV for this outer fold
    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_outer_dir / f"cv_results_outer_fold_{outer_fold}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Outer fold {outer_fold} complete!")
        print(f"Results saved to: {output_outer_dir}")
        print(f"Summary CSV saved to: {summary_path}")
        print(f"{'='*60}")
    else:
        print(f"\nNo results to save for outer fold {outer_fold}. Please check that data exists.")


if __name__ == "__main__":
    main()
