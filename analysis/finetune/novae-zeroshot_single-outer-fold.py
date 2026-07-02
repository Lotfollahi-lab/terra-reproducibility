"""
Novae Zero-shot KNN Evaluation for a Single Outer Fold.

This script performs nested cross-validation for zero-shot KNN evaluation on a SINGLE outer fold:
- Processes one specified outer fold (for parallelization across jobs)
- Inner loop: searches over k values to find the best k for KNN
- Evaluates on the held-out test set for the specified outer fold

Usage:
    python reproducibility/analysis/finetune/novae-zeroshot_single-outer-fold.py --config reproducibility/config/finetuning/novae-zeroshot_xhs_bk51_niche-transfer.yaml --outer-fold 1

Uses pretrained Novae embeddings with KNN classification (no fine-tuning).
"""
import yaml
import pickle
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import torch
import novae
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter


def knn_classifier(
        features_train: np.ndarray,
        labels_train: np.ndarray,
        features_test: np.ndarray,
        labels_test: np.ndarray,
        k: int = 20,
        batch_size: Optional[int] = None,
        results_save_path: Optional[str] = None,
    ):
    """
    Simple KNN classifier using cosine similarity and majority voting.
    
    Parameters:
    ----------
    features_train: np.ndarray
        The features of the training data.
    labels_train: np.ndarray
        The labels of the training data.
    features_test: np.ndarray
        The features of the test data.
    labels_test: np.ndarray
        The labels of the test data.
    k: int
        The number of nearest neighbors to use.
    batch_size: int | None
        The batch size to use for the similarity computation. If None, the similarity is computed in a single pass.
    results_save_path: str | None
        The path to save the results. If None, the results are not saved.
    """
    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    features_train = torch.tensor(features_train, dtype=torch.float32, device=device)
    labels_train = torch.tensor(labels_train, dtype=torch.long, device=device)
    features_test = torch.tensor(features_test, dtype=torch.float32, device=device)
    labels_test = torch.tensor(labels_test, dtype=torch.long, device=device)

    # Normalize features to unit length (cosine similarity)
    features_train = F.normalize(features_train, dim=1)
    features_test = F.normalize(features_test, dim=1)

    num_test = features_test.size(0)

    if batch_size is None:
        similarity = torch.matmul(features_test, features_train.T)
        _, indices = similarity.topk(k, dim=1, largest=True, sorted=True)
        neighbor_labels = labels_train[indices]
    else:
        # Compute similarity batch-wise
        all_neighbor_labels = []
        for start_idx in range(0, num_test, batch_size):
            end_idx = min(start_idx + batch_size, num_test)
            batch_features = features_test[start_idx:end_idx]

            similarity = torch.matmul(batch_features, features_train.T)
            _, indices = similarity.topk(k, dim=1, largest=True, sorted=True)
            all_neighbor_labels.append(labels_train[indices])

            del similarity

        neighbor_labels = torch.cat(all_neighbor_labels, dim=0)

    # Perform majority vote
    predictions = torch.empty(num_test, dtype=labels_train.dtype, device=device)
    for i in range(num_test):
        # Count votes
        votes = Counter(neighbor_labels[i].tolist())
        predictions[i] = votes.most_common(1)[0][0]

    # Convert predictions and targets to NumPy for sklearn
    predictions_np = predictions.cpu().numpy()
    labels_test_np = labels_test.cpu().numpy()

    print("\n--- Evaluation Report on Test Set ---")
    cls_report = classification_report(
        labels_test_np, predictions_np, digits=4)
    print(cls_report)

    # Save to a .txt file
    if results_save_path:
        with open(results_save_path, "w") as f:
            f.write(cls_report)

        print("\n--- Evaluation Report saved. ---")

    return predictions_np


# ============================================================================
# Helper Functions
# ============================================================================

def compute_novae_embeddings(
    model: novae.Novae,
    adata: ad.AnnData,
    desc: str = "Computing Novae embeddings..."
) -> tuple[list[str], np.ndarray]:
    """
    Compute Novae embeddings for an AnnData object.
    
    Returns:
        cell_ids: List of cell IDs in the same order as embeddings
        embeddings: numpy array of shape (n_cells, embedding_dim)
    """
    print(f"  {desc}")
    
    # Compute zero-shot representations
    model.compute_representations(
        adata,
        zero_shot=True
    )
    print(adata)
    
    # Extract embeddings from obsm
    embeddings = adata.obsm['novae_latent']
    
    # Get cell IDs - use index if 'cell_id' column doesn't exist
    if 'cell_id' in adata.obs.columns:
        cell_ids = adata.obs['cell_id'].tolist()
    else:
        cell_ids = adata.obs.index.tolist()
    
    return cell_ids, embeddings


def collect_labels(
        adata: ad.AnnData,
        label_name: str,
        cell_ids: list[str]
    ) -> np.ndarray:
    """
    Collect labels for given cell IDs.
    
    Parameters:
    -----------
    adata: ad.AnnData
        AnnData object containing the data.
    label_name: str
        Name of the label column.
    cell_ids: list[str]
        List of cell IDs to collect labels for.
        
    Returns:
    -------
    labels: np.ndarray
        Array of labels for the given cell IDs.
    """
    assert 'cell_id' in adata.obs.columns, f"'cell_id' not found in adata.obs.columns"
    assert label_name in adata.obs.columns, f"'{label_name}' not found in adata.obs.columns"
    
    label_lookup = adata.obs.set_index('cell_id')[label_name]
    labels = np.array(label_lookup[cell_ids].values, dtype=np.int64)
    
    return labels


def save_classification_report_csv(
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
        save_path: Path,
        k: int
    ) -> None:
    """
    Save classification report as CSV.
    
    Parameters:
    -----------
    labels_true: np.ndarray
        Array of true labels.
    labels_pred: np.ndarray
        Array of predicted labels.
    save_path: Path
        Path to save the classification report.
    k: int
        Number of neighbors to use for KNN.
        
    Returns:
    -------
    None
    """
    report = classification_report(
        labels_true, labels_pred,
        output_dict=True,
        zero_division=0,
        digits=4
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    df['k'] = k
    df.index.name = 'class'
    
    # Save to CSV
    df.to_csv(save_path)
    print(f"    Saved classification report to: {save_path}")
    
    return


# ============================================================================
# Main Single Outer Fold Execution
# ============================================================================

def main():
    # ---------------------------------------------------------------------------------
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Zero-shot KNN evaluation for a single outer fold"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to config YAML file'
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

    # ---------------------------------------------------------------------------------    
    # Set data path
    data_root = Path(config['data_root'])
    data_dir = data_root / "data"

    # Load split metadata
    split_metadata_path = data_root / "split_metadata.pkl"
    print("Loading split metadata...")
    with open(split_metadata_path, 'rb') as f:
        split_info = pickle.load(f)
    print(f"Found {split_info['K_outer']} outer folds and {split_info['K_inner']} inner folds")

    # Validate outer fold index
    if outer_fold_to_run < 1 or outer_fold_to_run > split_info['K_outer']:
        raise ValueError(
            f"Invalid outer fold {outer_fold_to_run}. "
            f"Must be between 1 and {split_info['K_outer']}"
        )

    # ---------------------------------------------------------------------------------
    # Output directory for zeroshot results
    output_root = Path(config.get('output_root', str(data_root / "model_artifacts" / "novae-zeroshot")))
    output_root.mkdir(parents=True, exist_ok=True)
        
    # Determine label name and inject into config
    print(f"Found label column in split metadata: {split_info['label_column']}")
    label_column = split_info['label_column']
    label_name = f"{label_column}_cls_idx"

    # K values to test
    k_values = config.get('evaluation', {}).get('k_values', [4, 8, 12, 16, 20])
    knn_batch_size = config.get('evaluation', {}).get('knn_batch_size', None)
    
    print(f"K values to evaluate: {k_values}")
    print(f"KNN batch size: {knn_batch_size}")
    
    # ---------------------------------------------------------------------------------
    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    print(f"Using device: {device}")
    
    # ---------------------------------------------------------------------------------
    # Load Novae model once
    print("\n" + "="*60)
    print(f"Loading pretrained Novae model...")
    print("="*60)
    model = novae.Novae.from_pretrained('MICS-Lab/novae-human-0')
    
    # ---------------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------------
    # Load test data (outer fold)
    print(f"\nLoading test data for outer fold {outer_fold}...")
    test_adata_path = outer_fold_dir / "test.h5ad"
    
    if not test_adata_path.exists():
        raise FileNotFoundError(
            f"Test data not found for outer fold {outer_fold}. "
            f"Expected: {test_adata_path}"
        )
    
    test_adata = sc.read_h5ad(test_adata_path)
    test_cell_ids = test_adata.obs['cell_id'].tolist()

    if not (output_outer_dir / "test_features.npy").exists() or not (output_outer_dir / "test_labels.npy").exists():
        # Compute zero-shot representations
        model.compute_representations(
            test_adata,
            zero_shot=True
        )
        print(test_adata)
        test_features = test_adata.obsm['novae_latent']
        
        test_labels = collect_labels(
            adata=test_adata,
            label_name=label_name,
            cell_ids=test_cell_ids
        )
        
        # Save test features and labels
        np.save(output_outer_dir / "test_features.npy", test_features)
        np.save(output_outer_dir / "test_labels.npy", test_labels)
        print(f"Saved test features: {test_features.shape}, labels: {test_labels.shape}")
    
    else:
        test_features = np.load(output_outer_dir / "test_features.npy")
        test_labels = np.load(output_outer_dir / "test_labels.npy")
        print(f"Loaded test features: {test_features.shape}, labels: {test_labels.shape}")
    
    # ---------------------------------------------------------------------------------
    # Load train data (outer fold - this is train+val combined)
    print(f"\nLoading train data for outer fold {outer_fold}...")
    train_adata_path = outer_fold_dir / "train.h5ad"
    
    if not train_adata_path.exists():
        raise FileNotFoundError(
            f"Train data not found for outer fold {outer_fold}. "
            f"Expected: {train_adata_path}"
        )
    
    train_adata = sc.read_h5ad(train_adata_path)
    train_cell_ids = train_adata.obs['cell_id'].tolist()
    
    if not (output_outer_dir / "train_features.npy").exists() or not (output_outer_dir / "train_labels.npy").exists():
            
        model.compute_representations(
            train_adata,
            zero_shot=True
        )
        print(train_adata)

        train_features = train_adata.obsm['novae_latent']

        train_labels = collect_labels(
            adata=train_adata,
            label_name=label_name,
            cell_ids=train_cell_ids
        )
        # Save train features and labels
        np.save(output_outer_dir / "train_features.npy", train_features)
        np.save(output_outer_dir / "train_labels.npy", train_labels)
        print(f"Saved train features: {train_features.shape}, labels: {train_labels.shape}")
    
    else:
        train_features = np.load(output_outer_dir / "train_features.npy")
        train_labels = np.load(output_outer_dir / "train_labels.npy")
        print(f"Loaded train features: {train_features.shape}, labels: {train_labels.shape}")
    
    # ---------------------------------------------------------------------------------
    # Create cell_id to index mapping for efficient subsetting in inner folds
    # Inner fold train/val are subsets of outer fold train, so we can reuse embeddings
    train_cell_id_to_idx = {cid: i for i, cid in enumerate(train_cell_ids)}
    
    # Store best k values from inner folds
    inner_fold_best_ks = []
    inner_fold_results = []
    
    # Iterate over inner folds
    for inner_fold_info in outer_fold_info['inner_folds']:
        inner_fold = inner_fold_info['inner_fold']
        inner_fold_dir = outer_fold_dir / f"inner-fold_{inner_fold}"
        output_inner_dir = output_outer_dir / f"inner-fold_{inner_fold}"
        output_inner_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Processing Inner Fold {inner_fold}/{split_info['K_inner']}...")
        
        # Load val data (inner fold)
        print(f"    Subsetting val features from outer fold train...")
        val_adata_path = inner_fold_dir / "val.h5ad"
        
        if not val_adata_path.exists():
            print(f"      Warning: Val data not found for inner fold {inner_fold}. Skipping...")
            continue
        
        val_adata = sc.read_h5ad(val_adata_path)
        val_cell_ids = val_adata.obs['cell_id'].tolist()
        val_indices = [train_cell_id_to_idx[cid] for cid in val_cell_ids]
        val_features = train_features[val_indices]
        val_labels = train_labels[val_indices]
        del val_adata
        
        # Save val features and labels
        np.save(output_inner_dir / "val_features.npy", val_features)
        np.save(output_inner_dir / "val_labels.npy", val_labels)
        print(f"    Saved val features: {val_features.shape}, labels: {val_labels.shape}")
        
        # Subset train features from outer fold train (no need to re-run inference)
        print(f"    Subsetting train features from outer fold train...")
        train_inner_adata_path = inner_fold_dir / "train.h5ad"
        
        if not train_inner_adata_path.exists():
            print(f"      Warning: Train data not found for inner fold {inner_fold}. Skipping...")
            continue
        
        train_inner_adata = sc.read_h5ad(train_inner_adata_path)
        train_inner_cell_ids = train_inner_adata.obs['cell_id'].tolist()
        train_inner_indices = [train_cell_id_to_idx[cid] for cid in train_inner_cell_ids]
        train_inner_features = train_features[train_inner_indices]
        train_inner_labels = train_labels[train_inner_indices]
        del train_inner_adata
        
        # Save train features and labels
        np.save(output_inner_dir / "train_features.npy", train_inner_features)
        np.save(output_inner_dir / "train_labels.npy", train_inner_labels)
        print(f"    Saved train features: {train_inner_features.shape}, labels: {train_inner_labels.shape}")

    #     # KNN evaluation: Test different k values on val set
    #     print(f"    Evaluating KNN with k values {k_values} on val set...")
    #     val_accuracies = {}
    #     best_k_predictions = None
        
    #     for k in k_values:
    #         val_predictions = knn_classifier(
    #             features_train=train_inner_features,
    #             labels_train=train_inner_labels,
    #             features_test=val_features,
    #             labels_test=val_labels,
    #             k=k,
    #             batch_size=knn_batch_size
    #         )
    #         val_acc = accuracy_score(val_labels, val_predictions)
    #         val_accuracies[k] = val_acc
    #         print(f"      k={k}: val_accuracy={val_acc:.4f}")
            
    #         # Store predictions for best k
    #         if best_k_predictions is None or val_acc > max(val_accuracies.values()):
    #             best_k_predictions = val_predictions
        
    #     # Pick best k based on val accuracy for this inner fold
    #     best_k = max(val_accuracies, key=val_accuracies.get)
    #     best_val_acc = val_accuracies[best_k]
    #     print(f"    Best k={best_k} with val_accuracy={best_val_acc:.4f}")
        
    #     # Save classification report for best k on val set
    #     val_report_path = output_inner_dir / f"val_cls_report_k={best_k}.csv"
    #     save_classification_report_csv(
    #         labels_true=val_labels,
    #         labels_pred=best_k_predictions,
    #         save_path=val_report_path,
    #         k=best_k
    #     )
        
    #     # Store best k and results for this inner fold
    #     inner_fold_best_ks.append(best_k)
    #     inner_fold_results.append({
    #         'outer_fold': outer_fold,
    #         'inner_fold': inner_fold,
    #         'best_k': best_k,
    #         'val_accuracy': best_val_acc,
    #         **{f'k_{k}_val_acc': val_accuracies.get(k, np.nan) for k in k_values},
    #     })
    
    # # After all inner folds, aggregate best k (use most common, or mode)
    # if len(inner_fold_best_ks) > 0:
    #     best_k_counts = Counter(inner_fold_best_ks)
    #     aggregated_best_k = best_k_counts.most_common(1)[0][0]
    #     print(f"\n  Aggregated best k across all inner folds: {aggregated_best_k} (chosen {best_k_counts[aggregated_best_k]} times)")
        
    #     avg_val_accuracy = np.mean([r['val_accuracy'] for r in inner_fold_results])
    #     print(f"  Average validation accuracy across inner folds: {avg_val_accuracy:.4f}")
        
    #     # Evaluate on outer fold test set using aggregated best k
    #     print(f"\n  Evaluating on outer fold test set with aggregated best k={aggregated_best_k}...")
    #     test_predictions = knn_classifier(
    #         features_train=train_features,
    #         labels_train=train_labels,
    #         features_test=test_features,
    #         labels_test=test_labels,
    #         k=aggregated_best_k,
    #         batch_size=knn_batch_size
    #     )
        
    #     test_accuracy = accuracy_score(test_labels, test_predictions)
    #     print(f"  Test accuracy with k={aggregated_best_k}: {test_accuracy:.4f}")
        
    #     # Save classification report for test set
    #     test_report_path = output_outer_dir / f"test_cls_report_k={aggregated_best_k}.csv"
    #     save_classification_report_csv(
    #         labels_true=test_labels,
    #         labels_pred=test_predictions,
    #         save_path=test_report_path,
    #         k=aggregated_best_k
    #     )
        
    #     # Update all inner fold results with test accuracy and aggregated best k
    #     for result in inner_fold_results:
    #         result['aggregated_best_k'] = aggregated_best_k
    #         result['test_accuracy'] = test_accuracy
    #         result['avg_val_accuracy'] = avg_val_accuracy
        
    #     all_results.extend(inner_fold_results)
    
    # # Save summary CSV for this outer fold
    # if len(all_results) > 0:
    #     summary_df = pd.DataFrame(all_results)
    #     summary_path = output_outer_dir / f"cv_results_outer_fold_{outer_fold}.csv"
    #     summary_df.to_csv(summary_path, index=False)
    #     print(f"\n{'='*60}")
    #     print(f"Outer fold {outer_fold} complete!")
    #     print(f"Results saved to: {output_outer_dir}")
    #     print(f"Summary CSV saved to: {summary_path}")
    #     print(f"{'='*60}")
    # else:
    #     print(f"\nNo results to save for outer fold {outer_fold}. Please check that data exists.")


if __name__ == "__main__":
    main()
