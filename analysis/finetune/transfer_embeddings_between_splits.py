"""
Slice heldout embeddings into per-fold train/test features, matching cells via cell_id.

Two source modes:

  Mode 1 -- flat heldout npy (one model per call):
    python analysis/finetune/transfer_embeddings_between_splits.py \
        --models terra \
        --source-adata /nfs/team361/am84/TERRA/data/xhs/xhs_harmonized_nemo_heldout.h5ad \
        --source-embeddings /nfs/team361/am84/TERRA/data/xhs/terra_nbr_emb.npy \
        --source-label-column niche19 \
        --target-dir /nfs/team361/am84/TERRA/tasks/xhs_niche-transfer

  Mode 2 -- existing model_artifacts (transfer between tasks, multiple models per call):
    python analysis/finetune/transfer_embeddings_between_splits.py \
        --models terra scgpt-spatial novae \
        --source-dir /lustre/.../xhs_niche-transfer \
        --target-dir /lustre/.../xhs_disease-classification-patient

Valid --models choices: terra, terra-cell, terra-nbr, scgpt-spatial, novae, cellplm, nicheformer
Model name to output directory mapping:
  terra        -> model_artifacts/terra-zeroshot/
  terra-cell   -> model_artifacts/terra-cell/
  terra-nbr    -> model_artifacts/terra-nbr/
  scgpt-spatial -> model_artifacts/scgpt-spatial-zeroshot/
  novae        -> model_artifacts/novae-zeroshot/
  cellplm      -> model_artifacts/cellplm-zeroshot/
  nicheformer  -> model_artifacts/nicheformer-zeroshot/

Outputs per outer fold (all folds found automatically unless --outer-folds is set):
  {target-dir}/model_artifacts/{model-dir}/outer-fold_N/
    train_features.npy, train_labels.npy
    test_features.npy,  test_labels.npy

Target format is auto-detected:
  Lightweight -- outer-fold_N/train_cell_ids.npy present
  Full        -- data/outer-fold_N/train.h5ad present
"""

import argparse
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_model_dir_name(model_name: str) -> str:
    """Map short model name to full directory name."""
    mapping = {
        "terra": "terra-zeroshot",
        "terra-cell": "terra-cell",
        "terra-nbr": "terra-nbr",
        "scgpt-spatial": "scgpt-spatial-zeroshot",
        "novae": "novae-zeroshot",
        "cellplm": "cellplm-zeroshot",
        "nicheformer": "nicheformer-zeroshot",
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model name: {model_name}. Must be one of {list(mapping.keys())}")
    return mapping[model_name]


def load_embeddings_and_labels(
    source_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load embeddings and labels from source directory.

    Returns:
        (cellid_to_emb, cellid_to_label): Dictionaries mapping cell_id to embedding/label
    """
    train_emb_path = source_dir / "train_features.npy"
    test_emb_path = source_dir / "test_features.npy"
    train_label_path = source_dir / "train_labels.npy"
    test_label_path = source_dir / "test_labels.npy"
    
    # Load embeddings
    print(f"  Loading {train_emb_path}...")
    train_emb = np.load(train_emb_path)
    print(f"  Loading {test_emb_path}...")
    test_emb = np.load(test_emb_path)
    
    # Load labels
    print(f"  Loading {train_label_path}...")
    train_labels = np.load(train_label_path)
    print(f"  Loading {test_label_path}...")
    test_labels = np.load(test_label_path)
    
    # Load corresponding h5ad files to get cell_ids
    # source_dir is: .../model_artifacts/{model}/outer-fold_1
    # Need to go up to base dir, then into data/
    base_dir = source_dir.parent.parent.parent  # Go up from outer-fold_1 -> model -> model_artifacts -> base
    train_adata_path = base_dir / "data" / source_dir.name / "train.h5ad"
    test_adata_path = base_dir / "data" / source_dir.name / "test.h5ad"
    
    print(f"  Loading {train_adata_path}...")
    train_adata = sc.read_h5ad(train_adata_path)
    print(f"  Loading {test_adata_path}...")
    test_adata = sc.read_h5ad(test_adata_path)
    
    # Extract cell_ids
    train_ids = train_adata.obs["cell_id"].tolist()
    test_ids = test_adata.obs["cell_id"].tolist()
    
    # Sanity checks
    assert len(train_emb) == len(train_ids), f"Train embedding length ({len(train_emb)}) != train cell count ({len(train_ids)})"
    assert len(test_emb) == len(test_ids), f"Test embedding length ({len(test_emb)}) != test cell count ({len(test_ids)})"
    assert len(train_labels) == len(train_ids), f"Train label length ({len(train_labels)}) != train cell count ({len(train_ids)})"
    assert len(test_labels) == len(test_ids), f"Test label length ({len(test_labels)}) != test cell count ({len(test_ids)})"
    
    # Build mappings
    cellid_to_emb = {}
    cellid_to_label = {}
    
    for idx, (cid, emb) in enumerate(zip(train_ids, train_emb)):
        if cid in cellid_to_emb:
            raise ValueError(f"Duplicate cell_id in train set: {cid}")
        cellid_to_emb[cid] = emb
        cellid_to_label[cid] = train_labels[idx]
    
    for idx, (cid, emb) in enumerate(zip(test_ids, test_emb)):
        if cid in cellid_to_emb:
            raise ValueError(f"Duplicate cell_id found (in both train and test): {cid}")
        cellid_to_emb[cid] = emb
        cellid_to_label[cid] = test_labels[idx]
    
    print(f"  Loaded {len(cellid_to_emb)} total cells ({len(train_ids)} train, {len(test_ids)} test)")
    
    return cellid_to_emb, cellid_to_label


def load_embeddings_from_adata_and_npy(
    adata,
    embeddings_npy_path: Path,
    labels_npy_path: Optional[Path] = None,
    label_column: Optional[str] = None,
    cell_ids_npy_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load embeddings and labels from an AnnData object and npy file.

    Args:
        adata: AnnData object containing cell_ids in obs["cell_id"]
        embeddings_npy_path: Path to npy file containing embeddings
        labels_npy_path: Optional path to npy file containing labels (same row order as embeddings)
        label_column: Optional column name in adata.obs to use for labels
        cell_ids_npy_path: Optional path to npy file giving the cell_id row order of the
            embeddings. Required when embeddings are in a different row order than adata.obs
            (e.g. terra embeddings produced from a HuggingFace dataset iterator).

    Returns:
        (cellid_to_emb, cellid_to_label): Dictionaries mapping cell_id to embedding/label
    """
    if labels_npy_path is None and label_column is None:
        raise ValueError("Either labels_npy_path or label_column must be provided")
    if labels_npy_path is not None and label_column is not None:
        raise ValueError("Cannot specify both labels_npy_path and label_column")
    if "cell_id" not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'cell_id' column")

    print(f"  Loading embeddings from {embeddings_npy_path}...")
    embeddings = np.load(embeddings_npy_path)

    # cell_ids controls which cell maps to which embedding row
    if cell_ids_npy_path is not None:
        print(f"  Loading cell_ids from {cell_ids_npy_path}...")
        cell_ids = np.load(cell_ids_npy_path, allow_pickle=True).tolist()
    else:
        cell_ids = adata.obs["cell_id"].tolist()

    assert len(embeddings) == len(cell_ids), (
        f"Embedding length ({len(embeddings)}) != cell_ids length ({len(cell_ids)})"
    )

    if labels_npy_path is not None:
        print(f"  Loading labels from {labels_npy_path}...")
        labels_arr = np.load(labels_npy_path)
        assert len(labels_arr) == len(cell_ids), (
            f"Label length ({len(labels_arr)}) != cell_ids length ({len(cell_ids)})"
        )
        labels_list = labels_arr.tolist()
    else:
        print(f"  Extracting labels from adata.obs['{label_column}']...")
        if label_column not in adata.obs.columns:
            raise ValueError(f"Label column '{label_column}' not found in adata.obs")
        # Build lookup so h5ad obs order doesn't need to match embedding order
        adata_label_lookup = dict(zip(adata.obs["cell_id"].tolist(), adata.obs[label_column].tolist()))
        labels_list = [adata_label_lookup[cid] for cid in cell_ids]

    cellid_to_emb = {}
    cellid_to_label = {}
    for idx, cid in enumerate(cell_ids):
        if cid in cellid_to_emb:
            raise ValueError(f"Duplicate cell_id found: {cid}")
        cellid_to_emb[cid] = embeddings[idx]
        cellid_to_label[cid] = labels_list[idx]

    print(f"  Loaded {len(cellid_to_emb)} total cells")
    return cellid_to_emb, cellid_to_label


def load_all_source_embeddings(
    model_name: str,
    source_base: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load all embeddings from source outer-fold_1 (which contains all cells).
    
    Args:
        model_name: Short model name (terra, scgpt-spatial, novae)
        source_base: Base directory of source dataset
    
    Returns:
        (cellid_to_emb, cellid_to_label): Dictionaries mapping cell_id to embedding/label
    """
    model_dir = get_model_dir_name(model_name)

    # Always use outer-fold_1 from source (contains all cells)
    source_fold_dir = source_base / "model_artifacts" / model_dir / "outer-fold_1"

    if not source_fold_dir.exists():
        raise ValueError(f"Source directory does not exist: {source_fold_dir}")

    print(f"Loading source embeddings from {source_fold_dir}...")
    cellid_to_emb, cellid_to_label = load_embeddings_and_labels(source_fold_dir)
    
    return cellid_to_emb, cellid_to_label


def load_all_source_embeddings_from_adata(
    source_adata_path: Path,
    source_embeddings_path: Path,
    source_labels_path: Optional[Path] = None,
    label_column: Optional[str] = None,
    source_cell_ids_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load all embeddings from source adata and npy file.

    Args:
        source_adata_path: Path to source h5ad file
        source_embeddings_path: Path to source embeddings npy file
        source_labels_path: Optional path to source labels npy file
        label_column: Optional column name in adata.obs to use for labels
        source_cell_ids_path: Optional path to npy giving embedding row order.
            Required for terra embeddings (dataset iteration order ≠ h5ad obs order).

    Returns:
        (cellid_to_emb, cellid_to_label): Dictionaries mapping cell_id to embedding/label
    """
    if not source_adata_path.exists():
        raise ValueError(f"Source adata file does not exist: {source_adata_path}")
    if not source_embeddings_path.exists():
        raise ValueError(f"Source embeddings file does not exist: {source_embeddings_path}")

    print(f"Loading source adata from {source_adata_path}...")
    adata = sc.read_h5ad(source_adata_path)

    print(f"Loading source embeddings from {source_embeddings_path}...")
    cellid_to_emb, cellid_to_label = load_embeddings_from_adata_and_npy(
        adata=adata,
        embeddings_npy_path=source_embeddings_path,
        labels_npy_path=source_labels_path,
        label_column=label_column,
        cell_ids_npy_path=source_cell_ids_path,
    )

    return cellid_to_emb, cellid_to_label


def transfer_embeddings_for_fold(
    model_name: str,
    outer_fold: int,
    cellid_to_emb: Dict[str, np.ndarray],
    cellid_to_label: Dict[str, np.ndarray],
    target_base: Path,
) -> None:
    """
    Transfer embeddings for a single target outer fold using pre-loaded source mappings.
    
    Args:
        model_name: Short model name (terra, scgpt-spatial, novae)
        outer_fold: Outer fold number (1, 2, ...)
        cellid_to_emb: Pre-loaded mapping from cell_id to embedding
        cellid_to_label: Pre-loaded mapping from cell_id to label
        target_base: Base directory of target dataset (e.g., xhs_disease-classification-patient-batch)
    """
    model_dir = get_model_dir_name(model_name)

    print(f"\nProcessing {model_name} ({model_dir}) - outer-fold_{outer_fold}")

    # Target paths
    target_fold_dir = target_base / "model_artifacts" / model_dir / f"outer-fold_{outer_fold}"

    # Detect lightweight format (cell_ids.npy directly under outer-fold_N/)
    lightweight_fold_dir = target_base / f"outer-fold_{outer_fold}"
    lightweight = (lightweight_fold_dir / "train_cell_ids.npy").exists()

    if lightweight:
        print(f"  Loading target splits (lightweight) from {lightweight_fold_dir}...")
        train_ids_target = np.load(lightweight_fold_dir / "train_cell_ids.npy", allow_pickle=True).tolist()
        test_ids_target = np.load(lightweight_fold_dir / "test_cell_ids.npy", allow_pickle=True).tolist()
        preloaded_train_labels = np.load(lightweight_fold_dir / "train_labels.npy", allow_pickle=True)
        preloaded_test_labels = np.load(lightweight_fold_dir / "test_labels.npy", allow_pickle=True)
    else:
        target_data_dir = target_base / "data" / f"outer-fold_{outer_fold}"
        train_adata_path = target_data_dir / "train.h5ad"
        test_adata_path = target_data_dir / "test.h5ad"

        if not train_adata_path.exists():
            print(f"  WARNING: Target train file does not exist: {train_adata_path}")
            print(f"  Skipping...")
            return
        if not test_adata_path.exists():
            print(f"  WARNING: Target test file does not exist: {test_adata_path}")
            print(f"  Skipping...")
            return

        print(f"  Loading target splits from {target_data_dir}...")
        train_adata = sc.read_h5ad(train_adata_path)
        test_adata = sc.read_h5ad(test_adata_path)
        train_ids_target = train_adata.obs["cell_id"].tolist()
        test_ids_target = test_adata.obs["cell_id"].tolist()
        preloaded_train_labels = None
        preloaded_test_labels = None
    
    # Verify all cells exist in source
    missing_train = set(train_ids_target) - cellid_to_emb.keys()
    missing_test = set(test_ids_target) - cellid_to_emb.keys()
    
    if missing_train:
        raise ValueError(f"Missing train cells in source: {len(missing_train)} cells. First 10: {list(missing_train)[:10]}")
    if missing_test:
        raise ValueError(f"Missing test cells in source: {len(missing_test)} cells. First 10: {list(missing_test)[:10]}")
    
    # Reconstruct embeddings (and labels if not preloaded) in target order
    print(f"  Reconstructing embeddings for {len(train_ids_target)} train cells...")
    train_emb_target = np.stack([cellid_to_emb[cid] for cid in train_ids_target])
    train_labels_target = preloaded_train_labels if preloaded_train_labels is not None else np.array([cellid_to_label[cid] for cid in train_ids_target])

    print(f"  Reconstructing embeddings for {len(test_ids_target)} test cells...")
    test_emb_target = np.stack([cellid_to_emb[cid] for cid in test_ids_target])
    test_labels_target = preloaded_test_labels if preloaded_test_labels is not None else np.array([cellid_to_label[cid] for cid in test_ids_target])
    
    # Validate dimensions
    assert train_emb_target.shape[0] == len(train_ids_target), "Train embedding shape mismatch"
    assert test_emb_target.shape[0] == len(test_ids_target), "Test embedding shape mismatch"
    assert train_labels_target.shape[0] == len(train_ids_target), "Train label shape mismatch"
    assert test_labels_target.shape[0] == len(test_ids_target), "Test label shape mismatch"

    # Create target directory
    target_fold_dir.mkdir(parents=True, exist_ok=True)

    # Encode string labels to integers if needed (consistent encoding across train+test)
    if train_labels_target.dtype == object or train_labels_target.dtype.kind in ('U', 'S'):
        all_label_vals = np.concatenate([train_labels_target, test_labels_target])
        unique_labels = np.sort(np.unique(all_label_vals))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        train_labels_target = np.array([label_to_int[l] for l in train_labels_target], dtype=np.int64)
        test_labels_target = np.array([label_to_int[l] for l in test_labels_target], dtype=np.int64)
        np.save(target_fold_dir / "label_mapping.npy", unique_labels)
    
    # Save embeddings and labels
    train_emb_path = target_fold_dir / "train_features.npy"
    test_emb_path = target_fold_dir / "test_features.npy"
    train_label_path = target_fold_dir / "train_labels.npy"
    test_label_path = target_fold_dir / "test_labels.npy"
    
    print(f"  Saving to {target_fold_dir}...")
    print(f"    {train_emb_path} (shape: {train_emb_target.shape})")
    np.save(train_emb_path, train_emb_target.astype(np.float32))
    
    print(f"    {test_emb_path} (shape: {test_emb_target.shape})")
    np.save(test_emb_path, test_emb_target.astype(np.float32))
    
    print(f"    {train_label_path} (shape: {train_labels_target.shape})")
    np.save(train_label_path, train_labels_target)
    
    print(f"    {test_label_path} (shape: {test_labels_target.shape})")
    np.save(test_label_path, test_labels_target)
    
    print(f"  ✓ Completed {model_name} - outer-fold_{outer_fold}")


def find_outer_folds(target_base: Path) -> List[int]:
    """Find all outer-fold directories in target dataset (lightweight or full format)."""
    # Lightweight format: outer-fold_N/ directly under target_base
    folds = []
    for fold_dir in target_base.glob("outer-fold_*"):
        if (fold_dir / "train_cell_ids.npy").exists():
            folds.append(int(fold_dir.name.split("_")[-1]))
    if folds:
        return sorted(folds)

    # Full format: outer-fold_N/ under target_base/data/
    data_dir = target_base / "data"
    if not data_dir.exists():
        return []
    for fold_dir in data_dir.glob("outer-fold_*"):
        folds.append(int(fold_dir.name.split("_")[-1]))
    return sorted(folds)


def main():
    parser = argparse.ArgumentParser(
        description="Transfer embeddings between dataset splits by matching cell_ids"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=["terra", "terra-cell", "terra-nbr", "scgpt-spatial", "novae", "cellplm", "nicheformer"],
        help="Model names to transfer (can specify multiple)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Base directory of source dataset (e.g., .../xhs_niche-transfer). Required if --source-adata not provided.",
    )
    parser.add_argument(
        "--source-adata",
        type=str,
        default=None,
        help="Path to source h5ad file. Required if --source-dir not provided.",
    )
    parser.add_argument(
        "--source-embeddings",
        type=str,
        default=None,
        help="Path to source embeddings npy file. Required if --source-dir not provided.",
    )
    parser.add_argument(
        "--source-cell-ids",
        type=str,
        default=None,
        help=(
            "Path to npy file giving the cell_id row order of the source embeddings. "
            "Required for terra embeddings (dataset iteration order != h5ad obs order)."
        ),
    )
    parser.add_argument(
        "--source-labels",
        type=str,
        default=None,
        help="Path to source labels npy file (optional). Can use --source-label-column instead.",
    )
    parser.add_argument(
        "--source-label-column",
        type=str,
        default=None,
        help="Column name in source adata.obs to use for labels (optional). Can use --source-labels instead.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Base directory of target dataset (e.g., .../xhs_disease-classification-patient-batch)",
    )
    parser.add_argument(
        "--outer-folds",
        nargs="+",
        type=int,
        default=None,
        help="Specific outer folds to process (default: all found in target directory)",
    )
    
    args = parser.parse_args()
    
    # Validate that either source-dir or (source-adata + source-embeddings) is provided
    if args.source_dir is None:
        if args.source_adata is None or args.source_embeddings is None:
            raise ValueError(
                "Either --source-dir must be provided, or both --source-adata and --source-embeddings must be provided"
            )
    else:
        if args.source_adata is not None or args.source_embeddings is not None:
            raise ValueError(
                "Cannot specify both --source-dir and --source-adata/--source-embeddings. Use one or the other."
            )
    
    target_base = Path(args.target_dir)
    
    # Validate directories/files
    if args.source_dir is not None:
        source_base = Path(args.source_dir)
        if not source_base.exists():
            raise ValueError(f"Source directory does not exist: {source_base}")
    else:
        source_adata_path = Path(args.source_adata)
        source_embeddings_path = Path(args.source_embeddings)
        if not source_adata_path.exists():
            raise ValueError(f"Source adata file does not exist: {source_adata_path}")
        if not source_embeddings_path.exists():
            raise ValueError(f"Source embeddings file does not exist: {source_embeddings_path}")
        if args.source_labels is not None:
            source_labels_path = Path(args.source_labels)
            if not source_labels_path.exists():
                raise ValueError(f"Source labels file does not exist: {source_labels_path}")
        else:
            source_labels_path = None
    
    if not target_base.exists():
        raise ValueError(f"Target directory does not exist: {target_base}")
    
    # Find outer folds
    if args.outer_folds:
        outer_folds = args.outer_folds
    else:
        outer_folds = find_outer_folds(target_base)
        if not outer_folds:
            raise ValueError(f"No outer-fold directories found in {target_base / 'data'}")
    
    if args.source_dir:
        print(f"Source directory: {source_base}")
    else:
        print(f"Source adata: {source_adata_path}")
        print(f"Source embeddings: {source_embeddings_path}")
        if source_labels_path:
            print(f"Source labels: {source_labels_path}")
        elif args.source_label_column:
            print(f"Source label column: {args.source_label_column}")
    print(f"Target directory: {target_base}")
    print(f"Models: {args.models}")
    print(f"Outer folds: {outer_folds}")
    print("=" * 80)
    
    # Process each model
    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}")
        
        # Load all embeddings from source (once per model)
        try:
            if args.source_dir:
                cellid_to_emb, cellid_to_label = load_all_source_embeddings(
                    model_name=model_name,
                    source_base=source_base,
                )
            else:
                cellid_to_emb, cellid_to_label = load_all_source_embeddings_from_adata(
                    source_adata_path=source_adata_path,
                    source_embeddings_path=source_embeddings_path,
                    source_labels_path=source_labels_path,
                    label_column=args.source_label_column,
                )
        except Exception as e:
            print(f"\n  ✗ ERROR loading source embeddings for {model_name}: {e}")
            raise
        
        # Transfer to each target outer fold
        for outer_fold in outer_folds:
            try:
                transfer_embeddings_for_fold(
                    model_name=model_name,
                    outer_fold=outer_fold,
                    cellid_to_emb=cellid_to_emb,
                    cellid_to_label=cellid_to_label,
                    target_base=target_base,
                )
            except Exception as e:
                print(f"\n  ✗ ERROR processing {model_name} - outer-fold_{outer_fold}: {e}")
                raise
    
    print("\n" + "=" * 80)
    print("All transfers completed successfully!")


if __name__ == "__main__":
    main()
