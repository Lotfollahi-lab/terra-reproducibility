"""
Metrics of the gene panel subsampling experiment (gene_panel_subsampling.py).

All functions take embeddings as (n_cells, n_dims) arrays whose rows are the same cells in the same order, label
arrays with NaN or None for unlabelled cells, and a section array that names the tissue section of every cell.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             normalized_mutual_info_score)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def labelled_mask(labels: np.ndarray) -> np.ndarray:
    """True for cells with a label (not None, NaN or an empty string)."""
    return np.array([not (label is None or (isinstance(label, float) and np.isnan(label)) or label == '')
                     for label in labels], dtype=bool)


def exact_knn_indices(X: np.ndarray, sections: np.ndarray, k: int, n_jobs: int = 1) -> np.ndarray:
    """
    Exact k nearest neighbors (Euclidean) of every cell among the cells of its own section, as global row indices.

    The cell itself is not its own neighbor (`kneighbors()` without query points leaves the query point out).
    """
    indices = np.empty((X.shape[0], k), dtype=np.int64)
    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        if len(rows) <= k:
            raise ValueError(f'Section {section} has {len(rows)} cells, which is not more than k = {k}.')
        nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=n_jobs)
        nn.fit(X[rows])
        _, ind = nn.kneighbors()
        indices[rows] = rows[ind]
    return indices


def knn_overlap(indices_a: np.ndarray, indices_b: np.ndarray) -> np.ndarray:
    """Per cell, the fraction of its k nearest neighbors in representation a that are also among them in b."""
    if indices_a.shape != indices_b.shape:
        raise ValueError(f'kNN index arrays differ in shape: {indices_a.shape} vs {indices_b.shape}.')
    shared = (indices_a[:, :, None] == indices_b[:, None, :]).any(axis=2).sum(axis=1)
    return shared / indices_a.shape[1]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear centered kernel alignment (Kornblith et al., 2019) of two representations of the same cells; 1 means the
    representations agree up to rotation and isotropic scaling.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Y.T @ X, ord='fro') ** 2
    norm_x = np.linalg.norm(X.T @ X, ord='fro')
    norm_y = np.linalg.norm(Y.T @ Y, ord='fro')
    return float(cross / (norm_x * norm_y))


def paired_cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cosine similarity of every cell's embedding in X with its own embedding in Y (same space required)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    numerator = (X * Y).sum(axis=1)
    denominator = np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def label_scores(labels: np.ndarray, clusters: np.ndarray) -> dict:
    """NMI and ARI of clusters against labels, over the labelled cells."""
    mask = labelled_mask(labels)
    labels_str = np.asarray(labels, dtype=object)[mask].astype(str)
    return {'nmi': float(normalized_mutual_info_score(labels_str, clusters[mask], average_method='arithmetic')),
            'ari': float(adjusted_rand_score(labels_str, clusters[mask]))}


def partition_agreement(clusters_a: np.ndarray, clusters_b: np.ndarray) -> dict:
    """NMI and ARI between two partitions of the same cells."""
    return {'nmi': float(normalized_mutual_info_score(clusters_a, clusters_b, average_method='arithmetic')),
            'ari': float(adjusted_rand_score(clusters_a, clusters_b))}


def mean_pairwise_agreement(partitions_a: np.ndarray, partitions_b: np.ndarray, exclude_same_index: bool) -> dict:
    """
    Mean NMI and ARI over all pairs (i, j) of the rows of partitions_a (n_seeds_a, n_cells) and partitions_b
    (n_seeds_b, n_cells); with exclude_same_index, pairs with i == j are left out (seed-to-seed noise ceiling of one
    representation, where partitions_a is partitions_b).
    """
    nmis, aris = [], []
    for i, clusters_a in enumerate(partitions_a):
        for j, clusters_b in enumerate(partitions_b):
            if exclude_same_index and i == j:
                continue
            scores = partition_agreement(clusters_a, clusters_b)
            nmis.append(scores['nmi'])
            aris.append(scores['ari'])
    if not nmis:
        return {'nmi': float('nan'), 'ari': float('nan'), 'n_pairs': 0}
    return {'nmi': float(np.mean(nmis)), 'ari': float(np.mean(aris)), 'n_pairs': len(nmis)}


def leave_one_section_out_transfer(X_train: np.ndarray, X_test: np.ndarray, labels: np.ndarray,
                                   sections: np.ndarray, k: int, n_jobs: int = 1) -> list:
    """
    kNN label transfer between sections: for every section s, a kNN classifier trained on X_train of the labelled
    cells of all other sections predicts the labels of the labelled cells of s from X_test. X_train is X_test for
    transfer within one representation; X_train is the full-panel embedding and X_test a subset-panel embedding (same
    model, same space) for transfer across panels.

    Returns one dict per section with accuracy and macro F1 (classes present in the truth or the prediction).
    """
    mask = labelled_mask(labels)
    labels_str = np.asarray(labels, dtype=object).astype(str)
    results = []
    for section in np.unique(sections):
        train = mask & (sections != section)
        test = mask & (sections == section)
        if train.sum() == 0 or test.sum() == 0:
            continue
        classifier = KNeighborsClassifier(n_neighbors=min(k, int(train.sum())), algorithm='brute', n_jobs=n_jobs)
        classifier.fit(X_train[train], labels_str[train])
        predicted = classifier.predict(X_test[test])
        results.append({'section': str(section),
                        'n_test_cells': int(test.sum()),
                        'accuracy': float(accuracy_score(labels_str[test], predicted)),
                        'macro_f1': float(f1_score(labels_str[test], predicted, average='macro', zero_division=0))})
    return results


def n_classes_with_min_cells(labels: np.ndarray, min_cells: int) -> int:
    """Number of label classes with at least min_cells labelled cells (target number of Leiden clusters)."""
    mask = labelled_mask(labels)
    _, counts = np.unique(np.asarray(labels, dtype=object)[mask].astype(str), return_counts=True)
    return int((counts >= min_cells).sum())
