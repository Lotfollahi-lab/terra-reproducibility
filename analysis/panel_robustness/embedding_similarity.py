"""
Similarity of two embeddings of the same cells (e.g. one tissue embedded with two gene panels, or by two models).

Two kinds of metrics:
- Metrics that need one embedding space (the same model; a cell is compared with itself by coordinates): matched
  cosine similarity (raw and centered), its null (random other cell of the same section) and the gap between them,
  and cross-panel re-identification: for each cell, how many cells of its section are closer to it in the other
  embedding than its own counterpart (FOSCTTM, Liu et al. 2019; 0 = always the closest, 0.5 = random), and whether
  the counterpart is among the k closest.
- Metrics that are invariant to rotation and isotropic scaling, so they also compare different models: linear CKA,
  kNN overlap, representational similarity (Spearman correlation of pairwise distances) and the same matched-cell
  metrics after an orthogonal Procrustes alignment that is fit on one half of the cells and evaluated on the other.

All neighbor searches and re-identification use only the cells of the same section.
"""

import numpy as np


def normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def matched_cosine(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cosine similarity of each cell in X with the same cell in Y."""
    return (normalize_rows(X) * normalize_rows(Y)).sum(axis=1).astype(np.float64)


def null_cosine(X: np.ndarray, Y: np.ndarray, sections: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cosine similarity of each cell in X with a random other cell of its section in Y (null for matched_cosine)."""
    partner = np.empty(len(X), dtype=np.int64)
    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        if len(rows) < 2:
            raise ValueError(f'Section {section} has fewer than 2 cells.')
        shift = rng.integers(1, len(rows), size=len(rows))             # never the cell itself
        partner[rows] = rows[(np.arange(len(rows)) + shift) % len(rows)]
    return matched_cosine(X, Y[partner])


def _closer_counts(A: np.ndarray, B: np.ndarray, block_size: int, tolerance: float = 1e-5) -> np.ndarray:
    """
    For row i of A, the number of rows j != i of B with cos(A_i, B_j) > cos(A_i, B_i) + tolerance (rows normalized).
    The tolerance keeps float32 rounding from ranking a near-identical cell above an identical counterpart.
    """
    n = len(A)
    true = (A * B).sum(axis=1)
    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        S = A[start:stop] @ B.T
        counts[start:stop] = (S > true[start:stop, None] + tolerance).sum(axis=1)
    return counts


def reidentification(X: np.ndarray, Y: np.ndarray, sections: np.ndarray, ks=(1, 10),
                     block_size: int = 4096) -> dict:
    """
    Cross-panel re-identification within each section, in both directions (X -> Y and Y -> X), per cell:
    - foscttm: fraction of the other cells of the section that are closer (cosine) than the cell's counterpart,
      averaged over the two directions;
    - top<k>: whether the counterpart is among the k closest cells, averaged over the two directions.
    """
    A_all, B_all = normalize_rows(X), normalize_rows(Y)
    foscttm = np.empty(len(X))
    hits = {k: np.empty(len(X)) for k in ks}
    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        A, B = A_all[rows], B_all[rows]
        forward = _closer_counts(A, B, block_size)
        backward = _closer_counts(B, A, block_size)
        foscttm[rows] = (forward + backward) / (2 * max(len(rows) - 1, 1))
        for k in ks:
            hits[k][rows] = ((forward < k).astype(float) + (backward < k).astype(float)) / 2
    out = {'foscttm': foscttm}
    out.update({f'top{k}': hits[k] for k in ks})
    return out


def procrustes_fit(X: np.ndarray, Y: np.ndarray) -> dict:
    """Orthogonal Procrustes map of centered, unit-Frobenius-norm X onto Y (same number of dimensions)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f'Procrustes needs the same dimensions: {X.shape[1]} vs {Y.shape[1]}.')
    mean_x, mean_y = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mean_x, Y - mean_y
    scale_x, scale_y = np.linalg.norm(Xc), np.linalg.norm(Yc)
    U, _, Vt = np.linalg.svd((Xc / scale_x).T @ (Yc / scale_y), full_matrices=False)
    return {'mean_x': mean_x, 'mean_y': mean_y, 'scale_x': scale_x, 'scale_y': scale_y, 'R': U @ Vt}


def procrustes_transform(X: np.ndarray, Y: np.ndarray, fit: dict) -> tuple:
    """X mapped into Y's (centered, scaled) coordinates, and Y in the same coordinates."""
    A = ((np.asarray(X, dtype=np.float64) - fit['mean_x']) / fit['scale_x']) @ fit['R']
    B = (np.asarray(Y, dtype=np.float64) - fit['mean_y']) / fit['scale_y']
    return A.astype(np.float32), B.astype(np.float32)


def representational_similarity(X: np.ndarray, Y: np.ndarray, sections: np.ndarray, rng: np.random.Generator,
                                n_cells_per_section: int = 1500) -> dict:
    """Spearman correlation of the pairwise cosine distances of a random sample of cells per section."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    out = {}
    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        if len(rows) > n_cells_per_section:
            rows = np.sort(rng.choice(rows, n_cells_per_section, replace=False))
        out[str(section)] = float(spearmanr(pdist(X[rows], 'cosine'), pdist(Y[rows], 'cosine')).statistic)
    return out


def split_halves(sections: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of a random half of the cells of each section (the Procrustes fit cells)."""
    fit = np.zeros(len(sections), dtype=bool)
    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        fit[rng.choice(rows, len(rows) // 2, replace=False)] = True
    return fit
