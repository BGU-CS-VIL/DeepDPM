"""Clustering evaluation metrics."""

import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from typing import Dict


def compute_clustering_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """Compute clustering evaluation metrics.

    Args:
        predictions: (N,) predicted cluster assignments
        labels: (N,) ground truth labels

    Returns:
        Dictionary with metrics:
            - accuracy: Clustering accuracy (with optimal assignment)
            - nmi: Normalized Mutual Information
            - ari: Adjusted Rand Index
    """
    # Convert to numpy
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Compute metrics
    accuracy = clustering_accuracy(predictions, labels)
    nmi = normalized_mutual_info_score(labels, predictions, average_method='geometric')
    ari = adjusted_rand_score(labels, predictions)

    return {
        'accuracy': float(accuracy),
        'nmi': float(nmi),
        'ari': float(ari)
    }


def clustering_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute clustering accuracy with optimal assignment.

    Uses Hungarian algorithm to find best mapping between
    predicted clusters and ground truth labels.

    Args:
        y_pred: (N,) predicted cluster assignments
        y_true: (N,) ground truth labels

    Returns:
        Accuracy score (0 to 1)
    """
    # Build confusion matrix
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    n_samples = y_true.shape[0]

    # Remap labels to be 0-indexed and contiguous
    # This handles cases where K_pred > K_true or labels have gaps
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    # Create mapping dictionaries
    true_map = {old: new for new, old in enumerate(unique_true)}
    pred_map = {old: new for new, old in enumerate(unique_pred)}

    # Remap
    y_true_remapped = np.array([true_map[y] for y in y_true])
    y_pred_remapped = np.array([pred_map[y] for y in y_pred])

    n_clusters_true = len(unique_true)
    n_clusters_pred = len(unique_pred)

    # Confusion matrix
    n_classes = max(n_clusters_true, n_clusters_pred)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i in range(n_samples):
        confusion_matrix[y_pred_remapped[i], y_true_remapped[i]] += 1

    # Hungarian algorithm to find optimal assignment
    # Maximize sum of diagonal elements
    # linear_sum_assignment minimizes, so we negate
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    # Compute accuracy
    accuracy = confusion_matrix[row_ind, col_ind].sum() / n_samples

    return accuracy
