"""
Utility metrics and helpers.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def compute_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute AUC of ROC (official TAAC 2026 metric)."""
    try:
        return roc_auc_score(labels, preds)
    except ValueError:
        return 0.5


def compute_gauc(
    labels: np.ndarray,
    preds: np.ndarray,
    user_ids: np.ndarray,
) -> float:
    """
    Group AUC: compute per-user AUC and average (weighted by impressions).
    Useful for offline analysis even if not the official metric.
    """
    unique_users = np.unique(user_ids)
    total_weight = 0
    weighted_auc = 0

    for uid in unique_users:
        mask = user_ids == uid
        y = labels[mask]
        p = preds[mask]
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, p)
        weighted_auc += auc * len(y)
        total_weight += len(y)

    return weighted_auc / total_weight if total_weight > 0 else 0.5


def compute_logloss(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute binary cross-entropy (log loss)."""
    return log_loss(labels, np.clip(preds, 1e-7, 1 - 1e-7))
