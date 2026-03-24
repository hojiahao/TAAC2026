"""
Evaluation: AUC of ROC (the official TAAC 2026 metric).
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from typing import Dict, Tuple


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Returns:
        dict with "auc", "logloss", "num_samples", "pos_rate"
    """
    model.eval()
    all_labels = []
    all_preds = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits = model(batch)
        probs = torch.sigmoid(logits.squeeze(-1))  # (B,)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # AUC
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5  # only one class present

    # LogLoss
    eps = 1e-7
    all_preds_clipped = np.clip(all_preds, eps, 1 - eps)
    logloss = -np.mean(
        all_labels * np.log(all_preds_clipped) +
        (1 - all_labels) * np.log(1 - all_preds_clipped)
    )

    pos_rate = all_labels.mean()

    return {
        "auc": float(auc),
        "logloss": float(logloss),
        "num_samples": int(len(all_labels)),
        "pos_rate": float(pos_rate),
    }
