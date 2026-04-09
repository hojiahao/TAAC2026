"""
Evaluation: AUC of ROC (official TAAC 2026 metric).
Optimized: accumulate on GPU, transfer to CPU once at end.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from typing import Dict


@torch.no_grad()
def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits = model(batch)
        probs = torch.sigmoid(logits.squeeze(-1))

        # Accumulate as tensors (avoid per-batch CPU sync)
        all_preds.append(probs)
        all_labels.append(batch["label"])

    # Single CPU transfer at end
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5

    eps = 1e-7
    preds_clip = np.clip(all_preds, eps, 1 - eps)
    logloss = -np.mean(
        all_labels * np.log(preds_clip) +
        (1 - all_labels) * np.log(1 - preds_clip)
    )

    return {
        "auc": float(auc),
        "logloss": float(logloss),
        "num_samples": int(len(all_labels)),
        "pos_rate": float(all_labels.mean()),
    }
