"""
Loss functions for CVR prediction (AUC of ROC metric).

Strategy:
  BCE       → optimize probability calibration (proxy for AUC)
  PairAUC   → directly optimize pairwise ranking (the AUC definition)
  Combined  → BCE + PairAUC (industry best practice: calibration + ranking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance. FL(p) = -alpha * (1-p)^gamma * log(p)"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, action_types=None) -> torch.Tensor:
        probs = torch.sigmoid(logits.squeeze(-1))
        ce_loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets, reduction="none",
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class WeightedBCELoss(nn.Module):
    """BCE loss with per-sample weighting for conversion samples."""

    def __init__(self, conversion_weight: float = 1.0):
        super().__init__()
        self.conversion_weight = conversion_weight

    def forward(self, logits, targets, action_types=None):
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets, reduction="none",
        )
        if action_types is not None and self.conversion_weight != 1.0:
            weights = torch.ones_like(loss)
            weights[action_types == 2] = self.conversion_weight
            loss = loss * weights
        return loss.mean()


class PairwiseAUCLoss(nn.Module):
    """
    Pairwise AUC Loss — directly optimizes the AUC definition.

    AUC = P(score(positive) > score(negative))

    For each (positive, negative) pair in the batch:
        loss = -log(sigmoid(s_pos - s_neg))

    This is equivalent to BPR (Bayesian Personalized Ranking) loss.
    Directly maximizes the probability that positive samples rank higher.
    """

    def __init__(self, margin: float = 0.0, max_pairs: int = 256):
        super().__init__()
        self.margin = margin
        self.max_pairs = max_pairs  # cap pairs to avoid O(n*m) blow-up

    def forward(self, logits, targets, action_types=None):
        scores = logits.squeeze(-1)  # (B,)

        pos_mask = targets == 1.0
        neg_mask = targets == 0.0
        pos_scores = scores[pos_mask]  # (P,)
        neg_scores = scores[neg_mask]  # (N,)

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Sample pairs if too many (P*N can be huge)
        n_pos = min(len(pos_scores), self.max_pairs)
        n_neg = min(len(neg_scores), self.max_pairs)
        if len(pos_scores) > n_pos:
            idx = torch.randperm(len(pos_scores), device=scores.device)[:n_pos]
            pos_scores = pos_scores[idx]
        if len(neg_scores) > n_neg:
            idx = torch.randperm(len(neg_scores), device=scores.device)[:n_neg]
            neg_scores = neg_scores[idx]

        # All pairs: (n_pos, n_neg)
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0) - self.margin

        # BPR loss: -log(sigmoid(diff))
        loss = -F.logsigmoid(diff)
        return loss.mean()


class CombinedAUCLoss(nn.Module):
    """
    BCE + PairwiseAUC — industry best practice for AUC optimization.

    Total = alpha * BCE + (1-alpha) * PairwiseAUC

    BCE ensures probability calibration.
    PairwiseAUC directly pushes AUC higher.
    """

    def __init__(self, conversion_weight: float = 3.0, auc_weight: float = 0.5,
                 margin: float = 0.5, max_pairs: int = 256):
        super().__init__()
        self.bce = WeightedBCELoss(conversion_weight)
        self.pair_auc = PairwiseAUCLoss(margin, max_pairs)
        self.auc_weight = auc_weight

    def forward(self, logits, targets, action_types=None):
        bce_loss = self.bce(logits, targets, action_types)
        auc_loss = self.pair_auc(logits, targets)
        alpha = self.auc_weight
        return (1 - alpha) * bce_loss + alpha * auc_loss


def build_loss(config):
    tc = config.train
    if tc.loss_type == "focal":
        return FocalLoss(alpha=tc.focal_alpha, gamma=tc.focal_gamma)
    elif tc.loss_type == "auc":
        return CombinedAUCLoss(
            conversion_weight=tc.conversion_weight,
            auc_weight=tc.auc_loss_weight,
            margin=tc.auc_margin,
        )
    else:
        return WeightedBCELoss(conversion_weight=tc.conversion_weight)
