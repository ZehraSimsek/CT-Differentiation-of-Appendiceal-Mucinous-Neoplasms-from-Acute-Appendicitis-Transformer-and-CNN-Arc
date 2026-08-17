"""
losses/focal_loss.py — Clinical Focal Loss for Imbalanced Medical Image Classification
=======================================================================================
Implements the Focal Loss (Lin et al., 2017) adapted for clinical 3D medical image
classification with label smoothing and dynamic class weighting.

Complies with MULTI_RUN_PROTOCOL.md specifications:
- Focal gamma: 1.0 (or 2.0)
- Label smoothing: 0.05
- Multi-class / Binary CE compatible with [B, 2] logits and [B] target indices
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalFocalLoss(nn.Module):
    """
    Clinical Focal Loss operating on raw logits.
    
    Parameters
    ----------
    weight : Tensor, optional
        Per-class weights for balancing (alpha).
    gamma : float, default 1.0
        Focusing parameter. gamma = 0 recovers standard cross-entropy.
    label_smoothing : float, default 0.05
        Label smoothing factor in [0.0, 1.0].
    reduction : str, default "mean"
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 1.0,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Clinical Focal Loss.

        Parameters
        ----------
        logits : Tensor, shape (B, C)
            Raw (pre-softmax) model outputs.
        targets : Tensor, shape (B,)
            Ground-truth class indices.
        """
        # Cross entropy loss with label smoothing and optional class weighting
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # pt = exp(-ce)
        pt = torch.exp(-ce_loss)

        # Focal modulating factor: (1 - pt)^gamma
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Alias for backward compatibility
FocalLoss = ClinicalFocalLoss
