"""
losses/focal_loss.py — Clinical Focal Loss for Imbalanced Medical Image Classification
=======================================================================================
Implements the Focal Loss adapted for clinical 3D medical image classification.
Matches the Transformer models' ClinicalFocalLoss implementation exactly.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
class ClinicalFocalLoss(nn.Module):
    """
    Sensitivity-first training için tasarlandı.
    - Focal: Zor/nadir vakalara odak
    - pos_weight: Tümör sınıfına ekstra ceza
    - Label Smoothing: Aşırı güven önleme
    """
    def __init__(self, pos_weight=1.0, gamma=2.0, label_smoothing=0.05, num_classes=2, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.smoothing = label_smoothing
        self.num_classes = num_classes
        self.pos_weight = pos_weight
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        weight = torch.ones(self.num_classes, device=logits.device)
        weight[1] = self.pos_weight  
        log_prob = F.log_softmax(logits, dim=1)
        ce_loss = -(true_dist * log_prob * weight).sum(dim=1)
        prob = torch.softmax(logits, dim=1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()
FocalLoss = ClinicalFocalLoss
