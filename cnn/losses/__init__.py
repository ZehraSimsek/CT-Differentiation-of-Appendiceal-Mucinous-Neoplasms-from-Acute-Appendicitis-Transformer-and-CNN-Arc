"""
losses — Loss Function Factory
===============================
Exposes ``build_loss_fn(config)`` and ``ClinicalFocalLoss`` / ``FocalLoss``
for clinical 3D medical image classification.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from config import PipelineConfig
from losses.focal_loss import ClinicalFocalLoss, FocalLoss
import numpy as np
def pos_weight_from_labels(labels):
    """
    Dinamik pos_weight hesabı:
    Veri seti dengeli hale getirildiği için (Apandisit 127, Musinoz 117)
    artık manuel 1.5 cezası vermiyoruz. Doğrudan sınıf dağılımına göre hesaplıyoruz.
    """
    labels = np.array(labels)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    if n_pos == 0:
        return 1.0
    weight = float(n_neg) / float(n_pos)
    return weight
def build_loss_fn(config: PipelineConfig, pos_weight: float | None = None) -> nn.Module:
    """
    Factory — construct the training loss function based on config.
    """
    if pos_weight is None:
        pos_weight = 1.0
    if config.loss_fn == "focal":
        criterion = ClinicalFocalLoss(
            pos_weight=pos_weight,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            num_classes=config.num_classes,
            reduction="mean",
        )
    elif config.loss_fn == "ce":
        weight_tensor = torch.tensor([1.0, pos_weight], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=config.label_smoothing,
        )
    elif config.loss_fn == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss '{config.loss_fn}'. Choose from: 'focal', 'ce', 'bce'.")
    return criterion
