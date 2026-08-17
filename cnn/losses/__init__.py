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


def build_loss_fn(config: PipelineConfig, class_weights: torch.Tensor | None = None) -> nn.Module:
    """
    Factory — construct the training loss function based on config.

    Parameters
    ----------
    config : PipelineConfig
        Supplies ``loss_fn``, ``focal_gamma``, and ``label_smoothing``.
    class_weights : Tensor | None
        Optional class balancing weights.

    Returns
    -------
    nn.Module
        Loss module accepting (logits, targets).
    """
    if config.loss_fn == "focal":
        criterion = ClinicalFocalLoss(
            weight=class_weights,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            reduction="mean",
        )
    elif config.loss_fn == "ce":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing,
        )
    elif config.loss_fn == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss '{config.loss_fn}'. Choose from: 'focal', 'ce', 'bce'.")

    return criterion
