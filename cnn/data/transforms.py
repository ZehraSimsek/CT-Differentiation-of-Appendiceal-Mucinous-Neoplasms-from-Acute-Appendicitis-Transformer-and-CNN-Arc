"""
data/transforms.py — Training & Validation Transform Pipelines
===============================================================
Provides deterministic, reproducible image preprocessing pipelines
tailored for CT-derived 3-D volumetric classification.

Pure Preprocessing Strategy
---------------------------
To ensure clinical consistency and prevent diagnostic bias from artificial
distortions, zero stochastic data augmentations (flips, rotations, crops, jitter)
are applied. Both training and validation partitions use identical pipelines.
"""

from __future__ import annotations

from torchvision import transforms

from config import PipelineConfig


def get_train_transforms(cfg: PipelineConfig) -> transforms.Compose:
    """
    Pure preprocessing pipeline applied to the training split.
    Strictly zero stochastic spatial or intensity augmentations.
    """
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])


def get_val_transforms(cfg: PipelineConfig) -> transforms.Compose:
    """
    Pure preprocessing pipeline applied to the validation / test splits.
    Identical to the training pipeline to maintain evaluation consistency.
    """
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
    ])
