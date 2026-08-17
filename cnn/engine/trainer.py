"""
engine/trainer.py — Single-Epoch Training Loop with Gradient Accumulation
==========================================================================
Implements ``train_one_epoch`` with:
- Gradient Accumulation (accum_steps)
- Gradient Clipping (max_grad_norm)
- Progress reporting with tqdm
- Clinical loss calculation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PipelineConfig


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    config: PipelineConfig,
) -> float:
    """
    Execute one full training epoch with gradient accumulation.

    Parameters
    ----------
    model : nn.Module
        The classification model.
    dataloader : DataLoader
        Training-split DataLoader.
    criterion : nn.Module
        Loss function (ClinicalFocalLoss / CrossEntropyLoss).
    optimizer : Optimizer
        AdamW optimiser.
    device : torch.device
        Target compute device.
    epoch : int
        Current epoch index (1-indexed).
    config : PipelineConfig
        Global configuration.

    Returns
    -------
    float
        Mean unscaled training loss over the epoch.
    """
    model.train()

    running_loss = 0.0
    num_batches = len(dataloader)
    accum_steps = max(1, getattr(config, "accum_steps", 1))

    progress = tqdm(
        dataloader,
        desc=f"  Train Epoch {epoch:>3d}",
        unit="batch",
        leave=True,
        ncols=100,
    )

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)  # [B, 2]
        loss = criterion(logits, labels)

        # Gradient accumulation scaling
        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        # Step optimizer every accum_steps or at the end of epoch
        is_accum_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)
        if is_accum_step:
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()

        current_lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{current_lr:.2e}",
        )

    epoch_loss = running_loss / max(num_batches, 1)
    return epoch_loss
