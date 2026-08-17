"""
engine/evaluator.py — Validation / Test Evaluation Loop
========================================================
Implements ``evaluate`` — a complete inference pass over the validation
or test DataLoader in ``model.eval()`` mode with gradient calculation disabled.

Returns full prediction arrays and patient IDs for seamless saving of:
- best_val_predictions.csv
- oof_predictions.csv
- external_test_predictions.csv
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import MetricResult, compute_metrics
from utils.xai import GradCAM3D, overlay_heatmap_on_slice


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int | None = None,
    generate_xai: bool = False,
    xai_save_dir: str | None = None,
    desc_prefix: str = "Val",
    verbose_metrics: bool = True,
) -> tuple[float, MetricResult, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Run inference on the validation / test set and compute clinical metrics.

    Parameters
    ----------
    model : nn.Module
        The classification model.
    dataloader : DataLoader
        Evaluation DataLoader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    epoch : int | None
        Current epoch index (for display only).
    generate_xai : bool
        Whether to generate 3D Grad-CAM heatmaps.
    xai_save_dir : str | None
        Directory to save Grad-CAM visualizations.
    desc_prefix : str
        Prefix for tqdm bar (e.g. 'Val' or 'Test').
    verbose_metrics : bool
        Whether to pretty-print the metrics table.

    Returns
    -------
    val_loss : float
    metrics : MetricResult
    y_true : ndarray
    y_prob : ndarray (positive class probability)
    y_pred : ndarray (predicted class 0 or 1)
    patient_ids : list[str]
    """
    model.eval()

    running_loss = 0.0
    num_batches = 0

    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_pids: list[str] = []

    desc = f"  {desc_prefix}   Epoch {epoch:>3d}" if epoch is not None else f"  {desc_prefix}"
    progress = tqdm(
        dataloader,
        desc=desc,
        unit="batch",
        leave=True,
        ncols=100,
    )

    cam_generator = None
    if generate_xai:
        cam_generator = GradCAM3D(model)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        batch_pids = batch.get("patient_id", [f"patient_{i}" for i in range(images.size(0))])
        for p in batch_pids:
            all_pids.append(str(p.item() if isinstance(p, torch.Tensor) else p))

        with torch.set_grad_enabled(generate_xai):
            logits = model(images)  # [B, 2]
            loss = criterion(logits, labels)

            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            preds = torch.argmax(logits.detach(), dim=1)

            if generate_xai and xai_save_dir is not None:
                for i in range(images.size(0)):
                    single_img = images[i:i + 1]
                    true_label = labels[i].item()
                    label_name = "Musinoz" if true_label == 1 else "Apandisit"

                    cam = cam_generator.generate_cam(single_img, target_class=1)
                    if cam is not None:
                        orig_vol = single_img.squeeze().detach().cpu().numpy()
                        orig_vol = (orig_vol - orig_vol.min()) / (orig_vol.max() - orig_vol.min() + 1e-8)
                        pid = str(batch_pids[i].item() if isinstance(batch_pids[i], torch.Tensor) else batch_pids[i])

                        save_path = os.path.join(xai_save_dir, label_name, f"{pid}_cam.png")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        overlay_heatmap_on_slice(
                            original_volume=orig_vol,
                            heatmap_volume=cam,
                            save_path=save_path,
                        )

        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

        running_loss += loss.item()
        num_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}")

    if cam_generator is not None:
        cam_generator.remove_hooks()

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)

    val_loss = running_loss / max(num_batches, 1)

    metrics = compute_metrics(y_true, y_prob, y_pred=y_pred)
    if verbose_metrics:
        metrics.pretty_print(epoch=epoch)

    return val_loss, metrics, y_true, y_prob, y_pred, all_pids
