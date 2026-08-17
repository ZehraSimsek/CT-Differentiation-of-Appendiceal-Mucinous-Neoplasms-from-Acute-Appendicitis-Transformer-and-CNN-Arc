"""
utils/checkpoint.py — Model Checkpointing Manager
===================================================
Saves and loads ``.pth`` model state dictionaries using a dual strategy:
1. Best-metric save — persists the model whenever the monitored validation metric improves
   (respecting ``min_epochs_save``).
2. Periodic interval save — optionally saves a snapshot every N epochs.
"""

from __future__ import annotations

import os
from pathlib import Path
import torch
import torch.nn as nn

from config import PipelineConfig
from utils.metrics import MetricResult


class CheckpointManager:
    """
    Tracks validation metrics and persists the model on improvement.

    Parameters
    ----------
    save_dir : str | Path
        Target directory to save checkpoints (e.g., experiments_q1_128/model/run_01/fold_01).
    config : PipelineConfig | None
        Optional pipeline configuration.
    metric_name : str, default "auc_roc"
        Metric to track ("auc_roc", "f1", "sensitivity").
    min_epochs_save : int, default 3
        Minimum epoch count before saving best checkpoints.
    filename : str, default "best_model.pth"
        Name of the best checkpoint file.
    """

    def __init__(
        self,
        save_dir: str | Path,
        config: PipelineConfig | None = None,
        metric_name: str = "auc_roc",
        min_epochs_save: int = 3,
        filename: str = "best_model.pth",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if config is not None:
            self.metric_name = getattr(config, "checkpoint_metric", metric_name)
            self.min_epochs_save = getattr(config, "min_epochs_save", min_epochs_save)
            self.interval = getattr(config, "checkpoint_interval", 50)
        else:
            self.metric_name = metric_name
            self.min_epochs_save = min_epochs_save
            self.interval = 50

        self.best_value: float = -float("inf")
        self.best_epoch: int = -1
        self.save_path = self.save_dir / filename

    def step(
        self,
        model: nn.Module,
        metrics: MetricResult,
        epoch: int,
    ) -> bool:
        """
        Evaluate current metrics and save if improved.

        Returns True if a new best checkpoint was saved.
        """
        current_value = getattr(metrics, self.metric_name, 0.0)

        # Only save if we passed min_epochs_save and metric improved
        if epoch >= self.min_epochs_save and current_value > self.best_value:
            self.best_value = float(current_value)
            self.best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": self.best_epoch,
                    "best_value": self.best_value,
                    "metric_name": self.metric_name,
                },
                self.save_path,
            )

            print(
                f"  ✓ Best checkpoint   |  {self.metric_name} improved to "
                f"{current_value:.4f}  →  {self.save_path.name}"
            )
            return True

        if current_value > self.best_value and epoch < self.min_epochs_save:
            print(
                f"  ℹ Epoch {epoch} < min_epochs_save ({self.min_epochs_save}), "
                f"skipping save ({self.metric_name} = {current_value:.4f})"
            )
        else:
            print(
                f"  ✗ No improvement    |  {self.metric_name} = {current_value:.4f} "
                f"(best = {self.best_value:.4f} @ epoch {self.best_epoch})"
            )
        return False

    def load_best(self, model: nn.Module) -> nn.Module:
        """Restore the best saved weights into model."""
        if not self.save_path.is_file():
            raise FileNotFoundError(
                f"No checkpoint found at '{self.save_path}'. Run training first."
            )

        checkpoint = torch.load(self.save_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            self.best_epoch = checkpoint.get("best_epoch", -1)
            self.best_value = checkpoint.get("best_value", -1.0)
        else:
            model.load_state_dict(checkpoint)

        print(
            f"  ✓ Loaded best checkpoint from epoch {self.best_epoch} "
            f"({self.metric_name} = {self.best_value:.4f})"
        )
        return model
