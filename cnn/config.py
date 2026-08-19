"""
config.py — Centralised Pipeline Configuration
================================================
Single-source-of-truth dataclass for every tuneable hyper-parameter, path,
and behavioural flag used across the multi-run training pipeline.
Complies with MULTI_RUN_PROTOCOL.md ( Medical Image Classification Standards):
- N_RUNS = 3 independent runs with RUN_SEEDS = [42, 123, 456]
- Fixed 5-Fold Stratified Cross-Validation (CSV-defined)
- Standardised 3D volume shape: [C=1, D=32, H=128, W=128]
- Standardised 9-Step 3D Augmentation suite (Overfitting Regularization)
- Gradient accumulation (accum_steps=8 with batch_size=2 -> effective batch=16)
- Checkpoint directory: experiments__128/<model_name>/run_XX/fold_XX/
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Literal
@dataclass
class PipelineConfig:
    """Immutable configuration object passed to every pipeline component."""
    n_runs: int = 3
    run_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    n_folds: int = 5
    base_experiment_dir: str = "experiments__128"
    model_name: str = "unet_plusplus"
    epochs: int = 100
    early_stopping_patience: int = 25  
    min_epochs_save: int = 3           
    batch_size: int = 2
    accum_steps: int = 8               
    learning_rate: float = 1e-3        
    weight_decay: float = 5e-3         
    warmup_epochs: int = 5             
    dropout_rate: float = 0.4          
    max_grad_norm: float = 1.0         
    loss_fn: Literal["focal", "ce", "bce"] = "focal"
    focal_gamma: float = 1.0           
    label_smoothing: float = 0.05      
    expected_D: int = 32
    expected_H: int = 128
    expected_W: int = 128
    expected_C: int = 1
    augment_train: bool = True
    num_classes: int = 2
    pretrained: bool = True
    freeze_backbone: bool = False
    num_workers: int = 0               
    pin_memory: bool = True
    mucinous_v03_dir: str = "../datas_2d/Musinoz/musinoz_128"
    appendicitis_v03_dir: str = "../datas_2d/Appendisit/apandisit_128"
    data_dir: str = "../datas_2d"
    betas: tuple[float, float] = (0.9, 0.999)
    min_lr: float = 1e-6
    checkpoint_metric: Literal["auc_roc", "f1", "sensitivity"] = "auc_roc"
    checkpoint_interval: int = 50
    seed: int = 42
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    def apply_model_profile(self, model_name: str | None = None) -> None:
        """
        Seçilen modelin kapasitesine göre overfitting önleyici hiperparametre profilini uygular.
        """
        if model_name is not None:
            self.model_name = model_name
        m = self.model_name.lower()
        if m == "unet_plusplus":
            self.learning_rate = 1e-3
            self.weight_decay = 5e-3
            self.warmup_epochs = 5
            self.dropout_rate = 0.4
        elif m == "densenet121":
            self.learning_rate = 1e-4
            self.weight_decay = 5e-3
            self.warmup_epochs = 10
            self.dropout_rate = 0.4
        elif m == "efficientnet_b0":
            self.learning_rate = 1e-4
            self.weight_decay = 1e-2
            self.warmup_epochs = 10
            self.dropout_rate = 0.4
    def resolve_device(self) -> str:
        """Return the concrete device string based on hardware availability."""
        import torch
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    def get_run_dir(self, run_idx: int) -> str:
        """Returns the specific run directory for a given run index."""
        return os.path.join(self.base_experiment_dir, self.model_name, f"run_{run_idx:02d}")
    def get_model_experiment_dir(self) -> str:
        """Returns the root experiment directory for the active model."""
        return os.path.join(self.base_experiment_dir, self.model_name)
