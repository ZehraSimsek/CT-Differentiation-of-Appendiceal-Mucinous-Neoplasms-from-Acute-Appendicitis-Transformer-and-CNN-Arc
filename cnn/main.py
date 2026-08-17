"""
main.py — Multi-Run 3D CT Classification Pipeline
===================================================
Orchestrates the complete Q1 medical imaging multi-run validation protocol:
- N_RUNS = 3 independent runs with fixed seeds [42, 123, 456]
- Fixed 5-Fold Stratified Cross-Validation (CSV-defined)
- Gradient Accumulation (accum_steps=8 with batch_size=2 -> effective batch=16)
- Checkpoint persistence and training history per fold
- Out-of-Fold (OOF) aggregation and metric computation
- External Test set evaluation via 5-fold model ensembling
- Multi-Run statistical reporting with mean ± std across all runs

Complies with MULTI_RUN_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import PipelineConfig
from data.patient_dataset import PatientCTDataset
from engine.evaluator import evaluate
from engine.trainer import train_one_epoch
from losses import build_loss_fn
from models import build_model
from utils.checkpoint import CheckpointManager
from utils.metrics import MetricResult, compute_metrics
from utils.seed import seed_everything
from utils.visualization import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 1e-3,
) -> LambdaLR:
    """Creates a learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def parse_args() -> PipelineConfig:
    """Parse command line arguments and return PipelineConfig."""
    parser = argparse.ArgumentParser(
        description="Multi-Run 3D CT Classification Pipeline (MULTI_RUN_PROTOCOL.md)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet_plusplus",
        choices=["unet_plusplus", "densenet121", "efficientnet_b0"],
        help="Architecture variant",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Maximum epochs per fold")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-step mini-batch size")
    parser.add_argument("--accum_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'mps', 'cpu', 'auto')")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--n_runs", type=int, default=None, help="Number of independent runs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of seeds for independent runs (e.g. 42 123 456)",
    )

    args = parser.parse_args()
    config = PipelineConfig()

    config.apply_model_profile(args.model)

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.accum_steps is not None:
        config.accum_steps = args.accum_steps
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.patience is not None:
        config.early_stopping_patience = args.patience
    if args.device is not None:
        config.device = args.device
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.n_runs is not None:
        config.n_runs = args.n_runs
    if args.seeds is not None:
        config.run_seeds = list(args.seeds)
        config.n_runs = len(args.seeds)

    return config


def verify_data_files(config: PipelineConfig) -> None:
    """Verifies that all required fixed fold and external test CSV files exist."""
    print("=" * 75)
    print("  VERIFYING FIXED DATASET SPLITS (MULTI_RUN_PROTOCOL.md)")
    print("=" * 75)
    missing = []
    for fold in range(1, config.n_folds + 1):
        tr_csv = os.path.join(config.data_dir, f"fold_{fold}_train.csv")
        val_csv = os.path.join(config.data_dir, f"fold_{fold}_val.csv")
        if not os.path.isfile(tr_csv):
            missing.append(tr_csv)
        if not os.path.isfile(val_csv):
            missing.append(val_csv)

    ext_csv = os.path.join(config.data_dir, "external_test_set.csv")
    if not os.path.isfile(ext_csv):
        missing.append(ext_csv)

    if missing:
        print("  ❌ Missing required dataset files:")
        for m in missing:
            print(f"     - {m}")
        sys.exit(1)

    print("  ✓ All 5-Fold CSVs and External Test Set CSV verified successfully.")
    print("=" * 75 + "\n")


def train_single_fold(
    fold: int,
    run_idx: int,
    seed: int,
    config: PipelineConfig,
    device: torch.device,
    fold_dir: str,
) -> tuple[MetricResult, pd.DataFrame]:
    """Trains a single fold model and returns its best evaluation metrics and prediction df."""
    print("\n" + "-" * 75)
    print(f"  RUN {run_idx:02d}/{config.n_runs:02d} (Seed {seed})  |  FOLD {fold:02d}/{config.n_folds:02d}")
    print("-" * 75)

    tr_csv = os.path.join(config.data_dir, f"fold_{fold}_train.csv")
    val_csv = os.path.join(config.data_dir, f"fold_{fold}_val.csv")

    train_df = pd.read_csv(tr_csv)
    val_df = pd.read_csv(val_csv)

    # Class balance weights
    n_neg = (train_df["label"] == 0).sum()
    n_pos = (train_df["label"] == 1).sum()
    class_weights = None
    if n_pos > 0 and n_neg > 0:
        total_samples = len(train_df)
        w0 = total_samples / (2.0 * n_neg)
        w1 = total_samples / (2.0 * n_pos)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        print(f"  Class balance: Apandisit={n_neg}, Musinoz={n_pos} (Weights: [{w0:.2f}, {w1:.2f}])")

    train_dataset = PatientCTDataset(train_df, augment_train=config.augment_train, config=config)
    val_dataset = PatientCTDataset(val_df, augment_train=False, config=config)

    # Deterministic loader generator
    g = torch.Generator()
    g.manual_seed(seed + fold)

    # Drop single remainder sample in training to protect BatchNorm layers (e.g. batch_size=1)
    drop_last_train = (len(train_dataset) % config.batch_size == 1) if config.batch_size > 1 else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory if device.type == "cuda" else False,
        generator=g,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory if device.type == "cuda" else False,
    )

    # Build components
    model = build_model(config).to(device)
    criterion = build_loss_fn(config, class_weights=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
    )

    ckpt_manager = CheckpointManager(
        save_dir=fold_dir,
        config=config,
        metric_name=config.checkpoint_metric,
        min_epochs_save=config.min_epochs_save,
        filename="best_model.pth",
    )

    history_records = []
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
        )

        val_loss, val_metrics, y_true, y_prob, y_pred, pids = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            verbose_metrics=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_metrics.auc_roc,
            "val_sensitivity": val_metrics.sensitivity,
            "val_specificity": val_metrics.specificity,
            "val_f1": val_metrics.f1,
            "val_accuracy": val_metrics.accuracy,
            "val_brier": val_metrics.brier_score,
            "learning_rate": current_lr,
        })

        improved = ckpt_manager.step(model, val_metrics, epoch)
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience and epoch >= config.min_epochs_save:
            print(f"  🛑 Early stopping triggered at epoch {epoch} (patience={config.early_stopping_patience})")
            break

    # Save training history CSV
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(os.path.join(fold_dir, "training_history.csv"), index=False)

    # Load best model for final evaluation
    model = ckpt_manager.load_best(model)

    print(f"\n  Final Best Checkpoint Evaluation for Fold {fold}:")
    val_loss, best_metrics, y_true, y_prob, y_pred, pids = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        epoch=ckpt_manager.best_epoch,
        verbose_metrics=True,
    )

    # Save best validation predictions
    val_pred_df = pd.DataFrame({
        "patient_id": pids,
        "true_label": y_true,
        "pred_prob": y_prob,
        "pred_label": y_pred,
        "fold": fold,
    })
    val_pred_df.to_csv(os.path.join(fold_dir, "best_val_predictions.csv"), index=False)

    # Save fold performance plots
    plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=os.path.join(fold_dir, "confusion_matrix.png"),
        title=f"Run {run_idx} Fold {fold} Confusion Matrix",
    )
    plot_roc_curve(
        y_true,
        y_prob,
        auc_score=best_metrics.auc_roc,
        save_path=os.path.join(fold_dir, "roc_curve.png"),
        title=f"Run {run_idx} Fold {fold} ROC Curve",
    )
    plot_pr_curve(
        y_true,
        y_prob,
        save_path=os.path.join(fold_dir, "pr_curve.png"),
        title=f"Run {run_idx} Fold {fold} PR Curve",
    )

    return best_metrics, val_pred_df


def evaluate_external_test_ensemble(
    run_idx: int,
    run_dir: str,
    config: PipelineConfig,
    device: torch.device,
) -> tuple[MetricResult, pd.DataFrame]:
    """Loads all 5 fold models for this run, computes ensemble predictions on external test set."""
    ext_csv = os.path.join(config.data_dir, "external_test_set.csv")
    ext_df = pd.read_csv(ext_csv)

    ext_dataset = PatientCTDataset(ext_df, augment_train=False, config=config)
    ext_loader = DataLoader(
        ext_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    criterion = build_loss_fn(config)
    fold_probs = []

    for fold in range(1, config.n_folds + 1):
        fold_dir = os.path.join(run_dir, f"fold_{fold:02d}")
        ckpt_path = os.path.join(fold_dir, "best_model.pth")

        model = build_model(config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

        _, _, y_true, y_prob, _, pids = evaluate(
            model=model,
            dataloader=ext_loader,
            criterion=criterion,
            device=device,
            verbose_metrics=False,
        )
        fold_probs.append(y_prob)

    all_fold_probs = np.array(fold_probs)  # [5, N]
    ensemble_prob = np.mean(all_fold_probs, axis=0)  # [N]
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    test_metrics = compute_metrics(y_true, ensemble_prob, y_pred=ensemble_pred)

    ext_dir = os.path.join(run_dir, "external_test")
    os.makedirs(ext_dir, exist_ok=True)

    # Save predictions
    pred_dict = {
        "patient_id": pids,
        "true_label": y_true,
    }
    for f_idx in range(config.n_folds):
        pred_dict[f"fold_{f_idx + 1:02d}_prob"] = all_fold_probs[f_idx]
    pred_dict["ensemble_prob"] = ensemble_prob
    pred_dict["ensemble_pred"] = ensemble_pred

    pred_df = pd.DataFrame(pred_dict)
    pred_df.to_csv(os.path.join(ext_dir, "external_test_predictions.csv"), index=False)

    # Save metrics
    metrics_df = pd.DataFrame([test_metrics.to_dict()])
    metrics_df.to_csv(os.path.join(ext_dir, "external_test_metrics.csv"), index=False)

    # Save plots
    plot_confusion_matrix(
        y_true,
        ensemble_pred,
        save_path=os.path.join(ext_dir, "confusion_matrix.png"),
        title=f"Run {run_idx} External Test Confusion Matrix (5-Fold Ensemble)",
    )
    plot_roc_curve(
        y_true,
        ensemble_prob,
        auc_score=test_metrics.auc_roc,
        save_path=os.path.join(ext_dir, "roc_curve.png"),
        title=f"Run {run_idx} External Test ROC (5-Fold Ensemble)",
    )
    plot_pr_curve(
        y_true,
        ensemble_prob,
        save_path=os.path.join(ext_dir, "pr_curve.png"),
        title=f"Run {run_idx} External Test PR (5-Fold Ensemble)",
    )

    return test_metrics, pred_df


def main() -> None:
    """Main Multi-Run execution entrypoint."""
    config = parse_args()
    device_str = config.resolve_device()
    device = torch.device(device_str)

    model_exp_dir = config.get_model_experiment_dir()
    os.makedirs(model_exp_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("  3D CT CLASSIFICATION — MULTI-RUN PROTOCOL (MULTI_RUN_PROTOCOL.md)")
    print("=" * 75)
    print(f"  Model Architecture      : {config.model_name}")
    print(f"  Independent Runs (N)    : {config.n_runs} (Seeds: {config.run_seeds})")
    print(f"  Cross-Validation Folds  : {config.n_folds} (Fixed Stratified CSVs)")
    print(f"  Max Epochs / Patience   : {config.epochs} / {config.early_stopping_patience}")
    print(f"  Batch Size / Accum      : {config.batch_size} * {config.accum_steps} = {config.batch_size * config.accum_steps} (Effective)")
    print(f"  Loss Function           : {config.loss_fn} (gamma={config.focal_gamma}, smoothing={config.label_smoothing})")
    print(f"  Compute Device          : {device_str.upper()}")
    print(f"  Experiment Directory    : {model_exp_dir}")
    print("=" * 75 + "\n")

    verify_data_files(config)

    run_records = []

    for run_idx, seed in enumerate(config.run_seeds, start=1):
        print("\n" + "#" * 75)
        print(f"  STARTING RUN {run_idx:02d} / {config.n_runs:02d}  |  SEED = {seed}")
        print("#" * 75)

        seed_everything(seed)
        run_dir = config.get_run_dir(run_idx)
        os.makedirs(run_dir, exist_ok=True)

        fold_preds = []
        for fold in range(1, config.n_folds + 1):
            fold_dir = os.path.join(run_dir, f"fold_{fold:02d}")
            _, val_pred_df = train_single_fold(
                fold=fold,
                run_idx=run_idx,
                seed=seed,
                config=config,
                device=device,
                fold_dir=fold_dir,
            )
            fold_preds.append(val_pred_df)

        # ---------------------------------------------------------------------
        # 1. Out-of-Fold (OOF) Aggregation
        # ---------------------------------------------------------------------
        oof_dir = os.path.join(run_dir, "aggregate_oof")
        os.makedirs(oof_dir, exist_ok=True)

        oof_df = pd.concat(fold_preds, ignore_index=True)
        oof_df.to_csv(os.path.join(oof_dir, "oof_predictions.csv"), index=False)

        oof_metrics = compute_metrics(
            oof_df["true_label"].values,
            oof_df["pred_prob"].values,
            y_pred=oof_df["pred_label"].values,
        )

        oof_metrics_df = pd.DataFrame([oof_metrics.to_dict()])
        oof_metrics_df.to_csv(os.path.join(oof_dir, "oof_metrics.csv"), index=False)

        plot_confusion_matrix(
            oof_df["true_label"].values,
            oof_df["pred_label"].values,
            save_path=os.path.join(oof_dir, "confusion_matrix.png"),
            title=f"Run {run_idx} Out-Of-Fold Confusion Matrix",
        )
        plot_roc_curve(
            oof_df["true_label"].values,
            oof_df["pred_prob"].values,
            auc_score=oof_metrics.auc_roc,
            save_path=os.path.join(oof_dir, "roc_curve.png"),
            title=f"Run {run_idx} Out-Of-Fold ROC Curve",
        )
        plot_pr_curve(
            oof_df["true_label"].values,
            oof_df["pred_prob"].values,
            save_path=os.path.join(oof_dir, "pr_curve.png"),
            title=f"Run {run_idx} Out-Of-Fold PR Curve",
        )

        print("\n" + "=" * 70)
        print(f"  RUN {run_idx:02d} (Seed {seed}) -- OOF Validation Summary")
        print("=" * 70)
        oof_metrics.pretty_print()

        # ---------------------------------------------------------------------
        # 2. External Test Evaluation (5-Fold Ensemble)
        # ---------------------------------------------------------------------
        ext_metrics, _ = evaluate_external_test_ensemble(
            run_idx=run_idx,
            run_dir=run_dir,
            config=config,
            device=device,
        )

        print("\n" + "=" * 70)
        print(f"  RUN {run_idx:02d} (Seed {seed}) -- External Test 5-Fold Ensemble Summary")
        print("=" * 70)
        ext_metrics.pretty_print()

        # Record run row
        run_records.append({
            "run_idx": run_idx,
            "seed": seed,
            # OOF Metrics
            "oof_auc": oof_metrics.auc_roc,
            "oof_sensitivity": oof_metrics.sensitivity,
            "oof_specificity": oof_metrics.specificity,
            "oof_f1": oof_metrics.f1,
            "oof_accuracy": oof_metrics.accuracy,
            "oof_brier": oof_metrics.brier_score,
            # External Test Metrics
            "ext_auc": ext_metrics.auc_roc,
            "ext_sensitivity": ext_metrics.sensitivity,
            "ext_specificity": ext_metrics.specificity,
            "ext_f1": ext_metrics.f1,
            "ext_accuracy": ext_metrics.accuracy,
            "ext_brier": ext_metrics.brier_score,
        })

    # =========================================================================
    # MULTI-RUN STATISTICAL SUMMARY & REPORTING (mean ± std)
    # =========================================================================
    summary_df = pd.DataFrame(run_records)

    oof_auc = summary_df["oof_auc"].values
    oof_sens = summary_df["oof_sensitivity"].values
    oof_spec = summary_df["oof_specificity"].values
    oof_f1 = summary_df["oof_f1"].values
    oof_acc = summary_df["oof_accuracy"].values
    oof_brier = summary_df["oof_brier"].values

    ext_auc = summary_df["ext_auc"].values
    ext_sens = summary_df["ext_sensitivity"].values
    ext_spec = summary_df["ext_specificity"].values
    ext_f1 = summary_df["ext_f1"].values
    ext_acc = summary_df["ext_accuracy"].values
    ext_brier = summary_df["ext_brier"].values

    summary_csv_path = os.path.join(model_exp_dir, "multi_run_summary.csv")
    summary_xlsx_path = os.path.join(model_exp_dir, "multi_run_summary.xlsx")
    log_txt_path = os.path.join(model_exp_dir, "train_log.txt")

    summary_df.to_csv(summary_csv_path, index=False)
    try:
        summary_df.to_excel(summary_xlsx_path, index=False)
    except Exception:
        pass

    # Build final report string
    report_lines = [
        "",
        "=" * 75,
        f"  FINAL MULTI-RUN STATISTICAL REPORT — {config.model_name.upper()}",
        f"  (n={config.n_runs} independent runs × {config.n_folds}-Fold Stratified CV, seeds={config.run_seeds})",
        "=" * 75,
        "  1. VALIDATION OUT-OF-FOLD (OOF) PERFORMANCE:",
        f"     AUC-ROC      = {oof_auc.mean():.4f} ± {oof_auc.std():.4f} (Individual: {[round(x, 4) for x in oof_auc]})",
        f"     Sensitivity  = {oof_sens.mean():.4f} ± {oof_sens.std():.4f}",
        f"     Specificity  = {oof_spec.mean():.4f} ± {oof_spec.std():.4f}",
        f"     F1-Score     = {oof_f1.mean():.4f} ± {oof_f1.std():.4f}",
        f"     Accuracy     = {oof_acc.mean():.4f} ± {oof_acc.std():.4f}",
        f"     Brier Score  = {oof_brier.mean():.4f} ± {oof_brier.std():.4f}",
        "-" * 75,
        "  2. EXTERNAL TEST SET (5-FOLD ENSEMBLE) PERFORMANCE:",
        f"     AUC-ROC      = {ext_auc.mean():.4f} ± {ext_auc.std():.4f} (Individual: {[round(x, 4) for x in ext_auc]})",
        f"     Sensitivity  = {ext_sens.mean():.4f} ± {ext_sens.std():.4f}",
        f"     Specificity  = {ext_spec.mean():.4f} ± {ext_spec.std():.4f}",
        f"     F1-Score     = {ext_f1.mean():.4f} ± {ext_f1.std():.4f}",
        f"     Accuracy     = {ext_acc.mean():.4f} ± {ext_acc.std():.4f}",
        f"     Brier Score  = {ext_brier.mean():.4f} ± {ext_brier.std():.4f}",
        "=" * 75,
        f"  Results saved to: {summary_csv_path}",
        "=" * 75,
        "",
    ]

    report_str = "\n".join(report_lines)
    print(report_str)

    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write(report_str)


if __name__ == "__main__":
    main()
