"""
utils/visualization.py — Clinical Reporting Visualizations
===========================================================
Generates publication-quality performance plots (ROC, PR, Confusion Matrix)
using matplotlib and seaborn.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def set_plot_style() -> None:
    """Configures global matplotlib and seaborn styles for academic plotting."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.edgecolor"] = "#333333"


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
    threshold: float | None = None,
) -> None:
    """
    Generates and saves a styled Confusion Matrix with clinical class names.

    Parameters
    ----------
    threshold : float | None
        If provided, appended to the title as "(Thresh=X.XXX)".
    """
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Append threshold info to title
    full_title = title
    if threshold is not None:
        full_title = f"{title} (Thresh={threshold:.3f})"

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
    )
    
    ax.set_title(full_title, pad=15, weight="bold")
    ax.set_xlabel("Predicted Label", weight="bold", labelpad=10)
    ax.set_ylabel("True Label", weight="bold", labelpad=10)
    ax.set_xticklabels(["Appendisit", "Musinoz"])
    ax.set_yticklabels(["Appendisit", "Musinoz"])

    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_score: float,
    save_path: str,
    title: str = "Receiver Operating Characteristic",
) -> None:
    """
    Generates and saves a styled ROC Curve.
    """
    set_plot_style()
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except ValueError:
        # Handle edge cases where there is only one class
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#d32f2f", lw=2.5, label=f"AUC = {auc_score * 100:.2f}%")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", weight="bold", labelpad=10)
    plt.ylabel("True Positive Rate (Sensitivity)", weight="bold", labelpad=10)
    plt.title(title, pad=15, weight="bold")
    plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    title: str = "Precision-Recall Curve",
) -> None:
    """
    Generates and saves a styled Precision-Recall Curve.
    """
    set_plot_style()
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
    except ValueError:
        precision, recall, pr_auc = np.array([0, 1]), np.array([0, 1]), 0.0

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color="#1976d2", lw=2.5, label=f"PR-AUC = {pr_auc * 100:.2f}%")
    
    # Calculate baseline
    baseline = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0.5
    plt.plot([0, 1], [baseline, baseline], color="gray", lw=1.5, linestyle="--", label=f"Baseline = {baseline * 100:.1f}%")

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Recall (Sensitivity)", weight="bold", labelpad=10)
    plt.ylabel("Precision", weight="bold", labelpad=10)
    plt.title(title, pad=15, weight="bold")
    plt.legend(loc="lower left", frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
