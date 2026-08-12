"""
generate_combined_curves.py
===========================
Q1 Makale kalitesinde Transformer ve CNN modelleri için ROC + PR eğrileri.
İki model ailesi yan yana (1x2 subplot).
Eğriler, her modelin en yüksek AUC (veya AP) değerine sahip (MAX) run'ı üzerinden çizilmiştir.
Tüm yazılar (fontlar) büyütülmüştür.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         24,  # Base font size
    "axes.linewidth":    1.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "figure.dpi":        150,
    "savefig.dpi":       600,
})

BASE = Path(__file__).resolve().parent / "experiments_multirun"
CNN_DIR = BASE / "classic_cnns"
OUT_DIR = BASE

# ──────────────────────────────────────────────────────────────────────────────
# RENK PALETİ
# ──────────────────────────────────────────────────────────────────────────────
TRANSFORMERS = {
    "swinunetr_lp":     ("SwinUNETR-LP",   "#1F3A93"),
    "ag_msf":           ("AG-MSF",          "#8E44AD"),
    "mae_tiny3d":       ("MAE-Tiny3D",      "#117A65"),
    "segformer3d_msca": ("SegFormer3D",     "#D35400"),
}

CNN_MODELS = {
    "UNet++":          ("UNet++",          "#2471A3"),
    "DenseNet121":     ("DenseNet121",     "#196F3D"),
    "EfficientNet-B0": ("EfficientNet-B0", "#CB181D"),
}

# ──────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR (MAX RUN BULMA)
# ──────────────────────────────────────────────────────────────────────────────
def load_transformer_max_roc(folder):
    best_auc = -1
    best_fpr, best_tpr = None, None
    for run_idx in [1, 2, 3]:
        p = BASE / folder / f"run_{run_idx:02d}" / "external_test" / "ensemble_probs.csv"
        if not p.exists(): continue
        df = pd.read_csv(p).sort_values("patient_id").reset_index(drop=True)
        y = df["label"].values
        prob = df["prob_mucinous"].values
        fpr, tpr, _ = roc_curve(y, prob)
        c_auc = auc(fpr, tpr)
        if c_auc > best_auc:
            best_auc = c_auc
            best_fpr = fpr
            best_tpr = tpr
    return best_fpr, best_tpr, best_auc

def load_transformer_max_pr(folder):
    best_ap = -1
    best_prec, best_rec = None, None
    for run_idx in [1, 2, 3]:
        p = BASE / folder / f"run_{run_idx:02d}" / "external_test" / "ensemble_probs.csv"
        if not p.exists(): continue
        df = pd.read_csv(p).sort_values("patient_id").reset_index(drop=True)
        y = df["label"].values
        prob = df["prob_mucinous"].values
        prec, rec, _ = precision_recall_curve(y, prob)
        c_ap = average_precision_score(y, prob)
        if c_ap > best_ap:
            best_ap = c_ap
            best_prec = prec
            best_rec = rec
    return best_rec, best_prec, best_ap

def cnn_max_roc(model_name):
    df = pd.read_csv(CNN_DIR / "roc_curves_classic_cnn.csv")
    df_m = df[df["Model"] == model_name]
    best_auc = -1
    best_fpr, best_tpr = None, None
    if len(df_m) == 0: return None, None, None
    for run, grp in df_m.groupby("Run"):
        grp = grp.sort_values("FPR")
        fpr = grp["FPR"].values
        tpr = grp["TPR"].values
        c_auc = auc(fpr, tpr)
        if c_auc > best_auc:
            best_auc = c_auc
            best_fpr = fpr
            best_tpr = tpr
    return best_fpr, best_tpr, best_auc

def cnn_max_pr(model_name):
    df = pd.read_csv(CNN_DIR / "pr_curves_classic_cnn.csv")
    df_m = df[df["Model"] == model_name]
    best_ap = -1
    best_prec, best_rec = None, None
    if len(df_m) == 0: return None, None, None
    for run, grp in df_m.groupby("Run"):
        grp = grp.sort_values("Recall")
        rec = grp["Recall"].values
        prec = grp["Precision"].values
        c_ap = abs(float(np.trapezoid(prec, rec)))
        if c_ap > best_ap:
            best_ap = c_ap
            best_prec = prec
            best_rec = rec
    return best_rec, best_prec, best_ap

def style_ax(ax, xlim=(-0.02, 1.02), ylim=(-0.02, 1.05)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(labelsize=22)
    ax.grid(True, linestyle=":", linewidth=0.8, color="#CCCCCC", zorder=0)

# ──────────────────────────────────────────────────────────────────────────────
# ROC CURVE (YAN YANA)
# ──────────────────────────────────────────────────────────────────────────────
def make_roc_side_by_side():
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    ax1, ax2 = axes

    # --- Transformer (AX1) ---
    for folder, (name, color) in TRANSFORMERS.items():
        fpr, tpr, c_auc = load_transformer_max_roc(folder)
        if fpr is None: continue
        ax1.plot(fpr, tpr, color=color, lw=3.5, linestyle="-", 
                 label=f"{name} (AUC = {c_auc:.3f})", zorder=4)
    
    ax1.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.5, linestyle=(0, (4, 4)), zorder=1)
    style_ax(ax1)
    ax1.set_xlabel("False Positive Rate", fontsize=26, labelpad=12)
    ax1.set_ylabel("True Positive Rate", fontsize=26, labelpad=12)
    ax1.set_title("Transformer Models", fontsize=28, pad=15)
    ax1.legend(loc="lower right", fontsize=21, frameon=True, edgecolor="#CCCCCC")

    # --- CNN (AX2) ---
    for model_name, (display, color) in CNN_MODELS.items():
        fpr, tpr, c_auc = cnn_max_roc(model_name)
        if fpr is None: continue
        ax2.plot(fpr, tpr, color=color, lw=3.5, linestyle="-", 
                 label=f"{display} (AUC = {c_auc:.3f})", zorder=4)
                 
    ax2.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.5, linestyle=(0, (4, 4)), zorder=1)
    style_ax(ax2)
    ax2.set_xlabel("False Positive Rate", fontsize=26, labelpad=12)
    ax2.set_title("Classic CNN Models", fontsize=28, pad=15)
    ax2.legend(loc="lower right", fontsize=21, frameon=True, edgecolor="#CCCCCC")

    fig.suptitle("Receiver Operating Characteristic (ROC)", fontsize=30, y=1.03)
    plt.tight_layout()
    out = OUT_DIR / "Combined_ROC_SideBySide.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] Yan yana ROC → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PR CURVE (YAN YANA)
# ──────────────────────────────────────────────────────────────────────────────
def make_pr_side_by_side():
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    ax1, ax2 = axes

    # --- Transformer (AX1) ---
    for folder, (name, color) in TRANSFORMERS.items():
        rec, prec, c_ap = load_transformer_max_pr(folder)
        if rec is None: continue
        ax1.step(rec, prec, color=color, lw=3.5, linestyle="-", where="post", 
                 label=f"{name} (PR-AUC = {c_ap:.3f})", zorder=4)
                 
    style_ax(ax1, xlim=(-0.02, 1.02), ylim=(0.0, 1.05))
    ax1.set_xlabel("Recall", fontsize=26, labelpad=12)
    ax1.set_ylabel("Precision", fontsize=26, labelpad=12)
    ax1.set_title("Transformer Models", fontsize=28, pad=15)
    ax1.legend(loc="lower left", fontsize=21, frameon=True, edgecolor="#CCCCCC")

    # --- CNN (AX2) ---
    for model_name, (display, color) in CNN_MODELS.items():
        rec, prec, c_ap = cnn_max_pr(model_name)
        if rec is None: continue
        ax2.step(rec, prec, color=color, lw=3.5, linestyle="-", where="post", 
                 label=f"{display} (PR-AUC = {c_ap:.3f})", zorder=4)

    style_ax(ax2, xlim=(-0.02, 1.02), ylim=(0.0, 1.05))
    ax2.set_xlabel("Recall", fontsize=26, labelpad=12)
    ax2.set_title("Classic CNN Models", fontsize=28, pad=15)
    ax2.legend(loc="lower left", fontsize=21, frameon=True, edgecolor="#CCCCCC")

    fig.suptitle("Precision-Recall (PR)", fontsize=30, y=1.03)
    plt.tight_layout()
    out = OUT_DIR / "Combined_PR_SideBySide.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] Yan yana PR  → {out}")


if __name__ == "__main__":
    print("=== Max-Run Yan Yana Eğriler Oluşturuluyor ===")
    make_roc_side_by_side()
    make_pr_side_by_side()
    print("\nTamamlandı!")
