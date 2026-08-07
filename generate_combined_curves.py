"""
generate_combined_curves.py
===========================
Q1 Makale kalitesinde 7 model ROC + PR eğrisi.
Nature Medicine / Radiology dergi formatı.
Her run ayrı ince kesik çizgi, mean kalın düz çizgi (CNN grubunuzla aynı stil).
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
    "font.size":         11,
    "axes.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "figure.dpi":        150,
    "savefig.dpi":       300,
})

BASE    = Path("experiments_multirun")
CNN_DIR = BASE / "classic_cnns"
OUT_DIR = BASE

# ──────────────────────────────────────────────────────────────────────────────
# RENK PALETİ
# Transformer → Her biri tamamen farklı, koyu + okunaklı
# CNN → Sizin mevcut plot renkleriyle birebir uyumlu (mavi/yeşil/kırmızı)
# ──────────────────────────────────────────────────────────────────────────────
TRANSFORMERS = {
    "swinunetr_lp":     ("SwinUNETR-LP",   "#1F3A93"),  # Lacivert
    "ag_msf":           ("AG-MSF",          "#8E44AD"),  # Mor
    "mae_tiny3d":       ("MAE-Tiny3D",      "#117A65"),  # Koyu teal/yeşil
    "segformer3d_msca": ("SegFormer3D",     "#D35400"),  # Koyu turuncu
}

CNN_MODELS = {
    "UNet++":          ("UNet++",          "#2471A3"),  # Mavi (CNN plotunuzdaki gibi)
    "DenseNet121":     ("DenseNet121",     "#196F3D"),  # Koyu yeşil
    "EfficientNet-B0": ("EfficientNet-B0", "#CB181D"),  # Koyu kırmızı
}


# ──────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────────────────────
def load_transformer_runs_roc(folder):
    """Her run için ayrı ROC eğrisi + mean ROC döndür."""
    common_fpr = np.linspace(0, 1, 300)
    run_tprs = []
    y_true_all = None
    probs_all = []

    for run_idx in [1, 2, 3]:
        p = BASE / folder / f"run_{run_idx:02d}" / "external_test" / "ensemble_probs.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).sort_values("patient_id").reset_index(drop=True)
        y = df["label"].values
        prob = df["prob_mucinous"].values
        if y_true_all is None:
            y_true_all = y
        fpr, tpr, _ = roc_curve(y, prob)
        run_tprs.append(np.interp(common_fpr, fpr, tpr))
        probs_all.append(prob)

    if not run_tprs:
        return None, None, None, None, None

    mean_tpr = np.mean(run_tprs, axis=0)
    mean_auc = auc(common_fpr, mean_tpr)
    mean_probs = np.mean(probs_all, axis=0)
    return common_fpr, run_tprs, mean_tpr, mean_auc, (y_true_all, mean_probs)


def load_transformer_runs_pr(folder):
    """Her run için ayrı PR eğrisi + mean PR döndür."""
    common_rec = np.linspace(0, 1, 300)
    run_precs = []
    y_true_all = None
    probs_all = []

    for run_idx in [1, 2, 3]:
        p = BASE / folder / f"run_{run_idx:02d}" / "external_test" / "ensemble_probs.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).sort_values("patient_id").reset_index(drop=True)
        y = df["label"].values
        prob = df["prob_mucinous"].values
        if y_true_all is None:
            y_true_all = y
        prec, rec, _ = precision_recall_curve(y, prob)
        run_precs.append(np.interp(common_rec, rec[::-1], prec[::-1]))
        probs_all.append(prob)

    if not run_precs:
        return None, None, None, None, None

    mean_prec = np.mean(run_precs, axis=0)
    mean_ap   = average_precision_score(y_true_all, np.mean(probs_all, axis=0))
    return common_rec, run_precs, mean_prec, mean_ap, y_true_all


def cnn_roc_runs(model_name):
    df = pd.read_csv(CNN_DIR / "roc_curves_classic_cnn.csv")
    df_m = df[df["Model"] == model_name]
    common_fpr = np.linspace(0, 1, 300)
    tprs = []
    for run, grp in df_m.groupby("Run"):
        grp = grp.sort_values("FPR")
        tprs.append(np.interp(common_fpr, grp["FPR"].values, grp["TPR"].values))
    mean_tpr = np.mean(tprs, axis=0)
    return common_fpr, tprs, mean_tpr, auc(common_fpr, mean_tpr)


def cnn_pr_runs(model_name):
    df = pd.read_csv(CNN_DIR / "pr_curves_classic_cnn.csv")
    df_m = df[df["Model"] == model_name]
    common_rec = np.linspace(0, 1, 300)
    precs = []
    for run, grp in df_m.groupby("Run"):
        grp = grp.sort_values("Recall")
        precs.append(np.interp(common_rec, grp["Recall"].values[::-1],
                               grp["Precision"].values[::-1]))
    mean_prec = np.mean(precs, axis=0)
    mean_ap   = float(np.trapezoid(mean_prec[::-1], common_rec[::-1]))
    return common_rec, precs, mean_prec, mean_ap


def style_ax(ax, xlim=(-0.02, 1.02), ylim=(-0.02, 1.05)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=":", linewidth=0.55, color="#CCCCCC", zorder=0)


# ──────────────────────────────────────────────────────────────────────────────
# ROC CURVE
# ──────────────────────────────────────────────────────────────────────────────
def make_roc():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    legend_handles = []

    # Transformer — her run ince kesik, mean kalın düz
    for folder, (name, color) in TRANSFORMERS.items():
        result = load_transformer_runs_roc(folder)
        if result[0] is None:
            print(f"  [SKIP] {name}")
            continue
        common_fpr, run_tprs, mean_tpr, mean_auc, _ = result
        for tpr in run_tprs:
            ax.plot(common_fpr, tpr, color=color, lw=0.9,
                    linestyle="--", alpha=0.35, zorder=2)
        line, = ax.plot(common_fpr, mean_tpr, color=color, lw=2.4,
                        linestyle="-", zorder=4)
        legend_handles.append((line, f"{name}  (AUC = {mean_auc:.3f})"))

    # CNN — her run ince kesik, mean kalın kesik (CNN grubunu ayrıştırmak için)
    for model_name, (display, color) in CNN_MODELS.items():
        try:
            common_fpr, run_tprs, mean_tpr, mean_auc = cnn_roc_runs(model_name)
            for tpr in run_tprs:
                ax.plot(common_fpr, tpr, color=color, lw=0.9,
                        linestyle="--", alpha=0.35, zorder=2)
            line, = ax.plot(common_fpr, mean_tpr, color=color, lw=2.4,
                            linestyle="--", zorder=4)
            legend_handles.append((line, f"{display}  (AUC = {mean_auc:.3f})"))
        except Exception as e:
            print(f"  [SKIP] {display}: {e}")

    # Şans çizgisi
    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.1,
            linestyle=(0, (4, 4)), zorder=1)

    style_ax(ax)
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=12, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, labelpad=8)
    ax.set_title(
        "External Validation — ROC Curves (n = 24, 3 Runs)\n"
        "Bold = 3-Run Mean  |  ─ Transformer Models  ╌ Classic CNN Models",
        fontsize=10.5, pad=12
    )

    tr_hdr  = Line2D([], [], linestyle="-",  color="#333333", lw=2.0)
    cnn_hdr = Line2D([], [], linestyle="--", color="#333333", lw=2.0)

    handles = [tr_hdr]  + [h for h, _ in legend_handles[:4]] + \
              [cnn_hdr] + [h for h, _ in legend_handles[4:]]
    labels  = ["Transformer Models"] + [l for _, l in legend_handles[:4]] + \
              ["Classic CNN Models"] + [l for _, l in legend_handles[4:]]

    ax.legend(handles, labels, loc="lower right", fontsize=9,
              frameon=True, framealpha=0.93, edgecolor="#CCCCCC",
              handlelength=2.2, handletextpad=0.8, labelspacing=0.45,
              borderpad=0.8)

    plt.tight_layout()
    out = OUT_DIR / "Combined_ROC_All7Models.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] ROC → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PR CURVE
# ──────────────────────────────────────────────────────────────────────────────
def make_pr():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    legend_handles = []

    for folder, (name, color) in TRANSFORMERS.items():
        result = load_transformer_runs_pr(folder)
        if result[0] is None:
            continue
        common_rec, run_precs, mean_prec, mean_ap, _ = result
        for prec in run_precs:
            ax.step(common_rec, prec, color=color, lw=0.9,
                    linestyle="--", alpha=0.35, where="post", zorder=2)
        line, = ax.step(common_rec, mean_prec, color=color, lw=2.4,
                        linestyle="-", where="post", zorder=4)
        legend_handles.append((line, f"{name}  (AP = {mean_ap:.3f})"))

    for model_name, (display, color) in CNN_MODELS.items():
        try:
            common_rec, run_precs, mean_prec, mean_ap = cnn_pr_runs(model_name)
            for prec in run_precs:
                ax.step(common_rec, prec, color=color, lw=0.9,
                        linestyle="--", alpha=0.35, where="post", zorder=2)
            line, = ax.step(common_rec, mean_prec, color=color, lw=2.4,
                            linestyle="--", where="post", zorder=4)
            legend_handles.append((line, f"{display}  (AP = {mean_ap:.3f})"))
        except Exception as e:
            print(f"  [SKIP] {display}: {e}")

    style_ax(ax, xlim=(-0.02, 1.02), ylim=(0.0, 1.05))
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12, labelpad=8)
    ax.set_ylabel("Precision (PPV)", fontsize=12, labelpad=8)
    ax.set_title(
        "External Validation — Precision–Recall Curves (n = 24, 3 Runs)\n"
        "Bold = 3-Run Mean  |  ─ Transformer Models  ╌ Classic CNN Models",
        fontsize=10.5, pad=12
    )

    tr_hdr  = Line2D([], [], linestyle="-",  color="#333333", lw=2.0)
    cnn_hdr = Line2D([], [], linestyle="--", color="#333333", lw=2.0)

    handles = [tr_hdr]  + [h for h, _ in legend_handles[:4]] + \
              [cnn_hdr] + [h for h, _ in legend_handles[4:]]
    labels  = ["Transformer Models"] + [l for _, l in legend_handles[:4]] + \
              ["Classic CNN Models"] + [l for _, l in legend_handles[4:]]

    ax.legend(handles, labels, loc="lower left", fontsize=9,
              frameon=True, framealpha=0.93, edgecolor="#CCCCCC",
              handlelength=2.2, handletextpad=0.8, labelspacing=0.45,
              borderpad=0.8)

    plt.tight_layout()
    out = OUT_DIR / "Combined_PR_All7Models.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] PR  → {out}")


if __name__ == "__main__":
    print("=== Q1 Combined Curves oluşturuluyor ===")
    make_roc()
    make_pr()
    print("\nTamamlandı!")
