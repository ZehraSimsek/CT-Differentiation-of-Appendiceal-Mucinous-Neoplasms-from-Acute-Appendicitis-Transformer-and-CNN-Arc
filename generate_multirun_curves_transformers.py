import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path

BASE = Path("experiments_multirun")

MODELS = {
    "swinunetr_lp":     ("SwinUNETR-LP", "blue"),
    "ag_msf":           ("AG-MSF", "green"),
    "mae_tiny3d":       ("MAE-Tiny3D", "orange"),
    "segformer3d_msca": ("SegFormer3D", "red")
}

def load_mean_probs(folder):
    all_probs = []
    ref_df = None
    for run_idx in [1, 2, 3]:
        p = BASE / folder / f"run_{run_idx:02d}" / "external_test" / "ensemble_probs.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).sort_values("patient_id").reset_index(drop=True)
        all_probs.append(df["prob_mucinous"].values)
        if ref_df is None:
            ref_df = df[["patient_id", "label"]]
    
    if not all_probs:
        return None, None
    mean_probs = np.mean(all_probs, axis=0)
    return ref_df["label"].values, mean_probs

def main():
    plt.style.use('default')
    
    # 1. ROC CURVE
    plt.figure(figsize=(8, 6))
    for folder, (name, color) in MODELS.items():
        y_true, probs = load_mean_probs(folder)
        if probs is not None:
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color, lw=2.5)
            
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Transformer Family: External Test ROC Curves (n=24, 3 Runs)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(BASE / 'Transformer_ROC_Curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PR CURVE
    plt.figure(figsize=(8, 6))
    for folder, (name, color) in MODELS.items():
        y_true, probs = load_mean_probs(folder)
        if probs is not None:
            precision, recall, _ = precision_recall_curve(y_true, probs)
            ap = average_precision_score(y_true, probs)
            # PR Curve is usually plotted with step
            plt.step(recall, precision, where='post', label=f"{name} (AP = {ap:.3f})", color=color, lw=2.5)
            
    plt.xlim([-0.02, 1.02])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Transformer Family: External Test PR Curves (n=24, 3 Runs)')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(BASE / 'Transformer_PR_Curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Transformer_ROC_Curves.png ve Transformer_PR_Curves.png oluşturuldu!")

if __name__ == '__main__':
    main()
