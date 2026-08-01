"""
MAE-Tiny Final (All-Data) external test sonuçlarını istenen sensitivity hedefine göre raporlar.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve

from shared_utils import compute_binary_metrics, compute_bootstrap_ci, print_full_metrics_table, plot_confusion_matrix

CSV = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/experiments_q1_128/mae_tinytransformer_final/external_test/ensemble_probs.csv")
OUT = CSV.parent


def threshold_for_sens(y_true, y_prob, target_sens=0.85):
    fpr, tpr, thrs = roc_curve(y_true, y_prob)
    valid = np.where(tpr >= target_sens)[0]
    if len(valid) == 0:
        return float(thrs[0]) if len(thrs) else 0.5
    return float(thrs[valid[0]])


def main():
    if not CSV.exists():
        print(f"Uyarı: MAE-Tiny Final modeli henüz eğitilmemiş (Dosya yok: {CSV})")
        return
    df = pd.read_csv(CSV)
    y_true = df["label"].values
    y_prob = df["prob_mucinous"].values

    rows = []
    for target in [0.80, 0.85]:
        thr = threshold_for_sens(y_true, y_prob, target)
        m, cm, _ = compute_binary_metrics(y_true, y_prob, threshold=thr)
        ci = compute_bootstrap_ci(y_true, y_prob, threshold=thr)
        print_full_metrics_table(m, ci, f"MAE-Tiny Final @SENS≥{target:.0%}", f"thr={thr:.3f}")
        plot_confusion_matrix(cm, f"MAE-Tiny Final SENS≥{target:.0%} thr={thr:.3f}",
                              save_path=OUT / f"cm_sens_ge_{int(target*100)}.png")
        rows.append({"target_sens": target, "threshold": thr, **m, **ci})

    pd.DataFrame(rows).to_csv(OUT / "report_by_sensitivity.csv", index=False)
    print(f"\nKaydedildi: {OUT / 'report_by_sensitivity.csv'}")


if __name__ == "__main__":
    main()
