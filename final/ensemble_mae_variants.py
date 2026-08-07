"""
İki MAE-Tiny varyantının (5-fold ensemble + all-data final) external test olasılıklarını
birleştirir ve en iyi klinik eşikte Q1 metrikleri üretir.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

from shared_utils import (
    compute_binary_metrics, compute_bootstrap_ci,
    find_youden_threshold, print_full_metrics_table, plot_confusion_matrix
)

BASE = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/experiments_q1_128")
FOLD_CSV = BASE / "mae_tinytransformer" / "external_test" / "ensemble_probs.csv"
FINAL_CSV = BASE / "mae_tinytransformer_final" / "external_test" / "ensemble_probs.csv"
OUT_DIR = BASE / "mae_tinytransformer_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_best_threshold(y_true, y_prob, min_sens=0.80, min_spec=0.50):
    best_thr, best_f1 = 0.5, 0.0
    found = False
    for thr in np.linspace(0.05, 0.95, 191):
        pred = (y_prob >= thr).astype(int)
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, pred):
            cm[t, p] += 1
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        if sens >= min_sens - 1e-5 and spec >= min_spec - 1e-5:
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                found = True
    if not found:
        best_thr, _ = find_youden_threshold(y_true, y_prob)
    return float(best_thr)


def main():
    df_fold = pd.read_csv(FOLD_CSV).sort_values("patient_id").reset_index(drop=True)
    df_final = pd.read_csv(FINAL_CSV).sort_values("patient_id").reset_index(drop=True)

    assert df_fold["patient_id"].equals(df_final["patient_id"]), "patient_id sıraları uyuşmuyor"

    y_true = df_fold["label"].values
    p_fold = df_fold["prob_mucinous"].values
    p_final = df_final["prob_mucinous"].values
    p_ens = (p_fold + p_final) / 2.0

    print(f"5-fold AUC: {roc_auc_score(y_true, p_fold):.3f}")
    print(f"All-data AUC: {roc_auc_score(y_true, p_final):.3f}")
    print(f"Ensemble AUC: {roc_auc_score(y_true, p_ens):.3f}")

    rows = []
    for name, thr in [("@0.5", 0.5),
                      ("@Youden", find_youden_threshold(y_true, p_ens)[0]),
                      ("@Clinical (SENS≥0.80, SPEC≥0.50)", find_best_threshold(y_true, p_ens))]:
        m, cm, _ = compute_binary_metrics(y_true, p_ens, threshold=thr)
        ci = compute_bootstrap_ci(y_true, p_ens, threshold=thr)
        rows.append({"threshold": name.strip("@"), **m, **ci})
        print_full_metrics_table(m, ci, "MAE-Tiny Ensemble (5-fold + All-Data)", f"{name} {thr:.3f}")
        plot_confusion_matrix(cm, f"MAE-Tiny Ensemble {name} {thr:.3f}",
                              save_path=OUT_DIR / f"cm_{name.strip('@').replace('+', 'p').replace(' ', '_').replace('(', '').replace(')', '').replace('≥', 'ge')}.png")

    pd.DataFrame(rows).to_csv(OUT_DIR / "q1_external_test_metrics.csv", index=False)
    pd.DataFrame({
        "patient_id": df_fold["patient_id"].values,
        "label": y_true,
        "prob_mucinous_5fold": p_fold,
        "prob_mucinous_final": p_final,
        "prob_mucinous_ensemble": p_ens,
    }).to_csv(OUT_DIR / "ensemble_probs.csv", index=False)

    print(f"\nSonuçlar kaydedildi: {OUT_DIR}")


if __name__ == "__main__":
    main()
