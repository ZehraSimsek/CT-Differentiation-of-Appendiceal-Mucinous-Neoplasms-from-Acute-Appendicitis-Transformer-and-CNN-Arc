"""
MAE-TinyTransformer3D External Test çıktısını OOF üzerinden kalibre edip
 daha dengeli Q1 metrikleri üretir.

Çıktılar:
    experiments_q1_128/mae_tinytransformer/external_test/calibrated_q1_metrics.csv
    experiments_q1_128/mae_tinytransformer/external_test/calibrated_ensemble_probs.csv
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from shared_utils import (
    compute_binary_metrics, compute_bootstrap_ci,
    find_youden_threshold, print_full_metrics_table, plot_confusion_matrix
)

BASE = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/experiments_q1_128/mae_tinytransformer")
OOF_CSV = BASE / "aggregate_oof" / "oof_predictions.csv"
EXT_CSV = BASE / "external_test" / "ensemble_probs.csv"
OUT_DIR = BASE / "external_test"


def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def find_constraint_threshold(y_true, y_prob, min_sens=0.85, min_spec=0.50):
    """OOF üzerinde SENS>=min_sens VE SPEC>=min_spec sağlayan, F1'i maksimize eden eşik."""
    from sklearn.metrics import f1_score
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
        # Çift kısıt sağlanamazsa Youden dön
        best_thr, _ = find_youden_threshold(y_true, y_prob)
    return float(best_thr)


def main():
    oof = pd.read_csv(OOF_CSV)
    ext = pd.read_csv(EXT_CSV)

    y_oof = oof["label"].values
    p_oof = oof["prob_mucinous"].values
    y_ext = ext["label"].values
    p_ext = ext["prob_mucinous"].values

    print(f"OOF: {len(oof)} hasta | External: {len(ext)} hasta")
    print(f"OOF AUC: {roc_auc_score(y_oof, p_oof):.3f}")
    print(f"Raw External AUC: {roc_auc_score(y_ext, p_ext):.3f}")

    # 1) Platt scaling: logit(prob) üzerinde LR fit et
    logit_oof = safe_logit(p_oof).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e10, solver="lbfgs")
    calibrator.fit(logit_oof, y_oof)

    logit_ext = safe_logit(p_ext).reshape(-1, 1)
    p_oof_cal = calibrator.predict_proba(logit_oof)[:, 1]
    p_ext_cal = calibrator.predict_proba(logit_ext)[:, 1]

    print(f"Calibrated OOF AUC: {roc_auc_score(y_oof, p_oof_cal):.3f}")
    print(f"Calibrated External AUC: {roc_auc_score(y_ext, p_ext_cal):.3f}")

    # 2) Eşiği OOF üzerinden seç (klinik kısıtlar + F1)
    thr = find_constraint_threshold(y_oof, p_oof_cal, min_sens=0.80, min_spec=0.50)
    print(f"\nOOF'dan seçilen eşik: {thr:.3f}")

    # 3) External test metrikleri
    m, cm, _ = compute_binary_metrics(y_ext, p_ext_cal, threshold=thr)
    ci = compute_bootstrap_ci(y_ext, p_ext_cal, threshold=thr)
    print_full_metrics_table(m, ci, "MAE-Tiny (Platt Calibrated)", f"OOF-derived {thr:.3f}")

    # 4) Kaydet
    cal_df = pd.DataFrame({
        "patient_id": ext["patient_id"].values,
        "label": y_ext,
        "prob_mucinous_raw": p_ext,
        "prob_mucinous_calibrated": p_ext_cal,
    })
    cal_df.to_csv(OUT_DIR / "calibrated_ensemble_probs.csv", index=False)

    summary = pd.DataFrame([{"model": "MAE-Tiny (Platt Calibrated)", "threshold": f"OOF-derived {thr:.3f}", **m, **ci}])
    summary.to_csv(OUT_DIR / "calibrated_q1_metrics.csv", index=False)
    plot_confusion_matrix(cm, f"MAE-Tiny Calibrated @ {thr:.3f}", save_path=OUT_DIR / "cm_calibrated.png")

    print(f"\nKaydedildi:\n  {OUT_DIR / 'calibrated_q1_metrics.csv'}\n  {OUT_DIR / 'calibrated_ensemble_probs.csv'}")


if __name__ == "__main__":
    main()
