"""
Mevcut modellerin external test olasılıklarını birleştirir.
Otomatik olarak hangi modellerin çıktısı varsa onları alır.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

from shared_utils import (
    compute_binary_metrics, compute_bootstrap_ci,
    find_youden_threshold, print_full_metrics_table, plot_confusion_matrix, plot_roc_pr
)

EXP_DIR = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/experiments_q1_128")
OUT_DIR = EXP_DIR / "cross_model_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {}
for p in EXP_DIR.glob("*/external_test/ensemble_probs.csv"):
    model_name = p.parent.parent.name
    MODEL_PATHS[model_name] = p


def find_clinical_threshold(y_true, y_prob, min_sens=0.80, min_spec=0.50):
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
    available = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            available[name] = pd.read_csv(path).sort_values("patient_id").reset_index(drop=True)
            print(f"✓ {name} yüklendi: {path}")

    if not available:
        print("Hiçbir modelin external_test/ensemble_probs.csv dosyası bulunamadı.")
        return

    # Hizalama kontrolü
    base = list(available.values())[0]
    for name, df in available.items():
        if not df["patient_id"].equals(base["patient_id"]):
            print(f"{name} patient_id sırası uyuşmuyor, atlanıyor.")
            continue

    y_true = base["label"].values
    probs = {n: df["prob_mucinous"].values for n, df in available.items()}

    # Her modelin tek başına metrikleri
    print("\n" + "=" * 70)
    print("TEK MODEL METRIKLERI (External Test @Youden)")
    print("=" * 70)
    for name, p in probs.items():
        auc = roc_auc_score(y_true, p)
        thr, _ = find_youden_threshold(y_true, p)
        m, _, _ = compute_binary_metrics(y_true, p, threshold=thr)
        print(f"{name:25s} AUC={auc:.3f} SENS={m['sensitivity']:.3f} SPEC={m['specificity']:.3f} F1={m['f1']:.3f}")

    # Basit ortalama ensemble
    ens_prob = np.mean(list(probs.values()), axis=0)
    ens_name = "Cross-Model Ensemble (Average)"

    print("\n" + "=" * 70)
    print("CROSS-MODEL ENSEMBLE")
    print("=" * 70)

    rows = []
    for label, thr in [("@0.5", 0.5),
                       ("@Youden", find_youden_threshold(y_true, ens_prob)[0]),
                       ("@Clinical (SENS≥0.80, SPEC≥0.50)", find_clinical_threshold(y_true, ens_prob))]:
        m, cm, _ = compute_binary_metrics(y_true, ens_prob, threshold=thr)
        ci = compute_bootstrap_ci(y_true, ens_prob, threshold=thr)
        print_full_metrics_table(m, ci, ens_name, f"{label} {thr:.3f}")
        plot_confusion_matrix(cm, f"{ens_name} {label} {thr:.3f}",
                              save_path=OUT_DIR / f"cm_{label.strip('@').replace('+', 'p').replace(' ', '_').replace('(', '').replace(')', '').replace('≥', 'ge')}.png")
        rows.append({"model": ens_name, "threshold": label.strip("@"), **m, **ci})

    pd.DataFrame(rows).to_csv(OUT_DIR / "cross_model_ensemble_metrics.csv", index=False)
    pd.DataFrame({
        "patient_id": base["patient_id"].values,
        "label": y_true,
        "prob_mucinous": ens_prob,
        **{f"prob_{n.replace(' ', '_').replace('-', '_')}": p for n, p in probs.items()}
    }).to_csv(OUT_DIR / "ensemble_probs.csv", index=False)
    plot_roc_pr(y_true, ens_prob, "cross_model_ensemble", OUT_DIR, opt_threshold=find_youden_threshold(y_true, ens_prob)[0])

    print(f"\nSonuçlar kaydedildi: {OUT_DIR}")


if __name__ == "__main__":
    main()
