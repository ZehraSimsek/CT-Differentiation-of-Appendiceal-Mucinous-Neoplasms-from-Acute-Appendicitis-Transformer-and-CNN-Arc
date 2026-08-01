"""
Master Q1 Karşılaştırma Tablosu — tüm modeller tek tabloda + DeLong pairwise testleri
==========================================================================================
experiments/<model>/external_test/{q1_external_test_metrics.csv, ensemble_probs.csv}
dosyalarını bulan HER modeli otomatik toplar (kaç model tamamlanmışsa o kadarını).
Eksik/henüz eğitilmemiş modelleri atlar, hata vermez — o yüzden tüm 3 transformer +
2 klasik baseline bitmeden de ara-rapor olarak çalıştırılabilir.

Üretir:
    experiments/q1_master_comparison.csv   — tüm modellerin ensemble@Youden metrikleri
    experiments/q1_delong_pairwise.csv     — ikili AUC karşılaştırmaları (DeLong p-value)

Çalıştırma:
    cd segformer && python generate_q1_master_table.py
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from shared_utils import delong_roc_test, compute_calibration

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments_q1_128"

DISPLAY_NAMES = {
    "segformer3d": "SegFormer3D-MSCA",
    "attention_swinunetr": "Attention-SwinUNETR",
    "mae_tinytransformer": "MAE-TinyTransformer3D",
    "swinunetr_linearprobe": "SwinUNETR Linear-Probe",
    "swinunetr_lp": "SwinUNETR Linear-Probe",
    "swinunetr_baseline": "SwinUNETR Baseline",
    "baseline_logreg": "Radiomics-lite + LogReg",
    "baseline_rf": "Radiomics-lite + RandomForest",
}


def find_completed_models():
    """external_test/ensemble_probs.csv olan HER model klasörünü bulur."""
    found = {}
    if not EXPERIMENTS_DIR.exists():
        return found
    for model_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        probs_path = model_dir / "external_test" / "ensemble_probs.csv"
        metrics_path = model_dir / "external_test" / "q1_external_test_metrics.csv"
        if probs_path.exists() and metrics_path.exists():
            found[model_dir.name] = {"probs": probs_path, "metrics": metrics_path}
    return found


def main():
    models = find_completed_models()
    if not models:
        print("Henüz hiçbir modelin external_test/ensemble_probs.csv dosyası yok. "
              "Önce en az bir train_*.py script'ini çalıştırın.")
        return

    print(f"Bulunan {len(models)} tamamlanmış model: {list(models.keys())}\n")

    # ---- 1) Master metrics table (Ensemble @Youden satırı her modelden) ----
    master_rows = []
    probs_data = {}
    for model_key, paths in models.items():
        display_name = DISPLAY_NAMES.get(model_key, model_key)
        metrics_df = pd.read_csv(paths["metrics"])
        
        if "fold" not in metrics_df.columns:
            print(f"  [uyarı] {model_key}: 'fold' kolonu bulunamadı, atlanıyor.")
            continue
            
        youden_row = metrics_df[metrics_df["fold"] == "Ensemble (@Youden)"]
        if youden_row.empty:
            print(f"  [uyarı] {model_key}: 'Ensemble (@Youden)' satırı bulunamadı, atlanıyor.")
            continue
        youden_row = youden_row.iloc[0].to_dict()

        probs_df = pd.read_csv(paths["probs"])
        probs_data[model_key] = probs_df

        cal = compute_calibration(probs_df["label"].values, probs_df["prob_mucinous"].values)

        master_rows.append({
            "model": display_name,
            "n_test": len(probs_df),
            "auc_roc": youden_row.get("auc_roc"),
            "auc_ci": f"[{youden_row.get('auc_ci_lo', float('nan')):.3f}-{youden_row.get('auc_ci_hi', float('nan')):.3f}]"
                      if pd.notna(youden_row.get("auc_ci_lo")) else "-",
            "sensitivity": youden_row.get("sensitivity"),
            "specificity": youden_row.get("specificity"),
            "ppv": youden_row.get("ppv"),
            "npv": youden_row.get("npv"),
            "f1": youden_row.get("f1"),
            "accuracy": youden_row.get("accuracy"),
            "brier_score": cal["brier_score"],
            "ece": cal["ece"],
            "threshold_youden": youden_row.get("threshold_used"),
        })

    master_df = pd.DataFrame(master_rows).sort_values("auc_roc", ascending=False)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(EXPERIMENTS_DIR / "q1_master_comparison.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("=" * 100)
    print("Q1 MASTER KARŞILAŞTIRMA TABLOSU (Ensemble @Youden, External Test)")
    print("=" * 100)
    print(master_df.round(3).to_string(index=False))

    # ---- 2) DeLong pairwise AUC karşılaştırmaları (SADECE aynı hasta kümesinde geçerli) ----
    print("\n" + "=" * 100)
    print("DELONG PAIRWISE AUC KARŞILAŞTIRMASI (external test, aynı 37 hasta)")
    print("=" * 100)

    delong_rows = []
    model_keys = list(probs_data.keys())
    for m1, m2 in combinations(model_keys, 2):
        df1 = probs_data[m1].sort_values("patient_id").reset_index(drop=True)
        df2 = probs_data[m2].sort_values("patient_id").reset_index(drop=True)

        if not df1["patient_id"].equals(df2["patient_id"]):
            common = sorted(set(df1["patient_id"]) & set(df2["patient_id"]))
            df1 = df1[df1["patient_id"].isin(common)].sort_values("patient_id").reset_index(drop=True)
            df2 = df2[df2["patient_id"].isin(common)].sort_values("patient_id").reset_index(drop=True)

        if len(df1) == 0 or not df1["label"].equals(df2["label"]):
            print(f"  [atlandı] {m1} vs {m2}: ortak/uyumlu hasta kümesi yok.")
            continue

        auc1, auc2, z, p = delong_roc_test(df1["label"].values, df1["prob_mucinous"].values,
                                            df2["prob_mucinous"].values)
        delong_rows.append({
            "model_1": DISPLAY_NAMES.get(m1, m1), "auc_1": round(auc1, 4),
            "model_2": DISPLAY_NAMES.get(m2, m2), "auc_2": round(auc2, 4),
            "auc_diff": round(auc1 - auc2, 4), "z": round(z, 4), "p_value": round(p, 5),
            "significant_p<0.05": p < 0.05,
        })

    delong_df = pd.DataFrame(delong_rows)
    if len(delong_df):
        delong_df.to_csv(EXPERIMENTS_DIR / "q1_delong_pairwise.csv", index=False)
        print(delong_df.to_string(index=False))
    else:
        print("  (En az 2 model gerekli; henüz yeterli model tamamlanmadı.)")

    print(f"\nKaydedildi: {EXPERIMENTS_DIR / 'q1_master_comparison.csv'}")
    if len(delong_df):
        print(f"Kaydedildi: {EXPERIMENTS_DIR / 'q1_delong_pairwise.csv'}")

    return master_df, delong_df if len(delong_rows) else None


if __name__ == "__main__":
    main()
