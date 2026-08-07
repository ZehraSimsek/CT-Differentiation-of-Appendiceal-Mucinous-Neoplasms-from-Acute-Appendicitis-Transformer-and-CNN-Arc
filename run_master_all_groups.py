"""
run_master_all_groups.py
========================
Tüm 7 modelin (Grup A ve Grup B) birleştiği makinede çalıştırılacak FINAL script.
experiments_multirun/ altındaki tüm modelleri tarar.
1) Her modelin Run 1, Run 2, Run 3 AUC değerlerini ve ortalamasını (mean ± std) çıkarır.
2) Tüm run'ların ortalama ensemble olasılıkları üzerinden nihai Sensitivity, Specificity vb. metrikleri hesaplar.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("final"))

import numpy as np
import pandas as pd
from pathlib import Path
from shared_utils import compute_calibration, compute_binary_metrics, find_youden_threshold, compute_bootstrap_ci

BASE = Path("experiments_multirun")

# Beklenen tüm modeller
ALL_MODELS = {
    "swinunetr_lp":     "SwinUNETR-LP",
    "ag_msf":           "AG-MSF",
    "mae_tiny3d":       "MAE-Tiny3D",
    "segformer3d_msca": "SegFormer3D-MSCA",
    "unetpp":           "UNet++",
    "densenet121":      "DenseNet121",
    "efficientnet_b0":  "EfficientNet-B0",
}

def load_run_aucs(folder):
    """Her modelin Run 1, 2, 3 için Youden ensemble AUC değerlerini çeker."""
    aucs = []
    for run_idx in [1, 2, 3]:
        p = BASE / folder / f"run_{run_idx:02d}" / "external_test" / "q1_external_test_metrics.csv"
        if not p.exists():
            aucs.append(np.nan)
            continue
        try:
            df = pd.read_csv(p)
            yrow = df[df['fold'].str.contains('Youden', na=False)]
            if not yrow.empty:
                aucs.append(float(yrow.iloc[0]['auc_roc']))
            else:
                aucs.append(np.nan)
        except Exception:
            aucs.append(np.nan)
    return aucs

def load_mean_probs(folder):
    """3 run'ın ensemble olasılıklarının hasta bazında ortalamasını döndür."""
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
    print("=== Q1 MASTER EVALUATION BAŞLIYOR ===")
    
    run_details = []
    metric_rows = []
    y_true_ref = None
    
    for folder, name in ALL_MODELS.items():
        # Run AUC detayları
        aucs = load_run_aucs(folder)
        
        # Ortalama olasılıklar (Genel metrikler için)
        y_true, probs = load_mean_probs(folder)
        
        if probs is not None:
            if y_true_ref is None:
                y_true_ref = y_true
                
            # AUC İstatistikleri
            valid_aucs = [a for a in aucs if not np.isnan(a)]
            mean_auc = np.mean(valid_aucs) if valid_aucs else np.nan
            std_auc = np.std(valid_aucs) if valid_aucs else np.nan
            
            run_details.append({
                "Model": name,
                "Run_1_AUC": round(aucs[0], 4) if len(aucs)>0 and not np.isnan(aucs[0]) else "-",
                "Run_2_AUC": round(aucs[1], 4) if len(aucs)>1 and not np.isnan(aucs[1]) else "-",
                "Run_3_AUC": round(aucs[2], 4) if len(aucs)>2 and not np.isnan(aucs[2]) else "-",
                "AUC_Mean±Std": f"{mean_auc:.3f} ± {std_auc:.3f}" if not np.isnan(mean_auc) else "-"
            })
            
            # Ana Metrikler
            thr, _ = find_youden_threshold(y_true, probs)
            m, _, _ = compute_binary_metrics(y_true, probs, threshold=thr)
            ci = compute_bootstrap_ci(y_true, probs, threshold=thr, n_bootstraps=2000)
            cal = compute_calibration(y_true, probs)
            
            metric_rows.append({
                "Model": name,
                "Mean_Ensemble_AUC": round(m["auc_roc"], 3),
                "AUC_CI": f"[{ci['auc_ci_lo']:.3f}-{ci['auc_ci_hi']:.3f}]",
                "Sens": round(m["sensitivity"], 3),
                "Spec": round(m["specificity"], 3),
                "F1": round(m["f1"], 3),
                "Brier": round(cal["brier_score"], 3),
                "ECE": round(cal["ece"], 3),
                "TP": m["tp"], "TN": m["tn"], "FP": m["fp"], "FN": m["fn"],
            })
            print(f"✅ İşlendi: {name}")
        else:
            print(f"❌ Bulunamadı: {name} (Veri klasöründe eksik)")

    if not run_details:
        print("\n[HATA] Model klasörleri bulunamadı.")
        return

    # 1) Run Detayları Tablosu
    run_df = pd.DataFrame(run_details)
    run_df.to_csv(BASE / "Q1_MASTER_run_details.csv", index=False)
    
    print("\n" + "="*70)
    print("1. RUN BAZLI AUC DETAYLARI")
    print("="*70)
    print(run_df.to_string(index=False))
    
    # 2) Ana Metrik Tablosu
    metrics_df = pd.DataFrame(metric_rows).sort_values("Mean_Ensemble_AUC", ascending=False)
    metrics_df.to_csv(BASE / "Q1_MASTER_full_metrics.csv", index=False)
    
    print("\n" + "="*70)
    print("2. ORTALAMA ENSEMBLE METRİKLERİ (Tüm Runlar Birleştirilmiş, n=24)")
    print("="*70)
    # Konsola daha okunabilir yazdırmak için bazı kolonları seçelim
    display_df = metrics_df[["Model", "Mean_Ensemble_AUC", "Sens", "Spec", "F1", "Brier", "ECE", "TP", "TN", "FP", "FN"]]
    print(display_df.to_string(index=False))

    print(f"\n[KAYDEDİLDİ] Dosyalar {BASE} dizinine kaydedildi:")
    print("  - Q1_MASTER_run_details.csv")
    print("  - Q1_MASTER_full_metrics.csv")

if __name__ == "__main__":
    main()
