"""
generate_group_a_tables.py
==========================
Grup A (Transformer) modelleri için Makale Tablolarını üretir.
- Tablo A-a: Run-bazlı AUC (n=24, 3 run)
- Tablo A-b (Kısım 1): T=0.50 Eşik Performansı
- Tablo A-b (Kısım 2): Youden-J Optimal Eşik Performansı
- Tablo A-c: T=0.50 Kalibrasyon ve TP/TN Detayları
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("final"))

import numpy as np
import pandas as pd
from pathlib import Path
from shared_utils import compute_calibration, compute_binary_metrics, find_youden_threshold, compute_bootstrap_ci

BASE = Path("experiments_multirun")

GROUP_A_MODELS = {
    "swinunetr_lp":     "SwinUNETR-LP",
    "ag_msf":           "AG-MSF",
    "mae_tiny3d":       "MAE-Tiny3D",
    "segformer3d_msca": "SegFormer3D-MSCA",
}

def load_run_aucs(folder):
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
        except:
            aucs.append(np.nan)
    return aucs

def load_mean_probs(folder):
    all_probs = []
    ref_df = None
    for run_idx in [1, 2, 3]:
        p = BASE / folder / f"run_{run_idx:02d}" / "external_test" / "ensemble_probs.csv"
        if not p.exists(): continue
        df = pd.read_csv(p).sort_values("patient_id").reset_index(drop=True)
        all_probs.append(df["prob_mucinous"].values)
        if ref_df is None: ref_df = df[["patient_id", "label"]]
    if not all_probs: return None, None
    return ref_df["label"].values, np.mean(all_probs, axis=0)

def main():
    table_a = []
    table_b1 = []
    table_b2 = []
    table_c = []
    
    y_true_ref = None
    
    for folder, name in GROUP_A_MODELS.items():
        aucs = load_run_aucs(folder)
        y_true, probs = load_mean_probs(folder)
        
        if probs is None:
            continue
            
        if y_true_ref is None:
            y_true_ref = y_true
            
        # Tablo A: Run Bazlı
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        mean_auc = np.mean(valid_aucs) if valid_aucs else np.nan
        std_auc = np.std(valid_aucs) if valid_aucs else np.nan
        
        table_a.append({
            "Model": name,
            "Run 1 (seed=42)": f"{aucs[0]:.4f}" if len(aucs)>0 and not np.isnan(aucs[0]) else "-",
            "Run 2 (seed=123)": f"{aucs[1]:.4f}" if len(aucs)>1 and not np.isnan(aucs[1]) else "-",
            "Run 3 (seed=456)": f"{aucs[2]:.4f}" if len(aucs)>2 and not np.isnan(aucs[2]) else "-",
            "AUC (ort.±std)": f"{mean_auc:.3f} ± {std_auc:.3f}" if not np.isnan(mean_auc) else "-"
        })
        
        # Ortak Hesaplamalar
        # T=0.50 Metrikleri
        m_50, _, _ = compute_binary_metrics(y_true, probs, threshold=0.50)
        ci_50 = compute_bootstrap_ci(y_true, probs, threshold=0.50, n_bootstraps=2000)
        cal = compute_calibration(y_true, probs)
        bal_acc_50 = (m_50["sensitivity"] + m_50["specificity"]) / 2
        
        # Youden Metrikleri
        opt_thr, _ = find_youden_threshold(y_true, probs)
        m_y, _, _ = compute_binary_metrics(y_true, probs, threshold=opt_thr)
        bal_acc_y = (m_y["sensitivity"] + m_y["specificity"]) / 2
        
        # Tablo B-1 (T=0.50)
        table_b1.append({
            "Model": name,
            "AUC": f"{m_50['auc_roc']:.3f}",
            "%95 GA (AUC)": f"[{ci_50['auc_ci_lo']:.3f}–{ci_50['auc_ci_hi']:.3f}]",
            "Duyarlılık": f"{m_50['sensitivity']:.3f}",
            "Özgüllük": f"{m_50['specificity']:.3f}",
            "F1": f"{m_50['f1']:.3f}",
            "Dengeli Doğruluk": f"{bal_acc_50:.3f}"
        })
        
        # Tablo B-2 (Youden)
        table_b2.append({
            "Model": name,
            "Opt. Eşik": f"{opt_thr:.3f}",
            "AUC": f"{m_y['auc_roc']:.3f}",
            "Duyarlılık": f"{m_y['sensitivity']:.3f}",
            "Özgüllük": f"{m_y['specificity']:.3f}",
            "F1": f"{m_y['f1']:.3f}",
            "Dengeli Doğruluk": f"{bal_acc_y:.3f}",
            "TP": m_y["tp"], "TN": m_y["tn"], "FP": m_y["fp"], "FN": m_y["fn"]
        })
        
        # Tablo C (Kalibrasyon T=0.50)
        table_c.append({
            "Model": name,
            "Brier": f"{cal['brier_score']:.3f}",
            "ECE": f"{cal['ece']:.3f}",
            "TP": m_50["tp"], "TN": m_50["tn"], "FP": m_50["fp"], "FN": m_50["fn"]
        })
        
    # DataFramelere Çevir
    df_a = pd.DataFrame(table_a)
    df_b1 = pd.DataFrame(table_b1)
    df_b2 = pd.DataFrame(table_b2)
    df_c = pd.DataFrame(table_c)
    
    # CSV Kayıt
    df_a.to_csv(BASE / "TABLO_A_run_bazli.csv", index=False)
    df_b1.to_csv(BASE / "TABLO_B1_t50_performans.csv", index=False)
    df_b2.to_csv(BASE / "TABLO_B2_youden_performans.csv", index=False)
    df_c.to_csv(BASE / "TABLO_C_t50_kalibrasyon.csv", index=False)
    
    # Konsola Bas
    print("="*100)
    print("Tablo A. Transformer Modelleri ailesinde run-bazlı ensemble AUC kararlılığı (n=24, 3 run).")
    print("="*100)
    print(df_a.to_string(index=False))
    
    print("\n" + "="*100)
    print("Tablo B (Kısım 1). Dış test setinde varsayılan eşikteki (T=0.50) ortalama ayırt edicilik performansı.")
    print("="*100)
    print(df_b1.to_string(index=False))
    
    print("\n" + "="*100)
    print("Tablo B (Kısım 2). Dış test setinde Youden-J optimal karar eşiğindeki ortalama performans.")
    print("="*100)
    print(df_b2.to_string(index=False))
    
    print("\n" + "="*100)
    print("Tablo C. Transformer Modelleri ailesinin dış test setindeki (n=24) kalibrasyon ve sınıflandırma detayları (T=0.50).")
    print("="*100)
    print(df_c.to_string(index=False))

if __name__ == "__main__":
    main()
