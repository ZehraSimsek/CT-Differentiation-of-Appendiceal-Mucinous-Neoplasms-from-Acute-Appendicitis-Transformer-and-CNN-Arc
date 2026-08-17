import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)

MODELS = ["unet_plusplus", "densenet121", "efficientnet_b0"]
MODEL_DISPLAY_NAMES = {
    "unet_plusplus": "UNet++",
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNet-B0",
}
RUN_SEEDS = [42, 123, 456]
EXP_ROOT = "experiments_q1_128"
OUT_DIR = "paper_tables_and_curves"
os.makedirs(OUT_DIR, exist_ok=True)

def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Include upper boundary in last bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
            bin_data.append({
                "bin_idx": i + 1,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_size": int(bin_size),
                "bin_acc": float(bin_acc),
                "bin_conf": float(bin_conf),
                "calibration_gap": float(np.abs(bin_acc - bin_conf))
            })
        else:
            bin_data.append({
                "bin_idx": i + 1,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_size": 0,
                "bin_acc": 0.0,
                "bin_conf": float((bin_lower + bin_upper) / 2.0),
                "calibration_gap": 0.0
            })
    return float(ece), bin_data

def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=2000, ci=95, rng_seed=42):
    """Compute non-parametric Bootstrap 95% Confidence Interval for AUC."""
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
        
    alpha = (100 - ci) / 2.0
    lower = np.percentile(bootstrapped_scores, alpha)
    upper = np.percentile(bootstrapped_scores, 100 - alpha)
    return float(lower), float(upper)

def find_optimal_threshold_youden(y_true, y_prob):
    """Find optimal threshold via Youden's J statistic (Sensitivity + Specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])
    best_thresh = min(max(best_thresh, 0.01), 0.99)
    return best_thresh

def get_metrics_at_threshold(y_true, y_prob, threshold=0.5):
    """Compute all evaluation metrics at a specific decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = (sens + spec) / 2.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ece, _ = compute_ece(y_true, y_prob)
    
    return {
        "threshold": float(threshold),
        "auc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(bacc),
        "f1": float(f1),
        "accuracy": float(acc),
        "brier": float(brier),
        "ece": float(ece),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

# -----------------------------------------------------------------------------
# MAIN ANALYSIS LOOP
# -----------------------------------------------------------------------------
table_3a_rows = []
table_3b_default_rows = []
table_3b_optimal_rows = []
table_3c_details_rows = []
roc_curve_rows = []
pr_curve_rows = []
calibration_curve_rows = []

for m_key in MODELS:
    m_name = MODEL_DISPLAY_NAMES[m_key]
    run_aucs = []
    run_default_metrics = []
    run_optimal_metrics = []
    run_prob_dfs = []
    
    for r_idx, seed in enumerate(RUN_SEEDS, start=1):
        pred_csv = os.path.join(EXP_ROOT, m_key, f"run_0{r_idx}", "external_test", "external_test_predictions.csv")
        if not os.path.exists(pred_csv):
            print(f"Warning: {pred_csv} not found!")
            continue
        
        df = pd.read_csv(pred_csv)
        y_true = df["true_label"].values
        y_prob = df["ensemble_prob"].values
        run_prob_dfs.append(df)
        
        # 1. AUC
        auc_val = float(roc_auc_score(y_true, y_prob))
        run_aucs.append(auc_val)
        
        # 2. Metrics @ 0.50
        def_m = get_metrics_at_threshold(y_true, y_prob, threshold=0.50)
        run_default_metrics.append(def_m)
        
        # 3. Optimal threshold & metrics
        opt_thresh = find_optimal_threshold_youden(y_true, y_prob)
        opt_m = get_metrics_at_threshold(y_true, y_prob, threshold=opt_thresh)
        run_optimal_metrics.append(opt_m)
        
        # 4. Curves
        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        for f, t, th in zip(fpr, tpr, roc_thresh):
            roc_curve_rows.append({
                "Model": m_name,
                "Model_Code": m_key,
                "Run": f"Run {r_idx} (seed={seed})",
                "FPR": float(f),
                "TPR": float(t),
                "Threshold": float(th)
            })
            
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)
        for p, r in zip(prec, rec):
            pr_curve_rows.append({
                "Model": m_name,
                "Model_Code": m_key,
                "Run": f"Run {r_idx} (seed={seed})",
                "Recall": float(r),
                "Precision": float(p)
            })
            
        ece_val, bin_details = compute_ece(y_true, y_prob)
        for b in bin_details:
            calibration_curve_rows.append({
                "Model": m_name,
                "Model_Code": m_key,
                "Run": f"Run {r_idx} (seed={seed})",
                "ECE_Run": ece_val,
                **b
            })

    # Summary across 3 runs for Table 3a
    mean_auc = np.mean(run_aucs)
    std_auc = np.std(run_aucs, ddof=1) if len(run_aucs) > 1 else 0.0
    table_3a_rows.append({
        "Model": m_name,
        "Run 1 (seed=42)": f"{run_aucs[0]:.4f}",
        "Run 2 (seed=123)": f"{run_aucs[1]:.4f}",
        "Run 3 (seed=456)": f"{run_aucs[2]:.4f}",
        "AUC (ort.±std)": f"{mean_auc:.3f} ± {std_auc:.3f}",
    })
    
    # Combined probabilities across 3 runs for Grand Ensemble and Bootstrap CI
    grand_probs = np.mean([df["ensemble_prob"].values for df in run_prob_dfs], axis=0)
    y_true_ext = run_prob_dfs[0]["true_label"].values
    ci_low, ci_high = bootstrap_auc_ci(y_true_ext, grand_probs, n_bootstraps=2000, ci=95)
    
    # Run averages
    avg_sens_def = np.mean([m["sensitivity"] for m in run_default_metrics])
    avg_spec_def = np.mean([m["specificity"] for m in run_default_metrics])
    avg_f1_def = np.mean([m["f1"] for m in run_default_metrics])
    avg_bacc_def = np.mean([m["balanced_accuracy"] for m in run_default_metrics])
    avg_brier_def = np.mean([m["brier"] for m in run_default_metrics])
    avg_ece_def = np.mean([m["ece"] for m in run_default_metrics])
    
    grand_def = get_metrics_at_threshold(y_true_ext, grand_probs, threshold=0.50)
    grand_opt_thresh = find_optimal_threshold_youden(y_true_ext, grand_probs)
    grand_opt = get_metrics_at_threshold(y_true_ext, grand_probs, threshold=grand_opt_thresh)
    
    # Table 2b/3b format for Default Threshold
    table_3b_default_rows.append({
        "Model": m_name,
        "AUC": f"{mean_auc:.3f}",
        "%95 GA (AUC)": f"[{ci_low:.3f}–{ci_high:.3f}]",
        "Duyarlılık": f"{avg_sens_def:.3f}",
        "Özgüllük": f"{avg_spec_def:.3f}",
        "F1": f"{avg_f1_def:.3f}",
        "Dengeli Doğruluk": f"{avg_bacc_def:.3f}",
    })
    
    # Optimal Threshold metrics
    avg_opt_thresh = np.mean([m["threshold"] for m in run_optimal_metrics])
    avg_sens_opt = np.mean([m["sensitivity"] for m in run_optimal_metrics])
    avg_spec_opt = np.mean([m["specificity"] for m in run_optimal_metrics])
    avg_f1_opt = np.mean([m["f1"] for m in run_optimal_metrics])
    avg_bacc_opt = np.mean([m["balanced_accuracy"] for m in run_optimal_metrics])
    
    table_3b_optimal_rows.append({
        "Model": m_name,
        "Opt. Eşik": f"{avg_opt_thresh:.3f}",
        "AUC": f"{mean_auc:.3f}",
        "Duyarlılık": f"{avg_sens_opt:.3f}",
        "Özgüllük": f"{avg_spec_opt:.3f}",
        "F1": f"{avg_f1_opt:.3f}",
        "Dengeli Doğruluk": f"{avg_bacc_opt:.3f}",
        "TP": int(np.round(np.mean([m["tp"] for m in run_optimal_metrics]))),
        "TN": int(np.round(np.mean([m["tn"] for m in run_optimal_metrics]))),
        "FP": int(np.round(np.mean([m["fp"] for m in run_optimal_metrics]))),
        "FN": int(np.round(np.mean([m["fn"] for m in run_optimal_metrics]))),
    })
    
    # Table 2c/3c format (Calibration & Confusion Matrix Details @ 0.50)
    table_3c_details_rows.append({
        "Model": m_name,
        "Brier": f"{avg_brier_def:.3f}",
        "ECE": f"{avg_ece_def:.3f}",
        "TP": int(np.round(np.mean([m["tp"] for m in run_default_metrics]))),
        "TN": int(np.round(np.mean([m["tn"] for m in run_default_metrics]))),
        "FP": int(np.round(np.mean([m["fp"] for m in run_default_metrics]))),
        "FN": int(np.round(np.mean([m["fn"] for m in run_default_metrics]))),
    })

# Save DataFrames to CSV
df_3a = pd.DataFrame(table_3a_rows)
df_3a.to_csv(os.path.join(OUT_DIR, "tablo_3a_run_bazli_auc_kararliligi.csv"), index=False)

df_3b_def = pd.DataFrame(table_3b_default_rows)
df_3b_def.to_csv(os.path.join(OUT_DIR, "tablo_3b_varsayilan_esik_performans.csv"), index=False)

df_3b_opt = pd.DataFrame(table_3b_optimal_rows)
df_3b_opt.to_csv(os.path.join(OUT_DIR, "tablo_3b_optimal_esik_performans.csv"), index=False)

df_3c = pd.DataFrame(table_3c_details_rows)
df_3c.to_csv(os.path.join(OUT_DIR, "tablo_3c_kalibrasyon_ve_detaylar.csv"), index=False)

df_roc = pd.DataFrame(roc_curve_rows)
df_roc.to_csv(os.path.join(OUT_DIR, "roc_curves_classic_cnn.csv"), index=False)

df_pr = pd.DataFrame(pr_curve_rows)
df_pr.to_csv(os.path.join(OUT_DIR, "pr_curves_classic_cnn.csv"), index=False)

df_calib = pd.DataFrame(calibration_curve_rows)
df_calib.to_csv(os.path.join(OUT_DIR, "calibration_curves_classic_cnn.csv"), index=False)

# Master Excel Workbook
excel_path = os.path.join(OUT_DIR, "Zehra_Hoca_Classic_CNN_Makale_Tablolari_ve_Verileri.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    df_3a.to_excel(writer, sheet_name="Tablo_3a_AUC_Kararliligi", index=False)
    df_3b_def.to_excel(writer, sheet_name="Tablo_3b_Varsayilan_Esik", index=False)
    df_3b_opt.to_excel(writer, sheet_name="Tablo_3b_Optimal_Esik", index=False)
    df_3c.to_excel(writer, sheet_name="Tablo_3c_Kalibrasyon", index=False)
    df_roc.to_excel(writer, sheet_name="ROC_Egrileri_Ham_Veri", index=False)
    df_pr.to_excel(writer, sheet_name="PR_Egrileri_Ham_Veri", index=False)
    df_calib.to_excel(writer, sheet_name="Kalibrasyon_ECE_Detay", index=False)

print("Calculation completed successfully! All files generated in:", OUT_DIR)
