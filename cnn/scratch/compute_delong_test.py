import os
import sys
import gc
import shutil
import zipfile
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from config import PipelineConfig
from data.patient_dataset import PatientCTDataset
from models import build_model

def append_h5_paths(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    h5_paths = []
    for _, row in df.iterrows():
        pid = str(row['patient_id'])
        label = int(row['label'])
        folder = config.mucinous_v03_dir if label == 1 else config.appendicitis_v03_dir
        path = os.path.join(folder, f"{pid}.h5")
        h5_paths.append(path)
    df['h5_path'] = h5_paths
    return df

# DeLong test implementation
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        tx[r, :] = tz[r, :m]
        ty[r, :] = tz[r, m:]

    tz_sum_m = np.sum(tx, axis=1)
    aucs = (tz_sum_m - m * (m + 1) / 2.0) / (m * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_delong_p_value(ground_truth, pred1, pred2):
    preds = np.vstack([pred1, pred2])
    order = np.argsort(ground_truth)[::-1] # 1s first
    ground_truth_sorted = np.array(ground_truth)[order]
    preds_sorted = preds[:, order]
    
    label_1_count = int(np.sum(ground_truth_sorted == 1))
    aucs, cov = fast_delong(preds_sorted, label_1_count)
    
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], diff, 0.0, 1.0
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], diff, z, p

def run_evaluation_with_folds(model_name, ckpt_dir, test_loader, device, drop_rate):
    cfg = PipelineConfig(model_name=model_name, dropout_rate=drop_rate, pretrained=False)
    all_fold_probs = []
    
    for fold in range(1, 6):
        model = build_model(cfg)
        model = model.to(device)
        ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}", f"best_{model_name}_auc_roc.pth")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                fold_probs.extend(probs)
                
        all_fold_probs.append(np.array(fold_probs))
        del model
        gc.collect()
        
    ensemble_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
    return all_fold_probs, ensemble_probs

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)
    
    config = PipelineConfig()
    
    test_df = pd.read_csv("datas/external_test_set.csv")
    test_df = append_h5_paths(test_df, config)
    
    test_ds = PatientCTDataset(df=test_df, augment_train=False, config=config)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)
    
    models_to_eval = [
        ("1_UNet_PlusPlus", "unet_plusplus", "checkpoints/unet_plusplus/deneme_1", 0.4),
        ("2_DenseNet121", "densenet121", "checkpoints/densenet121/deneme_2", 0.4),
        ("3_EfficientNet_B0", "efficientnet_b0", "checkpoints/efficientnet_b0/deneme_1", 0.3)
    ]
    
    base_dest = "external_test_outputs"
    os.makedirs(base_dest, exist_ok=True)
    
    pids = test_df['patient_id'].astype(str).tolist()
    labels = test_df['label'].astype(int).tolist()
    class_names = ["Musinoz" if l == 1 else "Apandisit" for l in labels]
    
    master_df = pd.DataFrame({
        "patient_id": pids,
        "true_label": labels,
        "true_class": class_names
    })
    
    model_ensemble_probs = {}
    
    for folder_name, m_type, src_dir, drop_rate in models_to_eval:
        print(f"\n--- Processing {folder_name} ---", flush=True)
        model_dest = os.path.join(base_dest, folder_name)
        os.makedirs(model_dest, exist_ok=True)
        
        # 1. Copy ensemble, folds, gradcam, metrics
        ens_src = os.path.join(src_dir, "ensemble")
        if os.path.exists(ens_src):
            shutil.copytree(ens_src, os.path.join(model_dest, "Ensemble"), dirs_exist_ok=True)
            
        folds_dest = os.path.join(model_dest, "Fold_Bazli")
        os.makedirs(folds_dest, exist_ok=True)
        for fold in range(1, 6):
            fold_src = os.path.join(src_dir, f"fold_{fold}")
            f_dest = os.path.join(folds_dest, f"Fold_{fold}")
            if os.path.exists(fold_src):
                os.makedirs(f_dest, exist_ok=True)
                for f in os.listdir(fold_src):
                    if f.endswith(".png"):
                        shutil.copy2(os.path.join(fold_src, f), os.path.join(f_dest, f))
                        
        xai_src = os.path.join(src_dir, "external_test_xai")
        if os.path.exists(xai_src):
            shutil.copytree(xai_src, os.path.join(model_dest, "GradCAM_XAI"), dirs_exist_ok=True)
            
        for meta in ["metrics_summary.csv", "metrics_summary.xlsx"]:
            mp = os.path.join(src_dir, meta)
            if os.path.exists(mp):
                shutil.copy2(mp, os.path.join(model_dest, meta))
                
        # 2. Extract Probabilities
        fold_probs_list, ens_probs = run_evaluation_with_folds(m_type, src_dir, test_loader, device, drop_rate)
        clean_name = folder_name.split("_", 1)[1]
        
        for fold_idx, f_prob in enumerate(fold_probs_list, start=1):
            master_df[f"{clean_name}_Fold_{fold_idx}_Prob"] = np.round(f_prob, 6)
            
        master_df[f"{clean_name}_Ensemble_Prob"] = np.round(ens_probs, 6)
        model_ensemble_probs[clean_name] = ens_probs
        
        # Save individual model CSV & Excel
        sub_cols = ["patient_id", "true_label", "true_class"] + [c for c in master_df.columns if clean_name in c]
        master_df[sub_cols].to_csv(os.path.join(model_dest, f"{clean_name.lower()}_test_probabilities.csv"), index=False)
        master_df[sub_cols].to_excel(os.path.join(model_dest, f"{clean_name.lower()}_test_probabilities.xlsx"), index=False)
        print(f"✓ Saved {clean_name} probabilities (AUC = {roc_auc_score(labels, ens_probs):.4f})", flush=True)
        
    # Save master probabilities CSV & Excel in root
    master_csv_path = os.path.join(base_dest, "test_predictions_probabilities_all_models.csv")
    master_excel_path = os.path.join(base_dest, "test_predictions_probabilities_all_models.xlsx")
    master_df.to_csv(master_csv_path, index=False)
    master_df.to_excel(master_excel_path, index=False)
    print(f"\n✓ Saved Master Probabilities to: {master_csv_path}", flush=True)
    
    # Save ROC curve coordinates
    roc_rows = []
    for prefix, ens_probs in model_ensemble_probs.items():
        fpr, tpr, thresholds = roc_curve(labels, ens_probs)
        for f, t, th in zip(fpr, tpr, thresholds):
            roc_rows.append({
                "model": prefix,
                "fpr_1_minus_specificity": round(f, 6),
                "tpr_sensitivity": round(t, 6),
                "threshold": round(th, 6)
            })
    roc_coords_df = pd.DataFrame(roc_rows)
    roc_coords_csv = os.path.join(base_dest, "roc_curve_coordinates.csv")
    roc_coords_xlsx = os.path.join(base_dest, "roc_curve_coordinates.xlsx")
    roc_coords_df.to_csv(roc_coords_csv, index=False)
    roc_coords_df.to_excel(roc_coords_xlsx, index=False)
    print(f"✓ Saved ROC curve coordinates to: {roc_coords_csv}", flush=True)
    
    # Save PR curve coordinates
    pr_rows = []
    for prefix, ens_probs in model_ensemble_probs.items():
        precision, recall, thresholds = precision_recall_curve(labels, ens_probs)
        for p, r in zip(precision, recall):
            pr_rows.append({
                "model": prefix,
                "recall_sensitivity": round(r, 6),
                "precision": round(p, 6)
            })
    pr_coords_df = pd.DataFrame(pr_rows)
    pr_coords_csv = os.path.join(base_dest, "pr_curve_coordinates.csv")
    pr_coords_xlsx = os.path.join(base_dest, "pr_curve_coordinates.xlsx")
    pr_coords_df.to_csv(pr_coords_csv, index=False)
    pr_coords_df.to_excel(pr_coords_xlsx, index=False)
    print(f"✓ Saved PR curve coordinates to: {pr_coords_csv}", flush=True)
    
    # Compute DeLong Statistical Tests
    print("\n--- Computing DeLong Statistical Tests ---", flush=True)
    delong_results = []
    pairs = [
        ("UNet_PlusPlus", "DenseNet121"),
        ("UNet_PlusPlus", "EfficientNet_B0"),
        ("DenseNet121", "EfficientNet_B0")
    ]
    for m1, m2 in pairs:
        auc1, auc2, diff, z, p = calc_delong_p_value(labels, model_ensemble_probs[m1], model_ensemble_probs[m2])
        significance = "Anlamlı Fark Var (p < 0.05)" if p < 0.05 else "Anlamlı Fark Yok (p >= 0.05)"
        delong_results.append({
            "Karsilastirma": f"{m1} vs {m2}",
            "Model_1_AUC": round(auc1, 4),
            "Model_2_AUC": round(auc2, 4),
            "AUC_Farki": round(diff, 4),
            "Z_Score": round(z, 4),
            "p_degeri": round(p, 4),
            "Istatistiksel_Yorum": significance
        })
        print(f"  • {m1} (AUC={auc1:.4f}) vs {m2} (AUC={auc2:.4f}): Diff={diff:+.4f}, Z={z:.3f}, p={p:.4f} -> {significance}", flush=True)
        
    delong_df = pd.DataFrame(delong_results)
    delong_df.to_csv(os.path.join(base_dest, "delong_statistical_test_results.csv"), index=False)
    delong_df.to_excel(os.path.join(base_dest, "delong_statistical_test_results.xlsx"), index=False)
    print("✓ Saved DeLong test results to CSV & Excel.", flush=True)

    # Create ZIP archive
    zip_filename = "external_test_outputs.zip"
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        
    print("\n--- Creating Final ZIP Package ---", flush=True)
    file_count = 0
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dest):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dest)
                zipf.write(file_path, arcname)
                file_count += 1
                
    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"\n============================================================")
    print(f"✓ MASTER ZIP CREATED: {zip_filename} ({zip_size_mb:.2f} MB)")
    print(f"✓ Total files packaged: {file_count}")
    print(f"============================================================", flush=True)

if __name__ == "__main__":
    main()
