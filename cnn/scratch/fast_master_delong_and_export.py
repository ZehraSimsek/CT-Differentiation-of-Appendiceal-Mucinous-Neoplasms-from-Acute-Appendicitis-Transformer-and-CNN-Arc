import os
import sys
import gc
import shutil
import zipfile
import h5py
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from config import PipelineConfig
from models import build_model

def load_patient_volume(filepath: str) -> torch.Tensor:
    with h5py.File(filepath, "r") as f:
        volume = f["image"][:].astype(np.float32)
        
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = np.transpose(volume, (3, 0, 1, 2))
    elif volume.ndim == 3:
        volume = np.expand_dims(volume, axis=0)
        
    volume = np.clip(volume, a_min=-150.0, a_max=250.0)
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    if std_val > 1e-5:
        volume = (volume - mean_val) / std_val
    else:
        volume = volume - mean_val
        
    return torch.tensor(volume, dtype=torch.float32).unsqueeze(0) # [1, 1, D, H, W]

# DeLong test functions
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

def main():
    torch.set_num_threads(8)
    device = torch.device("cpu")
    print(f"Using device: {device} with {torch.get_num_threads()} CPU threads", flush=True)
    
    config = PipelineConfig()
    
    # 1. Load test dataframe
    test_df = pd.read_csv("datas/external_test_set.csv")
    
    print("Pre-loading & pre-processing 24 test CT volumes into memory...", flush=True)
    cached_images = []
    pids = []
    labels = []
    
    for _, row in test_df.iterrows():
        pid = str(row['patient_id'])
        label = int(row['label'])
        folder = config.mucinous_v03_dir if label == 1 else config.appendicitis_v03_dir
        path = os.path.join(folder, f"{pid}.h5")
        
        vol_tensor = load_patient_volume(path).to(device)
        cached_images.append(vol_tensor)
        pids.append(pid)
        labels.append(label)
        
    print(f"✓ Cached {len(cached_images)} patient volumes in RAM.", flush=True)
    
    class_names = ["Musinoz" if l == 1 else "Apandisit" for l in labels]
    master_df = pd.DataFrame({
        "patient_id": pids,
        "true_label": labels,
        "true_class": class_names
    })
    
    models_to_eval = [
        ("1_UNet_PlusPlus", "unet_plusplus", "checkpoints/unet_plusplus/deneme_1", 0.4),
        ("2_DenseNet121", "densenet121", "checkpoints/densenet121/deneme_2", 0.4),
        ("3_EfficientNet_B0", "efficientnet_b0", "checkpoints/efficientnet_b0/deneme_1", 0.3)
    ]
    
    base_dest = "external_test_outputs"
    os.makedirs(base_dest, exist_ok=True)
    model_ensemble_probs = {}
    
    for folder_name, m_type, src_dir, drop_rate in models_to_eval:
        clean_name = folder_name.split("_", 1)[1]
        print(f"\n--- Processing {clean_name} ({src_dir}) ---", flush=True)
        model_dest = os.path.join(base_dest, folder_name)
        os.makedirs(model_dest, exist_ok=True)
        
        # Copy ensemble, fold plots, gradcam, metrics
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
                
        # Inference across 5 folds
        cfg = PipelineConfig(model_name=m_type, dropout_rate=drop_rate, pretrained=False)
        all_fold_probs = []
        
        for fold in range(1, 6):
            model = build_model(cfg)
            model = model.to(device)
            ckpt_path = os.path.join(src_dir, f"fold_{fold}", f"best_{m_type}_auc_roc.pth")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            
            fold_probs = []
            with torch.no_grad():
                for img_t in cached_images:
                    out = model(img_t)
                    if isinstance(out, tuple):
                        out = out[0]
                    p = torch.softmax(out, dim=1)[:, 1].item()
                    fold_probs.append(p)
                    
            fold_probs = np.array(fold_probs)
            all_fold_probs.append(fold_probs)
            master_df[f"{clean_name}_Fold_{fold}_Prob"] = np.round(fold_probs, 6)
            
            del model
            gc.collect()
            
        ens_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
        master_df[f"{clean_name}_Ensemble_Prob"] = np.round(ens_probs, 6)
        model_ensemble_probs[clean_name] = ens_probs
        
        # Save individual model CSV & Excel inside model directory
        sub_cols = ["patient_id", "true_label", "true_class"] + [c for c in master_df.columns if clean_name in c]
        sub_df = master_df[sub_cols]
        sub_df.to_csv(os.path.join(model_dest, f"{clean_name.lower()}_test_probabilities.csv"), index=False)
        sub_df.to_excel(os.path.join(model_dest, f"{clean_name.lower()}_test_probabilities.xlsx"), index=False)
        print(f"✓ {clean_name} Done! Ensemble AUC = {roc_auc_score(labels, ens_probs):.4f}", flush=True)
        
    print("\n--- 3. SAVING MASTER CSV & EXCEL FILES ---", flush=True)
    # Master Probabilities
    csv_master = os.path.join(base_dest, "test_predictions_probabilities_all_models.csv")
    xlsx_master = os.path.join(base_dest, "test_predictions_probabilities_all_models.xlsx")
    master_df.to_csv(csv_master, index=False)
    master_df.to_excel(xlsx_master, index=False)
    print(f"✓ Master Probabilities CSV: {csv_master}", flush=True)
    print(f"✓ Master Probabilities Excel: {xlsx_master}", flush=True)
    
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
    print(f"✓ ROC Curve Coordinates: {roc_coords_csv}", flush=True)
    
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
    print(f"✓ PR Curve Coordinates: {pr_coords_csv}", flush=True)
    
    # Compute DeLong Statistical Tests
    print("\n--- 4. COMPUTING DELONG STATISTICAL TESTS ---", flush=True)
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
        
    print("\n--- 5. CREATING FINAL ZIP ARCHIVE ---", flush=True)
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
