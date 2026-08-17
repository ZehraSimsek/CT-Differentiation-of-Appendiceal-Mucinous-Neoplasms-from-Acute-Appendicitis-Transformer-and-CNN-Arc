import os
import sys
import gc
import shutil
import zipfile
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve

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

def run_evaluation_for_model(model_name: str, ckpt_dir: str, test_loader: DataLoader, device: torch.device, dropout: float = 0.3):
    cfg = PipelineConfig(model_name=model_name, dropout_rate=dropout, pretrained=False)
    all_fold_probs = []
    y_true_test = []
    
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
        fold_y = []
        with torch.no_grad():
            for volumes, labels in test_loader:
                volumes = volumes.to(device)
                outputs = model(volumes)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                fold_probs.extend(probs)
                fold_y.extend(labels.numpy())
                
        all_fold_probs.append(np.array(fold_probs))
        y_true_test = fold_y
        del model
        gc.collect()
        
    ensemble_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
    return y_true_test, all_fold_probs, ensemble_probs

def main():
    print("=== STARTING FULL EXTRACTION & PACKAGING ===", flush=True)
    device = torch.device("cpu")
    config = PipelineConfig()
    
    base_dest = "external_test_outputs"
    if os.path.exists(base_dest):
        shutil.rmtree(base_dest)
    os.makedirs(base_dest, exist_ok=True)
    
    # 1. Load test data
    test_df = pd.read_csv("datas/external_test_set.csv")
    test_df = append_h5_paths(test_df, config)
    
    test_ds = PatientCTDataset(df=test_df, augment_train=False, config=config)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    
    pids = test_df['patient_id'].astype(str).tolist()
    labels = test_df['label'].astype(int).tolist()
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
    
    model_ensemble_probs = {}
    
    # 2. Copy artifacts & Extract probabilities
    for folder_name, m_type, src_dir, drop_rate in models_to_eval:
        print(f"\n--- Processing {folder_name} ({m_type}) ---", flush=True)
        model_dest = os.path.join(base_dest, folder_name)
        os.makedirs(model_dest, exist_ok=True)
        
        # Copy Ensemble folder
        ens_src = os.path.join(src_dir, "ensemble")
        if os.path.exists(ens_src):
            shutil.copytree(ens_src, os.path.join(model_dest, "Ensemble"))
            print("  ✓ Copied Ensemble plots", flush=True)
            
        # Copy Fold-based folders
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
        print("  ✓ Copied 5-Fold plots", flush=True)
        
        # Copy GradCAM XAI
        xai_src = os.path.join(src_dir, "external_test_xai")
        if os.path.exists(xai_src):
            shutil.copytree(xai_src, os.path.join(model_dest, "GradCAM_XAI"))
            print("  ✓ Copied Grad-CAM heatmaps", flush=True)
            
        # Copy Metrics summaries
        for meta in ["metrics_summary.csv", "metrics_summary.xlsx"]:
            mp = os.path.join(src_dir, meta)
            if os.path.exists(mp):
                shutil.copy2(mp, os.path.join(model_dest, meta))
        print("  ✓ Copied metrics summaries", flush=True)
        
        # Extract predictions
        print("  -> Calculating predictions for test set...", flush=True)
        y_t, fold_probs_list, ens_probs = run_evaluation_for_model(m_type, src_dir, test_loader, device, drop_rate)
        
        clean_name = folder_name.split("_", 1)[1]
        for fold_idx, f_prob in enumerate(fold_probs_list, start=1):
            master_df[f"{clean_name}_Fold_{fold_idx}_Prob"] = np.round(f_prob, 6)
            
        master_df[f"{clean_name}_Ensemble_Prob"] = np.round(ens_probs, 6)
        model_ensemble_probs[clean_name] = ens_probs
        
        # Save individual model CSV & Excel
        sub_cols = ["patient_id", "true_label", "true_class"] + [c for c in master_df.columns if clean_name in c]
        sub_csv_path = os.path.join(model_dest, f"{clean_name.lower()}_test_probabilities.csv")
        sub_xlsx_path = os.path.join(model_dest, f"{clean_name.lower()}_test_probabilities.xlsx")
        master_df[sub_cols].to_csv(sub_csv_path, index=False)
        master_df[sub_cols].to_excel(sub_xlsx_path, index=False)
        print(f"  ✓ Saved {sub_csv_path}", flush=True)
        
    # 3. Save Master probabilities files (CSV & Excel)
    print("\n--- Saving Master Files ---", flush=True)
    master_csv = os.path.join(base_dest, "test_predictions_probabilities_all_models.csv")
    master_xlsx = os.path.join(base_dest, "test_predictions_probabilities_all_models.xlsx")
    master_df.to_csv(master_csv, index=False)
    master_df.to_excel(master_xlsx, index=False)
    print(f"✓ Master Probabilities CSV: {master_csv}", flush=True)
    print(f"✓ Master Probabilities Excel: {master_xlsx}", flush=True)
    
    # 4. Save ROC Coordinates CSV & Excel
    roc_rows = []
    for model_name, ens_probs in model_ensemble_probs.items():
        fpr, tpr, thresholds = roc_curve(labels, ens_probs)
        for f, t, th in zip(fpr, tpr, thresholds):
            roc_rows.append({
                "model": model_name,
                "fpr_1_minus_specificity": round(f, 6),
                "tpr_sensitivity": round(t, 6),
                "threshold": round(th, 6)
            })
    roc_coords_df = pd.DataFrame(roc_rows)
    roc_csv = os.path.join(base_dest, "roc_curve_coordinates.csv")
    roc_xlsx = os.path.join(base_dest, "roc_curve_coordinates.xlsx")
    roc_coords_df.to_csv(roc_csv, index=False)
    roc_coords_df.to_excel(roc_xlsx, index=False)
    print(f"✓ ROC Curve Coordinates CSV: {roc_csv}", flush=True)
    
    # 5. Save PR Coordinates CSV & Excel
    pr_rows = []
    for model_name, ens_probs in model_ensemble_probs.items():
        precision, recall, thresholds = precision_recall_curve(labels, ens_probs)
        for p, r in zip(precision, recall):
            pr_rows.append({
                "model": model_name,
                "recall_sensitivity": round(r, 6),
                "precision": round(p, 6)
            })
    pr_coords_df = pd.DataFrame(pr_rows)
    pr_csv = os.path.join(base_dest, "pr_curve_coordinates.csv")
    pr_xlsx = os.path.join(base_dest, "pr_curve_coordinates.xlsx")
    pr_coords_df.to_csv(pr_csv, index=False)
    pr_coords_df.to_excel(pr_xlsx, index=False)
    print(f"✓ PR Curve Coordinates CSV: {pr_csv}", flush=True)
    
    # 6. Create ZIP archive
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
    print(f"✓ SUCCESS! Created {zip_filename} ({zip_size_mb:.2f} MB)")
    print(f"✓ Total files packaged: {file_count}")
    print(f"============================================================", flush=True)

if __name__ == "__main__":
    main()
