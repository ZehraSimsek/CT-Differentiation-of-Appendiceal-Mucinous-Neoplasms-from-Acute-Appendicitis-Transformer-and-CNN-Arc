import re

file_path = "generate_multirun_gradcam.py"
with open(file_path, "r") as f:
    content = f.read()

# Modify load_model to also load patient predictions
load_model_new = """
def load_model(name, builder, wrapper_cls, device):
    model = builder().to(device)
    ckpt_path = EXP_DIR / name / "run_01" / "fold_01" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"HATA: {ckpt_path} bulunamadı!")
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    
    threshold = 0.5
    csv_path = EXP_DIR / name / "run_01" / "external_test" / "q1_external_test_metrics.csv"
    if csv_path.exists():
        df_val = pd.read_csv(csv_path)
        row = df_val[df_val["fold"].astype(str).str.contains("Ensemble \(@Youden\)", na=False)]
        if not row.empty:
            threshold = float(row["threshold"].iloc[0])

    # Load patient predictions from ensemble_probs.csv
    patient_probs = {}
    probs_path = EXP_DIR / name / "run_01" / "external_test" / "ensemble_probs.csv"
    if probs_path.exists():
        df_probs = pd.read_csv(probs_path)
        for _, r in df_probs.iterrows():
            patient_probs[r["patient_id"]] = float(r["prob_mucinous"])

    wrapper = wrapper_cls(model)
    wrapper.threshold = threshold
    wrapper.patient_probs = patient_probs
    return wrapper
"""
content = re.sub(r'def load_model\(.*?:.*?return wrapper', load_model_new.strip(), content, flags=re.DOTALL)

# Modify the loop to use wrapper.patient_probs
loop_old = """
                    prob = c_data["prob"]
                    cam_slice = cam_3d[slice_idx]
                    
                    threshold = extractor.threshold
                    pred_str = "Mucinous" if prob >= threshold else "Appendicitis"
"""

loop_new = """
                    # GERÇEK ENSEMBLE TAHMİNİ KULLANILACAK (Grad-CAM'den gelen prob yerine)
                    model_probs = extractor.patient_probs
                    if pid in model_probs:
                        prob = model_probs[pid]
                    else:
                        prob = c_data["prob"] # fallback
                        
                    cam_slice = cam_3d[slice_idx]
                    
                    threshold = extractor.threshold
                    pred_str = "Mucinous" if prob >= threshold else "Appendicitis"
"""
content = content.replace(loop_old.strip(), loop_new.strip())

with open(file_path, "w") as f:
    f.write(content)
