# Test: doğru prob + threshold kombinasyonu gösteriyor mu?
import pandas as pd
from pathlib import Path

EXP_DIR = Path("experiments_multirun")
MODEL_DIRS = {
    "SwinUNETR-LP": "swinunetr_lp",
    "AG-MSF": "ag_msf",
    "MAE-Tiny3D": "mae_tiny3d",
    "SegFormer3D": "segformer3d_msca",
}

for m_name, folder in MODEL_DIRS.items():
    probs_path = EXP_DIR / folder / "run_01" / "external_test" / "ensemble_probs.csv"
    metrics_path = EXP_DIR / folder / "run_01" / "external_test" / "q1_external_test_metrics.csv"
    
    df_p = pd.read_csv(probs_path)
    df_m = pd.read_csv(metrics_path)
    yrow = df_m[df_m["fold"].astype(str).str.contains("Youden", na=False)]
    threshold = float(yrow["threshold"].iloc[0])
    
    df_p["pred"] = (df_p["prob_mucinous"] >= threshold).map({True: "Mucinous", False: "Appendicitis"})
    df_p["true"] = df_p["label"].map({1: "Mucinous", 0: "Appendicitis"})
    df_p["correct"] = df_p["pred"] == df_p["true"]
    
    print(f"\n{m_name} (thr={threshold:.2f}):")
    print(df_p[df_p["true"]=="Appendicitis"][["patient_id","prob_mucinous","pred","correct"]].to_string(index=False))
