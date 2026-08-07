import pandas as pd
from pathlib import Path

EXP_DIR = Path("experiments_multirun")
for name in ["swinunetr_lp", "ag_msf", "mae_tiny3d", "segformer3d_msca"]:
    csv_path = EXP_DIR / name / "run_01" / "fold_01" / "best_val_predictions.csv"
    if csv_path.exists():
        df_val = pd.read_csv(csv_path)
        if "youden_threshold" in df_val.columns:
            threshold = float(df_val["youden_threshold"].iloc[0])
            print(f"{name}: youden_threshold = {threshold}")
        elif "_threshold_used" in df_val.columns:
            threshold = float(df_val["_threshold_used"].iloc[0])
            print(f"{name}: _threshold_used = {threshold}")
        else:
            print(f"{name}: NO THRESHOLD COLUMN FOUND in {df_val.columns}")
    else:
        print(f"{name}: PATH NOT FOUND {csv_path}")
