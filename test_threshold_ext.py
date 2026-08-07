import pandas as pd
from pathlib import Path

EXP_DIR = Path("experiments_multirun")
for name in ["swinunetr_lp", "ag_msf", "mae_tiny3d", "segformer3d_msca"]:
    csv_path = EXP_DIR / name / "run_01" / "external_test" / "q1_external_test_metrics.csv"
    if csv_path.exists():
        df_val = pd.read_csv(csv_path)
        row = df_val[df_val["fold"] == "Ensemble (@Youden)"]
        if not row.empty:
            threshold = float(row["threshold"].iloc[0])
            print(f"{name} (External Ensemble Youden): {threshold}")
        else:
            print(f"{name}: NO Ensemble ROW")
    else:
        print(f"{name}: PATH NOT FOUND {csv_path}")
