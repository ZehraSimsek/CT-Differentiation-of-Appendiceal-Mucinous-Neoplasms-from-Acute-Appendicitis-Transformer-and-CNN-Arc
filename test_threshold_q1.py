import pandas as pd
from pathlib import Path

EXP_DIR = Path("final/experiments_q1_128")
if not EXP_DIR.exists():
    EXP_DIR = Path("experiments_q1_128")
    
for name in ["swinunetr_lp", "attention_swinunetr", "mae_tinytransformer", "segformer3d"]:
    csv_path = EXP_DIR / name / "external_test" / "q1_external_test_metrics.csv"
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
