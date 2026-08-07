import pandas as pd
from pathlib import Path
EXP_DIR = Path("experiments_multirun")
name = "swinunetr_lp"

probs_path = EXP_DIR / name / "run_01" / "external_test" / "ensemble_probs.csv"
if probs_path.exists():
    df = pd.read_csv(probs_path)
    print(df.head())
