import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE_DIR = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS")
MUSINOZ_DIR = BASE_DIR / "musinoz_2d" / "v02_slice_bbox_resize_128_D32"
APANDISIT_DIR = BASE_DIR / "apandisit_2d" / "v02_slice_bbox_resize_128_D32"

OUT_DIR = BASE_DIR / "segformer" / "datas_2d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = []

# Musinoz files
for f in MUSINOZ_DIR.glob("*.h5"):
    pid = f.stem
    data.append({
        "patient_id": pid,
        "h5_path": str(f),
        "label": 1,
        "label_name": "Mucinous"
    })

# Apandisit files
for f in APANDISIT_DIR.glob("*.h5"):
    pid = f.stem
    data.append({
        "patient_id": pid,
        "h5_path": str(f),
        "label": 0,
        "label_name": "Appendicitis"
    })

df = pd.DataFrame(data)

import re
from sklearn.model_selection import StratifiedGroupKFold

# Extract base patient ID to prevent leakage (e.g. necati_ala and necati_ala_2 belong to same group)
def get_base_id(pid):
    return re.sub(r'_[0-9]+$', '', pid)

df["group_id"] = df["patient_id"].apply(get_base_id)

# Split out ~16.6% for External Test Set using StratifiedGroupKFold to ensure CLASS BALANCE!
sgkf_test = StratifiedGroupKFold(n_splits=6)
train_val_idx, test_idx = next(sgkf_test.split(df, df["label"], groups=df["group_id"]))

train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# Save test set
test_df.drop(columns=["group_id"]).to_csv(OUT_DIR / "external_test_set.csv", index=False)
print(f"External test set saved: {len(test_df)} patients.")
print(f"Test Set Class Counts:\n{test_df['label_name'].value_counts()}")

# 5-Fold STRATIFIED Grouped Split on train_val_df
sgkf_val = StratifiedGroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(sgkf_val.split(train_val_df, train_val_df["label"], groups=train_val_df["group_id"]), 1):
    train_fold = train_val_df.iloc[train_idx]
    val_fold = train_val_df.iloc[val_idx]
    
    train_fold.drop(columns=["group_id"]).to_csv(OUT_DIR / f"fold_{fold}_train.csv", index=False)
    val_fold.drop(columns=["group_id"]).to_csv(OUT_DIR / f"fold_{fold}_val.csv", index=False)
    
    print(f"Fold {fold} - Train: {len(train_fold)}, Val: {len(val_fold)}")

print(f"All CSVs created in {OUT_DIR} with STRICT Patient-Level Leakage Prevention")
