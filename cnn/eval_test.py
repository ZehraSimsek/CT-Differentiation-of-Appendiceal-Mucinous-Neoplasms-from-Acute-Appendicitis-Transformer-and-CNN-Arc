import os
import sys
import torch
import pandas as pd
from models import build_model
import h5py
import numpy as np
import warnings
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv("../datas_2d/external_test_set.csv")
models = {}
models_config = [
    {"name": "unet_plusplus", "run": "run_01"},
    {"name": "densenet121", "run": "run_01"},
    {"name": "efficientnet_b0", "run": "run_01"}
]
for cfg in models_config:
    m = cfg["name"]
    run = cfg["run"]
    class Args: pass
    args = Args()
    args.model_name = m
    args.in_channels = 1
    args.num_classes = 2
    args.pretrained = False
    args.dropout_rate = 0.0
    mdl = build_model(args).to(device)
    ckpt = torch.load(f"experiments__128/{m}/{run}/fold_01/best_model.pth", map_location=device)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.eval()
    models[m] = mdl
correct_counts = {}
for idx, row in df.iterrows():
    pid = row['patient_id']
    label = row['label']
    h5_path = row['h5_path'].replace("../datas/", "../datas_2d/")
    if not os.path.exists(h5_path):
        continue
    with h5py.File(h5_path, 'r') as f:
        img = f['image'][:]
    img = torch.tensor(img, dtype=torch.float32).squeeze(-1).unsqueeze(0).unsqueeze(0).to(device) 
    preds = {}
    with torch.no_grad():
        for m_name, mdl in models.items():
            out = mdl(img)
            pred_cls = out.argmax(dim=1).item()
            preds[m_name] = pred_cls
    print(f"{pid} (Label: {label}): {preds}")
