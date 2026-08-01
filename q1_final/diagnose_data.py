import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

from shared_utils import AppendixH5Dataset, SHARED_CONFIG, compute_binary_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATAS_DIR = os.path.abspath("../datas")

def build_simple_cnn():
    return nn.Sequential(
        nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
        nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
        nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
        nn.Flatten(), nn.Linear(64, 2)
    ).to(DEVICE)

def eval_model(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(DEVICE)
            y = b["label"].to(DEVICE)
            p = torch.softmax(model(x), 1)[:, 1]
            ys.extend(y.cpu().numpy()); ps.extend(p.cpu().numpy())
    return np.array(ys), np.array(ps)

if __name__ == "__main__":
    cfg = dict(SHARED_CONFIG)
    cfg["batch_size"] = 4
    tr = pd.read_csv(os.path.join(DATAS_DIR, "fold_1_train.csv"))
    val = pd.read_csv(os.path.join(DATAS_DIR, "fold_1_val.csv"))
    print(f"Train: {len(tr)} | Val: {len(val)}")

    tr_ds = AppendixH5Dataset(tr, augment=False, config=cfg)
    val_ds = AppendixH5Dataset(val, augment=False, config=cfg)
    tr_loader = DataLoader(tr_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    x0, y0 = tr_ds[0]["image"], tr_ds[0]["label"]
    print(f"Sample shape: {x0.shape}, label: {y0.item()}, min/max/mean: {x0.min():.3f}/{x0.max():.3f}/{x0.mean():.3f}")

    model = build_simple_cnn()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, 11):
        model.train()
        total_loss = 0
        for b in tr_loader:
            x = b["image"].to(DEVICE); y = b["label"].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            total_loss += loss.item()
        yt, yp = eval_model(model, val_loader)
        auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float('nan')
        print(f"Epoch {ep:02d} | train_loss={total_loss/len(tr_loader):.4f} | val_auc={auc:.4f}")
