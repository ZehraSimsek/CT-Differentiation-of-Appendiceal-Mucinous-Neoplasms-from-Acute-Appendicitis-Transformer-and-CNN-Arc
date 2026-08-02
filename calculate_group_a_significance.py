import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import math

# Add the directory to the path to import shared_utils
sys.path.insert(0, "/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer")
from shared_utils import _fast_delong, _norm_cdf

EXPERIMENTS_DIR = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/experiments_q1_128")

def single_model_delong_test_vs_chance(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    label_1_count = int(y_true_sorted.sum())
    
    # We pass the predictions as a 1xN array
    preds = np.asarray(y_prob)[order].reshape(1, -1)
    aucs, delongcov = _fast_delong(preds, label_1_count)
    
    auc = aucs[0]
    var = delongcov if np.isscalar(delongcov) or delongcov.ndim == 0 else delongcov[0, 0]
    
    if var <= 0:
        return float(auc), 0.0, 1.0
        
    z = (auc - 0.50) / np.sqrt(var)
    p = float(2 * (1 - _norm_cdf(abs(z))))
    
    return float(auc), float(z), p

def main():
    found = {}
    if not EXPERIMENTS_DIR.exists():
        print(f"Dir not found: {EXPERIMENTS_DIR}")
        return
    for model_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        probs_path = model_dir / "external_test" / "ensemble_probs.csv"
        if probs_path.exists():
            found[model_dir.name] = probs_path
            
    print("Single Model Significance against AUC=0.50:")
    print("="*60)
    for model_name, probs_path in found.items():
        df = pd.read_csv(probs_path)
        y_true = df["label"].values
        y_prob = df["prob_mucinous"].values
        
        auc, z, p = single_model_delong_test_vs_chance(y_true, y_prob)
        print(f"{model_name:25s} | AUC: {auc:.3f} | z={z:.3f}, p={p:.3e}")

if __name__ == '__main__':
    main()
