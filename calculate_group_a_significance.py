import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

EXPERIMENTS_DIR = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/experiments_q1_128")

def single_model_vs_chance_hm(y_true, y_prob):
    # Hanley-McNeil approach for testing AUC vs 0.5 (Null Hypothesis Variance)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    
    n1 = np.sum(y_true == 1)
    n2 = np.sum(y_true == 0)
    
    # Calculate AUC using Mann-Whitney U statistic formulation for exactness
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    u_stat = sum([sum(p > neg) + 0.5 * sum(p == neg) for p in pos])
    auc = u_stat / (n1 * n2)
    
    # Variance under the null hypothesis H0: AUC = 0.5
    var_null = (n1 + n2 + 1) / (12 * n1 * n2)
    se_null = np.sqrt(var_null)
    
    z = (auc - 0.50) / se_null
    p = 2 * (1 - norm.cdf(abs(z)))
    
    return float(auc), float(z), float(p)

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
            
    print("Single Model Significance against AUC=0.50 (Hanley-McNeil Null Variance):")
    print("="*75)
    for model_name, probs_path in found.items():
        df = pd.read_csv(probs_path)
        y_true = df["label"].values
        y_prob = df["prob_mucinous"].values
        
        auc, z, p = single_model_vs_chance_hm(y_true, y_prob)
        print(f"{model_name:25s} | AUC: {auc:.3f} | z={z:.3f}, p={p:.3e}")

if __name__ == '__main__':
    main()
