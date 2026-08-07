import numpy as np
from scipy.stats import norm, mannwhitneyu
import pandas as pd
from pathlib import Path

EXPERIMENTS_DIR = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/experiments_q1_128")

print("Exact Mann-Whitney U test vs Chance:")
for model_dir in sorted(EXPERIMENTS_DIR.iterdir()):
    if not model_dir.is_dir(): continue
    probs_path = model_dir / "external_test" / "ensemble_probs.csv"
    if not probs_path.exists(): continue
    
    df = pd.read_csv(probs_path)
    y_true = df["label"].values
    y_prob = df["prob_mucinous"].values
    
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    
    # scipy mannwhitneyu
    res = mannwhitneyu(pos, neg, alternative='two-sided')
    
    # manual AUC and Hanley-McNeil Z
    auc = res.statistic / (len(pos) * len(neg))
    n1, n2 = len(pos), len(neg)
    var_null = (n1 + n2 + 1) / (12 * n1 * n2)
    se_null = np.sqrt(var_null)
    z_hm = (auc - 0.5) / se_null
    p_hm = 2 * (1 - norm.cdf(abs(z_hm)))
    
    print(f"{model_dir.name:20s} | AUC: {auc:.3f} | MWU p-val: {res.pvalue:.4f} | Z: {z_hm:.3f}, p_HM: {p_hm:.4f}")
