import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("final"))

from shared_utils import (
    SHARED_CONFIG, AppendixH5Dataset, evaluate_model, compute_calibration
)

import final.train_swinunetr_linearprobe as swin_lp
import final.train_attention_swinunetr as attn_swin
import final.train_mae_tinytransformer as mae_tiny
import final.train_segformer3d as segformer3d

EXP_DIR = Path("experiments_q1_128")
TEST_CSV = "datas/external_test_set.csv"

MODELS = [
    ("SwinUNETR-LP", "swinunetr_lp", swin_lp.build_model),
    ("AG-MSF", "attention_swinunetr", attn_swin.build_model),
    ("MAE-Tiny3D", "mae_tinytransformer", mae_tiny.build_model),
    ("SegFormer3D-MSCA", "segformer3d", segformer3d.build_model)
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_df = pd.read_csv(TEST_CSV)
    test_ds = AppendixH5Dataset(test_df, augment=False, config=SHARED_CONFIG)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True
    )
    criterion = torch.nn.CrossEntropyLoss()
    y_true = test_df["label"].values

    results = []

    for model_display, dir_name, builder in MODELS:
        print(f"\nProcessing {model_display}...")
        base_dir = EXP_DIR / dir_name
        
        for fold in range(1, 6):
            ckpt_path = base_dir / f"fold_{fold:02d}" / "best_model.pt"
            if not ckpt_path.exists():
                print(f"  [MISSING] {ckpt_path}")
                continue
                
            model = builder().to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            model.eval()
            
            with torch.no_grad():
                _, _, _, _, pred_df = evaluate_model(model, test_loader, criterion, device)
            
            y_prob = pred_df["prob_mucinous"].values
            cal = compute_calibration(y_true, y_prob)
            brier = cal["brier_score"]
            ece = cal["ece"]
            
            print(f"  Fold {fold}: Brier={brier:.3f}, ECE={ece:.3f}")
            results.append({
                "model": model_display,
                "fold": fold,
                "brier": brier,
                "ece": ece
            })
            del model
            torch.cuda.empty_cache()

    df_res = pd.DataFrame(results)
    df_res.to_csv("fold_calibration_results.csv", index=False)
    print("\n[DONE] Saved to fold_calibration_results.csv")

if __name__ == "__main__":
    main()
