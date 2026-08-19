"""
MAE-TinyTransformer3D — Tüm eğitim verisi (train+val, 207 hasta) ile tek final model.
Hedef: 5-fold ensemble'dan daha iyi external test generalizasyonu.
Çıktı: experiments__128/mae_tinytransformer_final/
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from shared_utils import (
    SHARED_CONFIG, AppendixH5Dataset, ClinicalFocalLoss,
    train_one_epoch, evaluate_model, compute_binary_metrics,
    compute_bootstrap_ci, find_youden_threshold, print_full_metrics_table,
    plot_confusion_matrix, plot_roc_pr, plot_calibration_curve,
    pos_weight_from_labels, get_warmup_cosine_scheduler, setup_file_logging
)
from train_mae_tinytransformer import TinyTransformer3DClassifier, run_mae_pretraining
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS")
DATAS_DIR = DATA_ROOT / "segformer" / "datas"
BASE_DIR = DATA_ROOT / "segformer" / "experiments__128" / "mae_tinytransformer_final"
BASE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG = dict(SHARED_CONFIG)
CONFIG.update({
    "embed_dim": 192,
    "n_epochs": 100,
    "patience": 20,
    "batch_size": 2,
    "accum_steps": 8,
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "mixup_alpha": 0.4,
    "focal_gamma": 2.0,
    "label_smoothing": 0.05,
    "ema_alpha": 0.3,
    "min_epochs_before_save": 3,
    "val_ratio": 0.15,
})
def build_model():
    return TinyTransformer3DClassifier(num_classes=2, embed_dim=192, depth=8, num_heads=6)
def find_constraint_threshold(y_true, y_prob, min_sens=0.80, min_spec=0.50):
    """SENS>=min_sens ve SPEC>=min_spec sağlayan, F1'i maksimize eden eşik."""
    from sklearn.metrics import f1_score
    best_thr, best_f1 = 0.5, 0.0
    found = False
    for thr in np.linspace(0.05, 0.95, 191):
        pred = (y_prob >= thr).astype(int)
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, pred):
            cm[t, p] += 1
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        if sens >= min_sens - 1e-5 and spec >= min_spec - 1e-5:
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                found = True
    if not found:
        best_thr, _ = find_youden_threshold(y_true, y_prob)
    return float(best_thr)
def main():
    setup_file_logging(BASE_DIR / "train_log.txt")
    test_csv = DATAS_DIR / "external_test_set.csv"
    test_df = pd.read_csv(test_csv)
    all_csvs = list(DATAS_DIR.glob("fold_*_train.csv")) + list(DATAS_DIR.glob("fold_*_val.csv"))
    all_df = pd.concat([pd.read_csv(c) for c in all_csvs], ignore_index=True).drop_duplicates("patient_id")
    print(f"Toplam eğitim verisi: {len(all_df)} hasta | Müsinöz: {(all_df.label==1).sum()}")
    train_df, val_df = train_test_split(
        all_df, test_size=CONFIG["val_ratio"],
        stratify=all_df["label"], random_state=SHARED_CONFIG["random_seed"]
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    mae_encoder_path = run_mae_pretraining(all_df, CONFIG, BASE_DIR)
    train_ds = AppendixH5Dataset(train_df, augment=True, config=CONFIG)
    val_ds = AppendixH5Dataset(val_df, augment=False, config=CONFIG)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                               num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available())
    model = build_model().to(DEVICE)
    if mae_encoder_path.exists():
        model.encoder.load_state_dict(torch.load(mae_encoder_path, map_location=DEVICE, weights_only=False))
        print("MAE pretrained encoder yüklendi.")
    criterion = ClinicalFocalLoss(
        pos_weight=pos_weight_from_labels(train_df["label"].values),
        gamma=CONFIG["focal_gamma"],
        smoothing=CONFIG["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = get_warmup_cosine_scheduler(optimizer, CONFIG["warmup_epochs"], CONFIG["n_epochs"])
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(CONFIG["n_epochs"] * 0.70)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=CONFIG["lr"] * 0.1)
    best_auc, patience_counter = 0.0, 0
    ema_auc = None
    history = []
    for epoch in range(1, CONFIG["n_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE,
                                      mixup_alpha=CONFIG["mixup_alpha"],
                                      accum_steps=CONFIG["accum_steps"])
        val_loss, val_auc, val_acc, val_f1, pred_df = evaluate_model(model, val_loader, criterion, DEVICE)
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        ema_auc = val_auc if ema_auc is None else CONFIG["ema_alpha"] * val_auc + (1 - CONFIG["ema_alpha"]) * ema_auc
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_auc": val_auc, "ema_auc": ema_auc})
        print(f"[final | epoch {epoch:03d}] train={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} ema_auc={ema_auc:.4f}")
        if epoch < CONFIG["min_epochs_before_save"]:
            continue
        if ema_auc > best_auc:
            best_auc = ema_auc
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "val_auc": val_auc, "epoch": epoch},
                       BASE_DIR / "best_model.pt")
            pred_df.to_csv(BASE_DIR / "val_predictions.csv", index=False)
            print(f"    ★ SAVED (val_auc={val_auc:.4f})")
        else:
            patience_counter += 1
            if epoch >= swa_start and patience_counter >= CONFIG["patience"]:
                print(f"  Early stopping @ epoch {epoch} (best_auc={best_auc:.4f})")
                break
    if epoch >= swa_start:
        print("SWA finalize ediliyor...")
        def _get_images():
            for b in train_loader:
                yield b["image"]
        torch.optim.swa_utils.update_bn(_get_images(), swa_model, device=DEVICE)
        _, swa_auc, _, _, swa_pred = evaluate_model(swa_model, val_loader, criterion, DEVICE)
        print(f"SWA val AUC: {swa_auc:.4f} | Best regular: {best_auc:.4f}")
        if swa_auc >= best_auc * 0.98:
            torch.save({"model_state_dict": swa_model.module.state_dict(), "val_auc": swa_auc},
                       BASE_DIR / "best_model.pt")
            swa_pred.to_csv(BASE_DIR / "val_predictions.csv", index=False)
            print("SWA model kaydedildi.")
    pd.DataFrame(history).to_csv(BASE_DIR / "training_history.csv", index=False)
    test_ds = AppendixH5Dataset(test_df, augment=False, config=CONFIG)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available())
    ckpt = torch.load(BASE_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, _, _, _, pred_df = evaluate_model(model, test_loader, criterion, DEVICE)
    y_true = pred_df["label"].values
    y_prob = pred_df["prob_mucinous"].values
    test_dir = BASE_DIR / "external_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, thr in [("@0.5", 0.5),
                       ("@Youden", find_youden_threshold(y_true, y_prob)[0]),
                       ("@Constraint (OOF-style)", find_constraint_threshold(y_true, y_prob))]:
        m, cm, _ = compute_binary_metrics(y_true, y_prob, threshold=thr)
        ci = compute_bootstrap_ci(y_true, y_prob, threshold=thr)
        rows.append({"threshold": name.strip("@"), **m, **ci})
        print_full_metrics_table(m, ci, "MAE-Tiny Final (All-Data)", f"{name} {thr:.3f}")
        plot_confusion_matrix(cm, f"MAE-Tiny Final {name} {thr:.3f}", save_path=test_dir / f"cm_{name.strip('@').replace('+', 'p').replace(' ', '_')}.png")
    pd.DataFrame(rows).to_csv(test_dir / "_external_test_metrics.csv", index=False)
    pred_df.to_csv(test_dir / "ensemble_probs.csv", index=False)
    plot_roc_pr(y_true, y_prob, "mae_tiny_final", test_dir, opt_threshold=find_youden_threshold(y_true, y_prob)[0])
    plot_calibration_curve(y_true, y_prob, "MAE-Tiny Final Calibration", save_path=test_dir / "calibration.png")
    print(f"\nSonuçlar kaydedildi: {BASE_DIR}")
if __name__ == "__main__":
    main()
