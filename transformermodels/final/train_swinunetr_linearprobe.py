"""
SwinUNETR Linear Probe — Appendisit/Müsinöz Sınıflandırma
Çalıştırma:
    cd segformer && python train_swinunetr_linearprobe.py

Çıktılar:
    experiments/swinunetr_lp/fold_0X/best_model.pt
    experiments/swinunetr_lp/aggregate_oof/
    experiments/swinunetr_lp/external_test/external_test_metrics.csv
    experiments/swinunetr_lp/train_log.txt
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
from shared_utils import *  # noqa
from monai.networks.nets import SwinUNETR

torch.manual_seed(SHARED_CONFIG["random_seed"])
np.random.seed(SHARED_CONFIG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "swinunetr_lp"
DATA_ROOT  = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS")
DATAS_DIR  = DATA_ROOT / "segformer" / "datas"
BASE_DIR   = DATA_ROOT / "segformer" / "experiments_128" / MODEL_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = dict(SHARED_CONFIG)
CONFIG.update({
    "model_name":        MODEL_NAME,
    "feature_size":      48,
    "n_epochs":          100,
    "patience":          25,
    "lr":                1e-3,   # Backbone donduruldu → head hızlı öğrenebilir
    "weight_decay":      5e-3,
    "warmup_epochs":     5,
    "min_sens_floor":    0.80,
    "min_spec_floor":    0.50,
    "sensitivity_first": True,
    "focal_gamma":       1.0,
    "label_smoothing":   0.05,
    "batch_size":        2,
    "accum_steps":       8,
})


# ============================================================
# Model: SwinUNETR Linear Probe
# Backbone tamamen dondurulmuş → sadece ~50K param eğitiliyor
# Son 2 SwinViT katmanı kullanılıyor (en semantik özellikler)
# LayerNorm: batch_size=4'te BatchNorm'dan çok daha stabil
# ============================================================
class SwinLinearProbe(nn.Module):
    def __init__(self, num_classes=2, feature_size=48):
        super().__init__()
        self.backbone = SwinUNETR(
            in_channels=1, out_channels=14,
            feature_size=feature_size,
            use_checkpoint=True, spatial_dims=3,
        )
        # Backbone TAMAMEN donduruldu
        for p in self.backbone.parameters():
            p.requires_grad = False

        dim4 = feature_size * 16  # 768 — son SwinViT katmanı
        dim3 = feature_size * 8   # 384 — sondan bir önceki

        self.gap = nn.AdaptiveAvgPool3d(1)

        # Son 2 katman → projeksiyon → LayerNorm → Dropout → 2
        self.proj4 = nn.Linear(dim4, 64)
        self.proj3 = nn.Linear(dim3, 32)
        self.head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():  # Backbone gradyan yok
            hidden = self.backbone.swinViT(x, self.backbone.normalize)
        f3 = self.gap(hidden[3]).flatten(1)   # [B, 384]
        f4 = self.gap(hidden[4]).flatten(1)   # [B, 768]
        p3 = F.gelu(self.proj3(f3))           # [B, 32]
        p4 = F.gelu(self.proj4(f4))           # [B, 64]
        feat = torch.cat([p3, p4], dim=1)     # [B, 96]
        return self.head(feat)


def build_model():
    return SwinLinearProbe(feature_size=CONFIG["feature_size"])


def load_pretrained_swin(model):
    candidates = [
        os.path.expanduser("~/models/model_swinvit.pt"),
        str(DATA_ROOT / "segformer/model_swinvit.pt"),
    ]
    ckpt_path = next((p for p in candidates if Path(p).exists()), None)
    if not ckpt_path:
        print("  [WARN] Pretrained ağırlık bulunamadı. Random init ile devam.")
        return model

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    target = model.backbone.swinViT.state_dict()
    new_sd, loaded, skipped = {}, 0, 0
    for k, v in sd.items():
        k2 = k.replace("swinViT.", "").replace("module.", "")
        if k2 in target and target[k2].shape == v.shape:
            new_sd[k2] = v; loaded += 1
        else:
            skipped += 1
    model.backbone.swinViT.load_state_dict(new_sd, strict=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Pretrained: {loaded} katman yüklendi, {skipped} atlandı.")
    print(f"  Eğitilebilir: {trainable:,} / {total:,} param ({100*trainable/total:.1f}%)")
    return model


# ============================================================
# Tek fold eğitimi
# ============================================================
def run_one_fold(train_df, val_df, fold_idx, config, output_dir):
    from sklearn.metrics import f1_score as _f1

    def set_seed(s):
        torch.manual_seed(s); np.random.seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    set_seed(config["random_seed"] + fold_idx)

    fold_dir = Path(output_dir) / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AppendixH5Dataset(train_df, augment=True,  config=config)
    val_ds   = AppendixH5Dataset(val_df,   augment=False, config=config)

    labels = train_df["label"].values.astype(int)
    cc = np.bincount(labels)
    print(f"  [Fold {fold_idx}] class_counts={cc} | backbone dondurulmuş (~50K param)")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"],
                              pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False, num_workers=config["num_workers"],
                              pin_memory=torch.cuda.is_available())

    model = build_model().to(DEVICE)
    model = load_pretrained_swin(model)

    # Sadece head parametreleri
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = get_warmup_cosine_scheduler(optimizer, config["warmup_epochs"], config["n_epochs"])

    criterion = ClinicalFocalLoss(
        pos_weight=pos_weight_from_labels(train_df["label"].values),
        gamma=config.get("focal_gamma", 2.0),
        smoothing=config.get("label_smoothing", 0.05),
    )

    best_score, patience_cnt = -10.0, 0
    history = []

    for epoch in range(1, config["n_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, 
                                      accum_steps=config.get("accum_steps", 1))
        val_loss, val_auc, val_acc, _, pred_df = evaluate_model(
            model, val_loader, criterion, DEVICE)

        y_true = pred_df["label"].values
        y_prob = pred_df["prob_mucinous"].values
        threshold = float(pred_df["_threshold_used"].iloc[0]) \
            if "_threshold_used" in pred_df.columns else 0.5
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        val_f1_thr = float(_f1(y_true, y_pred, zero_division=0))
        composite  = clinical_composite(sens, val_auc, spec, f1=val_f1_thr)

        print(f"    [Prob] min={y_prob.min():.3f} max={y_prob.max():.3f} "
              f"| TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"[lp | fold {fold_idx} | epoch {epoch:03d}] "
              f"train={train_loss:.4f} val={val_loss:.4f} "
              f"auc={val_auc:.4f} thr={threshold:.3f}")
        print(f"  [Metrics] AUC:{val_auc:.3f} F1:{val_f1_thr:.3f} "
              f"SENS:{sens:.3f} SPEC:{spec:.3f} | COMPOSITE:{composite:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "auc": val_auc,
                        "sens": sens, "spec": spec, "f1": val_f1_thr,
                        "composite": composite})

        scheduler.step()

        # Çift kısıt model seçimi (Penalty-based)
        min_sens = config.get("min_sens_floor", 0.80)
        min_spec = config.get("min_spec_floor", 0.50)
        meets_constraints = (sens >= min_sens - 1e-5) and (spec >= min_spec - 1e-5)
        
        # Eğer kısıtlar sağlanırsa saf composite skoru kullan. Sağlanmazsa büyük bir ceza kes (-2.0).
        score_for_selection = composite if meets_constraints else (composite - 2.0)

        if score_for_selection > best_score:
            best_score = score_for_selection
            torch.save({"model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "composite": composite,
                        "epoch": epoch}, fold_dir / "best_model.pt")
            patience_cnt = 0
            if meets_constraints:
                print(f"    ★ SAVED DUAL-PASS (SENS={sens:.3f}≥{min_sens} "
                      f"SPEC={spec:.3f}≥{min_spec} | F1={val_f1_thr:.3f} | comp={composite:.4f})")
            else:
                print(f"    ☆ SAVED FALLBACK (SENS={sens:.3f} SPEC={spec:.3f} | comp={composite:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= config["patience"]:
                print(f"  Early stopping @ epoch {epoch} | best_score={best_score:.4f}")
                break

    import json
    with open(fold_dir / "history.json", "w") as fh:
        json.dump(history, fh)

    # Final evaluation
    ckpt = torch.load(fold_dir / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    _, _, _, _, pdf_f = evaluate_model(model, val_loader, criterion, DEVICE)
    youden_thr, _ = find_youden_threshold(pdf_f["label"].values, pdf_f["prob_mucinous"].values)
    ci_f = compute_bootstrap_ci(pdf_f["label"].values, pdf_f["prob_mucinous"].values, youden_thr)
    m_f, cm_f, _ = compute_binary_metrics(
        pdf_f["label"].values, pdf_f["prob_mucinous"].values, youden_thr)
    print_full_metrics_table(m_f, ci_f, f"SwinLP Fold {fold_idx}", f"Youden {youden_thr:.3f}")
    plot_confusion_matrix(cm_f, f"SwinLP Fold {fold_idx}",
                          save_path=fold_dir / f"cm_fold{fold_idx}.png")
    pdf_f["fold"] = fold_idx
    pdf_f["youden_threshold"] = youden_thr
    pdf_f.to_csv(fold_dir / "val_predictions.csv", index=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return m_f, ci_f, pdf_f, history


# ============================================================
# Main: 5-Fold + OOF + External Test
# ============================================================
def main():
    setup_file_logging(BASE_DIR / "train_log.txt")

    test_csv = DATAS_DIR / "external_test_set.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"external_test_set.csv bulunamadı: {test_csv}")

    print("=" * 80)
    print(f"SwinUNETR Linear Probe — {len(pd.read_csv(test_csv))} external test hasta")
    print(f"Backbone: DONDURULMUŞ | Head: ~50K param | LR: {CONFIG['lr']}")
    print("=" * 80)

    all_preds = []
    for fold_idx in range(1, CONFIG["n_splits"] + 1):
        train_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_train.csv")
        val_df   = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_val.csv")
        print(f"\n{'='*70}\nFOLD {fold_idx}/{CONFIG['n_splits']}\n{'='*70}")
        m, ci, pred, hist = run_one_fold(train_df, val_df, fold_idx, CONFIG, BASE_DIR)
        all_preds.append(pred)

    # Aggregate OOF
    oof = pd.concat(all_preds, ignore_index=True)
    yt, yp = oof["label"].values, oof["prob_mucinous"].values
    opt_thr, _ = find_youden_threshold(yt, yp)
    m_oof, cm_oof, _ = compute_binary_metrics(yt, yp, threshold=opt_thr)
    ci_oof = compute_bootstrap_ci(yt, yp, threshold=opt_thr)
    agg_dir = BASE_DIR / "aggregate_oof"
    agg_dir.mkdir(exist_ok=True)
    oof.to_csv(agg_dir / "oof_predictions.csv", index=False)
    plot_confusion_matrix(cm_oof, f"SwinLP OOF @Youden {opt_thr:.3f}",
                          save_path=agg_dir / "agg_cm_youden.png")
    plot_roc_pr(yt, yp, "swinlp_oof", agg_dir, opt_threshold=opt_thr)
    print("\nAGGREGATE OOF (5-Fold Cross-Validation):")
    print_full_metrics_table(m_oof, ci_oof, "SwinUNETR LinearProbe OOF", f"Youden {opt_thr:.3f}")

    # External test
    evaluate_external_test_ensemble(
        model_builder=build_model,
        base_dir=BASE_DIR,
        config=CONFIG,
        test_csv_path=test_csv,
        model_display_name="SwinUNETR-LinearProbe",
        n_folds=CONFIG["n_splits"],
    )


if __name__ == "__main__":
    main()
