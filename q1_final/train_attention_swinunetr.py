"""
Attention-Guided Multi-Scale Fusion SwinUNETR (AG-MSF) — Appendisit/Müsinöz Sınıflandırma
============================================================================================
Q1 makale karşılaştırma paketindeki 3 mimariden biri (pretrained SwinUNETR backbone).
- Backbone: MONAI SwinUNETR (abdominal CT'lerde önceden eğitilmiş, yerel model_swinvit.pt).
- Yenilik: Her 5 SwinViT aşamasından (48,96,192,384,768) global-pool edilen özellikler
  channel-attention ile ağırlıklandırılıp birleştiriliyor.
- Progressive fine-tuning: epoch<15 backbone donuk (sadece classifier), epoch>=15 tüm model.
- Stochastic Weight Averaging (SWA): son %30 epoch'ta aktif.

v06_attention_swinunetr_3d.ipynb'den standalone .py'ye taşındı; internetten indirmek yerine
zaten diskte bulunan segformer/model_swinvit.pt kullanılıyor (v04/v08'in de kullandığı dosya).

Çalıştırma:
    cd segformer && python train_attention_swinunetr.py
Çıktılar:
    experiments/attention_swinunetr/fold_0X/{best_model.pt, best_val_predictions.csv, cm_*.png}
    experiments/attention_swinunetr/aggregate_oof/...
    experiments/attention_swinunetr/external_test/q1_external_test_metrics.csv
    experiments/attention_swinunetr/train_log.txt
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
from shared_utils import *  # noqa: F401,F403

torch.manual_seed(SHARED_CONFIG["random_seed"])
np.random.seed(SHARED_CONFIG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "attention_swinunetr"
DATA_ROOT = Path(r"/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS")
DATAS_DIR = DATA_ROOT / "segformer" / "datas"
BASE_DIR = DATA_ROOT / "segformer" / "experiments_q1_128" / MODEL_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Yerelde zaten indirilmiş pretrained SwinViT ağırlıkları (v04/v08 ile ortak)
LOCAL_PRETRAINED_PATH = DATA_ROOT / "segformer" / "model_swinvit.pt"

CONFIG = dict(SHARED_CONFIG)
CONFIG["output_dir"] = str(BASE_DIR)
CONFIG["lr"] = 1e-4
CONFIG["n_epochs"] = 100
CONFIG["patience"] = 25
CONFIG["mixup_alpha"] = 0.05  # Çok hafif Mixup
CONFIG["accum_steps"] = 4    # efektif batch_size = 4*4 = 16 (gürültülü gradyanı azaltmak için)
CONFIG["ema_alpha"] = 0.3    # checkpoint seçim metriğini yumuşatmak için EMA katsayısı
CONFIG["min_epochs_before_save"] = 3
CONFIG["weight_decay"] = 5e-3
CONFIG["focal_gamma"] = 1.0
CONFIG["label_smoothing"] = 0.05


# ============================================================
# Model: Attention-Guided Multi-Scale Fusion SwinUNETR
# ============================================================
from monai.networks.nets import SwinUNETR


class ChannelAttention(nn.Module):
    """Farklı derinliklerden gelen (multi-scale) özniteliklerin göreli önemini öğrenir."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.attention(x)
        return x * weight


class AttentionSwinUNETR3D(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, feature_size=48):
        super().__init__()
        self.backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=14,
            feature_size=feature_size,
            use_checkpoint=True,
            spatial_dims=3,
        )

        dims = [feature_size * (2 ** i) for i in range(5)]  # [48, 96, 192, 384, 768]
        fused_dim = sum(dims)  # 1488

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.channel_attention = ChannelAttention(in_channels=fused_dim, reduction=16)

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        hidden = self.backbone.swinViT(x, self.backbone.normalize)
        feats = []
        for i, h in enumerate(hidden):
            f = self.gap(h).flatten(1)
            f = self.norms[i](f)
            feats.append(f)

        fused = torch.cat(feats, dim=1)
        attended_features = self.channel_attention(fused)
        return self.classifier(attended_features)


def build_model():
    return AttentionSwinUNETR3D(num_classes=2, in_channels=CONFIG["expected_C"], feature_size=48)


def load_pretrained_backbone(model, path):
    if not path.exists():
        print(f"  UYARI: pretrained dosya bulunamadı ({path}), backbone rastgele init ile başlıyor.")
        return model
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    new_sd = {}
    for k, v in sd.items():
        new_k = "backbone." + k if not k.startswith("backbone.") else k
        if new_k in model_sd and model_sd[new_k].shape == v.shape:
            new_sd[new_k] = v
            loaded += 1
        else:
            skipped += 1
    model_sd.update(new_sd)
    model.load_state_dict(model_sd, strict=False)
    print(f"  Pretrained: {loaded} katman yüklendi, {skipped} atlandı.")
    return model


# ============================================================
# Tek fold eğitimi (progressive fine-tuning + SWA)
# ============================================================
def run_one_fold_attention_swinunetr(train_df, val_df, fold_idx, config, output_dir):
    fold_dir = Path(output_dir) / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AppendixH5Dataset(train_df, augment=True, config=config)
    val_ds = AppendixH5Dataset(val_df, augment=False, config=config)
    n0 = (train_df["label"] == 0).sum()
    n1 = (train_df["label"] == 1).sum()
    w = torch.tensor([n1 / (n0 + n1), n0 / (n0 + n1)], dtype=torch.float32).to(DEVICE)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                               num_workers=config["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"], pin_memory=True)

    model = build_model().to(DEVICE)
    model = load_pretrained_backbone(model, LOCAL_PRETRAINED_PATH)

    # Progressive Fine-Tuning: başlangıçta backbone donuk
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    criterion = ClinicalFocalLoss(pos_weight=pos_weight_from_labels(train_df["label"].values),
                                   gamma=2.0, smoothing=0.1)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=config["lr"], weight_decay=1e-2)
    scheduler = get_warmup_cosine_scheduler(optimizer, config.get("warmup_epochs", 10), config["n_epochs"])

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(config["n_epochs"] * 0.70)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=config["lr"] * 0.1)

    best_score, patience_counter = 0.0, 0
    ema_composite = None
    ema_alpha = config.get("ema_alpha", 0.3)
    min_epochs_before_save = config.get("min_epochs_before_save", 3)
    history = []
    epoch = 0

    for epoch in range(1, config["n_epochs"] + 1):
        if epoch == 15:
            print("  [Progressive FT] Epoch 15: Backbone çözülüyor, tam model ince ayara başlıyor.")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"] * 0.1, weight_decay=1e-2)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, 0.0,
                                      accum_steps=config.get("accum_steps", 1))
        val_loss, val_auc, val_acc, val_f1, pred_df = evaluate_model(model, val_loader, criterion, DEVICE)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_auc": val_auc, "val_acc": val_acc, "val_f1": val_f1})
        print(f"[attn-swinunetr | fold {fold_idx} | epoch {epoch:03d}] "
              f"train={train_loss:.4f} val_loss={val_loss:.4f} auc={val_auc:.4f} acc={val_acc:.4f}")

        yt_val = pred_df["label"].values
        yp_val = pred_df["prob_mucinous"].values
        opt_thr_val, _ = find_youden_threshold(yt_val, yp_val)
        m_val, _, _ = compute_binary_metrics(yt_val, yp_val, threshold=opt_thr_val)
        composite_score = m_val["composite"]

        # Tek epoch'luk composite küçük val fold'da çok gürültülü; checkpoint seçimini
        # EMA ile yumuşatıyoruz (linear-probe denemesindeki dalgalanmaya karşı önlem).
        ema_composite = composite_score if ema_composite is None else (
            ema_alpha * composite_score + (1 - ema_alpha) * ema_composite
        )

        print(f"  [Metrics] AUC:{m_val['auc_roc']:.3f} F1:{m_val['f1']:.3f} "
              f"SENS:{m_val['sensitivity']:.3f} SPEC:{m_val['specificity']:.3f} | "
              f"COMPOSITE:{composite_score:.4f} EMA:{ema_composite:.4f}")

        if epoch < min_epochs_before_save:
            pass
        elif ema_composite > best_score:
            best_score = ema_composite
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "val_auc": val_auc,
                        "composite_score": composite_score, "ema_composite": ema_composite},
                       fold_dir / "best_model.pt")
            pred_df.to_csv(fold_dir / "best_val_predictions.csv", index=False)
        else:
            patience_counter += 1
            # SWA fazına girmeden (epoch < swa_start) erken durdurma TETİKLENMEZ — aksi halde
            # SWA hiç çalışmadan patience ile eğitim kesiliyordu (gerçek bug, SWA'yı etkisiz kılıyordu).
            if epoch >= swa_start and patience_counter >= config["patience"]:
                print(f"  Early stopping @ epoch {epoch} (best EMA composite={best_score:.4f})")
                break

    if epoch >= swa_start:
        print("  SWA model finalize ediliyor...")
        def _get_images():
            for b in train_loader:
                yield b["image"]
        torch.optim.swa_utils.update_bn(_get_images(), swa_model, device=DEVICE)
        _, swa_auc, _, _, swa_pred = evaluate_model(swa_model, val_loader, criterion, DEVICE)

        yt_swa = swa_pred["label"].values
        yp_swa = swa_pred["prob_mucinous"].values
        opt_thr_swa, _ = find_youden_threshold(yt_swa, yp_swa)
        m_swa, _, _ = compute_binary_metrics(yt_swa, yp_swa, threshold=opt_thr_swa)
        swa_composite = m_swa["composite"]

        print(f"  SWA val COMPOSITE: {swa_composite:.4f} | Best EMA COMPOSITE: {best_score:.4f}")
        if swa_composite >= best_score * 0.98:
            torch.save({"model_state_dict": swa_model.module.state_dict(), "val_auc": swa_auc,
                        "composite_score": swa_composite}, fold_dir / "best_model.pt")
            swa_pred.to_csv(fold_dir / "best_val_predictions.csv", index=False)
            print("  SWA model kaydedildi (genelleme için tercih edildi).")

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_pred = pd.read_csv(fold_dir / "best_val_predictions.csv")
    yt, yp = best_pred["label"].values, best_pred["prob_mucinous"].values
    opt_thr, _ = find_youden_threshold(yt, yp)
    m_y, cm_y, _ = compute_binary_metrics(yt, yp, threshold=opt_thr)
    ci_y = compute_bootstrap_ci(yt, yp, threshold=opt_thr, n_bootstraps=config["n_bootstrap"])
    print_full_metrics_table(m_y, ci_y, f"AttentionSwinUNETR Fold {fold_idx}", f"Youden {opt_thr:.3f}")
    plot_confusion_matrix(cm_y, f"Attention-SwinUNETR Fold {fold_idx} @Youden {opt_thr:.3f}",
                           save_path=fold_dir / "cm_youden.png")
    plot_roc_pr(yt, yp, f"attention_swinunetr_fold{fold_idx}", fold_dir, opt_threshold=opt_thr)
    best_pred["fold"] = fold_idx
    best_pred["youden_threshold"] = opt_thr

    return m_y, ci_y, best_pred, pd.DataFrame(history)


# ============================================================
# 5-Fold Experiment + Aggregate OOF + External Test
# ============================================================
def main():
    setup_file_logging(BASE_DIR / "train_log.txt")

    test_csv_path = DATAS_DIR / "external_test_set.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Lütfen önce generate_master_splits.py çalıştırıp datas klasörünü oluşturun: {test_csv_path}")

    print("=" * 80)
    print(f"Attention-SwinUNETR (AG-MSF) — External Test: {len(pd.read_csv(test_csv_path))} hasta")
    print("=" * 80)

    all_preds = []
    for fold_idx in range(1, CONFIG["n_splits"] + 1):
        train_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_train.csv")
        val_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_val.csv")

        print(f"\n{'=' * 70}\nFOLD {fold_idx}/{CONFIG['n_splits']}\n{'=' * 70}")
        _, _, pred_f, _ = run_one_fold_attention_swinunetr(train_df, val_df, fold_idx, CONFIG, BASE_DIR)
        all_preds.append(pred_f)

    oof = pd.concat(all_preds, ignore_index=True)
    yt, yp = oof["label"].values, oof["prob_mucinous"].values
    opt_thr, _ = find_youden_threshold(yt, yp)
    m_oof, cm_oof, _ = compute_binary_metrics(yt, yp, threshold=opt_thr)
    ci_oof = compute_bootstrap_ci(yt, yp, threshold=opt_thr, n_bootstraps=CONFIG["n_bootstrap"])
    agg_dir = BASE_DIR / "aggregate_oof"
    agg_dir.mkdir(exist_ok=True)
    oof.to_csv(agg_dir / "oof_predictions.csv", index=False)
    plot_confusion_matrix(cm_oof, f"Attention-SwinUNETR Aggregate OOF @Youden {opt_thr:.3f}",
                           save_path=agg_dir / "agg_cm_youden.png")
    plot_roc_pr(yt, yp, "attention_swinunetr_aggregate_oof", agg_dir, opt_threshold=opt_thr)
    print("\nAGGREGATE OOF (5-Fold Cross-Validation):")
    print_full_metrics_table(m_oof, ci_oof, "Attention-SwinUNETR Aggregate OOF", f"Youden {opt_thr:.3f}")

    evaluate_external_test_ensemble(
        model_builder=build_model,
        base_dir=BASE_DIR,
        config=CONFIG,
        test_csv_path=test_csv_path,
        model_display_name="Attention-SwinUNETR",
        n_folds=CONFIG["n_splits"],
    )


if __name__ == "__main__":
    main()
