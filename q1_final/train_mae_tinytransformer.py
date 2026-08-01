"""
MAE-Pretrained TinyTransformer3D — Appendisit/Müsinöz Sınıflandırma
=======================================================================
HEDEF: Q1 Dergi Standardı (SENS >= 0.85, AUC >= 0.80)
Bu model, tıbbi 3D görevlerde aşırı öğrenmeyi (overfitting) tamamen engellemek üzere tasarlanmış,
self-supervised (kendi kendine öğrenen) Masked Autoencoder (MAE) ön eğitimine sahiptir.
Ön eğitimde apendiks anatomisi öğrenildikten sonra, ince ayar (fine-tuning) aşamasında
MixUp, SWA ve ClinicalFocalLoss gibi güçlü regülarizasyon teknikleriyle 
SOTA (State-of-the-Art) klinik performans hedeflenmektedir.

Çalıştırma:
    cd segformer && python train_mae_tinytransformer.py
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
from shared_utils import *  # noqa: F401,F403

torch.manual_seed(SHARED_CONFIG["random_seed"])
np.random.seed(SHARED_CONFIG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "mae_tinytransformer"
DATA_ROOT = Path(r"/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS")
DATAS_DIR = DATA_ROOT / "segformer" / "datas"
BASE_DIR = DATA_ROOT / "segformer" / "experiments_q1_128" / MODEL_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = dict(SHARED_CONFIG)
CONFIG.update({
    "model_name":        MODEL_NAME,
    "embed_dim":         192,
    "n_epochs":          100,
    "patience":          20,
    "batch_size":        2,
    "accum_steps":       8,
    "lr":                2e-4,
    "weight_decay":      1e-2,
    "mae_mask_ratio":    0.75,
    "mae_epochs":        80,
    "mixup_alpha":       0.4,       # Q1-Level Overfit Koruması
})
CONFIG["ema_alpha"] = 0.3   # checkpoint seçim metriğini yumuşatmak için EMA katsayısı
CONFIG["min_epochs_before_save"] = 3


# ============================================================
# 3D Sin-Cos Pozisyonel Embedding (sabit, öğrenilmeyen — MAE referans
# uygulamasındaki 2D şemanın 3 eksene genişletilmiş hali).
# Encoder'da pozisyonel bilgi YOKTU: mask token'lar self-attention'da
# birbirinden ayrışamıyordu (hepsi aynı çıktıyı üretiyordu). Bu ekleme
# hem sınıflandırıcıyı hem MAE rekonstrüksiyonunu güçlendirir.
# ============================================================
def _get_1d_sincos_pos_embed(dim, positions):
    assert dim % 2 == 0
    omega = np.arange(dim // 2, dtype=np.float32)
    omega /= dim / 2.0
    omega = 1.0 / (10000 ** omega)
    positions = positions.reshape(-1).astype(np.float32)
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # [M, dim]


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    """grid_size: (D', H', W') patch grid boyutu. embed_dim 3'e tam bölünmeli."""
    assert embed_dim % 3 == 0, "embed_dim 3 eksene bölünebilmeli"
    dim_each = embed_dim // 3
    gd, gh, gw = grid_size
    grid_d, grid_h, grid_w = np.meshgrid(
        np.arange(gd, dtype=np.float32), np.arange(gh, dtype=np.float32),
        np.arange(gw, dtype=np.float32), indexing="ij",
    )
    emb_d = _get_1d_sincos_pos_embed(dim_each, grid_d)
    emb_h = _get_1d_sincos_pos_embed(dim_each, grid_h)
    emb_w = _get_1d_sincos_pos_embed(dim_each, grid_w)
    return np.concatenate([emb_d, emb_h, emb_w], axis=1)  # [D'*H'*W', embed_dim]


# ============================================================
# TinyTransformer3D Encoder — CLS + Mean-Pool Fusion Classifier
# ============================================================
class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch=1, embed_dim=192, patch_size=(2, 8, 8)):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), (D, H, W)


class TransformerBlock(nn.Module):
    def __init__(self, dim=192, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformer3DEncoder(nn.Module):
    def __init__(self, in_ch=1, embed_dim=192, depth=8, num_heads=6, patch_size=(2, 8, 8),
                 img_size=(32, 128, 128)):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_ch, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        grid_size = tuple(img_size[i] // patch_size[i] for i in range(3))
        patch_pos = get_3d_sincos_pos_embed(embed_dim, grid_size)  # [N, embed_dim], sabit (öğrenilmez)
        self.register_buffer("patch_pos_embed", torch.from_numpy(patch_pos).float().unsqueeze(0))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

    def forward(self, x):
        tokens, shape = self.patch_embed(x)
        B = tokens.shape[0]
        tokens = tokens + self.patch_pos_embed
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos_embed
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens


class TinyTransformer3DClassifier(nn.Module):
    """CLS token + mean-pool fusion (embed_dim=192)."""
    def __init__(self, num_classes=2, embed_dim=192, depth=8, num_heads=6):
        super().__init__()
        self.encoder = TinyTransformer3DEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        tokens = self.encoder(x)
        cls_out = tokens[:, 0]
        mean_out = tokens[:, 1:].mean(dim=1)
        fused = torch.cat([cls_out, mean_out], dim=1)
        return self.head(fused)


def build_model():
    return TinyTransformer3DClassifier(num_classes=2, embed_dim=192, depth=8, num_heads=6)


# ============================================================
# MAE (Masked Autoencoder) — Stage 1: Self-supervised Pretraining
# ============================================================
class MAEDecoder(nn.Module):
    """
    NOT: Eski sürüm mask token'ları sıraya sadece EKLİYORDU (unshuffle yok, pozisyon yok) —
    tüm mask token'lar self-attention'a özdeş girip özdeş çıkıyordu (hangi yamayı
    rekonstrükte ettiğini ayırt edemiyordu). Bu sürüm ids_restore ile doğru sıraya
    geri koyup her pozisyona kendi sabit 3D sin-cos embedding'ini ekliyor.
    """
    def __init__(self, embed_dim=192, decoder_dim=96, patch_size=(2, 8, 8), depth=2,
                 img_size=(32, 128, 128)):
        super().__init__()
        self.embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.blocks = nn.ModuleList([TransformerBlock(decoder_dim, 2) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        self.pred = nn.Linear(decoder_dim, patch_vol)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        grid_size = tuple(img_size[i] // patch_size[i] for i in range(3))
        patch_pos = get_3d_sincos_pos_embed(decoder_dim, grid_size)
        self.register_buffer("patch_pos_embed", torch.from_numpy(patch_pos).float().unsqueeze(0))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

    def forward(self, encoded_tokens, ids_restore, num_patches):
        """encoded_tokens: [B, 1+vis, embed_dim] (CLS + görünür token'lar, ids_keep sırasında)."""
        B = encoded_tokens.shape[0]
        x = self.embed(encoded_tokens)  # [B, 1+vis, decoder_dim]
        vis_count = x.shape[1] - 1

        mask_tokens = self.mask_token.expand(B, num_patches - vis_count, -1)
        patches_shuffled = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # [B, N, D] (görünür+maske, karışık sıra)
        patches_orig = torch.gather(
            patches_shuffled, 1, ids_restore.unsqueeze(-1).expand(-1, -1, patches_shuffled.shape[-1])
        )  # orijinal yama sırasına geri getir
        patches_orig = patches_orig + self.patch_pos_embed  # artık doğru pozisyona sabit embedding ekle

        cls_tok = x[:, :1, :] + self.cls_pos_embed
        full = torch.cat([cls_tok, patches_orig], dim=1)
        for blk in self.blocks:
            full = blk(full)
        full = self.norm(full)
        return self.pred(full[:, 1:])  # [B, N, patch_vol] — ORİJİNAL yama sırasında


def patchify_pixels(x, patch_size):
    """[B,C,D,H,W] -> [B,N,patch_vol] ham (pikselden) patch'ler; conv patch_embed ile
    aynı D,H,W sırasını korur (rekonstrüksiyon hedefi embedded token değil, ham piksel olmalı)."""
    B, C, D, H, W = x.shape
    pd, ph, pw = patch_size
    x = x.reshape(B, C, D // pd, pd, H // ph, ph, W // pw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    x = x.reshape(B, (D // pd) * (H // ph) * (W // pw), C * pd * ph * pw)
    return x


class MAEModel(nn.Module):
    def __init__(self, mask_ratio=0.75, embed_dim=192, patch_size=(2, 8, 8)):
        super().__init__()
        self.encoder = TinyTransformer3DEncoder(embed_dim=embed_dim, patch_size=patch_size)
        self.decoder = MAEDecoder(embed_dim=embed_dim, patch_size=patch_size)
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def forward(self, x):
        embedded, _ = self.encoder.patch_embed(x)  # [B, N, embed_dim] — encoder girdisi
        embedded = embedded + self.encoder.patch_pos_embed  # maskelemeden ÖNCE pozisyon ekle
        with torch.no_grad():
            raw_patches = patchify_pixels(x, self.patch_size)  # [B, N, patch_vol] — rekonstrüksiyon hedefi (orijinal sıra)

        B, N, C = embedded.shape
        keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        ids_mask = ids_shuffle[:, keep:]

        vis_tokens = torch.gather(embedded, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
        target_masked = torch.gather(raw_patches, 1, ids_mask.unsqueeze(-1).expand(-1, -1, raw_patches.shape[-1]))

        cls = self.encoder.cls_token.expand(B, -1, -1) + self.encoder.cls_pos_embed
        enc_in = torch.cat([cls, vis_tokens], dim=1)
        for blk in self.encoder.blocks:
            enc_in = blk(enc_in)
        enc_in = self.encoder.norm(enc_in)

        # Decoder artık ORİJİNAL yama sırasında tahmin döndürüyor (unshuffle içeride yapılıyor);
        # maskeli hedefle karşılaştırmak için aynı ids_mask ile gather ediyoruz.
        pred_all = self.decoder(enc_in, ids_restore, N)  # [B, N, patch_vol]
        pred_masked = torch.gather(pred_all, 1, ids_mask.unsqueeze(-1).expand(-1, -1, pred_all.shape[-1]))

        loss = F.mse_loss(pred_masked, target_masked)
        return loss


def run_mae_pretraining(manifest_df, config, base_dir):
    encoder_path = base_dir / "mae_pretrained_encoder.pt"
    if encoder_path.exists():
        print(f"MAE pretrained encoder zaten var, pretraining atlanıyor: {encoder_path}")
        return encoder_path

    all_ds = AppendixH5Dataset(manifest_df, augment=True, config=config)
    mae_loader = DataLoader(all_ds, batch_size=config.get("batch_size", 2), shuffle=True,
                             num_workers=config["num_workers"], pin_memory=True, drop_last=True)

    mae_model = MAEModel(mask_ratio=config.get("mae_mask_ratio", config.get("mask_ratio", 0.75)), embed_dim=192).to(DEVICE)
    mae_opt = torch.optim.AdamW(mae_model.parameters(), lr=1e-4, weight_decay=1e-4)
    mae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(mae_opt, T_max=config["mae_epochs"])

    print("MAE Pretraining başlıyor...")
    for epoch in range(1, config["mae_epochs"] + 1):
        mae_model.train()
        total_loss = 0.0
        for batch in mae_loader:
            images = batch["image"].to(DEVICE)
            mae_opt.zero_grad()
            loss = mae_model(images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae_model.parameters(), 1.0)
            mae_opt.step()
            total_loss += loss.item()
        mae_sched.step()
        if epoch % 10 == 0 or epoch == config["mae_epochs"]:
            print(f"[MAE | epoch {epoch:03d}] loss={total_loss / len(mae_loader):.4f}")

    torch.save(mae_model.encoder.state_dict(), encoder_path)
    print(f"MAE pretrained encoder kaydedildi: {encoder_path}")
    del mae_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return encoder_path


# ============================================================
# Tek fold fine-tuning (MAE-pretrained encoder + SWA)
# ============================================================
def run_one_fold_mae(train_df, val_df, fold_idx, config, output_dir, mae_encoder_path):
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
    if mae_encoder_path.exists():
        model.encoder.load_state_dict(torch.load(mae_encoder_path, map_location=DEVICE, weights_only=False))
        print(f"  Fold {fold_idx}: MAE pretrained encoder yüklendi.")

    criterion = ClinicalFocalLoss(pos_weight=pos_weight_from_labels(train_df["label"].values),
                                   gamma=config.get("focal_gamma", 2.0),
                                   smoothing=config.get("label_smoothing", 0.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = get_warmup_cosine_scheduler(optimizer, config.get("warmup_epochs", 10), config["n_epochs"])

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(config["n_epochs"] * 0.70)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=config["lr"] * 0.1)

    best_score, patience_counter = -10.0, 0
    ema_score = None
    ema_alpha = config.get("ema_alpha", 0.3)
    min_epochs_before_save = config.get("min_epochs_before_save", 3)
    history = []
    epoch = 0

    from sklearn.metrics import f1_score as _f1
    for epoch in range(1, config["n_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE,
                                      config.get("mixup_alpha", 0.0),
                                      accum_steps=config.get("accum_steps", 1))
        val_loss, val_auc, val_acc, val_f1_raw, pred_df = evaluate_model(model, val_loader, criterion, DEVICE)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        yt_val = pred_df["label"].values
        yp_val = pred_df["prob_mucinous"].values
        
        # Sıkı Q1 Çift Kısıt (Dual Constraint) Eşik Değerini Kullan
        threshold = float(pred_df["_threshold_used"].iloc[0]) if "_threshold_used" in pred_df.columns else 0.5
        y_pred = (yp_val >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt_val, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        val_f1_thr = float(_f1(yt_val, y_pred, zero_division=0))
        composite = clinical_composite(sens, val_auc, spec, f1=val_f1_thr)

        min_sens = config.get("min_sens_floor", 0.80)
        min_spec = config.get("min_spec_floor", 0.50)
        meets_constraints = (sens >= min_sens - 1e-5) and (spec >= min_spec - 1e-5)
        
        # Ceza Sistemi: Kısıtları karşılamayan modeller yedek (fallback) olarak negatif skorda yarışır
        raw_score = composite if meets_constraints else (composite - 2.0)
        
        ema_score = raw_score if ema_score is None else (
            ema_alpha * raw_score + (1 - ema_alpha) * ema_score
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_auc": val_auc, "val_acc": val_acc, "val_f1": val_f1_thr, 
                         "composite": composite, "ema_score": ema_score})

        print(f"[mae-tiny | fold {fold_idx} | epoch {epoch:03d}] "
              f"train={train_loss:.4f} val_loss={val_loss:.4f} auc={val_auc:.4f} thr={threshold:.3f}")
        print(f"  [Metrics] AUC:{val_auc:.3f} F1:{val_f1_thr:.3f} "
              f"SENS:{sens:.3f} SPEC:{spec:.3f} | "
              f"COMPOSITE:{composite:.4f} EMA:{ema_score:.4f}")

        if epoch < min_epochs_before_save:
            pass
        elif ema_score > best_score:
            best_score = ema_score
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "val_auc": val_auc, "ema_score": ema_score},
                       fold_dir / "best_model.pt")
            pred_df.to_csv(fold_dir / "best_val_predictions.csv", index=False)
            
            if meets_constraints:
                print(f"    ★ SAVED DUAL-PASS (SENS={sens:.3f}≥{min_sens} "
                      f"SPEC={spec:.3f}≥{min_spec} | comp={composite:.4f})")
            else:
                print(f"    ☆ SAVED FALLBACK (SENS={sens:.3f} SPEC={spec:.3f} | comp={composite:.4f})")
        else:
            patience_counter += 1
            if epoch >= swa_start and patience_counter >= config["patience"]:
                print(f"  Early stopping @ epoch {epoch} (best EMA score={best_score:.4f})")
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
            torch.save({"model_state_dict": swa_model.module.state_dict(), "val_auc": swa_auc, "composite_score": swa_composite}, fold_dir / "best_model.pt")
            swa_pred.to_csv(fold_dir / "best_val_predictions.csv", index=False)
            print("  SWA model kaydedildi.")

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_pred = pd.read_csv(fold_dir / "best_val_predictions.csv")
    yt, yp = best_pred["label"].values, best_pred["prob_mucinous"].values
    opt_thr, _ = find_youden_threshold(yt, yp)
    m_y, cm_y, _ = compute_binary_metrics(yt, yp, threshold=opt_thr)
    ci_y = compute_bootstrap_ci(yt, yp, threshold=opt_thr, n_bootstraps=config["n_bootstrap"])
    print_full_metrics_table(m_y, ci_y, f"MAE-Tiny Fold {fold_idx}", f"Youden {opt_thr:.3f}")
    plot_confusion_matrix(cm_y, f"MAE-Tiny Fold {fold_idx} @Youden {opt_thr:.3f}",
                           save_path=fold_dir / "cm_youden.png")
    plot_roc_pr(yt, yp, f"mae_tiny_fold{fold_idx}", fold_dir, opt_threshold=opt_thr)
    best_pred["fold"] = fold_idx
    best_pred["youden_threshold"] = opt_thr

    return m_y, ci_y, best_pred, pd.DataFrame(history)


# ============================================================
# MAE Pretraining + 5-Fold Fine-tune + Aggregate OOF + External Test
# ============================================================
def main():
    setup_file_logging(BASE_DIR / "train_log.txt")

    test_csv_path = DATAS_DIR / "external_test_set.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Lütfen önce generate_master_splits.py çalıştırıp datas klasörünü oluşturun: {test_csv_path}")

    # MAE pretraining için SADECE train+val verisi (external test dahil EDİLMİYOR - SIZINTI ÖNLENDİ)
    all_csvs = list(DATAS_DIR.glob("fold_*_train.csv")) + list(DATAS_DIR.glob("fold_*_val.csv"))
    manifest_df = pd.concat([pd.read_csv(c) for c in all_csvs], ignore_index=True).drop_duplicates("patient_id")
    print(f"MAE pretraining manifest (Leak-Free): {len(manifest_df)} benzersiz hasta")

    mae_encoder_path = run_mae_pretraining(manifest_df, CONFIG, BASE_DIR)

    print("=" * 80)
    print(f"MAE-TinyTransformer3D — External Test: {len(pd.read_csv(test_csv_path))} hasta")
    print("=" * 80)

    all_preds = []
    for fold_idx in range(1, CONFIG["n_splits"] + 1):
        train_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_train.csv")
        val_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_val.csv")

        print(f"\n{'=' * 70}\nFOLD {fold_idx}/{CONFIG['n_splits']}\n{'=' * 70}")
        _, _, pred_f, _ = run_one_fold_mae(train_df, val_df, fold_idx, CONFIG, BASE_DIR, mae_encoder_path)
        all_preds.append(pred_f)

    oof = pd.concat(all_preds, ignore_index=True)
    yt, yp = oof["label"].values, oof["prob_mucinous"].values
    opt_thr, _ = find_youden_threshold(yt, yp)
    m_oof, cm_oof, _ = compute_binary_metrics(yt, yp, threshold=opt_thr)
    ci_oof = compute_bootstrap_ci(yt, yp, threshold=opt_thr, n_bootstraps=CONFIG["n_bootstrap"])
    agg_dir = BASE_DIR / "aggregate_oof"
    agg_dir.mkdir(exist_ok=True)
    oof.to_csv(agg_dir / "oof_predictions.csv", index=False)
    plot_confusion_matrix(cm_oof, f"MAE-Tiny Aggregate OOF @Youden {opt_thr:.3f}",
                           save_path=agg_dir / "agg_cm_youden.png")
    plot_roc_pr(yt, yp, "mae_tiny_aggregate_oof", agg_dir, opt_threshold=opt_thr)
    print("\nAGGREGATE OOF (5-Fold Cross-Validation):")
    print_full_metrics_table(m_oof, ci_oof, "MAE-Tiny Aggregate OOF", f"Youden {opt_thr:.3f}")

    evaluate_external_test_ensemble(
        model_builder=build_model,
        base_dir=BASE_DIR,
        config=CONFIG,
        test_csv_path=test_csv_path,
        model_display_name="MAE-TinyTransformer3D",
        n_folds=CONFIG["n_splits"],
    )


if __name__ == "__main__":
    main()
