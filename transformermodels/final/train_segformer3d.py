"""
SegFormer3D-MSCA (Multi-Scale Context Attention) — Appendisit/Müsinöz Sınıflandırma
Çalıştırma:
    cd segformer && python train_segformer3d.py
Çıktılar:
    experiments/segformer3d/fold_0X/{best_model.pt, val_predictions.csv, cm_*.png, ...}
    experiments/segformer3d/aggregate_oof/...
    experiments/segformer3d/external_test/external_test_metrics.csv
    experiments/segformer3d/train_log.txt
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
from shared_utils import *  
torch.manual_seed(SHARED_CONFIG["random_seed"])
np.random.seed(SHARED_CONFIG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "segformer3d"
DATA_ROOT = Path(r"/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS")
DATAS_DIR = DATA_ROOT / "segformer" / "datas"
BASE_DIR = DATA_ROOT / "segformer" / "experiments_128" / MODEL_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG = dict(SHARED_CONFIG)
CONFIG["output_dir"] = str(BASE_DIR)
CONFIG["lr"] = 3e-4
CONFIG["n_epochs"] = 100
CONFIG["patience"] = 25
CONFIG["batch_size"] = 2
CONFIG["accum_steps"] = 8
CONFIG["ema_alpha"] = 0.3
CONFIG["min_epochs_before_save"] = 3
CONFIG["mixup_alpha"] = 0.1      
CONFIG["weight_decay"] = 5e-3    
CONFIG["focal_gamma"] = 1.0
CONFIG["label_smoothing"] = 0.05
class OverlapPatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=3, stride=2, padding=1):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        return x
class EfficientSelfAttention3D(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W
        x_flat = x.flatten(2).transpose(1, 2)
        q = self.q(x_flat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = self.sr(x).flatten(2).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_flat).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(B, C, D, H, W)
        return out
class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * mask
class MixFFN3D(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.fc2 = nn.Conv3d(hidden_features, in_features, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.drop(self.act(x))
        x = self.fc2(x)
        x = self.drop(x)
        return x
class SegformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, drop_path=0.0, mlp_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention3D(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN3D(dim, dim * 4, dropout=mlp_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        B, C, D, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        norm1_x = self.norm1(x_flat).transpose(1, 2).reshape(B, C, D, H, W)
        x = x + self.drop_path(self.attn(norm1_x))
        x_flat2 = x.flatten(2).transpose(1, 2)
        norm2_x = self.norm2(x_flat2).transpose(1, 2).reshape(B, C, D, H, W)
        x = x + self.drop_path(self.mlp(norm2_x))
        return x
class CBAM3D(nn.Module):
    """3D Spatial-Channel Attention (CBAM-3D)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_cat))
        x = x * spatial_att
        return x
class SegFormer3DClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        embed_dims = [32, 64, 128, 256]
        num_heads = [2, 4, 8, 16]
        sr_ratios = [4, 2, 2, 1]
        depths = [2, 2, 3, 3]
        dpr = [x.item() for x in torch.linspace(0, 0.10, sum(depths))]
        def _blocks(dim, heads, sr, depth, start):
            return nn.Sequential(*[
                SegformerBlock3D(dim, heads, sr, drop_path=dpr[start + i])
                for i in range(depth)
            ])
        self.patch_embed1 = OverlapPatchEmbed3D(in_channels, embed_dims[0], patch_size=7, stride=2, padding=3)
        self.block1 = _blocks(embed_dims[0], num_heads[0], sr_ratios[0], depths[0], 0)
        self.patch_embed2 = OverlapPatchEmbed3D(embed_dims[0], embed_dims[1], patch_size=3, stride=2, padding=1)
        self.block2 = _blocks(embed_dims[1], num_heads[1], sr_ratios[1], depths[1], depths[0])
        self.patch_embed3 = OverlapPatchEmbed3D(embed_dims[1], embed_dims[2], patch_size=3, stride=2, padding=1)
        self.block3 = _blocks(embed_dims[2], num_heads[2], sr_ratios[2], depths[2], sum(depths[:2]))
        self.patch_embed4 = OverlapPatchEmbed3D(embed_dims[2], embed_dims[3], patch_size=3, stride=2, padding=1)
        self.block4 = _blocks(embed_dims[3], num_heads[3], sr_ratios[3], depths[3], sum(depths[:3]))
        self.cbam = CBAM3D(embed_dims[3])
        self.pool = nn.AdaptiveAvgPool3d(1)
        fusion_dim = embed_dims[2] + embed_dims[3]
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.patch_embed1(x)
        x = self.block1(x)
        x = self.patch_embed2(x)
        x = self.block2(x)
        x = self.patch_embed3(x)
        x3 = self.block3(x)
        x = self.patch_embed4(x3)
        x4 = self.block4(x)
        x4 = self.cbam(x4)
        feat3 = self.pool(x3).flatten(1)
        feat4 = self.pool(x4).flatten(1)
        fused = torch.cat([feat3, feat4], dim=1)
        return self.fc(fused)
def build_model():
    return SegFormer3DClassifier(in_channels=CONFIG["expected_C"], num_classes=2)
class SegFormer3DMAE(nn.Module):
    """SegFormer3D encoder + basit patch reconstruction decoder (MAE pretraining için)."""
    def __init__(self, in_channels=1, mask_ratio=0.75):
        super().__init__()
        embed_dims = [32, 64, 128, 256]
        num_heads  = [2, 4, 8, 16]
        sr_ratios  = [4, 2, 2, 1]
        depths = [2, 2, 3, 3]
        dpr = [x.item() for x in torch.linspace(0, 0.15, sum(depths))]
        self.mask_ratio = mask_ratio
        def _blocks(dim, heads, sr, depth, start):
            return nn.Sequential(*[
                SegformerBlock3D(dim, heads, sr, drop_path=dpr[start + i])
                for i in range(depth)
            ])
        self.patch_embed1 = OverlapPatchEmbed3D(in_channels,   embed_dims[0], 7, 2, 3)
        self.block1       = _blocks(embed_dims[0], num_heads[0], sr_ratios[0], depths[0], 0)
        self.patch_embed2 = OverlapPatchEmbed3D(embed_dims[0], embed_dims[1], 3, 2, 1)
        self.block2       = _blocks(embed_dims[1], num_heads[1], sr_ratios[1], depths[1], depths[0])
        self.patch_embed3 = OverlapPatchEmbed3D(embed_dims[1], embed_dims[2], 3, 2, 1)
        self.block3       = _blocks(embed_dims[2], num_heads[2], sr_ratios[2], depths[2], sum(depths[:2]))
        self.patch_embed4 = OverlapPatchEmbed3D(embed_dims[2], embed_dims[3], 3, 2, 1)
        self.block4       = _blocks(embed_dims[3], num_heads[3], sr_ratios[3], depths[3], sum(depths[:3]))
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dims[3], embed_dims[2], 2, 2),
            nn.GELU(),
            nn.ConvTranspose3d(embed_dims[2], embed_dims[1], 2, 2),
            nn.GELU(),
            nn.ConvTranspose3d(embed_dims[1], embed_dims[0], 2, 2),
            nn.GELU(),
            nn.ConvTranspose3d(embed_dims[0], in_channels,   2, 2),
        )
    def forward(self, x):
        B, C, D, H, W = x.shape
        mask = torch.rand(B, 1, D // 2, H // 2, W // 2, device=x.device) < self.mask_ratio
        mask_up = F.interpolate(mask.float(), size=(D, H, W), mode='nearest')
        x_masked = x * (1 - mask_up)
        f = self.patch_embed1(x_masked); f = self.block1(f)
        f = self.patch_embed2(f);        f = self.block2(f)
        f = self.patch_embed3(f);        f = self.block3(f)
        f = self.patch_embed4(f);        f = self.block4(f)
        recon = self.decoder(f)
        recon = F.interpolate(recon, size=(D, H, W), mode='trilinear', align_corners=False)
        recon_loss = F.mse_loss(recon * mask_up, x * mask_up)
        return recon_loss
def pretrain_segformer_mae(config, device, save_path):
    """Tüm 241 hastalık veriyle SegFormer3D encoder'ı MAE ile önceden eğitir."""
    if Path(save_path).exists():
        print(f"  SegFormer3D MAE encoder zaten var, pretraining atlanıyor: {save_path}")
        return
    print("\n" + "=" * 70)
    print("SegFormer3D MAE Pretraining başlıyor (241 hasta, self-supervised)...")
    print("=" * 70)
    all_csvs = list((DATAS_DIR).glob("fold_*_train.csv")) + list((DATAS_DIR).glob("fold_*_val.csv"))
    all_dfs  = [pd.read_csv(p) for p in all_csvs]
    pretrain_df = pd.concat(all_dfs).drop_duplicates("patient_id").reset_index(drop=True)
    print(f"  MAE pretraining manifest: {len(pretrain_df)} benzersiz hasta")
    pretrain_ds     = AppendixH5Dataset(pretrain_df, augment=True, config=config)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=config["batch_size"],
                                  shuffle=True, num_workers=config["num_workers"],
                                  pin_memory=torch.cuda.is_available(), drop_last=True)
    mae_model = SegFormer3DMAE(
        in_channels=config["expected_C"],
        mask_ratio=config.get("mae_mask_ratio", 0.75)
    ).to(device)
    optimizer = torch.optim.AdamW(mae_model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("mae_epochs", 60))
    mae_model.train()
    for ep in range(1, config.get("mae_epochs", 60) + 1):
        total_loss = 0.0
        for batch in pretrain_loader:
            imgs = batch["image"].to(device)
            optimizer.zero_grad()
            loss = mae_model(imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(pretrain_loader)
        scheduler.step()
        if ep % 10 == 0 or ep == 1:
            print(f"  [MAE pretrain | epoch {ep:03d}] recon_loss={avg:.4f}")
    encoder_sd = {
        k: v for k, v in mae_model.state_dict().items()
        if not k.startswith("decoder")
    }
    torch.save(encoder_sd, save_path)
    print(f"  SegFormer3D MAE encoder kaydedildi: {save_path}")
    del mae_model; torch.cuda.empty_cache()
def run_one_fold_segformer3d(train_df, val_df, fold_idx, config, output_dir):
    fold_dir = Path(output_dir) / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_ds = AppendixH5Dataset(train_df, augment=True, config=config)
    val_ds = AppendixH5Dataset(val_df, augment=False, config=config)
    n0 = (train_df["label"] == 0).sum()
    n1 = (train_df["label"] == 1).sum()
    print(f"  [Fold {fold_idx}] class_counts: NEG={n0}, POS={n1}")
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                               num_workers=config["num_workers"], pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"], pin_memory=torch.cuda.is_available())
    model = build_model().to(DEVICE)
    mae_encoder_path = BASE_DIR.parent / "segformer3d_mae" / "segformer3d_mae_encoder.pt"
    if mae_encoder_path.exists():
        pretrained_sd = torch.load(mae_encoder_path, map_location=DEVICE, weights_only=False)
        model_sd = model.state_dict()
        matched = {k: v for k, v in pretrained_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        model_sd.update(matched)
        model.load_state_dict(model_sd, strict=False)
        print(f"  [Fold {fold_idx}] MAE pretrained encoder yüklendi ({len(matched)} katman).")
    else:
        print(f"  [Fold {fold_idx}] MAE encoder bulunamadı, rastgele init ile devam.")
    criterion = ClinicalFocalLoss(
        pos_weight=pos_weight_from_labels(train_df["label"].values),
        gamma=config.get("focal_gamma", 2.0),
        smoothing=config.get("label_smoothing", 0.05))
    head_params = list(model.cbam.parameters()) + list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith('cbam') and not n.startswith('fc')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config["lr"] * 0.1},
        {'params': head_params, 'lr': config["lr"]}
    ], weight_decay=config["weight_decay"])
    scheduler = get_warmup_cosine_scheduler(optimizer, config.get("warmup_epochs", 10), config["n_epochs"])
    best_score, patience_counter = -10.0, 0
    ema_score = None
    ema_alpha = config.get("ema_alpha", 0.3)
    min_epochs_before_save = config.get("min_epochs_before_save", 3)
    history = []
    epoch = 0
    from sklearn.metrics import f1_score as _f1
    for epoch in range(1, config["n_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE,
                                      mixup_alpha=config.get("mixup_alpha", 0.0),
                                      accum_steps=config.get("accum_steps", 1))
        val_loss, val_auc, val_acc, val_f1_raw, pred_df = evaluate_model(model, val_loader, criterion, DEVICE)
        scheduler.step()
        yt_val = pred_df["label"].values
        yp_val = pred_df["prob_mucinous"].values
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
        raw_score = composite if meets_constraints else (composite - 2.0)
        ema_score = raw_score if ema_score is None else (
            ema_alpha * raw_score + (1 - ema_alpha) * ema_score
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_auc": val_auc, "val_acc": val_acc, "val_f1": val_f1_thr,
                         "composite": composite, "ema_score": ema_score})
        print(f"[segformer3d | fold {fold_idx} | epoch {epoch:03d}] "
              f"train={train_loss:.4f} val_loss={val_loss:.4f} auc={val_auc:.4f} thr={threshold:.3f}")
        print(f"  [Metrics] AUC:{val_auc:.3f} F1:{val_f1_thr:.3f} "
              f"SENS:{sens:.3f} SPEC:{spec:.3f} | "
              f"COMPOSITE:{composite:.4f} EMA:{ema_score:.4f}")
        if epoch < min_epochs_before_save:
            pass  
        elif ema_score > best_score:
            best_score = ema_score
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "val_auc": val_auc,
                        "composite_score": composite, "ema_score": ema_score,
                        "epoch": epoch}, fold_dir / "best_model.pt")
            pred_df.to_csv(fold_dir / "best_val_predictions.csv", index=False)
            if meets_constraints:
                print(f"    ★ SAVED DUAL-PASS (SENS={sens:.3f}≥{min_sens} "
                      f"SPEC={spec:.3f}≥{min_spec} | comp={composite:.4f})")
            else:
                print(f"    ☆ SAVED FALLBACK (SENS={sens:.3f} SPEC={spec:.3f} | comp={composite:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  Early stopping @ epoch {epoch} (best EMA score={best_score:.4f})")
                break
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    best_pred = pd.read_csv(fold_dir / "best_val_predictions.csv")
    yt, yp = best_pred["label"].values, best_pred["prob_mucinous"].values
    opt_thr, _ = find_youden_threshold(yt, yp)
    m_y, cm_y, _ = compute_binary_metrics(yt, yp, threshold=opt_thr)
    ci_y = compute_bootstrap_ci(yt, yp, threshold=opt_thr, n_bootstraps=config["n_bootstrap"])
    print_full_metrics_table(m_y, ci_y, f"SegFormer3D Fold {fold_idx}", f"Youden {opt_thr:.3f}")
    plot_confusion_matrix(cm_y, f"SegFormer3D Fold {fold_idx} @Youden {opt_thr:.3f}",
                           save_path=fold_dir / "cm_youden.png")
    plot_roc_pr(yt, yp, f"segformer3d_fold{fold_idx}", fold_dir, opt_threshold=opt_thr)
    best_pred["fold"] = fold_idx
    best_pred["youden_threshold"] = opt_thr
    return m_y, ci_y, best_pred, pd.DataFrame(history)
def main():
    setup_file_logging(BASE_DIR / "train_log.txt")
    test_csv_path = DATAS_DIR / "external_test_set.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Lütfen önce generate_master_splits.py çalıştırıp datas klasörünü oluşturun: {test_csv_path}")
    mae_save_dir = BASE_DIR.parent / "segformer3d_mae"
    mae_save_dir.mkdir(parents=True, exist_ok=True)
    pretrain_segformer_mae(
        config=CONFIG,
        device=DEVICE,
        save_path=mae_save_dir / "segformer3d_mae_encoder.pt"
    )
    print("=" * 80)
    print(f"SegFormer3D-MSCA — External Test: {len(pd.read_csv(test_csv_path))} hasta")
    print("=" * 80)
    all_preds = []
    for fold_idx in range(1, CONFIG["n_splits"] + 1):
        train_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_train.csv")
        val_df = pd.read_csv(DATAS_DIR / f"fold_{fold_idx}_val.csv")
        print(f"\n{'=' * 70}\nFOLD {fold_idx}/{CONFIG['n_splits']}\n{'=' * 70}")
        _, _, pred_f, _ = run_one_fold_segformer3d(train_df, val_df, fold_idx, CONFIG, BASE_DIR)
        all_preds.append(pred_f)
    oof = pd.concat(all_preds, ignore_index=True)
    yt, yp = oof["label"].values, oof["prob_mucinous"].values
    opt_thr, _ = find_youden_threshold(yt, yp)
    m_oof, cm_oof, _ = compute_binary_metrics(yt, yp, threshold=opt_thr)
    ci_oof = compute_bootstrap_ci(yt, yp, threshold=opt_thr, n_bootstraps=CONFIG["n_bootstrap"])
    agg_dir = BASE_DIR / "aggregate_oof"
    agg_dir.mkdir(exist_ok=True)
    oof.to_csv(agg_dir / "oof_predictions.csv", index=False)
    plot_confusion_matrix(cm_oof, f"SegFormer3D Aggregate OOF @Youden {opt_thr:.3f}",
                           save_path=agg_dir / "agg_cm_youden.png")
    plot_roc_pr(yt, yp, "segformer3d_aggregate_oof", agg_dir, opt_threshold=opt_thr)
    print("\nAGGREGATE OOF (5-Fold Cross-Validation):")
    print_full_metrics_table(m_oof, ci_oof, "SegFormer3D Aggregate OOF", f"Youden {opt_thr:.3f}")
    evaluate_external_test_ensemble(
        model_builder=build_model,
        base_dir=BASE_DIR,
        config=CONFIG,
        test_csv_path=test_csv_path,
        model_display_name="SegFormer3D-MSCA",
        n_folds=CONFIG["n_splits"],
    )
if __name__ == "__main__":
    main()
