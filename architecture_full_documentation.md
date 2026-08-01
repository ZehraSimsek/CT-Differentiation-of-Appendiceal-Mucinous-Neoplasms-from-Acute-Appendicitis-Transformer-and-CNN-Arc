# 3D Apandisit/Müsinöz Sınıflandırma Projesi — Tam Mimari Dokümantasyonu

## PROJE GENEL BAKIŞ

**Görev:** CT görüntülerinden Apandisit (Class 0) vs Müsinöz Tümör (Class 1) ikili sınıflandırması  
**Veri:** 241 hasta toplam (165 labeled + external test)  
**Veri Formatı:** H5 dosyaları, shape=(1, 32, 128, 128) (C, D, H, W)  
**Hedef Metrik:** AUC ≥ 0.80, Sensitivity ≥ 0.85 (Q1 dergi standardı)  
**Validasyon:** 5-Fold Stratified Cross-Validation + Bağımsız External Test Seti

---

## BÖLÜM 1: ORTAK ALTYAPI (shared_utils.py)

### 1.1 Veri Pipeline

```
HDF5 Dosyası
    └── h5py.File → vol = f["image"][()] → float32
        └── Boyut Normalizasyonu:
            • 4D (D,H,W,C) → transpose → (C,D,H,W)
            • 3D → unsqueeze(0) → (1,D,H,W)
            • 2D → unsqueeze(0,0) → (1,1,H,W)
        └── Z-Score Normalizasyon: (vol - mean) / (std + 1e-8)
        └── Tensor Dönüşümü → torch.Tensor
```

### 1.2 3D Augmentasyon (Sadece Train, augment=True)

```
Girdi Tensörü (C,D,H,W)
    ├── Rastgele Flip [p=0.5]: Derinlik(D), Yükseklik(H), Genişlik(W) eksenleri
    ├── 90° Rotasyon [p=0.4]: k∈{1,2,3}, dims=[H,W]
    ├── Yoğunluk Kaydırma [p=0.6]: shift ∈ [-0.25, +0.25]
    ├── Yoğunluk Ölçekleme [p=0.6]: scale ∈ [0.75, 1.25]
    ├── Gaussian Gürültü [p=0.6]: σ=0.02
    ├── 3D Cutout [p=0.4]: (4, 16, 16) boyutlu sıfırlama bloğu
    └── Derinlik Kırpma+Yeniden Boyutlandırma [p=0.3]:
        • Baş ve kuyruktan D//6 kadar kesme
        • F.interpolate trilinear → orijinal boyuta geri
```

### 1.3 ClinicalFocalLoss

```
Logits [B, 2] + Labels [B]
    │
    ├── Label Smoothing (smoothing=0.10):
    │   true_dist[i][label] = 1.0 - smoothing
    │   true_dist[i][other] = smoothing / (C-1)
    │
    ├── Class Weight Ağırlığı:
    │   weight[0] = 1.0 (Apandisit - Negatif)
    │   weight[1] = pos_weight (Müsinöz - Pozitif) → sabit 1.5
    │
    ├── Weighted Cross Entropy:
    │   log_prob = log_softmax(logits)
    │   ce = -(true_dist * log_prob * weight).sum(dim=1)
    │
    ├── Focal Ağırlık:
    │   pt = softmax(logits)[:, label]
    │   focal_w = (1 - pt)^gamma  [gamma=1.5 veya 2.0]
    │
    └── Kayıp = mean(focal_w * ce)
```

### 1.4 Threshold Seçim Stratejisi (Sensitivity-First)

```
Val Tahminleri → ROC Eğrisi → [FPR[], TPR[], THR[]]
    │
    ├── ADIM 1 — ÇİFT KISIT:
    │   Koşul: SENS ≥ 0.80 VE SPEC ≥ 0.50
    │   Uygun adaylar → F1 Maksimize et → Seçilen Eşik
    │
    ├── ADIM 2 — TEK KISIT (Çift kısıt başarısız):
    │   Koşul: SENS ≥ 0.80
    │   Uygun adaylar → SPEC Maksimize et → Seçilen Eşik
    │
    ├── ADIM 3 — YOUDEN J (Her iki kısıt da başarısız):
    │   J = TPR - FPR → max(J) indeksine karşılık gelen eşik
    │
    └── Klip: threshold ∈ [0.15, 0.85]
```

### 1.5 Test-Time Augmentation (TTA) - Değerlendirme Sırasında

```
Giriş Hacmi x
    ├── Orijinal: model(x) → p1
    ├── Derinlik Flip: model(flip(x, dim=2)) → p2
    ├── Yükseklik Flip: model(flip(x, dim=3)) → p3
    └── Genişlik Flip: model(flip(x, dim=4)) → p4
        │
        └── Ortalama: prob = (p1+p2+p3+p4) / 4.0
```

### 1.6 Eğitim Döngüsü (Genel)

```
Her Epoch:
    ├── model.train()
    ├── Her Batch:
    │   ├── images → DEVICE, labels → DEVICE
    │   ├── outputs = model(images)
    │   ├── Deep Supervision desteği: isinstance(outputs, tuple) kontrolü
    │   │   ├── final_loss = criterion(final_out, labels)
    │   │   └── ds_loss = Σ(weight[j] * criterion(ds_out[j], labels))
    │   │       weights = [0.1, 0.1, 0.2, 0.3, 0.3]
    │   ├── (loss / accum_steps).backward()
    │   ├── Gradient Clip: max_norm=1.0
    │   └── optimizer.step() [her accum_steps adımda]
    └── Epoch Loss = total_loss / n_batches
```

### 1.7 Model Seçim Protokolü

```
Her Epoch Sonunda:
    ├── composite = 0.35*SENS + 0.25*F1 + 0.25*AUC + 0.15*SPEC
    ├── Kısıt Kontrolü: SENS≥0.80 AND SPEC≥0.50?
    │   ├── Evet: raw_score = composite
    │   └── Hayır: raw_score = composite - 2.0 (ceza)
    ├── EMA Yumuşatma: ema = 0.3*raw + 0.7*prev_ema
    ├── epoch < 3: kaydetme (ısınma periyodu)
    └── ema_score > best_score → Checkpoint Kaydet
```

### 1.8 Warmup + Cosine Annealing Scheduler

```
LR Planlaması:
    ├── epoch < warmup_epochs (10):
    │   lr_ratio = epoch / warmup_epochs  [min 0.01]
    └── epoch ≥ warmup_epochs:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr_ratio = 0.5 * (1 + cos(π * progress))  [min 0.01]
```

### 1.9 Stochastic Weight Averaging (SWA) - Tüm Modellerde

```
epoch < swa_start (epoch 70/100):
    └── Normal Warmup+Cosine LR
epoch ≥ swa_start:
    ├── swa_model.update_parameters(model)
    └── SWALR: lr = base_lr * 0.1 (sabit düşük lr)
Son Değerlendirme:
    ├── update_bn(train_loader, swa_model)
    ├── swa_composite = evaluate(swa_model)
    └── swa_composite ≥ best_score * 0.98 → SWA Modeli Kaydet
```

### 1.10 External Test Ensemble Protokolü

```
Tüm 5 Fold'un Checkpoint'leri
    │
    ├── Her Fold:
    │   ├── model.load(fold_XX/best_model.pt)
    │   ├── TTA ile Tahmin → fold_probs[]
    │   └── Youden threshold ile fold metrikleri
    │
    ├── Ensemble Olasılık: ens = mean(fold_probs, axis=0)
    │
    ├── @0.5 Threshold → Metrikler + 95% Bootstrap CI
    ├── @Youden Threshold → Metrikler + 95% Bootstrap CI
    └── @90%+ Sensitivity Threshold → Metrikler + 95% Bootstrap CI
        │
        ├── Confusion Matrix PNG
        ├── ROC + PR Eğrileri PNG
        ├── Kalibrasyon Eğrisi PNG (Brier + ECE)
        └── q1_external_test_metrics.csv
```

### 1.11 DeLong AUC Karşılaştırma Testi

```
İki Model (M1, M2) Aynı Test Hastaları:
    ├── Sort by y_true (pozitifler önce)
    ├── fast_delong: midrank hesabı → AUC kovaryans matrisi
    ├── auc_diff = AUC_M1 - AUC_M2
    ├── z = auc_diff / sqrt(var)
    └── p_value = 2 * (1 - Φ(|z|))
        └── p < 0.05 → İstatistiksel olarak anlamlı fark
```

### 1.12 Kalibrasyon Metrikleri

```
Tahmin Olasılıkları + Gerçek Etiketler:
    ├── Brier Score = mean((prob - label)^2)
    ├── ECE (Expected Calibration Error):
    │   • 10 bin içine prob'ları yerleştir
    │   • Her bin: |mean_predicted - observed_freq| * (n_bin/N)
    │   • ECE = sum(bin ECE'leri)
    └── Reliability Diagram PNG
```

### 1.13 Bootstrap CI (%95)

```
y_true, y_prob, threshold → n_bootstrap=2000 iterasyon:
    ├── Rastgele örnekleme (yerine koyarak)
    ├── Her bootstrap: AUC, SENS, SPEC, PPV, NPV, ACC, F1
    └── CI: [percentile(2.5), percentile(97.5)]
```
# Bölüm 2: MODEL MİMARİLERİ

## MODEL 1: SegFormer3D-MSCA (Multi-Scale Context Attention)
**Dosya:** `train_segformer3d.py`  
**Konfigürasyon:** lr=2e-4, batch=2, accum=8, epochs=100, mixup_alpha=0.4, weight_decay=1e-2

### 2.1 OverlapPatchEmbed3D
```
Girdi: (B, C_in, D, H, W)
    └── Conv3d(C_in→embed_dim, kernel=patch_size, stride=stride, pad=padding)
    └── Flatten spatial: (B, embed_dim, D', H', W') → (B, D'H'W', embed_dim)
    └── LayerNorm(embed_dim)
    └── Reshape geri: (B, embed_dim, D', H', W')
```

### 2.2 EfficientSelfAttention3D (SR-ratio ile verimli attention)
```
Girdi: (B, C, D, H, W)
    ├── x_flat = flatten+transpose → (B, N, C)  [N=D*H*W]
    ├── Q = Linear(C→C)(x_flat) → reshape → (B, heads, N, C//heads)
    ├── SR_ratio > 1:
    │   └── sr_x = Conv3d(C,C, k=sr_ratio, s=sr_ratio)(x) → flatten → LayerNorm
    │   └── KV = Linear(C→2C)(sr_x) → reshape → (B, 2, heads, N_reduced, C//heads)
    ├── Attention = softmax(Q @ K^T / sqrt(head_dim))
    ├── Out = Attention @ V → reshape → (B, N, C)
    └── proj = Linear(C→C) → reshape → (B, C, D, H, W)
```

### 2.3 DropPath (Stochastic Depth)
```
Eğitim: keep_prob = 1 - drop_prob
    └── mask = Bernoulli(keep_prob) → x / keep_prob * mask
Test: identity (bypass)
```

### 2.4 MixFFN3D (Karışık Feed-Forward)
```
Girdi (B, C, D, H, W):
    ├── Conv3d(C→4C, k=1)  [pointwise]
    ├── Conv3d(4C→4C, k=3, pad=1, groups=4C)  [depthwise]
    ├── GELU + Dropout(0.1)
    └── Conv3d(4C→C, k=1)  [pointwise]
```

### 2.5 SegformerBlock3D
```
Girdi x:
    ├── norm1(x) → attn → drop_path → x = x + attn_out
    └── norm2(x) → mlp → drop_path → x = x + mlp_out
```

### 2.6 CBAM3D (Channel-Spatial Attention)
```
Girdi (B, C, D, H, W):
    ├── CHANNEL ATTENTION:
    │   ├── avg_pool = AdaptiveAvgPool3d(1) → (B,C,1,1,1)
    │   ├── max_pool = AdaptiveMaxPool3d(1) → (B,C,1,1,1)
    │   ├── fc: Conv3d(C→C//16→C) her ikisine uygulanır
    │   ├── channel_att = Sigmoid(avg_fc + max_fc)
    │   └── x = x * channel_att
    └── SPATIAL ATTENTION:
        ├── avg_spatial = mean(x, dim=1) → (B,1,D,H,W)
        ├── max_spatial = max(x, dim=1) → (B,1,D,H,W)
        ├── cat → (B,2,D,H,W)
        ├── Conv3d(2→1, k=7, pad=3)
        ├── spatial_att = Sigmoid(conv_out)
        └── x = x * spatial_att
```

### 2.7 SegFormer3DClassifier — Tam İleri Geçiş
```
Girdi (B,1,32,128,128):
    │
    ├── Stage 1: OverlapPatchEmbed3D(1→32, k=7,s=2,p=3) → (B,32,16,64,64)
    │            2x SegformerBlock3D(dim=32, heads=2, sr=4, dpr=[0.00,0.015])
    │
    ├── Stage 2: OverlapPatchEmbed3D(32→64, k=3,s=2,p=1) → (B,64,8,32,32)
    │            2x SegformerBlock3D(dim=64, heads=4, sr=2, dpr=[0.03,0.045])
    │
    ├── Stage 3: OverlapPatchEmbed3D(64→128, k=3,s=2,p=1) → (B,128,4,16,16)
    │            3x SegformerBlock3D(dim=128, heads=8, sr=2, dpr=[0.06,0.075,0.09])
    │            → x3 [kaydedildi: multi-scale fusion için]
    │
    ├── Stage 4: OverlapPatchEmbed3D(128→256, k=3,s=2,p=1) → (B,256,2,8,8)
    │            3x SegformerBlock3D(dim=256, heads=16, sr=1, dpr=[0.105,0.12,0.135])
    │            → x4
    │
    ├── CBAM3D(256) → x4_attended
    │
    ├── MULTI-SCALE FUSION:
    │   ├── feat3 = AdaptiveAvgPool3d(1)(x3) → flatten → (B,128)
    │   ├── feat4 = AdaptiveAvgPool3d(1)(x4_attended) → flatten → (B,256)
    │   └── fused = cat([feat3, feat4], dim=1) → (B,384)
    │
    └── CLASSIFIER HEAD:
        ├── Linear(384→256) + BN + GELU + Dropout(0.5)
        ├── Linear(256→128) + BN + GELU + Dropout(0.3)
        └── Linear(128→2) → logits (B,2)
```

### 2.8 SegFormer3D MAE Pretraining (Stage 1)
```
Tüm 241 Hasta (etiketsiz)
    │
    ├── SegFormer3DMAE encoder (aynı 4 stage mimarisi)
    ├── Maskeleme: mask_ratio=0.75
    │   └── random mask (B,1,D//2,H//2,W//2) → upsample → (B,1,D,H,W)
    │   └── x_masked = x * (1 - mask_up)
    ├── Encode x_masked → f (B,256,2,8,8)
    ├── Decoder (ConvTranspose3d):
    │   ├── 256→128 (upsample x2)
    │   ├── 128→64  (upsample x2)
    │   ├── 64→32   (upsample x2)
    │   └── 32→1    (upsample x2) → (B,1,D,H,W)
    ├── F.interpolate → orijinal boyuta
    ├── Kayıp = MSE(recon * mask, x * mask)  [sadece maskeli bölgeler]
    ├── AdamW lr=1e-3 + CosineAnnealingLR, 60 epoch
    └── Encoder ağırlıkları → segformer3d_mae_encoder.pt
```

### 2.9 Fine-tuning Optimizer (Layer-wise LR)
```
Backbone params (patch_embed + blocks):  lr = 2e-4 * 0.1 = 2e-5
Head params (cbam + fc):                 lr = 2e-4
    └── AdamW(weight_decay=1e-2)
    └── Warmup(10 epoch) + Cosine Annealing
```

---

## MODEL 2: MAE-Pretrained TinyTransformer3D
**Dosya:** `train_mae_tinytransformer.py`  
**Konfigürasyon:** embed_dim=192, depth=8, heads=6, lr=2e-4, batch=2, accum=8, epochs=100

### 2.10 3D Sin-Cos Pozisyonel Encoding
```
Grid (D', H', W'):
    ├── embed_dim 3'e bölünür → dim_each = 64
    ├── Her eksen için 1D sin-cos: [M, 64]
    │   omega = 1 / 10000^(2k/dim)
    │   out[m,k] = m * omega[k]
    │   → [sin(out), cos(out)]
    └── Birleştir: [D'*H'*W', embed_dim=192]
        (sabit buffer, öğrenilmez)
```

### 2.11 PatchEmbed3D
```
Girdi (B,1,32,128,128):
    └── Conv3d(1→192, kernel=(2,8,8), stride=(2,8,8))
        → (B,192,16,16,16)  [D'=16, H'=16, W'=16]
    └── flatten: (B, 4096, 192)
    └── LayerNorm(192)
    → tokens (B, N=4096, 192) + grid_shape (16,16,16)
```

### 2.12 TransformerBlock
```
Girdi x (B, N+1, 192):
    ├── norm1(x) → MultiheadAttention(192, heads=6, dropout=0.2)
    ├── x = x + attn_out
    ├── norm2(x) → MLP:
    │   Linear(192→768) + GELU + Dropout(0.2)
    │   Linear(768→192) + Dropout(0.2)
    └── x = x + mlp_out
```

### 2.13 TinyTransformer3DEncoder
```
Girdi (B,1,32,128,128):
    ├── PatchEmbed3D → tokens (B,4096,192)
    ├── tokens += patch_pos_embed (sin-cos, sabit)
    ├── CLS token (B,1,192) += cls_pos_embed
    ├── tokens = cat([cls, tokens]) → (B,4097,192)
    ├── Dropout(0.1)
    ├── 8x TransformerBlock(dim=192, heads=6)
    └── LayerNorm → tokens (B,4097,192)
```

### 2.14 TinyTransformer3DClassifier
```
Encoder çıktısı (B,4097,192):
    ├── cls_out = tokens[:,0]       → (B,192)
    ├── mean_out = tokens[:,1:].mean(dim=1) → (B,192)
    ├── fused = cat([cls_out, mean_out]) → (B,384)
    └── HEAD:
        ├── LayerNorm(384)
        ├── Linear(384→256) + GELU + Dropout(0.4)
        ├── Linear(256→128) + GELU + Dropout(0.2)
        └── Linear(128→2) → logits
```

### 2.15 MAE Pretraining — MAEDecoder (Stage 1)
```
Tüm 241 Hasta (etiketsiz, + external test)
    │
    ├── mask_ratio = 0.60 (daha az maske → zengin anatomik temsil)
    ├── Rastgele permütasyon: ids_shuffle, ids_restore, ids_keep, ids_mask
    ├── Visible tokens = embed[ids_keep] → (B, keep, 192)
    ├── Hedef: patchify_pixels(x) → ham piksel yamaları (B, N, patch_vol)
    │
    ├── Encoder: [cls + visible_tokens] → 8x TransformerBlock → (B, 1+keep, 192)
    │
    ├── MAEDecoder:
    │   ├── Linear(192→96)
    │   ├── mask_token (öğrenilir, B, N-keep, 96)
    │   ├── Unshuffle: gather(ids_restore) → orijinal sıra
    │   ├── += patch_pos_embed (sin-cos, 96 dim)
    │   ├── [cls + patches] → 2x TransformerBlock(96, heads=2)
    │   ├── LayerNorm
    │   └── Linear(96→patch_vol) → pred_all (B, N, patch_vol)
    │
    ├── Kayıp = MSE(pred_masked, target_masked)
    │   [Sadece maskeli yamalar karşılaştırılır]
    ├── AdamW lr=1e-4, CosineAnnealingLR, 80 epoch
    └── encoder.state_dict() → mae_pretrained_encoder.pt
```

### 2.16 Fine-tuning (MAE → Classifier)
```
mae_pretrained_encoder.pt → model.encoder.load_state_dict()
    └── ClinicalFocalLoss(pos_weight=1.5, gamma=1.5, smoothing=0.10)
    └── AdamW(lr=2e-4, wd=1e-2) — TÜM parametreler (backbone dahil)
    └── Warmup(10) + Cosine + SWA(epoch 70+)
```

---

## MODEL 3: Attention-SwinUNETR (AG-MSF)
**Dosya:** `train_attention_swinunetr.py`  
**Konfigürasyon:** lr=1e-4, batch=4, accum=4, epochs=100, mixup=0.1

### 2.17 SwinUNETR Backbone (MONAI)
```
Pretrained: model_swinvit.pt (abdominal CT önceden eğitilmiş)
    └── SwinViT 5 Aşama:
        Stage 0: feature_size=48   → (B,48,D,H,W)
        Stage 1: feature_size=96   → (B,96,D/2,H/2,W/2)
        Stage 2: feature_size=192  → (B,192,D/4,H/4,W/4)
        Stage 3: feature_size=384  → (B,384,D/8,H/8,W/8)
        Stage 4: feature_size=768  → (B,768,D/16,H/16,W/16)
```

### 2.18 ChannelAttention
```
Birleştirilmiş özellik vektörü x (B,1488):
    ├── Linear(1488→93) + GELU
    ├── Linear(93→1488) + Sigmoid
    └── attended = x * attention_weights
```

### 2.19 AttentionSwinUNETR3D — Tam İleri Geçiş
```
Girdi (B,1,32,128,128):
    ├── hidden = backbone.swinViT(x)
    │   → [h0,h1,h2,h3,h4] (5 aşama çıktısı)
    │
    ├── MULTI-SCALE GLOBAL POOLING:
    │   ├── f0 = GAP(h0) → flatten → LayerNorm(48)  → (B,48)
    │   ├── f1 = GAP(h1) → flatten → LayerNorm(96)  → (B,96)
    │   ├── f2 = GAP(h2) → flatten → LayerNorm(192) → (B,192)
    │   ├── f3 = GAP(h3) → flatten → LayerNorm(384) → (B,384)
    │   └── f4 = GAP(h4) → flatten → LayerNorm(768) → (B,768)
    │
    ├── fused = cat([f0..f4]) → (B,1488)
    ├── attended = ChannelAttention(1488)(fused)
    │
    └── CLASSIFIER:
        ├── LayerNorm(1488)
        ├── Linear(1488→512) + GELU + Dropout(0.40)
        ├── Linear(512→128) + GELU + Dropout(0.20)
        └── Linear(128→2) → logits
```

### 2.20 Progressive Fine-tuning Stratejisi
```
epoch 1-14: Backbone DONDURULMUŞ
    └── Sadece classifier + channel_attention eğitiliyor
    └── optimizer: AdamW(classifier_params, lr=1e-4)

epoch 15+: Backbone ÇÖZÜLÜYOR
    └── Tüm parametreler eğitiliyor
    └── optimizer yeniden oluşturuluyor: AdamW(all_params, lr=1e-5)
    └── [Backbone daha küçük lr ile güncelleniyor]
```

---

## MODEL 4: SwinUNETR Linear Probe
**Dosya:** `train_swinunetr_linearprobe.py`  
**Konfigürasyon:** lr=5e-4, batch=2, accum=8, epochs=100, ~50K param

### 2.21 SwinLinearProbe Mimarisi
```
Pretrained SwinViT (TAMAMEN DONDURULMUŞ):
    └── torch.no_grad() ile hidden states çıkarılır
        Stage 3: h3 → (B,384,D/8,H/8,W/8)
        Stage 4: h4 → (B,768,D/16,H/16,W/16)
    │
    ├── f3 = GAP(h3) → flatten → (B,384)
    ├── f4 = GAP(h4) → flatten → (B,768)
    ├── p3 = GELU(Linear(384→32)(f3)) → (B,32)
    ├── p4 = GELU(Linear(768→64)(f4)) → (B,64)
    ├── feat = cat([p3,p4]) → (B,96)
    └── HEAD:
        ├── LayerNorm(96)
        ├── Linear(96→128) + GELU + Dropout(0.50)
        └── Linear(128→2) → logits

Eğitilebilir: ~50K parametre (sadece proj3, proj4, head)
Dondurulmuş: ~28M parametre (tüm SwinViT)
```

---

## MODEL 5: Radiomics-lite + Klasik ML Baseline
**Dosya:** `baseline_radiomics_ml.py`

### 2.22 Öznitelik Çıkarımı
```
HDF5 → vol (D,H,W):
    │
    ├── FIRST-ORDER (12 öznitelik):
    │   mean, std, min, max, median
    │   p10, p25, p75, p90
    │   skewness, kurtosis, entropy
    │
    ├── GLCM DOKU (6x öznitelik, orta 8 aksiyel kesit):
    │   contrast, dissimilarity, homogeneity
    │   energy, correlation, ASM
    │   [distances=[1,2], angles=[0,45,90,135], levels=32]
    │
    └── META ÖZNİTELİKLER (H5 attrs'dan, 5 öznitelik):
        roi_volume_fraction, n_valid_slices_used
        z_coverage_fraction, pixel_spacing_x, slice_spacing

Toplam: ~23 öznitelik/hasta → radiomics_lite_features.csv cache
```

### 2.23 Baseline Modeller
```
baseline_logreg:
    ├── StandardScaler()
    └── LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)

baseline_rf:
    └── RandomForestClassifier(n_estimators=300, max_depth=5,
                               class_weight="balanced", seed=42)
```

---

## BÖLÜM 3: 5-FOLD CROSS-VALIDATION PARADİGMASI

### 3.1 Veri Bölme Stratejisi
```
241 Hasta Toplam:
    ├── Stratified 5-Fold:
    │   ├── Her fold: ~160 train, ~40 val
    │   └── Bölme hasta-seviyesinde (data leakage yok)
    └── Bağımsız External Test: ~37 hasta
        └── ASLA eğitimde kullanılmaz

Her Fold Dosyaları (datas/):
    ├── fold_1_train.csv → fold_1_val.csv
    ├── fold_2_train.csv → fold_2_val.csv
    ├── ...
    └── external_test_set.csv
```

### 3.2 Aggregate OOF (Out-of-Fold) Sonuçları
```
Fold 1 val tahminleri
    + Fold 2 val tahminleri
    + Fold 3 val tahminleri
    + Fold 4 val tahminleri
    + Fold 5 val tahminleri
    = OOF DataFrame (tüm 165 hasta, her biri bir kez val'da)
        → Youden threshold → Metrikler
        → %95 Bootstrap CI
        → ROC/PR eğrileri
        → Confusion matrix
```

---

## BÖLÜM 4: Q1 MASTER KARŞILAŞTIRMA TABLOSU

### 4.1 generate_q1_master_table.py
```
experiments/ klasörü taranır:
    Her alt klasörde:
        external_test/ensemble_probs.csv    → mevcut?
        external_test/q1_external_test_metrics.csv → mevcut?
    
    Bulunan modeller:
        ├── segformer3d         → "SegFormer3D-MSCA"
        ├── attention_swinunetr → "Attention-SwinUNETR"
        ├── mae_tinytransformer → "MAE-TinyTransformer3D"
        ├── swinunetr_lp        → "SwinUNETR Linear-Probe"
        ├── baseline_logreg     → "Radiomics-lite + LogReg"
        └── baseline_rf         → "Radiomics-lite + RandomForest"
    
    Her model → "Ensemble (@Youden)" satırı çekilir:
        AUC-ROC, [CI], Sensitivity, Specificity
        PPV, NPV, F1, Accuracy, Brier, ECE
    
    DeLong pairwise: NxN model çifti → z, p_value
    
    Çıktı:
        experiments/q1_master_comparison.csv
        experiments/q1_delong_pairwise.csv
```

---

## BÖLÜM 5: HYPERPARAMETER KARŞILAŞTIRMA TABLOSU

| Parametre | SegFormer3D-MSCA | MAE-TinyTransformer | Attention-SwinUNETR | SwinUNETR-LP |
|-----------|-----------------|---------------------|---------------------|--------------|
| LR | 2e-4 | 2e-4 | 1e-4 | 5e-4 |
| Backbone LR | 2e-5 (0.1x) | 2e-4 (tam) | 1e-5 (epoch15+) | Dondurulmuş |
| Batch Size | 2 | 2 | 4 | 2 |
| Grad Accum | 8 | 8 | 4 | 8 |
| Eff. Batch | 16 | 16 | 16 | 16 |
| Epochs | 100 | 100 | 100 | 100 |
| Patience | 25 | 25 | 25 | 25 |
| SWA Start | Epoch 70 | Epoch 70 | Epoch 70 | Epoch 50 |
| Focal γ | 1.5 | 1.5 | 2.0 | 1.5 |
| Label Smooth | 0.10 | 0.10 | 0.10 | 0.10 |
| pos_weight | 1.5 | 1.5 | 1.5 | 1.5 |
| MixUp α | 0.4 | 0.4 | 0.1 | 0.0 |
| Weight Decay | 1e-2 | 1e-2 | 1e-2 | 1e-2 |
| Warmup | 10 epoch | 10 epoch | 10 epoch | 5 epoch |
| EMA α | 0.3 | 0.3 | 0.3 | - |
| Params (M) | ~3M | ~15M | ~29M | ~0.05M |
| Pretrain | MAE (60ep) | MAE (80ep) | SwinViT-CT | SwinViT-CT |
| Min SENS | 0.80 | 0.80 | 0.80 | 0.85 |
| Min SPEC | 0.50 | 0.50 | 0.50 | 0.60 |

---

## BÖLÜM 6: TAM EĞİTİM PIPELINE AKIŞI (HER MODEL İÇİN)

```
STAGE 0: Veri Hazırlığı
    generate_master_splits.py
        └── 241 hasta → stratified 5-fold + external test
        └── CSV'ler → datas/ klasörüne kaydedilir

STAGE 1: Pretraining (SegFormer3D ve MAE-Tiny için)
    241 hasta (labels ignored):
        ├── SegFormer3D: MAE encoder (mask_ratio=0.75, 60 epoch)
        └── MAE-Tiny: MAEModel (mask_ratio=0.60, 80 epoch)

STAGE 2: 5-Fold Cross-Validation
    For fold in [1,2,3,4,5]:
        ├── train_df = fold_X_train.csv
        ├── val_df = fold_X_val.csv
        ├── DataLoader(train, augment=True)
        ├── DataLoader(val, augment=False)
        ├── Model oluştur + Pretrained ağırlık yükle
        ├── For epoch in [1..100]:
        │   ├── train_one_epoch (grad accum, clip)
        │   ├── evaluate_model (TTA, threshold seçim)
        │   ├── composite hesapla + EMA yumuşat
        │   ├── Checkpoint seçimi (dual-constraint penalty)
        │   ├── SWA güncelle (epoch≥70)
        │   └── Early stopping (epoch≥70 AND patience≥25)
        ├── SWA finalize + BN update
        ├── SWA vs best EMA karşılaştır → final checkpoint
        └── Fold metrikleri + PNG kaydedilir

STAGE 3: Aggregate OOF
    Tüm fold val tahminleri birleştirilir
        └── Youden threshold → OOF metrikleri + CI

STAGE 4: External Test Ensemble
    5 fold checkpoint:
        ├── Her fold: test seti üzerinde TTA tahmini
        ├── Ensemble = mean(5 fold probs)
        ├── @0.5, @Youden, @90%+Sens → metrikler + CI
        ├── Confusion matrix + ROC/PR + Kalibrasyon PNG
        └── q1_external_test_metrics.csv

STAGE 5: Q1 Master Tablo
    generate_q1_master_table.py
        ├── Tüm model ensemble_probs.csv topla
        ├── Master metrik tablosu (AUC sıralı)
        └── DeLong pairwise p-value matrisi
```
