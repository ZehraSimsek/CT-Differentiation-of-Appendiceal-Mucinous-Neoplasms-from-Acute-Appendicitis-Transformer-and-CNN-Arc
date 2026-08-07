# 📋 Çok Tekrarlı Eğitim Protokolü (Multi-Run Protocol)
## Q1 Makale — Tüm Grup A ve Grup B Modelleri İçin Geçerlidir

---

## ⚠️ EN KRİTİK KURAL — FOLD ATAMASI DEĞİŞMEZ

> **`fold_1_train.csv` … `fold_5_val.csv` ve `external_test_set.csv` dosyaları bir kez üretildi ve artık KESİNLİKLE DEĞİŞTİRİLMEZ.**  
> `create_new_dataset_csvs.py` bir daha çalıştırılmaz.

Seed'ler **hangi hastanın hangi fold'a düştüğünü** hiçbir zaman etkilemez.  
Fold atamaları CSV dosyalarında sabit yazılıdır; eğitim başında sadece **okunur**.

Seed'lerin kontrol ettiği şeyler:
- Model ağırlıklarının başlangıç değerleri (weight initialization)
- Batch sırası (DataLoader shuffle)
- Augmentasyon rastgeleliği

---

## 🎯 Amaç

5-Fold CV tek başına yalnızca **veri bölünmesi varyansını** ölçer.  
Buna ek olarak **3 bağımsız run**, model ağırlığı başlangıcı ve batch sıralamasından kaynaklanan varyansı da yakalar.

**Sonuç formatı:** `AUC = 0.853 ± 0.012  (n=3 runs × 5-fold CV)`

Bu format Q1 hakemlerine:
- ✅ Sonuçların tekrarlanabilir olduğunu
- ✅ Rastgele şansa bağlı olmadığını
- ✅ Mimariler arası karşılaştırmanın kontrollü ve adil yapıldığını kanıtlar.

---

## 🔑 Seed Kuralı — EN ÖNEMLİ KURAL

> **Tüm modeller (SwinUNETR-LP, AG-MSF, MAE-Tiny3D, SegFormer3D-MSCA, UNet++, DenseNet121, EfficientNet-B0) aynı seed'leri kullanır.**

```python
N_RUNS   = 3
RUN_SEEDS = [42, 123, 456]
```

| Model | Run 1 | Run 2 | Run 3 |
|---|:---:|:---:|:---:|
| SwinUNETR-LP      | seed=**42** | seed=**123** | seed=**456** |
| AG-MSF            | seed=**42** | seed=**123** | seed=**456** |
| MAE-Tiny3D        | seed=**42** | seed=**123** | seed=**456** |
| SegFormer3D-MSCA  | seed=**42** | seed=**123** | seed=**456** |
| UNet++            | seed=**42** | seed=**123** | seed=**456** |
| DenseNet121       | seed=**42** | seed=**123** | seed=**456** |
| EfficientNet-B0   | seed=**42** | seed=**123** | seed=**456** |

### Neden aynı seed?
Amacımız modeller arası **adil karşılaştırma**. Tek değişken mimari olmalı, diğer her şey sabit:
- Fold bölünmesi → Zaten sabit (`fold_1_train.csv` vb. önceden üretildi, değiştirilmez)
- Model init seed → Aynı seed → Sabit
- Batch sırası → Aynı seed → Sabit

---

## 📂 Çıktı Klasör Yapısı

Her model kendi `experiments_*/` klasörü içinde şu yapıya sahip olacak:

```
experiments_q1_128/
└── <model_adı>/
    ├── run_01/           ← seed=42
    │   ├── fold_01/
    │   │   ├── best_model.pt
    │   │   ├── best_val_predictions.csv
    │   │   └── training_history.csv
    │   ├── fold_02/ … fold_05/
    │   └── aggregate_oof/
    │       └── oof_predictions.csv
    ├── run_02/           ← seed=123
    │   └── (aynı yapı)
    ├── run_03/           ← seed=456
    │   └── (aynı yapı)
    └── train_log.txt
```

---

## 🖥️ Her Eğitim Betiğine Eklenmesi Gereken Kod

Her modelin eğitim dosyasının (`train_*.py`) `main()` fonksiyonu başına aşağıdaki blok eklenecek:

```python
# ============================================================
# MULTI-RUN SABITLERI — Tüm modeller için aynı, değiştirme!
# ============================================================
N_RUNS    = 3
RUN_SEEDS = [42, 123, 456]

def main():
    ...
    run_ext_aucs = []

    for run_idx, seed in enumerate(RUN_SEEDS, start=1):
        print(f"\n{'#'*80}")
        print(f"  RUN {run_idx}/{N_RUNS}  |  Seed={seed}")
        print(f"{'#'*80}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        # Python hash seed için (tam deterministik):
        # import random; random.seed(seed)

        run_dir = BASE_DIR / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- 5-fold döngüsü burada çalışır ---
        all_preds = []
        for fold_idx in range(1, CONFIG["n_splits"] + 1):
            ...

    # --- ÖZET ---
    arr = np.array(run_ext_aucs)
    print(f"AUC = {arr.mean():.3f} ± {arr.std():.3f}  (n={N_RUNS} runs × 5-fold CV)")
```

---

## 📊 Makale Metodoloji Bölümü İçin Taslak Metin

> *"To assess the stability of model performance with respect to random initialization and mini-batch ordering, all models were trained with three independent runs. Each run employed an identical set of random seeds (42, 123, 456 for runs 1–3, respectively), applied uniformly across all architectures to ensure a controlled and fair comparison. Cross-validation fold assignments were fixed prior to all experiments and held constant across all models and runs. Final performance metrics are reported as mean ± standard deviation computed over the three runs."*

---

## 🔬 Augmentasyon Protokolü — 4 Model İçin Birebir Aynı

> Tüm modeller `AppendixH5Dataset` (shared_utils.py) üzerinden aynı augmentasyon pipeline'ını kullanır.  
> **Hiçbir modele özel augmentasyon eklenmez.**

### Preprocessing (Augmentasyon Öncesi — Her Zaman Uygulanır)
```
Z-score normalizasyon: vol = (vol - vol.mean()) / (vol.std() + 1e-8)
Giriş boyutu: (1, 32, 128, 128)  →  [Channel, Depth, Height, Width]
```

### Augmentasyon (Sadece Train Split — augment=True)

| # | İşlem | Uygulama Olasılığı | Parametre |
|---|---|:---:|---|
| 1 | **3D Axial Flip** (Depth ekseni) | p=0.50 | `torch.flip(dim=1)` |
| 2 | **3D Horizontal Flip** (H ekseni) | p=0.50 | `torch.flip(dim=2)` |
| 3 | **3D Vertical Flip** (W ekseni) | p=0.50 | `torch.flip(dim=3)` |
| 4 | **90° Rotasyon Simülasyonu** (H-W düzleminde) | p=0.40 | k ∈ {1,2,3} |
| 5 | **Intensity Shift** | p=0.60 | shift ∈ [-0.25, +0.25] |
| 6 | **Intensity Scale** | p=0.60 | scale ∈ [0.75, 1.25] |
| 7 | **Gaussian Noise** | p=0.60 | σ=0.02 |
| 8 | **3D Cutout** (Random Occlusion) | p=0.40 | patch: 4×16×16 voksel |
| 9 | **Depth Crop & Resize** | p=0.30 | ±D/6 kırpma, trilinear geri boyutlandırma |

### Val / External Test
```
augment=False → Sadece preprocessing (Z-score), augmentasyon uygulanmaz
TTA → KAPALI (use_tta=False)
```

---

## ⚙️ Hiperparametre Standardizasyonu

> Aşağıdaki parametreler **sabit tutulur**. Mimari kısıtlar nedeniyle farklı olan değerler not ile belirtilmiştir.

| Parametre | SwinUNETR-LP | AG-MSF | MAE-Tiny3D | SegFormer3D-MSCA |
|---|:---:|:---:|:---:|:---:|
| **n_epochs** | 100 | 100 | 100 | 100 |
| **patience** | 25 | 25 | 20 | 25 |
| **batch_size** | 2 | 2 | 2 | 2 |
| **accum_steps** | 8 | 4 | 8 | 8 |
| **Efektif batch** | 16 | 16 | 16 | 16 |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW |
| **LR** | 1e-3* | 1e-4 | 2e-4 | 3e-4 |
| **Weight Decay** | 5e-3 | 5e-3 | 1e-2 | 1e-2 |
| **Scheduler** | Warmup+Cosine | Warmup+Cosine | Warmup+Cosine | Warmup+Cosine |
| **Warmup Epochs** | 5 | 10 | 10 | 10 |
| **Loss** | ClinicalFocalLoss | ClinicalFocalLoss | ClinicalFocalLoss | ClinicalFocalLoss |
| **Focal γ** | 1.0 | 1.0 | 2.0 | 1.0 |
| **Label Smoothing** | 0.05 | 0.05 | 0.05 | 0.05 |
| **MixUp α** | — | 0.05 | 0.40 | 0.10 |
| **SWA** | ❌ Kaldırıldı | ❌ Kaldırıldı | ❌ Kaldırıldı | ❌ Kaldırıldı |
| **TTA** | ❌ Kaldırıldı | ❌ Kaldırıldı | ❌ Kaldırıldı | ❌ Kaldırıldı |
| **min_epochs_save** | 3 | 3 | 3 | 3 |

> \* SwinUNETR-LP backbone'u **dondurulmuştur** (frozen), yalnızca classification head eğitilir. Yüksek LR bu nedenle güvenlidir.

### Paylaşılan Sabit Parametreler (shared_utils.py → SHARED_CONFIG)

```python
"random_seed":       42          # Fold üretimi için (run seed'inden AYRI)
"n_splits":          5           # 5-fold CV
"num_workers":       4
"sensitivity_first": True
"min_sens_floor":    0.80        # Threshold seçiminde minimum SENS kısıtı
"min_spec_floor":    0.50        # Threshold seçiminde minimum SPEC kısıtı
"n_bootstrap":       2000        # 95% CI hesabı için bootstrap iterasyonu
"focal_gamma":       2.0         # ClinicalFocalLoss varsayılanı
"label_smoothing":   0.05
```

---



Eğitim başlamadan önce şu kodu çalıştırarak fold atamalarının değişmediğini doğrulayın:

```python
import pandas as pd, hashlib
from pathlib import Path

datas = Path("datas")
print("=== FOLD ATAMALARI MD5 KONTROLÜ ===")
for fold in range(1, 6):
    for split in ["train", "val"]:
        f = datas / f"fold_{fold}_{split}.csv"
        h = hashlib.md5(f.read_bytes()).hexdigest()
        print(f"fold_{fold}_{split}.csv → {h}")
```

Beklenen MD5 hash değerleri (referans — ilk üretimde not alın):

| Dosya | MD5 Hash |
|---|---|
| fold_1_train.csv | `f54cc614082867ec0528992dd86b7f6f` |
| fold_1_val.csv   | `af3865e3516b035cd4c5464c5cff1f7d` |
| fold_2_train.csv | `0d4bd47467e27b287cf89f96ec339a3d` |
| fold_2_val.csv   | `8aa6d207ddb025829be00dac74419631` |
| fold_3_train.csv | `7d20739202c7078d55ced20e5c84d20d` |
| fold_3_val.csv   | `6874d9dfcbcc86a2836da050aacb3fcf` |
| fold_4_train.csv | `0ec66b13e42e0b498659663348c43daf` |
| fold_4_val.csv   | `e8b1676ffa9d1f6ca6d19c43da146f29` |
| fold_5_train.csv | `0ee4fd63fe7ee3d93ebf35763cb8d108` |
| fold_5_val.csv   | `4827c02f3572091b7723ba3e0a43aed8` |

---

## ✅ Kontrol Listesi (Ekip için)

- [ ] `create_new_dataset_csvs.py` **bir daha çalıştırılmadı** (en kritik kural!)
- [ ] Fold MD5 hash'leri referans değerlerle eşleşiyor
- [ ] `external_test_set.csv` hiçbir modelin eğitimine dahil edilmedi
- [ ] `N_RUNS = 3`, `RUN_SEEDS = [42, 123, 456]` her eğitim betiğinde aynı
- [ ] Her model `run_01/`, `run_02/`, `run_03/` alt klasörlerine yazıyor
- [ ] MAE pretraining (`mae_pretrained_encoder.pt`) run'lar arasında paylaşılıyor (tek seferlik)
- [ ] Son raporda her model için `mean ± std` formatında AUC raporlandı

---

*Hazırlayan: Antigravity AI Asistanı | Tarih: Ağustos 2026*
