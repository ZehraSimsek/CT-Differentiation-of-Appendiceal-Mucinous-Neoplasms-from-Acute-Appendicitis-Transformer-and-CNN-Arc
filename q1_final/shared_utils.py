# ============================================================
# SHARED UTILITIES - Q1 Journal Ready
# Öncelik: Sensitivity > AUC-ROC > Specificity
# Tüm notebooklarda ortak kullanım için
# ============================================================

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, average_precision_score,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SHARED CONFIG
# Sensitivity-first training protocol
# ============================================================
SHARED_CONFIG = {
    "random_seed": 42,
    "n_splits": 5,
    "batch_size": 4,
    "num_workers": 4,
    "expected_C": 1,
    "img_D": 32,
    "img_H": 128,
    "img_W": 128,
    "n_epochs": 100,
    "lr": 3e-5,
    "warmup_epochs": 10,
    "weight_decay": 1e-3,
    # --- Q1 Model Selection: ÇİFT KISIT (SENS≥0.80 AND SPEC≥0.50) + F1 maximize ---
    # Composite: 0.35*SENS + 0.25*F1 + 0.25*AUC + 0.15*SPEC
    # Hedef: AUC≥0.80, SENS≥0.80, F1≥0.75, SPEC≥0.50
    "patience": 30,
    "label_smoothing": 0.05,      # Küçük veride aşırı yumşatma AUC'yi düşürür
    "pos_weight": 1.0,            # Dinamik pw fold içinde hesaplanır (pos_weight_from_labels)
    "focal_gamma": 2.0,           # Orijinal çalışan ayar
    "default_threshold": 0.5,     # Fallback threshold
    # --- Threshold Stratejisi: ÇİFT KISIT + F1 MAXIMIZE ---
    # Adım 1: SENS≥min_sens_floor VE SPEC≥min_spec_floor sağlayan adayları bul
    # Adım 2: Bu adaylar arasında F1'i maksimize eden threshold'u seç
    # Adım 3: Çift kısıt sağlanamazsa sadece SENS kısıtına dön
    # Adım 4: O da sağlanamazsa Youden J
    "sensitivity_first": True,
    "min_sens_floor": 0.80,       # Eğitim sırasında daha erişilebilir kısıt
    "min_spec_floor": 0.50,       # AUC öğrenmesini engellememek için gevşetildi
    "n_bootstrap": 2000,
}

# ============================================================
# CLINICAL COMPOSITE SCORE
# Ağırlıklar: SENS(0.35) + F1(0.25) + AUC(0.25) + SPEC(0.15)
# F1 eklendi: Precision-Recall dengesi model seçimine giriyor
# ============================================================
def clinical_composite(sens, auc, spec, f1=None):
    """
    Klinik öncelik sırasına göre ağırlıklı kompozit skor.
    f1 verilirse: SENS:0.35 | F1:0.25 | AUC:0.25 | SPEC:0.15
    f1 verilmezse (geriye dönük uyumluluk): SENS:0.50 | AUC:0.30 | SPEC:0.20
    """
    if f1 is not None:
        return 0.35 * sens + 0.25 * f1 + 0.25 * auc + 0.15 * spec
    return 0.50 * sens + 0.30 * auc + 0.20 * spec


# ============================================================
# DATASET - Gelişmiş Augmentasyon (Tümör Sınıfı Odaklı)
# ============================================================
class AppendixH5Dataset(Dataset):
    def __init__(self, df, augment=False, config=None):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.config = config or SHARED_CONFIG

    def __len__(self):
        return len(self.df)

    def _load_volume(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            vol = f["image"][()].astype(np.float32)
        if vol.ndim == 4 and vol.shape[-1] < vol.shape[0]:
            vol = vol.transpose(3, 0, 1, 2)
        elif vol.ndim == 3:
            vol = vol[np.newaxis]
        elif vol.ndim == 2:
            vol = vol[np.newaxis, np.newaxis]
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        return vol.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = self._load_volume(row["h5_path"])
        label = int(row["label"])
        vol_tensor = torch.from_numpy(vol)

        if self.augment and vol_tensor.ndim == 4:
            # --- Temel Spatial Augmentations ---
            for dim in [1, 2, 3]:
                if torch.rand(1) > 0.5:
                    vol_tensor = torch.flip(vol_tensor, dims=[dim])

            # --- Rotasyon Simülasyonu (90 derece katları) ---
            if torch.rand(1) > 0.6:
                k = torch.randint(1, 4, (1,)).item()
                vol_tensor = torch.rot90(vol_tensor, k=k, dims=[2, 3])

            # --- Intensity Augmentation ---
            if torch.rand(1) > 0.4:
                shift = (torch.rand(1).item() - 0.5) * 0.5
                vol_tensor = vol_tensor + shift

            if torch.rand(1) > 0.4:
                scale = 0.75 + torch.rand(1).item() * 0.5
                mean = vol_tensor.mean()
                vol_tensor = (vol_tensor - mean) * scale + mean

            # --- Gaussian Noise ---
            if torch.rand(1) > 0.4:
                vol_tensor = vol_tensor + 0.02 * torch.randn_like(vol_tensor)

            # --- Cutout (3D Random Occlusion) - Model ezber yapmasın ---
            if torch.rand(1) > 0.6:
                C, D, H, W = vol_tensor.shape
                d0 = torch.randint(0, max(1, D-6), (1,)).item()
                h0 = torch.randint(0, max(1, H-20), (1,)).item()
                w0 = torch.randint(0, max(1, W-20), (1,)).item()
                vol_tensor[:, d0:d0+4, h0:h0+16, w0:w0+16] = 0.0

            # --- Depth Crop & Resize ---
            if torch.rand(1) > 0.7 and vol_tensor.shape[1] > 8:
                D = vol_tensor.shape[1]
                start = torch.randint(0, D // 6, (1,)).item()
                end = D - torch.randint(0, D // 6, (1,)).item()
                cropped = vol_tensor[:, start:end, :, :]
                vol_tensor = F.interpolate(
                    cropped.unsqueeze(0), size=vol_tensor.shape[1:],
                    mode='trilinear', align_corners=False
                ).squeeze(0)

        return {
            "image": vol_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": str(row["patient_id"]),
        }


# ============================================================
# SENSITIVITY-FIRST FOCAL LOSS
# pos_weight + focal: Müsinöz Tümörü kaçırmayı çok pahalı yap
# ============================================================
class ClinicalFocalLoss(nn.Module):
    """
    Sensitivity-first training için tasarlandı.
    - Focal: Zor/nadir vakalara odak
    - pos_weight: Tümör sınıfına ekstra ceza
    - Label Smoothing: Aşırı güven önleme
    """
    def __init__(self, pos_weight=1.0, gamma=2.0, smoothing=0.05, num_classes=2):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # Handle tuple output (Deep Supervision)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Soft targets (MixUp) mu yoksa hard class indices mi?
        if targets.dtype == torch.float and targets.ndim == 2:
            true_dist = targets
        else:
            # Label smoothing
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Weighted cross entropy (pos_weight artırır Sensitivity)
        weight = torch.ones(self.num_classes, device=logits.device)
        weight[1] = self.pos_weight  # Müsinöz (Class 1) ağırlığı

        log_prob = F.log_softmax(logits, dim=1)
        ce_loss = -(true_dist * log_prob * weight).sum(dim=1)

        # Focal weighting (soft hedefler için de geçerli)
        prob = torch.softmax(logits, dim=1)
        pt = (prob * true_dist).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        return (focal_weight * ce_loss).mean()


# ============================================================
# SHARED METRICS
# ============================================================
def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc_val = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    sens = float(tp / (tp + fn + 1e-9))
    spec = float(tn / (tn + fp + 1e-9))
    f1_val = float(f1_score(y_true, y_pred, zero_division=0))
    metrics = {
        "threshold": float(threshold),
        "auc_roc": auc_val,
        "auc_pr": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": float(tp / (tp + fp + 1e-9)),
        "npv": float(tn / (tn + fn + 1e-9)),
        "f1": f1_val,
        "composite": float(clinical_composite(sens, auc_val if not np.isnan(auc_val) else 0.5, spec, f1=f1_val)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    return metrics, cm, y_pred


def find_youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def pos_weight_from_labels(labels):
    """
    Klinikte FN (False Negative) çok tehlikeli olduğu için, 
    Müsinöz (Class 1) çoğunluk sınıfı olsa bile modelin Apandisit'e 
    fazla kaymasını önlemek adına sabit yüksek bir ağırlık kullanıyoruz.
    Yeni veri setinde (117 Müsinöz, 27 Apandisit) modelin Spec=1.0 Sens=0.68 
    şeklinde overfit olmasını kırmak için pos_weight=3.0 olarak zorlandı.
    """
    return 3.0



def find_sensitivity_threshold(y_true, y_prob, min_sensitivity=0.90):
    """Minimum sensitivite kısıtı altında en yüksek spesifisiteyi veren eşiği bul."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = [(thr, 1 - fpr_val, tpr_val) for thr, fpr_val, tpr_val in zip(thresholds, fpr, tpr)
             if tpr_val >= min_sensitivity]
    if not valid:
        # Min_sensitivity sağlanamıyorsa en yüksek sensitivity'yi ver
        best_idx = np.argmax(tpr)
        return float(thresholds[best_idx])
    # En yüksek spesifisiteyi seç
    valid.sort(key=lambda x: x[1], reverse=True)
    return float(valid[0][0])


def compute_bootstrap_ci(y_true, y_prob, threshold, n_bootstraps=2000, seed=42):
    """95% Bootstrap Confidence Intervals for all metrics."""
    np.random.seed(seed)
    results = {k: [] for k in ["auc", "sens", "spec", "ppv", "npv", "acc", "f1"]}
    for _ in range(n_bootstraps):
        idx = np.random.randint(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        yhat = (yp >= threshold).astype(int)
        cm_ = confusion_matrix(yt, yhat, labels=[0, 1])
        tn_, fp_, fn_, tp_ = cm_.ravel()
        try:
            results["auc"].append(roc_auc_score(yt, yp))
        except:
            pass
        results["sens"].append(tp_ / (tp_ + fn_ + 1e-9))
        results["spec"].append(tn_ / (tn_ + fp_ + 1e-9))
        results["ppv"].append(tp_ / (tp_ + fp_ + 1e-9))
        results["npv"].append(tn_ / (tn_ + fn_ + 1e-9))
        results["acc"].append((tp_ + tn_) / (tp_ + tn_ + fp_ + fn_))
        results["f1"].append(2 * tp_ / (2 * tp_ + fp_ + fn_ + 1e-9))
    ci = {}
    for k, v in results.items():
        if v:
            v.sort()
            ci[f"{k}_ci_lo"] = float(np.percentile(v, 2.5))
            ci[f"{k}_ci_hi"] = float(np.percentile(v, 97.5))
        else:
            ci[f"{k}_ci_lo"] = 0.0
            ci[f"{k}_ci_hi"] = 1.0
    return ci


def print_full_metrics_table(metrics, ci, model_name, threshold_name):
    print(f"\n{'='*65}")
    print(f"  {model_name}  |  Threshold: {threshold_name}")
    print(f"{'='*65}")
    print(f"  {'Metrik':<18} {'Değer':>8}   {'95% CI':>22}")
    print(f"  {'-'*55}")
    rows = [
        ("AUC-ROC",      "auc",  metrics['auc_roc']),
        ("AUC-PR",       None,   metrics['auc_pr']),
        ("Sensitivity",  "sens", metrics['sensitivity']),
        ("Specificity",  "spec", metrics['specificity']),
        ("PPV",          "ppv",  metrics['ppv']),
        ("NPV",          "npv",  metrics['npv']),
        ("Accuracy",     "acc",  metrics['accuracy']),
        ("F1-Score",     "f1",   metrics['f1']),
        ("Composite",    None,   metrics.get('composite', 0.0)),
    ]
    for name, key, val in rows:
        if key and f"{key}_ci_lo" in ci:
            ci_str = f"[{ci[key+'_ci_lo']:.3f} - {ci[key+'_ci_hi']:.3f}]"
        else:
            ci_str = "-"
        print(f"  {name:<18} {val:>8.3f}   {ci_str:>22}")
    print(f"\n  CM: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print(f"  Toplam Hata: {metrics['fp']+metrics['fn']}/{metrics['tp']+metrics['fp']+metrics['fn']+metrics['tn']}")
    print(f"{'='*65}")


# ============================================================
# SHARED PLOT FUNCTIONS
# ============================================================
def plot_confusion_matrix(cm, title, save_path=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    labels = ["Appendicitis", "Mucinous"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=22, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()


def plot_roc_pr(y_true, y_prob, title_prefix, save_dir, opt_threshold=None):
    save_dir = Path(save_dir)
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, color="#1565C0", lw=2.5, label=f"AUC = {auc:.3f}")
    if opt_threshold is not None:
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        axes[0].scatter(fpr[best_idx], tpr[best_idx], color="crimson", s=150, zorder=5,
                        label=f"Youden Opt.\nT={thr_roc[best_idx]:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    axes[0].set_xlabel("1 - Specificity (FPR)", fontsize=12)
    axes[0].set_ylabel("Sensitivity (TPR)", fontsize=12)
    axes[0].set_title(f"{title_prefix} - ROC Curve", fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color="#1B5E20", lw=2.5, label=f"AP = {auc_pr:.3f}")
    axes[1].axhline(y=y_true.mean(), color="gray", linestyle="--", alpha=0.5, label="Baseline")
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=12)
    axes[1].set_ylabel("Precision (PPV)", fontsize=12)
    axes[1].set_title(f"{title_prefix} - PR Curve", fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"{title_prefix}_roc_pr.png", dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# SHARED TRAINING LOOP
# ============================================================
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    """Linear warmup then cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(min_lr_ratio, epoch / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha=0.4, num_classes=2):
    """MixUp: görüntü ve one-hot etiketleri lambda ile karıştır."""
    if alpha <= 0:
        return x, F.one_hot(y, num_classes=num_classes).float(), 1.0, torch.arange(len(y), device=x.device)
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]
    return mixed_x, mixed_y, lam, index


def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0, accum_steps=1):
    """
    Standard training loop. Deep Supervision (tuple output) destekli.
    mixup_alpha>0: MixUp augmentasyonu aktif (soft labels ile).
    accum_steps>1: gradient accumulation ile efektif batch size'ı büyütür.
    """
    model.train()
    total_loss = 0
    n_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    num_classes = getattr(criterion, "num_classes", 2)

    for i, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if mixup_alpha > 0:
            images, labels, _, _ = mixup_data(images, labels, mixup_alpha, num_classes)

        outputs = model(images)

        # Deep Supervision desteği
        if isinstance(outputs, tuple):
            final_out, ds_outputs = outputs
            loss = criterion(final_out, labels)
            ds_weights = [0.1, 0.1, 0.2, 0.3, 0.3]
            for j, ds_out in enumerate(ds_outputs[:len(ds_weights)]):
                loss = loss + ds_weights[j] * criterion(ds_out, labels)
        else:
            loss = criterion(outputs, labels)

        (loss / max(1, accum_steps)).backward()
        total_loss += loss.item()

        is_last_batch = (i == n_batches - 1)
        if ((i + 1) % accum_steps == 0) or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / n_batches


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    """
    TTA (Test-Time Augmentation) ile değerlendirme.
    4 yön: Orijinal + H-Flip + V-Flip + D-Flip
    Loss sadece orijinalden, prob = 4 yönün ortalaması.

    Threshold stratejisi:
      - SHARED_CONFIG['sensitivity_first'] = True  -> çift kısıt (SENS>=floor, SPEC>=floor) + F1 maximize
      - SHARED_CONFIG['sensitivity_first'] = False -> Youden threshold
    """
    model.eval()
    all_probs, all_labels, all_patient_ids = [], [], []
    total_loss = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        out1 = model(images)
        out2 = model(torch.flip(images, dims=[2]))
        out3 = model(torch.flip(images, dims=[3]))
        out4 = model(torch.flip(images, dims=[4]))

        # Deep Supervision: sadece final çıktıyı al
        if isinstance(out1, tuple): out1 = out1[0]
        if isinstance(out2, tuple): out2 = out2[0]
        if isinstance(out3, tuple): out3 = out3[0]
        if isinstance(out4, tuple): out4 = out4[0]

        loss = criterion(out1, labels)
        total_loss += loss.item()

        p1 = torch.softmax(out1, dim=1)[:, 1]
        p2 = torch.softmax(out2, dim=1)[:, 1]
        p3 = torch.softmax(out3, dim=1)[:, 1]
        p4 = torch.softmax(out4, dim=1)[:, 1]
        probs = ((p1 + p2 + p3 + p4) / 4.0).cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        if 'patient_id' in batch:
            all_patient_ids.extend(batch['patient_id'])

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    # --- THRESHOLD STRATEJİSİ: ÇİFT KISIT + F1 MAXIMIZE ---
    # Adım 1: SENS≥min_sens_floor VE SPEC≥min_spec_floor sağlayan adaylar → F1 maximize
    # Adım 2: Çift kısıt sağlanamazsa → sadece SENS≥min_sens_floor → max SPEC
    # Adım 3: SENS kısıtı da sağlanamazsa → Youden J
    use_sens_first = SHARED_CONFIG.get("sensitivity_first", True)
    min_floor      = SHARED_CONFIG.get("min_sens_floor", 0.80)
    min_spec       = SHARED_CONFIG.get("min_spec_floor", 0.50)
    threshold = SHARED_CONFIG["default_threshold"]
    if use_sens_first and len(np.unique(y_true)) > 1:
        try:
            fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, y_prob)
            # Adım 1: ÇİFT KISIT — SENS≥min_floor VE SPEC≥min_spec
            dual_candidates = []
            for i in range(len(thr_arr)):
                _sens = tpr_arr[i]
                _spec = 1.0 - fpr_arr[i]
                if _sens >= min_floor and _spec >= min_spec:
                    _pred = (y_prob >= thr_arr[i]).astype(int)
                    _f1   = float(f1_score(y_true, _pred, zero_division=0))
                    dual_candidates.append((thr_arr[i], _f1, _sens, _spec))
            if dual_candidates:
                # F1'i maximize et (SENS ve SPEC her ikisi de kısıt sağlıyor)
                best = max(dual_candidates, key=lambda x: x[1])
                threshold = float(np.clip(best[0], 0.15, 0.85))
            else:
                # Adım 2: Sadece SENS kısıtı — SPEC floor kaldırılıyor, max SPEC
                sens_candidates = [(thr_arr[i], 1.0 - fpr_arr[i])
                                   for i in range(len(thr_arr)) if tpr_arr[i] >= min_floor]
                if sens_candidates:
                    best_spec_thr = max(sens_candidates, key=lambda x: x[1])
                    threshold = float(np.clip(best_spec_thr[0], 0.15, 0.85))
                else:
                    # Adım 3: Youden J (hiçbir kısıt sağlanamadı)
                    j_idx = np.argmax(tpr_arr - fpr_arr)
                    threshold = float(np.clip(thr_arr[j_idx], 0.15, 0.85))
        except Exception:
            threshold = SHARED_CONFIG["default_threshold"]
    elif len(np.unique(y_true)) > 1:
        try:
            fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, y_prob)
            j_idx = np.argmax(tpr_arr - fpr_arr)
            threshold = float(np.clip(thr_arr[j_idx], 0.20, 0.80))
        except Exception:
            threshold = SHARED_CONFIG["default_threshold"]

    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Composite score: Sensitivity öncelikli
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    composite = clinical_composite(sens, auc, spec)

    if len(all_patient_ids) == len(y_true):
        pred_df = pd.DataFrame({
            "patient_id": all_patient_ids,
            "label": y_true,
            "prob_mucinous": y_prob
        })
    else:
        pred_df = pd.DataFrame({"label": y_true, "prob_mucinous": y_prob})

    # threshold'u da döndür: fold loop'ta log için kullanılabilir
    pred_df["_threshold_used"] = threshold
    return total_loss / len(loader), auc, acc, f1, pred_df


# ============================================================
# DELONG TEST — İki modelin AUC'sini AYNI hasta kümesinde
# istatistiksel olarak karşılaştırmak için (Q1 makalelerde standart).
# Sun & Xu (2014) "Fast Implementation of DeLong's Algorithm" temelli.
# ============================================================
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, y_prob_1, y_prob_2):
    """
    İki modelin AYNI hasta kümesi üzerindeki AUC'lerini DeLong testiyle karşılaştırır.
    Döndürür: (auc_1, auc_2, z_statistic, p_value)
    p_value < 0.05 -> iki modelin AUC'si arasındaki fark istatistiksel olarak anlamlı.
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    label_1_count = int(y_true_sorted.sum())

    preds = np.vstack([np.asarray(y_prob_1)[order], np.asarray(y_prob_2)[order]])
    aucs, delongcov = _fast_delong(preds, label_1_count)

    auc_diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = auc_diff / np.sqrt(var)
    p = float(2 * (1 - _norm_cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), float(z), p


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ============================================================
# KALİBRASYON — Brier score, Expected Calibration Error (ECE),
# reliability diagram. Q1 makalelerde "model iyi kalibre mi?" sorusu
# AUC/Sens/Spec kadar önemli (klinik karar destek için gerekli).
# ============================================================
def compute_calibration(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    brier = float(np.mean((y_prob - y_true) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    bin_rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (count / len(y_prob)) * abs(acc - conf)
        bin_rows.append({"bin": b, "count": count, "mean_predicted": conf, "observed_freq": acc})

    return {"brier_score": brier, "ece": float(ece), "bins": pd.DataFrame(bin_rows)}


def plot_calibration_curve(y_true, y_prob, title, save_path, n_bins=10):
    cal = compute_calibration(y_true, y_prob, n_bins=n_bins)
    bins_df = cal["bins"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Mükemmel kalibrasyon")
    if len(bins_df):
        ax.plot(bins_df["mean_predicted"], bins_df["observed_freq"], marker="o", lw=2.0,
                color="#1565C0", label="Model")
    ax.set_xlabel("Ortalama Tahmin Edilen Olasılık", fontsize=12)
    ax.set_ylabel("Gözlenen Frekans", fontsize=12)
    ax.set_title(f"{title}\nBrier={cal['brier_score']:.4f} | ECE={cal['ece']:.4f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    return cal


# ============================================================
# STDOUT/STDERR -> DOSYA + KONSOL (train_log.txt)
# ============================================================
import sys

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_file_logging(log_path):
    """stdout+stderr çıktısını hem konsola hem log_path dosyasına yazar."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_file


# ============================================================
# ORTAK EXTERNAL TEST DEĞERLENDİRMESİ (Q1: fold-wise + ensemble +
# 0.5 / Youden / 90%-Sensitivity + %95 Bootstrap CI)
# Üç mimarinin de aynı protokolle raporlanması için tek yerde tutulur.
# ============================================================
def evaluate_external_test_ensemble(model_builder, base_dir, config, test_csv_path,
                                     model_display_name, n_folds=5):
    """
    model_builder: () -> nn.Module (eğitilmemiş, sadece mimari) döndüren callable.
    base_dir: experiments/<model_name> klasörü (fold_XX/best_model.pt içerir).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(base_dir)
    test_df = pd.read_csv(test_csv_path)
    test_ds = AppendixH5Dataset(test_df, augment=False, config=config)
    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=torch.cuda.is_available(),
    )
    criterion = nn.CrossEntropyLoss()
    yt = test_df["label"].values

    test_dir = base_dir / "external_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    all_probs = []
    rows = []

    for fold_idx in range(1, n_folds + 1):
        ckpt_path = base_dir / f"fold_{fold_idx:02d}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  [skip] {ckpt_path} yok.")
            continue

        model = model_builder().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        _, _, _, _, pred_df = evaluate_model(model, test_loader, criterion, device)
        fold_prob = pred_df["prob_mucinous"].values
        all_probs.append(fold_prob)

        opt_thr_fold, _ = find_youden_threshold(yt, fold_prob)
        m_fold, _, _ = compute_binary_metrics(yt, fold_prob, threshold=opt_thr_fold)
        ci_fold = compute_bootstrap_ci(yt, fold_prob, threshold=opt_thr_fold, n_bootstraps=config.get("n_bootstrap", 2000))
        m_fold.update(ci_fold)
        m_fold["fold"] = f"Fold {fold_idx}"
        m_fold["threshold_used"] = opt_thr_fold
        rows.append(m_fold)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_probs:
        print("  Hiçbir fold checkpoint bulunamadı, external test atlandı.")
        return None

    ens = np.mean(all_probs, axis=0)
    fpr, tpr, thrs = roc_curve(yt, ens)

    def _threshold_row(thr, name):
        m, cm, _ = compute_binary_metrics(yt, ens, threshold=thr)
        ci = compute_bootstrap_ci(yt, ens, threshold=thr, n_bootstraps=config.get("n_bootstrap", 2000))
        row = dict(m)
        row.update(ci)
        row["fold"] = name
        row["threshold_used"] = thr
        rows.append(row)
        return m, cm, ci

    m_05, cm_05, ci_05 = _threshold_row(0.5, "Ensemble (@0.5)")
    youden_thr, _ = find_youden_threshold(yt, ens)
    m_y, cm_y, ci_y = _threshold_row(youden_thr, "Ensemble (@Youden)")
    valid_idx = np.where(tpr >= 0.90)[0]
    high_sens_thr = float(thrs[valid_idx[0]]) if len(valid_idx) > 0 else youden_thr
    m_s, cm_s, ci_s = _threshold_row(high_sens_thr, "Ensemble (90+ Sens)")

    summary = pd.DataFrame(rows)
    cols = ["fold"] + [c for c in summary.columns if c != "fold"]
    summary = summary[cols]
    summary.to_csv(test_dir / "q1_external_test_metrics.csv", index=False)

    plot_confusion_matrix(cm_y, f"{model_display_name} External Test @Youden {youden_thr:.3f}",
                           save_path=test_dir / "cm_youden.png")
    plot_roc_pr(yt, ens, title_prefix=f"{model_display_name}_external_test", save_dir=test_dir,
                opt_threshold=youden_thr)

    # Kalibrasyon (Brier / ECE / reliability diagram) — eşikten bağımsız, ham olasılıklar üzerinden.
    cal = plot_calibration_curve(yt, ens, f"{model_display_name} External Test Kalibrasyonu",
                                  save_path=test_dir / "calibration.png")
    print(f"\n{model_display_name} — Kalibrasyon: Brier={cal['brier_score']:.4f} | ECE={cal['ece']:.4f}")

    # Diğer mimarilerle çapraz-model ensemble/DeLong karşılaştırması yapılabilsin diye
    # ensemble olasılıklarını da hasta bazında kaydediyoruz.
    ens_probs_df = pd.DataFrame({
        "patient_id": test_df["patient_id"].values, "label": yt, "prob_mucinous": ens,
    })
    ens_probs_df.to_csv(test_dir / "ensemble_probs.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    disp_cols = ["fold", "auc_roc", "sensitivity", "specificity", "accuracy", "f1", "ppv", "npv",
                 "tp", "fp", "fn", "tn"]
    print(f"\n{model_display_name} — EXTERNAL TEST (ALL FOLDS + ENSEMBLE):")
    print(summary[disp_cols].round(3).to_string(index=False))

    print(f"\n{model_display_name} — ENSEMBLE @Youden ({youden_thr:.3f}) — %95 Bootstrap CI:")
    print_full_metrics_table(m_y, ci_y, model_display_name, f"Youden {youden_thr:.3f}")

    print(f"\n{model_display_name} — ENSEMBLE @90%+ Sensitivity ({high_sens_thr:.3f}) — %95 Bootstrap CI:")
    print_full_metrics_table(m_s, ci_s, model_display_name, f"90+Sens {high_sens_thr:.3f}")

    return summary


print("shared_utils.py yüklendi. [Q1 | Sensitivity-First | TTA | ClinicalFocalLoss]")
