import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "final"))

from shared_utils import SHARED_CONFIG, AppendixH5Dataset

import final.train_swinunetr_linearprobe as swin_lp
import final.train_attention_swinunetr as attn_swin
import final.train_mae_tinytransformer as mae_tiny
import final.train_segformer3d as segformer

# Yollar - Yeni Multi-Run Dizinine
EXP_DIR = Path("experiments_multirun_2d")
TEST_CSV = Path("datas_2d") / "external_test_set.csv"
OUT_DIR = EXP_DIR / "XAI_GradCAM_Master"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# WRAPPERS FOR GRAD-CAM HOOKS
# ==========================================================

class GradCAMExtractorBase:
    def get_cam(self, x, class_idx):
        logits = self.forward_hook(x)
        self.model.zero_grad()
        loss = logits[0, class_idx]
        loss.backward()
        
        grads = self.feature.grad[0] # [C, D, H, W]
        feats = self.feature[0]      # [C, D, H, W]
        
        alpha = grads.mean(dim=[1, 2, 3], keepdim=True)
        cam = (alpha * feats).sum(dim=0)
        cam = F.relu(cam)
        
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / cam.max()
            
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=x.shape[2:], mode="trilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy(), float(torch.softmax(logits, dim=1)[0, 1].item())


class SwinLPWrapper(GradCAMExtractorBase):
    def __init__(self, model):
        self.model = model
        for p in self.model.parameters(): p.requires_grad = True
            
    def forward_hook(self, x):
        hidden = self.model.backbone.swinViT(x, self.model.backbone.normalize)
        self.feature = hidden[4]
        self.feature.retain_grad()
        f3 = self.model.gap(hidden[3]).flatten(1)
        f4 = self.model.gap(self.feature).flatten(1)
        p3 = F.gelu(self.model.proj3(f3))
        p4 = F.gelu(self.model.proj4(f4))
        feat = torch.cat([p3, p4], dim=1)
        return self.model.head(feat)


class AttnSwinWrapper(GradCAMExtractorBase):
    def __init__(self, model):
        self.model = model
        for p in self.model.parameters(): p.requires_grad = True
            
    def forward_hook(self, x):
        hidden = self.model.backbone.swinViT(x, self.model.backbone.normalize)
        self.feature = hidden[4]
        self.feature.retain_grad()
        
        feats = []
        for i, h in enumerate(hidden):
            f = self.model.gap(h if i != 4 else self.feature).flatten(1)
            f = self.model.norms[i](f)
            feats.append(f)
        fused = torch.cat(feats, dim=1)
        attended = self.model.channel_attention(fused)
        return self.model.classifier(attended)


class MAETinyWrapper(GradCAMExtractorBase):
    def __init__(self, model):
        self.model = model
        for p in self.model.parameters(): p.requires_grad = True
            
    def forward_hook(self, x):
        # Forward pass through encoder
        tokens = self.model.encoder(x) # [B, N+1, C]
        cls_out = tokens[:, 0]
        spatial_tokens = tokens[:, 1:] # [B, 4096, 192]
        
        # D=16, H=16, W=16 -> (16*16*16 = 4096)
        B, N, C = spatial_tokens.shape
        spatial_3d = spatial_tokens.transpose(1, 2).view(B, C, 16, 16, 16)
        
        self.feature = spatial_3d
        self.feature.retain_grad()
        
        # Reconstruct for head to maintain gradient path
        re_flattened = self.feature.view(B, C, N).transpose(1, 2)
        mean_out = re_flattened.mean(dim=1)
        
        fused = torch.cat([cls_out, mean_out], dim=1)
        return self.model.head(fused)


class SegFormerWrapper(GradCAMExtractorBase):
    def __init__(self, model):
        self.model = model
        for p in self.model.parameters(): p.requires_grad = True
            
    def forward_hook(self, x):
        x = self.model.patch_embed1(x)
        x = self.model.block1(x)
        
        x = self.model.patch_embed2(x)
        x = self.model.block2(x)
        
        x = self.model.patch_embed3(x)
        x3 = self.model.block3(x)
        
        x = self.model.patch_embed4(x3)
        x4 = self.model.block4(x)
        x4 = self.model.cbam(x4)
        
        self.feature = x4
        self.feature.retain_grad()
        
        f3 = self.model.pool(x3).flatten(1)
        f4 = self.model.pool(self.feature).flatten(1)
        
        fused = torch.cat([f3, f4], dim=1)
        return self.model.fc(fused)

# ==========================================================

def load_model(name, builder, wrapper_cls, device):
    model = builder().to(device)
    ckpt_path = EXP_DIR / name / "run_01" / "fold_01" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"HATA: {ckpt_path} bulunamadı!")
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    wrapper = wrapper_cls(model)
    return wrapper


def main():
    print("=== Multi-Model Q1 Master Grad-CAM Başlatılıyor ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        "SwinUNETR-LP": load_model("swinunetr_lp", swin_lp.build_model, SwinLPWrapper, device),
        "AG-MSF": load_model("ag_msf", attn_swin.build_model, AttnSwinWrapper, device),
        "MAE-Tiny3D": load_model("mae_tiny3d", mae_tiny.build_model, MAETinyWrapper, device),
        "SegFormer3D": load_model("segformer3d_msca", segformer.build_model, SegFormerWrapper, device),
    }

    df = pd.read_csv(TEST_CSV)
    sample_cases = df.reset_index(drop=True)
    ds = AppendixH5Dataset(sample_cases, augment=False, config=SHARED_CONFIG)

    # Her model için ensemble tahminlerini ve Youden eşiğini önceden yükle
    MODEL_DIRS = {
        "SwinUNETR-LP":  "swinunetr_lp",
        "AG-MSF":        "ag_msf",
        "MAE-Tiny3D":    "mae_tiny3d",
        "SegFormer3D":   "segformer3d_msca",
    }
    # Her model için Youden eşiğini Tablo 2'den statik olarak alıyoruz.
    FROZEN_THRESHOLDS = {
        "SwinUNETR-LP":  0.624,
        "AG-MSF":        0.602,
        "MAE-Tiny3D":    0.562,
        "SegFormer3D":   0.548,
    }
    
    model_predictions = {}
    for m_name, folder in MODEL_DIRS.items():
        pid_probs = {}
        probs_path = EXP_DIR / folder / "run_01" / "aggregate_oof" / "oof_predictions.csv"
        # Since we want external test probability, wait. The artifact should show the predicted probability for this patient!
        # Actually, let's just use the forward pass probability computed below (c_data["prob"]).
        
        threshold = FROZEN_THRESHOLDS.get(m_name, 0.5)
        model_predictions[m_name] = {"probs": pid_probs, "threshold": threshold}
        print(f"  {m_name}: threshold={threshold:.3f}")
    
    for i in range(len(sample_cases)):
        row = sample_cases.iloc[i]
        pid = row["patient_id"]
        true_label = int(row["label"])
        label_str = "Mucinous" if true_label == 1 else "Appendicitis"
        
        print(f"İşleniyor: {pid} ({label_str})")
        vol_tensor = ds[i]["image"].unsqueeze(0).to(device)
        vol_tensor.requires_grad = True
        
        with h5py.File(row["h5_path"], "r") as f:
            mask_3d = f["mask"][:]
        
        img_3d = vol_tensor.squeeze().detach().cpu().numpy()
        D = img_3d.shape[0]  # Kesit sayısı (genelde 32)
        
        # 1. Önce modellerin 3D CAM verilerini tek seferde hesapla
        cam_data = {}
        for m_name, extractor in models.items():
            if extractor is not None:
                try:
                    cam_3d, prob = extractor.get_cam(vol_tensor, class_idx=1)
                    cam_data[m_name] = {"cam_3d": cam_3d, "prob": prob}
                except Exception as e:
                    print(f"  Hata {m_name} hesaplanırken: {e}")
                    cam_data[m_name] = None
            else:
                cam_data[m_name] = None
        
        # 2. Hasta için klasör oluştur
        pat_dir = OUT_DIR / f"{pid}_{label_str}"
        pat_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Her kesit için görsel oluştur
        for slice_idx in range(D):
            img_slice = img_3d[slice_idx]
            mask_slice = mask_3d[slice_idx]

            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            axes[0].imshow(img_slice, cmap="gray")
            if mask_slice.max() > 0:
                axes[0].contour(mask_slice, colors='yellow', linewidths=1.5, alpha=0.8)
                roi_text = " (ROI: Yellow)"
            else:
                roi_text = " (No ROI)"
            axes[0].set_title(f"Original CT{roi_text} - Slice {slice_idx+1}/{D}\nTrue: {label_str}", fontsize=24, fontweight='bold')
            axes[0].axis('off')
            
            col = 1
            for m_name in models.keys():
                c_data = cam_data.get(m_name)
                if c_data is None:
                    axes[col].set_title(f"{m_name}\n(Hata/Eksik)", fontsize=24, fontweight='bold')
                    axes[col].axis('off')
                else:
                    cam_3d_model = c_data["cam_3d"]
                    cam_slice = cam_3d_model[slice_idx]

                    # Bu modelin ensemble tahmini ve Youden eşiği
                    m_pred = model_predictions.get(m_name, {})
                    if pid in m_pred.get("probs", {}):
                        prob = m_pred["probs"][pid]
                    else:
                        prob = c_data["prob"]  # fallback: tek fold forward pass
                    m_threshold = m_pred.get("threshold", 0.5)

                    pred_str = "Mucinous" if prob >= m_threshold else "Appendicitis"
                    axes[col].imshow(img_slice, cmap="gray")
                    axes[col].imshow(cam_slice, cmap="jet", alpha=0.5)
                    color = "green" if (prob >= m_threshold and true_label == 1) or (prob < m_threshold and true_label == 0) else "red"
                    axes[col].set_title(f"{m_name}\nPred: {pred_str} ({prob:.2f})", color=color, fontsize=24, fontweight='bold')
                    axes[col].axis('off')
                col += 1
                
            save_path = pat_dir / f"slice_{slice_idx:02d}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            
        print(f"  --> Kaydedildi: {pat_dir.name}/ ({D} kesit)")
        
    print(f"\nTüm işlem tamamlandı! Çıktılar dizini: {OUT_DIR}")

if __name__ == "__main__":
    main()
