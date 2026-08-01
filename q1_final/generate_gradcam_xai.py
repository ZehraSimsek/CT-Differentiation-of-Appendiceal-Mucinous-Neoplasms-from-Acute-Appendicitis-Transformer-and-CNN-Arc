import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Proje dizinini yola ekle ki importlar çalışsın
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared_utils import SHARED_CONFIG, AppendixH5Dataset
from train_swinunetr_linearprobe import build_model

# Yollar
EXP_DIR = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/experiments_q1_128")
TEST_CSV = EXP_DIR.parent / "datas" / "external_test_set.csv"
MODEL_PATH = EXP_DIR / "swinunetr_lp" / "fold_01" / "best_model.pt"
OUT_DIR = EXP_DIR / "xai_gradcam"
OUT_DIR.mkdir(parents=True, exist_ok=True)

class HookBasedGradCAM:
    """SwinUNETR-LP için özel (Gradients by-pass) Grad-CAM üretici"""
    def __init__(self, model):
        self.model = model
        self.feature = None

        # Modeli gradyan takibine açıyoruz (eğitimde kapalıydı)
        for p in self.model.backbone.swinViT.parameters():
            p.requires_grad = True

    def forward_hook(self, x):
        # Orijinal forward, ancak no_grad() OLMADAN
        hidden = self.model.backbone.swinViT(x, self.model.backbone.normalize)
        self.feature = hidden[4] # Son Swin bloğu özellikleri [B, C, D, H, W]
        self.feature.retain_grad()
        
        f3 = self.model.gap(hidden[3]).flatten(1)
        f4 = self.model.gap(self.feature).flatten(1)
        p3 = F.gelu(self.model.proj3(f3))
        p4 = F.gelu(self.model.proj4(f4))
        feat = torch.cat([p3, p4], dim=1)
        return self.model.head(feat)

    def get_cam(self, x, class_idx):
        logits = self.forward_hook(x)
        self.model.zero_grad()
        loss = logits[0, class_idx]
        loss.backward()
        
        # Grad-CAM formülü
        grads = self.feature.grad[0] # [C, D, H, W]
        feats = self.feature[0]      # [C, D, H, W]
        
        alpha = grads.mean(dim=[1, 2, 3], keepdim=True) # [C, 1, 1, 1]
        cam = (alpha * feats).sum(dim=0) # [D, H, W]
        cam = F.relu(cam)
        
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / cam.max()
            
        # Orijinal 3D boyuta geri büyüt (Trilinear interpolation)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=x.shape[2:], mode="trilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()

def main():
    print("Grad-CAM XAI oluşturucu başlatılıyor...")
    if not MODEL_PATH.exists():
        print(f"Hata: Model ağırlığı bulunamadı -> {MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    cam_extractor = HookBasedGradCAM(model)

    df = pd.read_csv(TEST_CSV)
    # Sadece 2 müsinöz (pozitif) ve 2 apandisit (negatif) hasta seçelim
    pos_cases = df[df["label"] == 1].head(2)
    neg_cases = df[df["label"] == 0].head(2)
    sample_cases = pd.concat([pos_cases, neg_cases])

    ds = AppendixH5Dataset(sample_cases, augment=False, config=SHARED_CONFIG)
    
    for i in range(len(sample_cases)):
        row = sample_cases.iloc[i]
        pid = row["patient_id"]
        true_label = int(row["label"])
        label_str = "Mucinous" if true_label == 1 else "Appendicitis"
        print(f"İşleniyor: {pid} ({label_str})")

        vol_tensor = ds[i]["image"].unsqueeze(0).to(device)
        vol_tensor.requires_grad = True
        
        # Model tahmini
        logits = cam_extractor.forward_hook(vol_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred_label = 1 if prob >= 0.5 else 0
        pred_str = "Mucinous" if pred_label == 1 else "Appendicitis"
        
        # Gerçek sınıf için Grad-CAM üret
        cam_3d = cam_extractor.get_cam(vol_tensor, class_idx=1) # Hep 1'e (Tümör) olan ilgisine bakalım
        
        # Görselleştirme için orta kesiti bul
        img_3d = vol_tensor.squeeze().detach().cpu().numpy()
        D = img_3d.shape[0]
        mid_slice = D // 2
        
        img_slice = img_3d[mid_slice]
        cam_slice = cam_3d[mid_slice]
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_slice, cmap="gray")
        axes[0].set_title(f"Original MRI/CT Slice\nLabel: {label_str}")
        axes[0].axis('off')
        
        axes[1].imshow(cam_slice, cmap="jet")
        axes[1].set_title(f"Grad-CAM (Attention Map)\nFocus on 'Mucinous'")
        axes[1].axis('off')
        
        axes[2].imshow(img_slice, cmap="gray")
        axes[2].imshow(cam_slice, cmap="jet", alpha=0.5)
        axes[2].set_title(f"Overlay\nPred: {pred_str} (Prob: {prob:.2f})")
        axes[2].axis('off')
        
        save_path = OUT_DIR / f"gradcam_{pid}_{label_str}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Kaydedildi: {save_path}")

    print(f"\nİşlem tamamlandı! Görseller {OUT_DIR} klasöründe.")

if __name__ == "__main__":
    main()
