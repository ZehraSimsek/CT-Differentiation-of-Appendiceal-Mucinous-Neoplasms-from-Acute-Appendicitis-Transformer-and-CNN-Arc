import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_utils import SHARED_CONFIG, AppendixH5Dataset

# Modellerin importları
import train_swinunetr_linearprobe as swin_lp
import train_attention_swinunetr as attn_swin
import train_segformer3d as segformer

# Yollar
EXP_DIR = Path("/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/experiments_q1_128")
TEST_CSV = EXP_DIR.parent / "datas" / "external_test_set.csv"
OUT_DIR = EXP_DIR / "xai_gradcam_comparison"
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
        for p in self.model.backbone.swinViT.parameters(): p.requires_grad = True
            
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
        for p in self.model.backbone.swinViT.parameters(): p.requires_grad = True
            
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


def load_model(name, builder, wrapper_cls, device):
    model = builder().to(device)
    ckpt_path = EXP_DIR / name / "fold_01" / "best_model.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    
    threshold = 0.5
    csv_path = EXP_DIR / name / "external_test" / "q1_external_test_metrics.csv"
    if csv_path.exists():
        df_val = pd.read_csv(csv_path)
        row = df_val[df_val["fold"].astype(str).str.contains("Ensemble \(@Youden\)", na=False)]
        if not row.empty:
            threshold = float(row["threshold"].iloc[0])

    wrapper = wrapper_cls(model)
    wrapper.threshold = threshold
    return wrapper


def main():
    print("Multi-Model Grad-CAM Başlatılıyor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        "SwinUNETR-LP": load_model("swinunetr_lp", swin_lp.build_model, SwinLPWrapper, device),
        "Attention-Swin": load_model("attention_swinunetr", attn_swin.build_model, AttnSwinWrapper, device),
        "SegFormer3D": load_model("segformer3d", segformer.build_model, SegFormerWrapper, device),
    }

    df = pd.read_csv(TEST_CSV)
    sample_cases = df
    ds = AppendixH5Dataset(sample_cases, augment=False, config=SHARED_CONFIG)

    # Her model için gerçek ensemble tahminlerini yükle
    MODEL_DIRS_Q1 = {
        "SwinUNETR-LP":   "swinunetr_lp",
        "Attention-Swin": "attention_swinunetr",
        "SegFormer3D":    "segformer3d",
    }
    model_predictions = {}
    for m_name, folder in MODEL_DIRS_Q1.items():
        pid_probs = {}
        probs_path = EXP_DIR / folder / "external_test" / "ensemble_probs.csv"
        if probs_path.exists():
            df_p = pd.read_csv(probs_path)
            for _, r in df_p.iterrows():
                pid_probs[r["patient_id"]] = float(r["prob_mucinous"])

        threshold = 0.5
        metrics_path = EXP_DIR / folder / "external_test" / "q1_external_test_metrics.csv"
        if metrics_path.exists():
            df_m = pd.read_csv(metrics_path)
            yrow = df_m[df_m["fold"].astype(str).str.contains("Youden", na=False)]
            if not yrow.empty:
                threshold = float(yrow["threshold"].iloc[0])

        model_predictions[m_name] = {"probs": pid_probs, "threshold": threshold}
        print(f"  {m_name}: threshold={threshold:.3f}, {len(pid_probs)} hasta")
    
    for i in range(len(sample_cases)):
        row = sample_cases.iloc[i]
        pid = row["patient_id"]
        true_label = int(row["label"])
        label_str = "Mucinous" if true_label == 1 else "Appendicitis"
        
        print(f"\nİşleniyor: {pid} ({label_str})")
        vol_tensor = ds[i]["image"].unsqueeze(0).to(device)
        vol_tensor.requires_grad = True
        
        # Orijinal Maskeyi (ROI) Yükle
        with h5py.File(row["h5_path"], "r") as f:
            mask_3d = f["mask"][:]
        
        img_3d = vol_tensor.squeeze().detach().cpu().numpy()
        mid_slice = img_3d.shape[0] // 2
        img_slice = img_3d[mid_slice]
        mask_slice = mask_3d[mid_slice]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_slice, cmap="gray")
        if mask_slice.max() > 0:
            axes[0].contour(mask_slice, colors='yellow', linewidths=1.5, alpha=0.8)
            roi_text = " (ROI: Yellow)"
        else:
            roi_text = " (No ROI)"
            
        axes[0].set_title(f"Original MRI/CT{roi_text}\nTrue: {label_str}", fontsize=24)
        axes[0].axis('off')
        
        col = 1
        for m_name, extractor in models.items():
            if extractor is None:
                axes[col].set_title(f"{m_name}\n(Weights missing)", fontsize=24)
                axes[col].axis('off')
                col += 1
                continue
                
            try:
                cam_3d, _ = extractor.get_cam(vol_tensor, class_idx=1)
                cam_slice = cam_3d[mid_slice]

                # Gerçek ensemble tahmini
                m_pred = model_predictions.get(m_name, {})
                if pid in m_pred.get("probs", {}):
                    prob = m_pred["probs"][pid]
                else:
                    _, prob = extractor.get_cam(vol_tensor, class_idx=1)  # fallback
                m_threshold = m_pred.get("threshold", 0.5)

                pred_str = "Mucinous" if prob >= m_threshold else "Appendicitis"
                axes[col].imshow(img_slice, cmap="gray")
                axes[col].imshow(cam_slice, cmap="jet", alpha=0.5)
                if mask_slice.max() > 0:
                    axes[col].contour(mask_slice, colors='yellow', linewidths=1.5, alpha=0.8)

                color = "green" if (prob >= m_threshold and true_label == 1) or (prob < m_threshold and true_label == 0) else "red"
                axes[col].set_title(f"{m_name}\nPred: {pred_str} ({prob:.2f})", color=color, fontsize=24)
                axes[col].axis('off')
            except Exception as e:
                print(f"Hata {m_name}: {e}")
                axes[col].axis('off')
                
            col += 1
            
        save_path = OUT_DIR / f"comparison_{pid}_{label_str}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Kaydedildi: {save_path}")

if __name__ == "__main__":
    main()
