import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

csv_path = "/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/datas/external_test_set.csv"

def visualize_all_slices():
    df = pd.read_csv(csv_path)
    patient_row = df[df['label'] == 1].iloc[0]
    h5_path = patient_row['h5_path']
    patient_id = patient_row['patient_id']
    
    with h5py.File(h5_path, 'r') as f:
        img_3d = f["image"][...]
        mask_3d = f["mask"][...] if "mask" in f.keys() else np.zeros_like(img_3d)

    # Boyutları düzelt
    if img_3d.ndim == 4:
        if img_3d.shape[-1] == 1 or img_3d.shape[-1] == 3:
            img_3d = img_3d[..., 0]
        elif img_3d.shape[0] == 1:
            img_3d = img_3d[0]
            
    if mask_3d.ndim == 4:
        if mask_3d.shape[-1] == 1 or mask_3d.shape[-1] == 3:
            mask_3d = mask_3d[..., 0]
        elif mask_3d.shape[0] == 1:
            mask_3d = mask_3d[0]

    num_slices = img_3d.shape[0]
    print(f"\n{patient_id} hastasının Z ekseninde toplam {num_slices} adet kesiti (slice) bulunuyor.")
    
    # 32 slice'ı 4 satır x 8 sütun (veya num_slices'a göre uygun karekök) şeklinde çizelim
    cols = 8
    rows = math.ceil(num_slices / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f"All {num_slices} Axial Slices for Patient: {patient_id}", fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    for i in range(num_slices):
        axes[i].imshow(img_3d[i], cmap='gray')
        if mask_3d[i].max() > 0:
            axes[i].contour(mask_3d[i], colors='yellow', linewidths=1.0, alpha=0.8)
            axes[i].set_title(f"Slice {i+1}\n(ROI Detected)", color="red", fontsize=9)
        else:
            axes[i].set_title(f"Slice {i+1}", fontsize=9)
        axes[i].axis('off')
        
    # Boş kalan eksenleri gizle
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_path = f"patient_{patient_id}_all_slices.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"[BAŞARILI] Tüm kesitlerin yer aldığı montaj '{save_path}' olarak kaydedildi.")

if __name__ == "__main__":
    visualize_all_slices()
