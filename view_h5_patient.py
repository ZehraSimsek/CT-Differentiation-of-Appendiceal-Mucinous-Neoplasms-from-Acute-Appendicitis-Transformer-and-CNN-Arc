import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# CSV yolunu belirt
csv_path = "/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/segformer/datas/external_test_set.csv"

def visualize_h5():
    # İlk hastanın bilgisini al
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("CSV dosyası boş!")
        return

    # Rastgele veya ilk hastayı seçelim. Örnek olarak ilk müsinöz vakasını alalım:
    patient_row = df[df['label'] == ].iloc[0]
    h5_path = patient_row['h5_path']
    patient_id = patient_row['patient_id']
    label = "Mucinous" if patient_row['label'] == 1 else "Appendicitis"
    
    print(f"Hasta ID: {patient_id} | Etiket: {label}")
    print(f"H5 Yolu: {h5_path}")
    
    if not os.path.exists(h5_path):
        print(f"HATA: {h5_path} dosyası bulunamadı!")
        return

    # H5 dosyasını oku
    with h5py.File(h5_path, 'r') as f:
        print("H5 Dosyası İçindeki Anahtarlar (Keys):", list(f.keys()))
        
        # Görüntü (Volume)
        img_3d = f["image"][...]
        
        # Maske (Eğer varsa)
        mask_3d = f["mask"][...] if "mask" in f.keys() else np.zeros_like(img_3d)

    print(f"Orijinal Görüntü Boyutu: {img_3d.shape} | Max/Min Değer: {img_3d.max():.2f}/{img_3d.min():.2f}")
    
    # 3D veriler genelde (Depth, Height, Width) formatındadır.
    # Ortadaki kesitleri (slice) alalım.
    if img_3d.ndim == 4:
        # Eğer (D, H, W, 1) formatındaysa sondaki kanalı atalım
        if img_3d.shape[-1] == 1 or img_3d.shape[-1] == 3:
            img_3d = img_3d[..., 0]
        # Eğer (1, D, H, W) formatındaysa baştaki kanalı atalım
        elif img_3d.shape[0] == 1:
            img_3d = img_3d[0]
            
    if mask_3d.ndim == 4:
        if mask_3d.shape[-1] == 1 or mask_3d.shape[-1] == 3:
            mask_3d = mask_3d[..., 0]
        elif mask_3d.shape[0] == 1:
            mask_3d = mask_3d[0]
        
    D, H, W = img_3d.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    # Kesitleri ayarla (Axial, Coronal, Sagittal)
    slices = [
        ("Axial", img_3d[mid_d, :, :], mask_3d[mid_d, :, :]),
        ("Coronal", img_3d[:, mid_h, :], mask_3d[:, mid_h, :]),
        ("Sagittal", img_3d[:, :, mid_w], mask_3d[:, :, mid_w])
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Patient: {patient_id} ({label})", fontsize=16, fontweight='bold')

    for i, (name, img_slice, mask_slice) in enumerate(slices):
        # Üst Satır: Sadece Görüntü
        axes[0, i].imshow(img_slice, cmap="gray")
        axes[0, i].set_title(f"{name} Slice (Raw)")
        axes[0, i].axis('off')
        
        # Alt Satır: Görüntü + Maske Overlay
        axes[1, i].imshow(img_slice, cmap="gray")
        if mask_slice.max() > 0:
            axes[1, i].imshow(mask_slice, cmap="autumn", alpha=0.4)
            axes[1, i].contour(mask_slice, colors='yellow', linewidths=1.5, alpha=0.8)
        axes[1, i].set_title(f"{name} Slice (with ROI)")
        axes[1, i].axis('off')

    plt.tight_layout()
    save_path = f"patient_{patient_id}_h5_visualization.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"\n[BAŞARILI] H5 görselleştirmesi '{save_path}' dosyasına kaydedildi.")

if __name__ == "__main__":
    visualize_h5()
