import h5py
import numpy as np

f1_path = "/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/Musinoz/v03_native_canvas_128_D32/iman_tugce.h5"
f2_path = "/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/Appendisit/v03_native_canvas_128_D32/iman_tugce.h5"

with h5py.File(f1_path, 'r') as f1, h5py.File(f2_path, 'r') as f2:
    print("--- METADATA COMPARISON ---")
    keys_to_check = ['bbox_xyxy_original', 'roi_area_mm2_original', 'roi_centroid_xy_original', 'valid_z', 'label']
    for key in keys_to_check:
        if key in f1 and key in f2:
            val1 = f1[key][:] if isinstance(f1[key], h5py.Dataset) and f1[key].shape else f1[key][()]
            val2 = f2[key][:] if isinstance(f2[key], h5py.Dataset) and f2[key].shape else f2[key][()]
            print(f"{key}:")
            print(f"  Musinoz: {val1}")
            print(f"  Appendisit: {val2}")
