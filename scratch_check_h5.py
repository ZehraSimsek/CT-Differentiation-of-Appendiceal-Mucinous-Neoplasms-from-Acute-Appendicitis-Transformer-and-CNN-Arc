import h5py
import numpy as np

f1_path = "/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/Musinoz/v03_native_canvas_128_D32/iman_tugce.h5"
f2_path = "/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/Appendisit/v03_native_canvas_128_D32/iman_tugce.h5"

with h5py.File(f1_path, 'r') as f1, h5py.File(f2_path, 'r') as f2:
    print(f"File 1 keys: {list(f1.keys())}")
    print(f"File 2 keys: {list(f2.keys())}")
    
    for key in ['image', 'mask']:
        if key in f1 and key in f2:
            arr1 = f1[key][:]
            arr2 = f2[key][:]
            print(f"\n--- {key} ---")
            print(f"Shape 1: {arr1.shape}, dtype: {arr1.dtype}")
            print(f"Shape 2: {arr2.shape}, dtype: {arr2.dtype}")
            if arr1.shape == arr2.shape:
                diff = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
                print(f"Max diff: {diff.max()}")
                print(f"Mean diff: {diff.mean()}")
                print(f"Are arrays identical? {np.array_equal(arr1, arr2)}")
            else:
                print("Shapes are different!")
