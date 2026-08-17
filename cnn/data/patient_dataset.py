"""
data/patient_dataset.py — Volumetric 3D CT Dataset with Standardised Augmentations
===================================================================================
Loads 3D CT volumes from HDF5 (.h5) files and applies:
1. HU clipping [-150, 250] and Z-score standardisation: (vol - mean) / (std + 1e-8)
2. Exact 9-step 3D Augmentation suite during training (as specified in MULTI_RUN_PROTOCOL.md)
3. Deterministic evaluation transforms for Validation and External Test sets.
"""

from __future__ import annotations

import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PatientCTDataset(Dataset):
    """
    3D Medical CT Volumetric Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns: 'patient_id', 'h5_path', 'label'.
    augment_train : bool
        If True, applies the 9-step 3D stochastic augmentation protocol.
    config : Any, optional
        Pipeline configuration object.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        augment_train: bool = False,
        config: object | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.augment_train = augment_train
        self.config = config

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        filepath = str(row["h5_path"])
        label = int(row["label"])
        patient_id = str(row["patient_id"])

        if not os.path.exists(filepath):
            # Fallback 1: Resolve relative to local 'datas/' directory
            norm_path = filepath.replace("\\", "/")
            if "datas/" in norm_path:
                rel_part = norm_path[norm_path.index("datas/"):]
                if os.path.exists(rel_part):
                    filepath = rel_part

            # Fallback 2: Direct lookup by filename in subdirectories
            if not os.path.exists(filepath):
                fname = os.path.basename(filepath)
                cand_mus = os.path.join("datas", "Musinoz", "musinoz_128", fname)
                cand_app = os.path.join("datas", "Appendisit", "apandisit_128", fname)
                if os.path.exists(cand_mus):
                    filepath = cand_mus
                elif os.path.exists(cand_app):
                    filepath = cand_app

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing patient volume: {filepath}\n"
                f"Attempted search in local 'datas/' directory but file was not found.\n"
                f"Please ensure dataset .h5 files are located under 'datas/Musinoz/musinoz_128/' and 'datas/Appendisit/apandisit_128/'."
            )

        # 1. Read 3D Volume from HDF5
        with h5py.File(filepath, "r") as f:
            volume = f["image"][:]

        volume = volume.astype(np.float32)

        # 2. Ensure PyTorch [C, D, H, W] format
        if volume.ndim == 4 and volume.shape[-1] == 1:
            volume = np.transpose(volume, (3, 0, 1, 2))
        elif volume.ndim == 3:
            volume = np.expand_dims(volume, axis=0)

        # 3. Preprocessing (Always applied): HU clipping + Z-score normalisation
        volume = np.clip(volume, a_min=-150.0, a_max=250.0)
        mean_val = float(np.mean(volume))
        std_val = float(np.std(volume))
        volume = (volume - mean_val) / (std_val + 1e-8)

        vol_tensor = torch.tensor(volume, dtype=torch.float32)

        # 4. Standardised 3D Augmentation (Train Split only: MULTI_RUN_PROTOCOL.md)
        if self.augment_train and vol_tensor.ndim == 4:
            # 1. 3D Axial Flip (Depth axis, dim=1, p=0.50)
            if torch.rand(1).item() < 0.50:
                vol_tensor = torch.flip(vol_tensor, dims=[1])

            # 2. 3D Horizontal Flip (Height axis, dim=2, p=0.50)
            if torch.rand(1).item() < 0.50:
                vol_tensor = torch.flip(vol_tensor, dims=[2])

            # 3. 3D Vertical Flip (Width axis, dim=3, p=0.50)
            if torch.rand(1).item() < 0.50:
                vol_tensor = torch.flip(vol_tensor, dims=[3])

            # 4. 90-degree Rotation Simulation (H-W plane, p=0.40)
            if torch.rand(1).item() < 0.40:
                k = int(torch.randint(1, 4, (1,)).item())
                vol_tensor = torch.rot90(vol_tensor, k=k, dims=[2, 3])

            # 5. Intensity Shift (p=0.60, shift in [-0.25, +0.25])
            if torch.rand(1).item() < 0.60:
                shift = (torch.rand(1).item() - 0.50) * 0.50
                vol_tensor = vol_tensor + shift

            # 6. Intensity Scale (p=0.60, scale in [0.75, 1.25])
            if torch.rand(1).item() < 0.60:
                scale = 0.75 + torch.rand(1).item() * 0.50
                mean_t = vol_tensor.mean()
                vol_tensor = (vol_tensor - mean_t) * scale + mean_t

            # 7. Gaussian Noise (p=0.60, sigma=0.02)
            if torch.rand(1).item() < 0.60:
                vol_tensor = vol_tensor + 0.02 * torch.randn_like(vol_tensor)

            # 8. 3D Cutout / Random Occlusion (p=0.40, patch 4x16x16)
            if torch.rand(1).item() < 0.40:
                _, D, H, W = vol_tensor.shape
                d0 = torch.randint(0, max(1, D - 4), (1,)).item()
                h0 = torch.randint(0, max(1, H - 16), (1,)).item()
                w0 = torch.randint(0, max(1, W - 16), (1,)).item()
                vol_tensor[:, d0:d0 + 4, h0:h0 + 16, w0:w0 + 16] = 0.0

            # 9. Depth Crop & Trilinear Resize (p=0.30, +- D/6 crop)
            if torch.rand(1).item() < 0.30 and vol_tensor.shape[1] > 8:
                D = vol_tensor.shape[1]
                margin = max(1, D // 6)
                start = torch.randint(0, margin, (1,)).item()
                end = D - torch.randint(0, margin, (1,)).item()
                if end > start:
                    cropped = vol_tensor[:, start:end, :, :]
                    vol_tensor = F.interpolate(
                        cropped.unsqueeze(0),
                        size=vol_tensor.shape[1:],
                        mode="trilinear",
                        align_corners=False,
                    ).squeeze(0)

        return {
            "image": vol_tensor.contiguous(),
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id,
            "h5_path": filepath,
        }