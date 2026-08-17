"""
data/dataset.py — Custom CT Dataset & Dummy Data Simulator
==========================================================
Provides:
    1. ``CustomCTDataset`` — a reusable ``torch.utils.data.Dataset`` that
       wraps NumPy images + labels and applies torchvision transforms.
    2. ``generate_dummy_data()`` — synthesises class-conditional 2-D
       "CT-like" images so that the *entire pipeline can execute
       end-to-end without any real data*.

Dummy data design
-----------------
The simulator creates **visually distinguishable** synthetic images for the
two classes so that the training loop can demonstrate *actual learning*:

* **Class 0 — Acute Appendicitis** (negative / majority):
  Smooth, homogeneous base texture with a bright elliptical "inflammation"
  region.  Simulates the diffuse oedema pattern.

* **Class 1 — Appendiceal Mucinous Neoplasm** (positive / minority):
  Heterogeneous base with a low-density "cystic" ellipse and scattered
  high-density "calcification" specks.  Simulates the hallmark
  mucin-filled expansion seen in LAMN/HAMN.

When real DICOM / NIfTI data becomes available, simply replace the
data-loading logic inside ``generate_dummy_data`` (or write a new loader)
while keeping the ``CustomCTDataset`` interface unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import PipelineConfig


# =========================================================================== #
#  1. REUSABLE DATASET CLASS                                                   #
# =========================================================================== #

class CustomCTDataset(Dataset):
    """
    Generic dataset wrapper for 2-D CT image classification.

    Parameters
    ----------
    images : np.ndarray
        Array of shape ``(N, H, W, 3)`` with pixel values in ``[0, 1]``
        (float64 or float32).
    labels : np.ndarray
        Binary label array of shape ``(N,)`` with values in ``{0, 1}``.
    transform : torchvision.transforms.Compose | None
        Optional augmentation / normalisation pipeline.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose | None = None,
    ) -> None:
        assert len(images) == len(labels), (
            f"Image count ({len(images)}) ≠ label count ({len(labels)})"
        )
        self.images = images
        self.labels = labels
        self.transform = transform

    # ----- Required overrides -----------------------------------------------

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the sample at position *idx*.

        Returns
        -------
        image : Tensor, shape (C, H, W)
            Augmented / normalised image tensor.
        label : Tensor, shape ()
            Scalar binary label (0 or 1) as float32.
        """
        # NumPy array → PIL Image (expected by torchvision transforms)
        image_np = self.images[idx]  # (H, W, 3), float in [0, 1]
        image_pil = Image.fromarray(
            (image_np * 255.0).clip(0, 255).astype(np.uint8)
        )

        # Apply the transform pipeline (Resize → Augment → ToTensor → Norm)
        if self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image_tensor, label_tensor


# =========================================================================== #
#  2. DUMMY DATA SIMULATOR                                                     #
# =========================================================================== #

def _draw_ellipse(
    canvas: np.ndarray,
    cy: int,
    cx: int,
    ry: int,
    rx: int,
    intensity: float,
) -> None:
    """Draw a filled ellipse onto *canvas* (in-place) with the given intensity."""
    h, w = canvas.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy) ** 2 / max(ry ** 2, 1)) + ((xx - cx) ** 2 / max(rx ** 2, 1)) <= 1.0
    canvas[mask] = np.clip(intensity, 0.0, 1.0)


def _generate_class0_sample(
    h: int,
    w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Synthesise an **Acute Appendicitis** image (class 0).

    Pattern: smooth grey base + bright elliptical "inflammation" region.
    """
    # Smooth base (low-frequency noise ≈ homogeneous tissue)
    base = rng.uniform(0.25, 0.40, size=(h, w))
    base = np.stack([base] * 3, axis=-1)  # Greyscale → 3-channel

    # Bright ellipse simulating oedema / inflammatory mass
    cy = h // 2 + rng.integers(-20, 20)
    cx = w // 2 + rng.integers(-20, 20)
    ry = rng.integers(25, 50)
    rx = rng.integers(25, 50)
    intensity = rng.uniform(0.65, 0.85)
    for c in range(3):
        _draw_ellipse(base[:, :, c], cy, cx, ry, rx, intensity)

    # Add slight Gaussian noise
    noise = rng.normal(0, 0.02, size=base.shape)
    return np.clip(base + noise, 0.0, 1.0)


def _generate_class1_sample(
    h: int,
    w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Synthesise an **Appendiceal Mucinous Neoplasm** image (class 1).

    Pattern: heterogeneous base + low-density cystic ellipse + high-
    density calcification specks.
    """
    # Heterogeneous base (higher-frequency noise ≈ mixed tissue)
    base = rng.uniform(0.30, 0.55, size=(h, w))
    base = np.stack([base] * 3, axis=-1)

    # Low-density "cystic / mucinous" ellipse
    cy = h // 2 + rng.integers(-15, 15)
    cx = w // 2 + rng.integers(-15, 15)
    ry = rng.integers(30, 55)
    rx = rng.integers(30, 55)
    intensity = rng.uniform(0.15, 0.30)  # dark = fluid-density
    for c in range(3):
        _draw_ellipse(base[:, :, c], cy, cx, ry, rx, intensity)

    # Scattered high-density "calcification" specks
    n_specks = rng.integers(8, 25)
    for _ in range(n_specks):
        sy = rng.integers(cy - ry, cy + ry)
        sx = rng.integers(cx - rx, cx + rx)
        if 0 <= sy < h and 0 <= sx < w:
            r = rng.integers(1, 4)
            y_lo, y_hi = max(sy - r, 0), min(sy + r + 1, h)
            x_lo, x_hi = max(sx - r, 0), min(sx + r + 1, w)
            base[y_lo:y_hi, x_lo:x_hi] = rng.uniform(0.85, 1.0)

    # Higher noise floor (mucinous texture is more granular)
    noise = rng.normal(0, 0.04, size=base.shape)
    return np.clip(base + noise, 0.0, 1.0)


def generate_dummy_data(
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic CT-like images for pipeline smoke-testing.

    Returns
    -------
    train_images : ndarray, shape (N_train, H, W, 3)
    train_labels : ndarray, shape (N_train,)
    val_images   : ndarray, shape (N_val, H, W, 3)
    val_labels   : ndarray, shape (N_val,)
    """
    rng = np.random.default_rng(config.seed)
    h, w = config.image_size, config.image_size

    def _generate_split(
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        images = np.empty((n_samples, h, w, 3), dtype=np.float64)
        labels = np.empty(n_samples, dtype=np.int64)

        n_positive = int(n_samples * config.positive_class_ratio)
        n_negative = n_samples - n_positive

        # Generate negative samples (class 0 — Acute Appendicitis)
        for i in range(n_negative):
            images[i] = _generate_class0_sample(h, w, rng)
            labels[i] = 0

        # Generate positive samples (class 1 — Mucinous Neoplasm)
        for i in range(n_negative, n_samples):
            images[i] = _generate_class1_sample(h, w, rng)
            labels[i] = 1

        # Shuffle so positive / negative samples are interleaved
        perm = rng.permutation(n_samples)
        return images[perm], labels[perm]

    print(f"  ▸ Generating {config.num_train_samples} training samples …")
    train_images, train_labels = _generate_split(config.num_train_samples)

    print(f"  ▸ Generating {config.num_val_samples} validation samples …")
    val_images, val_labels = _generate_split(config.num_val_samples)

    # Summary statistics
    print(
        f"  ✓ Train: {len(train_labels)} samples  "
        f"({(train_labels == 1).sum()} positive, "
        f"{(train_labels == 0).sum()} negative)"
    )
    print(
        f"  ✓ Val:   {len(val_labels)} samples  "
        f"({(val_labels == 1).sum()} positive, "
        f"{(val_labels == 0).sum()} negative)"
    )

    return train_images, train_labels, val_images, val_labels


# =========================================================================== #
#  3. PATIENT-NAMED DUMMY DATA GENERATOR                                        #
# =========================================================================== #

def generate_dummy_patient_data(
    data_dir: str,
    n_positive_patients: int = 20,
    n_negative_patients: int = 20,
    min_images: int = 15,
    max_images: int = 40,
    image_size: int = 224,
    seed: int = 42,
) -> str:
    """
    Generate synthetic CT images on disk following the patient naming convention.

    Creates a directory of JPEG images named according to the clinical
    convention ``p_m_XXX_YYY.jpg`` (positive) and ``p_a_XXX_YYY.jpg``
    (negative), enabling end-to-end pipeline testing without real data.

    Parameters
    ----------
    data_dir : str
        Output directory for the generated images.
    n_positive_patients : int
        Number of mucinous (positive) patients to simulate.
    n_negative_patients : int
        Number of acute appendicitis (negative) patients to simulate.
    min_images : int
        Minimum number of images per patient.
    max_images : int
        Maximum number of images per patient.
    image_size : int
        Height and width of each generated image.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    str
        The ``data_dir`` path (for convenience).
    """
    import os

    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, w = image_size, image_size

    total_images = 0

    # --- Positive patients (p_m_*) ---
    for pid_num in range(1, n_positive_patients + 1):
        n_imgs = rng.integers(min_images, max_images + 1)
        patient_id = f"p_m_{pid_num:03d}"

        for frame_idx in range(1, n_imgs + 1):
            img_array = _generate_class1_sample(h, w, rng)
            img_pil = Image.fromarray(
                (img_array * 255.0).clip(0, 255).astype(np.uint8)
            )
            fname = f"{patient_id}_{frame_idx:03d}.jpg"
            img_pil.save(os.path.join(data_dir, fname), quality=95)
            total_images += 1

    # --- Negative patients (p_a_*) ---
    for pid_num in range(1, n_negative_patients + 1):
        n_imgs = rng.integers(min_images, max_images + 1)
        patient_id = f"p_a_{pid_num:03d}"

        for frame_idx in range(1, n_imgs + 1):
            img_array = _generate_class0_sample(h, w, rng)
            img_pil = Image.fromarray(
                (img_array * 255.0).clip(0, 255).astype(np.uint8)
            )
            fname = f"{patient_id}_{frame_idx:03d}.jpg"
            img_pil.save(os.path.join(data_dir, fname), quality=95)
            total_images += 1

    total_patients = n_positive_patients + n_negative_patients
    print(f"  ✓ Generated {total_images} images for {total_patients} patients")
    print(f"    → {n_positive_patients} positive (p_m_*), "
          f"{n_negative_patients} negative (p_a_*)")
    print(f"    → Images per patient: {min_images}–{max_images}")
    print(f"    → Output: {data_dir}")

    return data_dir
