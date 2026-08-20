#!/usr/bin/env python3
"""
visualize_xai_comparison.py — Multi-Model Grad-CAM & ROI Contour Visualizer
===========================================================================
Generates publication-quality side-by-side XAI comparison figures:
[Original CT (Yellow ROI)] | [Model 1 + CAM + ROI] | [Model 2 + CAM + ROI] | ...

Matching the visual layout and style of Dr. Zehra Şimşek's appendicitis study.
"""

from __future__ import annotations

import os
import sys
import argparse
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from config import PipelineConfig
from models import build_model
from utils.xai import GradCAM3D


# ============================================================================ #
#  DATA LOADING & PRE-PROCESSING                                                #
# ============================================================================ #

def load_patient_h5(filepath: str) -> tuple[torch.Tensor, np.ndarray, np.ndarray, int, str]:
    """
    Load H5 volume, ROI mask, and label.
    
    Returns:
    --------
    model_tensor : torch.Tensor of shape [1, 1, D, H, W] (normalized for model input)
    disp_volume  : np.ndarray of shape [D, H, W] (normalized [0, 1] for visual display)
    mask_volume  : np.ndarray of shape [D, H, W] (binary ROI mask {0, 1})
    true_label   : int (0: Appendicitis, 1: Mucinous)
    true_name    : str ("Appendicitis" or "Mucinous")
    """
    with h5py.File(filepath, "r") as f:
        img = f["image"][:].astype(np.float32)
        mask = f["mask"][:].astype(np.uint8) if "mask" in f else np.zeros(img.shape[:3], dtype=np.uint8)
        label = int(f["label"][0]) if "label" in f else 0

    # Ensure shape is [D, H, W]
    if img.ndim == 4 and img.shape[-1] == 1:
        raw_vol = img[..., 0]
    elif img.ndim == 4 and img.shape[0] == 1:
        raw_vol = img[0]
    elif img.ndim == 3:
        raw_vol = img
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask_vol = mask[..., 0]
    elif mask.ndim == 4 and mask.shape[0] == 1:
        mask_vol = mask[0]
    elif mask.ndim == 3:
        mask_vol = mask
    else:
        mask_vol = np.zeros_like(raw_vol, dtype=np.uint8)

    # 1. Prepare tensor for model inference & Grad-CAM [1, 1, D, H, W]
    vol_clipped = np.clip(raw_vol, a_min=-150.0, a_max=250.0)
    mean_val = np.mean(vol_clipped)
    std_val = np.std(vol_clipped)
    if std_val > 1e-5:
        vol_norm = (vol_clipped - mean_val) / std_val
    else:
        vol_norm = vol_clipped - mean_val
        
    model_tensor = torch.tensor(vol_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 2. Prepare visual display volume [0, 1]
    v_min, v_max = raw_vol.min(), raw_vol.max()
    if v_max - v_min > 1e-8:
        disp_volume = (raw_vol - v_min) / (v_max - v_min)
    else:
        disp_volume = np.zeros_like(raw_vol)

    true_name = "Mucinous" if label == 1 else "Appendicitis"

    return model_tensor, disp_volume, mask_vol, label, true_name


# ============================================================================ #
#  MODEL LOADING & GRAD-CAM EXTRACTION                                         #
# ============================================================================ #

class ModelEvaluator:
    """Wrapper to handle model inference and Grad-CAM generation."""
    def __init__(self, model_name: str, display_name: str, ckpt_path: str, device: torch.device):
        self.model_name = model_name
        self.display_name = display_name
        self.device = device
        
        cfg = PipelineConfig(model_name=model_name, pretrained=False)
        self.model = build_model(cfg).to(device)
        
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print(f"  ✓ Loaded weights for {display_name} from {ckpt_path}")
        else:
            print(f"  [WARNING] Checkpoint not found at {ckpt_path}, using initialized weights.")
            
        self.model.eval()

    def predict_and_cam(self, input_tensor: torch.Tensor, target_class: int = 1) -> tuple[float, int, str, np.ndarray]:
        """
        Runs inference and computes 3D Grad-CAM heatmap.
        
        Returns:
        --------
        prob_mucinous : float (probability of class 1 / Mucinous)
        pred_label    : int (0 or 1)
        pred_name     : str ("Appendicitis" or "Mucinous")
        cam_volume    : np.ndarray [D, H, W] in [0, 1]
        """
        input_t = input_tensor.to(self.device).clone()
        cam_gen = GradCAM3D(self.model)
        
        # 1. Grad-CAM forward & backward
        cam_vol = cam_gen.generate_cam(input_t, target_class=target_class)
        cam_gen.remove_hooks()
        
        # 2. Probability prediction
        with torch.no_grad():
            logits = self.model(input_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prob_mucinous = float(probs[1])
            pred_label = int(np.argmax(probs))
            pred_name = "Mucinous" if pred_label == 1 else "Appendicitis"

        if cam_vol is None:
            D, H, W = input_tensor.shape[2:]
            cam_vol = np.zeros((D, H, W), dtype=np.float32)

        return prob_mucinous, pred_label, pred_name, cam_vol


# ============================================================================ #
#  PLOTTING FUNCTION (DR. ZEHRA ŞİMŞEK EXACT PUBLICATION STYLE)                #
# ============================================================================ #

def render_xai_comparison(
    disp_volume: np.ndarray,
    mask_volume: np.ndarray,
    true_label: int,
    true_name: str,
    model_results: list[dict],
    slice_idx: int | None = None,
    save_path: str | None = None,
    show_plot: bool = False,
    colormap: str = "jet",
    draw_roi_on_orig: bool = True,
    alpha_cam: float = 0.5,
    title_fontsize: int = 20,
    crop_roi: bool = False,
    roi_padding: int = 15,
    lang: str = "en",
) -> tuple[plt.Figure, int]:
    """
    Renders multi-panel comparison figure matching Dr. Zehra Şimşek's publication layout:
    Panel 1: Original CT slice with Yellow ROI contour overlay (if present)
    Panels 2..N+1: Model Grad-CAM heatmaps (Green title if correct, Red if wrong)
    """
    FROZEN_THRESHOLDS = {
        "UNet++": 0.467,
        "DenseNet-121": 0.860,
        "EfficientNet-B0": 0.460,
        "MAE-Tiny3D": 0.562,
        "AG-MSF": 0.602,
        "SwinUNETR-LP": 0.624,
        "SegFormer3D-MSCA": 0.548
    }

    D = disp_volume.shape[0]

    # 1. Determine optimal slice index (slice with max ROI area, fallback to mid-slice)
    if slice_idx is None:
        mask_sums = mask_volume.sum(axis=(1, 2))
        if mask_sums.max() > 0:
            slice_idx = int(np.argmax(mask_sums))
        else:
            slice_idx = D // 2

    orig_slice = disp_volume[slice_idx]
    mask_slice = mask_volume[slice_idx]

    # Optional cropping around ROI
    if crop_roi and mask_slice.max() > 0:
        y_indices, x_indices = np.where(mask_slice > 0)
        y_min, y_max = max(0, y_indices.min() - roi_padding), min(orig_slice.shape[0], y_indices.max() + roi_padding + 1)
        x_min, x_max = max(0, x_indices.min() - roi_padding), min(orig_slice.shape[1], x_indices.max() + roi_padding + 1)
    else:
        y_min, y_max = 0, orig_slice.shape[0]
        x_min, x_max = 0, orig_slice.shape[1]

    n_panels = 1 + len(model_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 5.0), dpi=100)
    if n_panels == 1:
        axes = [axes]

    cropped_orig = orig_slice[y_min:y_max, x_min:x_max]
    cropped_mask = mask_slice[y_min:y_max, x_min:x_max]

    # Ground truth label string
    if lang == "en":
        label_str = "Mucinous" if true_label == 1 else "Appendicitis"
    else:
        label_str = "Mukozlu" if true_label == 1 else "Apandisit"

    # ---- PANEL 1: Original CT (with Yellow ROI Contour) ---------------------
    ax0 = axes[0]
    ax0.imshow(cropped_orig, cmap="gray")
    
    has_roi = (cropped_mask.max() > 0)
    if draw_roi_on_orig and has_roi:
        ax0.contour(cropped_mask > 0, levels=[0.5], colors=['yellow'], linewidths=1.8, alpha=0.85)
        roi_text = " (ROI: Yellow)" if lang == "en" else " (ROI: Sarı)"
    else:
        roi_text = " (No ROI)" if lang == "en" else " (ROI Yok)"

    if lang == "en":
        panel1_title = f"Original CT{roi_text} - Slice {slice_idx+1}/{D}\nTrue: {label_str}"
    else:
        panel1_title = f"Orijinal BT{roi_text} - Kesit {slice_idx+1}/{D}\nDoğru: {label_str}"

    ax0.set_title(panel1_title, fontsize=title_fontsize, fontweight="bold", color="black", pad=10)
    ax0.axis("off")

    # ---- PANELS 2..N+1: Model Grad-CAMs (Green=Correct, Red=Wrong) -----------
    for i, res in enumerate(model_results, start=1):
        ax = axes[i]
        cam_slice = res["cam"][slice_idx][y_min:y_max, x_min:x_max]
        prob = float(res["prob"]) # probability of class 1 (Mucinous)
        thr = FROZEN_THRESHOLDS.get(res["name"], 0.5)

        if lang == "en":
            pred_str = "Mucinous" if prob >= thr else "Appendicitis"
        else:
            pred_str = "Mukozlu" if prob >= thr else "Apandisit"

        ax.imshow(cropped_orig, cmap="gray")
        ax.imshow(cam_slice, cmap=colormap, alpha=alpha_cam)

        # Zehra Hoca's color rule: Green if correct prediction, Red if wrong prediction
        is_correct = ((prob >= thr and true_label == 1) or (prob < thr and true_label == 0))
        color = "green" if is_correct else "red"

        display_prob = prob if prob >= thr else (1.0 - prob)

        if lang == "en":
            title_text = f"{res['name']}\nPred: {pred_str} ({display_prob:.2f})"
        else:
            title_text = f"{res['name']}\nTahmin: {pred_str} ({display_prob:.2f})"

        ax.set_title(title_text, color=color, fontsize=title_fontsize, fontweight="bold", pad=10)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="white")
        print(f"  ✓ Saved figure: {save_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return fig, slice_idx


def render_xai_all_slices_grid(
    disp_volume: np.ndarray,
    mask_volume: np.ndarray,
    true_label: int,
    true_name: str,
    model_results: list[dict],
    save_path: str | None = None,
    save_per_slice_dir: str | None = None,
    patient_id: str = "",
    colormap: str = "jet",
    draw_roi_on_orig: bool = True,
    alpha_cam: float = 0.5,
    title_fontsize: int = 20,
    crop_roi: bool = False,
    roi_padding: int = 15,
    lang: str = "en",
) -> tuple[plt.Figure, int]:
    """
    Renders ALL axial slices (D slices) of the 3D CT volume:
    1. Full multi-row montage grid
    2. Individual slice PNGs in save_per_slice_dir (slice_00.png, slice_01.png...)
    """
    FROZEN_THRESHOLDS = {
        "UNet++": 0.467,
        "DenseNet-121": 0.860,
        "EfficientNet-B0": 0.460,
        "MAE-Tiny3D": 0.562,
        "AG-MSF": 0.602,
        "SwinUNETR-LP": 0.624,
        "SegFormer3D-MSCA": 0.548
    }

    D, H, W = disp_volume.shape
    n_models = len(model_results)
    n_panels = 1 + n_models

    # Consistent 3D cropping box across all slices if crop_roi is True
    if crop_roi and mask_volume.max() > 0:
        y_indices, x_indices = np.where(mask_volume.sum(axis=0) > 0)
        y_min = max(0, int(y_indices.min()) - roi_padding)
        y_max = min(H, int(y_indices.max()) + roi_padding + 1)
        x_min = max(0, int(x_indices.min()) - roi_padding)
        x_max = min(W, int(x_indices.max()) + roi_padding + 1)
    else:
        y_min, y_max = 0, H
        x_min, x_max = 0, W

    label_str = "Mucinous" if true_label == 1 else "Appendicitis" if lang == "en" else ("Mukozlu" if true_label == 1 else "Apandisit")

    # 1. Save individual slices matching Zehra Hoca's format (slice_00.png, slice_01.png...)
    if save_per_slice_dir:
        os.makedirs(save_per_slice_dir, exist_ok=True)
        for z in range(D):
            slice_path = os.path.join(save_per_slice_dir, f"slice_{z:02d}.png")
            render_xai_comparison(
                disp_volume=disp_volume,
                mask_volume=mask_volume,
                true_label=true_label,
                true_name=true_name,
                model_results=model_results,
                slice_idx=z,
                save_path=slice_path,
                colormap=colormap,
                draw_roi_on_orig=draw_roi_on_orig,
                alpha_cam=alpha_cam,
                title_fontsize=title_fontsize,
                crop_roi=crop_roi,
                roi_padding=roi_padding,
                lang=lang,
            )

    # 2. Render complete all-slices montage grid
    fig, axes = plt.subplots(D, n_panels, figsize=(5.0 * n_panels, 4.5 * D), dpi=100)
    if D == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_panels == 1:
        axes = np.expand_dims(axes, axis=1)

    for z in range(D):
        orig_slice = disp_volume[z]
        mask_slice = mask_volume[z]
        cropped_orig = orig_slice[y_min:y_max, x_min:x_max]
        cropped_mask = mask_slice[y_min:y_max, x_min:x_max]

        # --- Col 0: Original CT ---
        ax0 = axes[z, 0]
        ax0.imshow(cropped_orig, cmap="gray")
        if draw_roi_on_orig and cropped_mask.max() > 0:
            ax0.contour(cropped_mask > 0, levels=[0.5], colors=['yellow'], linewidths=1.8, alpha=0.85)

        roi_text = " (ROI: Yellow)" if (cropped_mask.max() > 0 and lang == "en") else (" (ROI: Sarı)" if cropped_mask.max() > 0 else "")
        if z == 0:
            p1_title = f"Original CT{roi_text} - Slice 1/{D}\nTrue: {label_str}" if lang == "en" else f"Orijinal BT{roi_text} - Kesit 1/{D}\nDoğru: {label_str}"
            ax0.set_title(p1_title, fontsize=title_fontsize, fontweight="bold", color="black", pad=8)
        else:
            ax0.set_title(f"Slice {z+1}/{D}{roi_text}" if lang == "en" else f"Kesit {z+1}/{D}{roi_text}", fontsize=title_fontsize - 3, fontweight="bold", color="black", pad=6)
        ax0.axis("off")

        # --- Cols 1..N: Models Grad-CAM ---
        for col_idx, res in enumerate(model_results, start=1):
            ax = axes[z, col_idx]
            cam_slice = res["cam"][z][y_min:y_max, x_min:x_max]
            prob = float(res["prob"])
            thr = FROZEN_THRESHOLDS.get(res["name"], 0.5)

            if lang == "en":
                pred_str = "Mucinous" if prob >= thr else "Appendicitis"
            else:
                pred_str = "Mukozlu" if prob >= thr else "Apandisit"

            ax.imshow(cropped_orig, cmap="gray")
            ax.imshow(cam_slice, cmap=colormap, alpha=alpha_cam)

            is_correct = ((prob >= thr and true_label == 1) or (prob < thr and true_label == 0))
            color = "green" if is_correct else "red"
            display_prob = prob if prob >= thr else (1.0 - prob)

            if z == 0:
                if lang == "en":
                    title_text = f"{res['name']}\nPred: {pred_str} ({display_prob:.2f})"
                else:
                    title_text = f"{res['name']}\nTahmin: {pred_str} ({display_prob:.2f})"
                ax.set_title(title_text, color=color, fontsize=title_fontsize, fontweight="bold", pad=8)
            ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="white")
        print(f"  ✓ Saved ALL-SLICES montage ({D} slices) to: {save_path}")

    plt.close(fig)
    return fig, D


# ============================================================================ #
#  MAIN RUNNER                                                                 #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description="Multi-Model 3D Grad-CAM & ROI Comparison Generator (Publication Standard)")
    parser.add_argument("--patient", type=str, default=None, help="Patient ID or H5 filename to process")
    parser.add_argument("--all", action="store_true", help="Process all patients in external test set")
    parser.add_argument("--test_csv", type=str, default="datas_2d/external_test_set.csv", help="Path to external test CSV")
    parser.add_argument("--output_dir", type=str, default="visualizations/xai_comparisons", help="Output directory for PNGs")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--colormap", type=str, default="jet", help="Heatmap colormap (jet, turbo, inferno, viridis)")
    parser.add_argument("--crop_roi", action="store_true", help="Zoom / crop closely around the ROI mask")
    parser.add_argument("--no_roi", action="store_true", help="Disable yellow ROI contour line on the original CT panel")
    parser.add_argument("--slice_idx", type=int, default=None, help="Explicit single axial slice index (default: process ALL slices)")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "tr"], help="Language for titles: en (English, default) or tr (Turkish)")
    parser.add_argument("--fontsize", type=int, default=20, help="Title font size (Default: 20pt)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["unet_plusplus", "densenet121", "efficientnet_b0"],
        choices=["unet_plusplus", "densenet121", "efficientnet_b0"],
        help="Models to compare (Default: All 3 models: unet_plusplus densenet121 efficientnet_b0)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Resolve Checkpoints
    def resolve_ckpt(model_name: str) -> str:
        exp_candidates = [
            f"experiments_q1_128/{model_name}/run_01/fold_01/best_model.pth",
            f"experiments_q1_128/{model_name}/run_01/fold_02/best_model.pth",
        ]
        for c in exp_candidates:
            if os.path.exists(c):
                return c
        matches = glob.glob(f"experiments_q1_128/{model_name}/**/best_model.pth", recursive=True)
        if matches:
            return matches[0]
        legacy_matches = glob.glob(f"checkpoints/{model_name}/**/best_*.pth", recursive=True)
        if legacy_matches:
            return legacy_matches[0]
        return f"experiments_q1_128/{model_name}/run_01/fold_01/best_model.pth"

    display_name_map = {
        "unet_plusplus": "UNet++",
        "densenet121": "DenseNet-121",
        "efficientnet_b0": "EfficientNet-B0",
    }

    models_config = [
        (m, display_name_map.get(m, m), resolve_ckpt(m))
        for m in args.models
    ]

    print("\n" + "=" * 70)
    print(f"  MULTI-MODEL 3D GRAD-CAM MASTER PIPELINE (PUBLICATION STANDARD)")
    print(f"  Layout: [Original CT (ROI: Yellow)] + [{len(args.models)} Model Grad-CAM]")
    print(f"  Selected Models: {', '.join([display_name_map.get(m, m) for m in args.models])}")
    print(f"  Language: {args.lang.upper()} | Font Size: {args.fontsize}pt | DPI: 100")
    print("=" * 70)
    print("Loading models from experiment checkpoints...")
    evaluators = []
    for m_type, d_name, ckpt in models_config:
        evaluators.append(ModelEvaluator(m_type, d_name, ckpt, device))

    def resolve_patient_h5(pid_raw: str, label: int | None = None) -> str | None:
        pid = str(pid_raw).replace(".h5", "").strip()
        
        # Check in the 2D folders directly
        candidates = [
            f"../musinoz_2d/v02_slice_bbox_resize_128_D32/{pid}.h5",
            f"../apandisit_2d/v02_slice_bbox_resize_128_D32/{pid}.h5",
            f"musinoz_2d/v02_slice_bbox_resize_128_D32/{pid}.h5",
            f"apandisit_2d/v02_slice_bbox_resize_128_D32/{pid}.h5",
            f"/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/musinoz_2d/v02_slice_bbox_resize_128_D32/{pid}.h5",
            f"/home/zera/Downloads/Appendiks varyasyon3 DS-20260713T105239Z-2-001/Appendiks varyasyon3 DS/apandisit_2d/v02_slice_bbox_resize_128_D32/{pid}.h5"
        ]

        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    # Identify patients to process
    test_csv_path = args.test_csv
    if not os.path.exists(test_csv_path):
        fixed_csv = "datas/external_test_set_fixed.csv"
        if os.path.exists(fixed_csv):
            test_csv_path = fixed_csv

    patient_files = []
    if args.all and os.path.exists(test_csv_path):
        df = pd.read_csv(test_csv_path)
        for _, row in df.iterrows():
            pid = str(row["patient_id"])
            lbl = int(row["label"]) if "label" in row else None
            h5_path = resolve_patient_h5(pid, lbl)
            if h5_path:
                patient_files.append((pid, h5_path, lbl))
            else:
                print(f"  [WARNING] Could not find H5 for test patient: {pid}")
    elif args.patient:
        pid = args.patient.replace(".h5", "")
        h5_path = resolve_patient_h5(pid)
        if h5_path:
            patient_files.append((pid, h5_path, None))
        else:
            print(f"[ERROR] Could not find H5 file for patient '{pid}' in datas/")
            sys.exit(1)
    else:
        # Default priority: Aydemir Meryem and Çınar Beyza first
        for p in ["aydemir_meryem", "cınar_beyza"]:
            h5_path = resolve_patient_h5(p)
            if h5_path:
                patient_files.append((p, h5_path, None))

    # ======================================================================== #
    #  PRIORITIZATION: Aydemir Meryem & Çınar Beyza FIRST!                     #
    # ======================================================================== #
    priority_order = ["aydemir_meryem", "cınar_beyza", "cinar_beyza", "beyza_cinar"]
    def get_priority_key(item):
        pid = str(item[0]).lower()
        for idx, key in enumerate(priority_order):
            if key in pid:
                return (0, idx)
        return (1, pid)

    patient_files.sort(key=get_priority_key)

    print(f"\nProcessing {len(patient_files)} patient(s)...")
    print(f"Priority Order:")
    for idx, (p, _, _) in enumerate(patient_files[:5], 1):
        print(f"  {idx}. {p}")
    if len(patient_files) > 5:
        print(f"  ... and {len(patient_files) - 5} more patients.")

    for pid, path, _ in patient_files:
        t_tensor, disp_vol, mask_vol, true_label, true_name = load_patient_h5(path)
        D_slices = disp_vol.shape[0]
        label_tag = "Mucinous" if true_label == 1 else "Appendicitis"
        print(f"\n--- Patient: {pid} ({label_tag}) ---")
        print(f"  -> Path: {path}")
        print(f"  -> Volume shape: {disp_vol.shape} ({D_slices} axial slices)")
        
        model_results = []
        for ev in evaluators:
            prob, pred_lbl, pred_name, cam_vol = ev.predict_and_cam(t_tensor, target_class=1)
            model_results.append({
                "name": ev.display_name,
                "pred_name": pred_name,
                "prob": prob,
                "cam": cam_vol
            })
            pred_en = "Mucinous" if prob >= 0.5 else "Appendicitis"
            display_prob = prob if prob >= 0.5 else (1.0 - prob)
            print(f"  -> {ev.display_name:<15s}: Pred={pred_en:<12s} (P={display_prob:.4f})")

        # Create patient output folder: {pid}_{label_tag} matching publication format
        patient_slice_folder = os.path.join(args.output_dir, f"{pid}_{label_tag}")

        if args.slice_idx is not None:
            # Single slice
            out_png = os.path.join(args.output_dir, f"{pid}_slice_{args.slice_idx}.png")
            render_xai_comparison(
                disp_volume=disp_vol,
                mask_volume=mask_vol,
                true_label=true_label,
                true_name=true_name,
                model_results=model_results,
                slice_idx=args.slice_idx,
                save_path=out_png,
                colormap=args.colormap,
                draw_roi_on_orig=(not args.no_roi),
                alpha_cam=0.5,
                title_fontsize=args.fontsize,
                crop_roi=args.crop_roi,
                lang=args.lang,
            )
        else:
            # ALL SLICES:
            # 1. Montage grid
            grid_out_png = os.path.join(args.output_dir, f"{pid}_all_slices_grid.png")
            
            render_xai_all_slices_grid(
                disp_volume=disp_vol,
                mask_volume=mask_vol,
                true_label=true_label,
                true_name=true_name,
                model_results=model_results,
                save_path=grid_out_png,
                save_per_slice_dir=patient_slice_folder,
                patient_id=pid,
                colormap=args.colormap,
                draw_roi_on_orig=(not args.no_roi),
                alpha_cam=0.5,
                title_fontsize=args.fontsize,
                crop_roi=args.crop_roi,
                lang=args.lang,
            )

            # 2. Maximum ROI Slice (Quick Summary figure)
            max_roi_png = os.path.join(args.output_dir, f"{pid}_xai_comparison.png")
            _, best_z = render_xai_comparison(
                disp_volume=disp_vol,
                mask_volume=mask_vol,
                true_label=true_label,
                true_name=true_name,
                model_results=model_results,
                slice_idx=None,
                save_path=max_roi_png,
                colormap=args.colormap,
                draw_roi_on_orig=(not args.no_roi),
                alpha_cam=0.5,
                title_fontsize=args.fontsize,
                crop_roi=args.crop_roi,
                lang=args.lang,
            )
            print(f"  ✓ Processed ALL {D_slices} slices (Saved to: {patient_slice_folder}/ + {grid_out_png})")

    print("\n" + "=" * 70)
    print(f"  DONE! All 3D Grad-CAM visualizations saved to: {args.output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
