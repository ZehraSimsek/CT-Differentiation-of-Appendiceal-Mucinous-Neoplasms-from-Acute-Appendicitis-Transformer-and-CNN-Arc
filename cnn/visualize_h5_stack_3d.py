#!/usr/bin/env python3
"""
visualize_h5_stack_3d.py — 3D Vertical-Panel CT Visualization (3 Patients)
===========================================================================
Loads ONE middle slice from each of 3 specific patient H5 files and renders
them as standing vertical glass-plate panels in a single 3D stack.

Patient order (front → back):
    1. aydemir_meryem   — Müsinöz
    2. altin_aynur      — Apandisit  (centre panel)
    3. lala_medine_qurban — Müsinöz

Output: visualizations/h5_input_stack_3d.png (transparent background)
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LightSource


# ============================================================================ #
#  PATIENT DEFINITIONS (hard-coded per request)                                  #
# ============================================================================ #
PATIENTS = [
    # (filename, directory, diagnosis)  — order defines front→back stacking
    ("aydemir_meryem.h5",       "datas/Musinoz/musinoz_128",       "Müsinöz"),
    ("altin_aynur.h5",          "datas/Appendisit/apandisit_128",  "Apandisit"),
    ("lala_medine_qurban.h5",   "datas/Musinoz/musinoz_128",       "Müsinöz"),
]

OUTPUT_DIR = "visualizations"


# ============================================================================ #
#  DATA HELPERS                                                                  #
# ============================================================================ #

def load_and_normalise(filepath: str, key: str = "image") -> np.ndarray:
    """
    Load 3D volume from HDF5 and normalise to [0, 1] float32.
    Handles shapes: [D,H,W], [1,D,H,W], [D,H,W,1].
    """
    with h5py.File(filepath, "r") as f:
        if key not in f:
            available = list(f.keys())
            raise KeyError(
                f"Key '{key}' not found in {filepath}. Available: {available}"
            )
        volume = f[key][:].astype(np.float32)

    # Squeeze singleton channel dimensions
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]
    elif volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume after squeezing, got shape {volume.shape}"
        )

    # Min-max normalisation → [0, 1]
    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min > 1e-8:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume


def extract_middle_slice(volume: np.ndarray) -> tuple[int, np.ndarray]:
    """Return the middle axial slice (D // 2) from a [D, H, W] volume."""
    mid = volume.shape[0] // 2
    return mid, volume[mid]


# ============================================================================ #
#  3D VERTICAL-PANEL RENDERER                                                    #
# ============================================================================ #

def render_vertical_panels(
    panels: list[tuple[str, str, int, np.ndarray]],
    output_path: str = "h5_input_stack_3d.png",
    azimuth: float = -55.0,
    elevation: float = 20.0,
    panel_spacing: float = 80.0,
) -> None:
    """
    Render grayscale slices as VERTICAL standing panels staggered in depth.

    Parameters
    ----------
    panels : list of (patient_name, diagnosis, depth_idx, 2D_slice)
    output_path : save path
    azimuth, elevation : camera angles
    panel_spacing : distance between panels along the depth (X) axis
    """
    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # ---- Fully transparent background & no axes ---------------------------
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_axis_off()

    # Remove all panes, grid lines, and tick marks
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_visible(False)
        axis.set_ticks([])
    ax.grid(False)

    # ---- Camera view -------------------------------------------------------
    ax.view_init(elev=elevation, azim=azimuth)

    # ---- Draw each patient's slice as a vertical standing panel ------------
    n_panels = len(panels)
    ls = LightSource(azdeg=315, altdeg=45)

    for panel_idx, (patient_name, diagnosis, depth_idx, img_2d) in enumerate(panels):
        H, W = img_2d.shape

        # Coordinate system for a VERTICAL panel:
        #   X = depth axis (panel stagger direction)
        #   Y = horizontal extent of the image (width)
        #   Z = vertical extent of the image (height — stands upright)
        y = np.arange(W)
        z = np.arange(H)
        Y, Z = np.meshgrid(y, z)

        # Each panel sits at a fixed X position
        x_pos = panel_idx * panel_spacing
        X = np.full_like(Y, dtype=float, fill_value=x_pos)

        # Flip Z so the image isn't upside-down
        Z_plot = Z.max() - Z

        # Grayscale → RGBA with subtle shading
        shaded = ls.shade(
            img_2d,
            cmap=plt.cm.gray,
            vmin=0.0,
            vmax=1.0,
            blend_mode="soft",
        )

        # Plot the vertical surface
        ax.plot_surface(
            X, Y, Z_plot,
            facecolors=shaded,
            rstride=1,
            cstride=1,
            antialiased=True,
            shade=False,
        )

        # ---- Clean dark border edges around the panel ----------------------
        edge_color = (0.12, 0.12, 0.12, 0.85)
        lw = 0.7
        ax.plot([x_pos, x_pos], [0, W - 1], [0, 0],
                color=edge_color, lw=lw)
        ax.plot([x_pos, x_pos], [0, W - 1], [H - 1, H - 1],
                color=edge_color, lw=lw)
        ax.plot([x_pos, x_pos], [0, 0], [0, H - 1],
                color=edge_color, lw=lw)
        ax.plot([x_pos, x_pos], [W - 1, W - 1], [0, H - 1],
                color=edge_color, lw=lw)

    # ---- Axis limits with padding ------------------------------------------
    sample_h, sample_w = panels[0][3].shape
    total_depth = (n_panels - 1) * panel_spacing
    ax.set_xlim(-panel_spacing * 0.3, total_depth + panel_spacing * 0.3)
    ax.set_ylim(-10, sample_w + 10)
    ax.set_zlim(-10, sample_h + 10)

    ax.set_box_aspect([
        total_depth * 0.5 if total_depth > 0 else 1,
        sample_w,
        sample_h,
    ])

    # ---- Save with full transparency --------------------------------------
    plt.savefig(
        output_path,
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    print(f"  ✓ Saved → {output_path}")


# ============================================================================ #
#  MAIN                                                                          #
# ============================================================================ #

def main() -> None:
    print("\n" + "=" * 60)
    print("  3D CT VERTICAL-PANEL VISUALISER")
    print("  1 slice per patient  ×  3 patients  →  1 stack")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load one middle slice from each patient ---------------------------
    panels: list[tuple[str, str, int, np.ndarray]] = []

    print()
    for filename, directory, diagnosis in PATIENTS:
        filepath = os.path.join(directory, filename)
        patient_name = os.path.splitext(filename)[0]

        if not os.path.exists(filepath):
            print(f"  [ERROR] File not found: {filepath}")
            sys.exit(1)

        volume = load_and_normalise(filepath, key="image")
        D, H, W = volume.shape
        depth_idx, middle_slice = extract_middle_slice(volume)

        panels.append((patient_name, diagnosis, depth_idx, middle_slice))

        print(f"  ✓ {patient_name:<25s}  ({diagnosis:<10s})  "
              f"Volume: [{D},{H},{W}]  →  slice @ depth={depth_idx}")

    # ---- Render single combined stack --------------------------------------
    print()
    out_path = os.path.join(OUTPUT_DIR, "h5_input_stack_3d.png")
    render_vertical_panels(panels, output_path=out_path)

    print("\n" + "=" * 60)
    print(f"  Done — combined stack saved to {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
