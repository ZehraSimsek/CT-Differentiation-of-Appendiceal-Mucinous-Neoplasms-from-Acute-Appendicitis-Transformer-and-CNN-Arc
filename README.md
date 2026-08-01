# 3D Medical Image Classification for Appendicitis and Mucinous Tumors

This repository contains the official PyTorch implementation for our Q1-submitted manuscript on 3D Appendicitis and Mucinous Cystadenoma classification from volumetric CT scans.

## 📌 Features & Highlights
* **Four Advanced 3D Architectures:** Implementations of SwinUNETR Linear-Probe, Attention-SwinUNETR (AG-MSF), MAE-TinyTransformer3D, and SegFormer3D-MSCA.
* **Leak-Free Self-Supervised Pretraining:** Rigorous patient-level splitting (119 Train/Val, 24 External Test) ensuring zero data leakage during MAE and Contrastive pretraining stages.
* **Clinical Constraint Optimization:** A dual-constraint fallback mechanism during training that strictly enforces Sensitivity $\ge$ 0.80 and Specificity $\ge$ 0.50 before maximizing the F1-Score.
* **High-Performance Protocols:** Incorporates Stochastic Weight Averaging (SWA), Validation Metric Exponential Moving Average (EMA), 4-Direction Test-Time Augmentation (TTA), and 5-Fold Ensembling.

## 🏗️ Repository Structure
```
├── q1_final/
│   ├── train_attention_swinunetr.py   # AG-MSF Training Pipeline
│   ├── train_mae_tinytransformer.py   # MAE-Tiny3D Training Pipeline
│   ├── train_segformer3d.py           # SegFormer3D-MSCA Training Pipeline
│   ├── train_swinunetr_linearprobe.py # SwinUNETR-LP Training Pipeline
│   └── cross_model_ensemble_q1.py     # Final Validation & Ensemble Scripts
├── shared_utils.py                    # Augmentations, Datasets, Metrics, and Losses
└── create_new_dataset_csvs.py         # Stratified Leak-Free Patient Splitting
```

## 🚀 Quick Start
### 1. Requirements
Ensure you have Python 3.10+ and the required packages installed:
```bash
pip install torch torchvision torchaudio
pip install monai h5py pandas numpy scikit-learn matplotlib seaborn
```

### 2. Dataset Preparation
The dataset expects 3D volumetric HDF5 files (`.h5`). The patient-level split is handled automatically via:
```bash
python create_new_dataset_csvs.py
```

### 3. Training
To train the models using the strict Q1 protocols (5-Fold Cross-Validation, Gradient Accumulation, SWA):
```bash
python q1_final/train_attention_swinunetr.py
python q1_final/train_mae_tinytransformer.py
python q1_final/train_segformer3d.py
python q1_final/train_swinunetr_linearprobe.py
```
*Note: Due to VRAM constraints, gradient accumulation (effective batch size = 16) is heavily utilized. A GPU with at least 12GB VRAM is recommended.*

## 📊 Evaluation
The evaluation is automatically triggered at the end of the 5-fold training. It utilizes a **4-Direction TTA** and calculates the $95\%$ Bootstrap Confidence Intervals for the external test set.

## 🔗 Citation
*(Citation details will be updated upon acceptance)*
