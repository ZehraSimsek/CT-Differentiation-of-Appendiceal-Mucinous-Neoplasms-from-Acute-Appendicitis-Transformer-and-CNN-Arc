# 3D Medical Image Classification for Appendiceal Tumors

This repository contains the official PyTorch implementation for our Q1-submitted manuscript on the 3D classification of Acute Appendicitis and Mucinous Neoplasms from volumetric CT scans. 

Our codebase implements a robust, leak-free, and highly interpretable pipeline benchmarking state-of-the-art **3D CNNs** against advanced **3D Vision Transformers**.

## 📌 Features & Highlights
* **Comprehensive 3D Architectures:** 
  * **Transformers:** SwinUNETR Linear-Probe, Attention-SwinUNETR (AG-MSF), MAE-TinyTransformer3D, and SegFormer3D-MSCA.
  * **CNNs:** 3D adaptations of UNet++ (Classifier), DenseNet121, and EfficientNet-B0.
* **Rigorous Evaluation Protocol:** 5-Fold Cross-Validation combined with multiple randomized runs (totaling 60 model trainings for transformers) to ensure strict statistical significance and confidence bounds.
* **Clinical Constraint Optimization:** A dual-constraint fallback mechanism during training that strictly enforces Clinical Sensitivity and Specificity floors before maximizing the composite metric.
* **Explainable AI (XAI):** Built-in 3D Grad-CAM generation pipeline to visualize model focal points slice-by-slice, ensuring anatomical grounding rather than shortcut learning.

## 🏗️ Repository Structure
The repository is modularly divided into two distinct families of models:

```
├── cnn/                               # 3D CNN Architectures & Baselines
│   ├── models/                        # UNet++, DenseNet121, EfficientNet-B0
│   ├── engine/                        # Training & Evaluation Loops
│   ├── main.py                        # Training Entrypoint for CNNs
│   └── eval_test.py                   # External Testing & Grad-CAM Entrypoint
│
└── transformermodels/                 # 3D Transformer Architectures
    ├── final/                         # Core Model Implementations
    │   ├── train_attention_swinunetr.py
    │   ├── train_mae_tinytransformer.py
    │   ├── train_segformer3d.py
    │   └── train_swinunetr_linearprobe.py
    ├── run_all_multirun_2d.py         # Multi-run (60 models) Entrypoint
    └── shared_utils.py                # Augmentations, Datasets, and Metrics
```

## 🚀 Quick Start
### 1. Requirements
Ensure you have Python 3.10+ and the required packages installed:
```bash
pip install torch torchvision torchaudio
pip install monai h5py pandas numpy scikit-learn matplotlib seaborn tqdm
```

### 2. Dataset Preparation
The dataset expects 3D volumetric HDF5 files (`.h5`). The patient-level split (training, validation, and external testing) is predefined in fold CSV files to absolutely prevent data leakage across slices.

### 3. Training & Inference
**To train CNN models:**
```bash
cd cnn
python main.py --model unet --batch_size 4 --epochs 100
```
**To run the Transformer multi-run benchmark (5-fold × 3 runs = 60 trainings):**
```bash
python transformermodels/run_all_multirun_2d.py
```
*Note: Due to 3D volume VRAM constraints, gradient accumulation is heavily utilized. A GPU with at least 12GB VRAM is recommended.*

## 📊 Evaluation & Interpretability
The evaluation computes rigorous metrics including **Brier Score, Expected Calibration Error (ECE), and DeLong's Test** for ROC AUC significance. 

Furthermore, the inference scripts automatically generate 3D Grad-CAM overlays mapped onto the original HU-windowed CT slices to facilitate clinical validation of the models' decision-making processes.

## 🔗 Citation
*(Citation details will be updated upon acceptance)*
