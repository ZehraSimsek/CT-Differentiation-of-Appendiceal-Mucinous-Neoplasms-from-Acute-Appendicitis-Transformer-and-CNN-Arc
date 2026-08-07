import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==========================================
# CONFIGURATION
# ==========================================
# CSV files expected in this directory. 
# Update the paths/filenames based on your uploaded CSVs.
CSV_DIR = "Grafik_Cizim_Verileri_CSV"

# Define the models, their exact CSV filenames, group (A or B), and display names.
# You can update 'filename' once you upload the Group B CSVs.
MODELS_CONFIG = {
    # Group A: Transformers (Blue/Cool shades)
    "segformer3d": {
        "filename": "segformer3d_ensemble_probs.csv",
        "display_name": "SegFormer3D",
        "color": "#1f77b4",  # Deep Blue
        "linestyle": "-"
    },
    "mae_tiny": {
        "filename": "mae_tinytransformer_ensemble_probs.csv",
        "display_name": "MAE-TinyTransformer3D",
        "color": "#00a8e8",  # Cyan / Light Blue
        "linestyle": "-"
    },
    "attention_swinunetr": {
        "filename": "attention_swinunetr_ensemble_probs.csv",
        "display_name": "Attention-SwinUNETR",
        "color": "#00509d",  # Dark Blue
        "linestyle": "-"
    },
    "swinunetr_lp": {
        "filename": "swinunetr_lp_ensemble_probs.csv",
        "display_name": "SwinUNETR (Linear Probe)",
        "color": "#89c2d9",  # Pale Blue
        "linestyle": "-"
    },
    
    # Group B: Radiomics / CNNs (Red/Warm shades)
    # TODO: Update these filenames with the exact names of your uploaded CSVs
    "radiomics_svm": {
        "filename": "radiomics_svm_probs.csv", 
        "display_name": "Radiomics + SVM",
        "color": "#d62828",  # Red
        "linestyle": "--"
    },
    "radiomics_rf": {
        "filename": "radiomics_rf_probs.csv", 
        "display_name": "Radiomics + RF",
        "color": "#f77f00",  # Orange
        "linestyle": "--"
    },
    "cnn_3d": {
        "filename": "cnn_3d_probs.csv", 
        "display_name": "3D CNN Baseline",
        "color": "#e07a5f",  # Light Red / Coral
        "linestyle": "--"
    }
}

OUTPUT_FILE = "Figure4_Combined_ROC_7Models.pdf"
OUTPUT_FILE_PNG = "Figure4_Combined_ROC_7Models.png"

def main():
    plt.figure(figsize=(10, 8))
    
    # Plot formatting for Q1 Journal Quality
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    
    lines = []
    labels = []

    for model_key, config in MODELS_CONFIG.items():
        csv_path = os.path.join(CSV_DIR, config["filename"])
        
        if not os.path.exists(csv_path):
            print(f"[WARNING] File not found: {csv_path}. Skipping {config['display_name']}.")
            continue
            
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Identify columns (assuming 'label' and 'prob_mucinous' or similar)
            label_col = 'label' if 'label' in df.columns else df.columns[1]
            prob_col = 'prob_mucinous' if 'prob_mucinous' in df.columns else df.columns[2]
            
            y_true = df[label_col].values
            y_prob = df[prob_col].values
            
            # Calculate ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Plot curve
            line, = plt.plot(
                fpr, tpr, 
                color=config["color"], 
                linestyle=config["linestyle"], 
                lw=2.5,
                alpha=0.9
            )
            
            lines.append(line)
            labels.append(f'{config["display_name"]} (AUC = {roc_auc:.3f})')
            print(f"[SUCCESS] Plotted {config['display_name']} with AUC {roc_auc:.3f}")
            
        except Exception as e:
            print(f"[ERROR] Could not process {csv_path}: {e}")

    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':', label='Random Guess (AUC = 0.500)')
    
    # Aesthetics
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title('Figure 4: Comparative ROC Analysis of All Models', fontweight='bold', pad=15)
    
    # Grid and styling
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend
    plt.legend(lines + [plt.gca().lines[-1]], labels + ['Random Guess (AUC = 0.500)'],
               loc="lower right", frameon=True, edgecolor='black', fontsize=12)

    # Save outputs
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_FILE_PNG, dpi=300, format='png', bbox_inches='tight')
    print(f"\n[DONE] Saved combined ROC plots to {OUTPUT_FILE} and {OUTPUT_FILE_PNG}")

if __name__ == "__main__":
    main()
