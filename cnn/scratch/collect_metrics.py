import os
import pandas as pd
import glob

def collect():
    checkpoints_dir = "/Users/tunahanferramuzekelik/Desktop/ct-classification-pipeline/ct-classification-pipeline/checkpoints"
    csv_paths = glob.glob(os.path.join(checkpoints_dir, "*", "deneme_*", "metrics_summary.csv"))
    
    output_lines = []
    
    for path in sorted(csv_paths):
        # Extract model and deneme from path
        parts = path.split(os.sep)
        model_name = parts[-3]
        deneme_name = parts[-2]
        
        output_lines.append(f"\n### Model: {model_name} | Trial: {deneme_name}")
        output_lines.append("-" * 60)
        
        try:
            df = pd.read_csv(path)
            # Reformat columns for easier reading
            # Let's select key columns: Unnamed: 0 (Fold), auc_roc, accuracy, balanced_accuracy, sensitivity, specificity, precision, f1, optimal_threshold, opt_sensitivity, opt_specificity, opt_f1, opt_accuracy, opt_balanced_accuracy
            cols = [
                'Unnamed: 0', 'auc_roc', 'accuracy', 'balanced_accuracy', 
                'sensitivity', 'specificity', 'precision', 'f1', 
                'optimal_threshold', 'opt_sensitivity', 'opt_specificity', 'opt_f1', 'opt_accuracy', 'opt_balanced_accuracy'
            ]
            # Ensure columns exist
            cols = [c for c in cols if c in df.columns]
            sub_df = df[cols].copy()
            
            # Rename Unnamed: 0 to Fold
            sub_df.rename(columns={'Unnamed: 0': 'Fold'}, inplace=True)
            
            # Format numbers
            for col in sub_df.columns:
                if col != 'Fold':
                    sub_df[col] = sub_df[col].map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
            
            # Convert to markdown
            output_lines.append(sub_df.to_markdown(index=False))
            output_lines.append("\n")
        except Exception as e:
            output_lines.append(f"Error reading {path}: {e}\n")
            
    # Write to a file
    out_file = "/Users/tunahanferramuzekelik/Desktop/ct-classification-pipeline/ct-classification-pipeline/scratch/metrics_report_summary.md"
    with open(out_file, "w") as f:
        f.write("\n".join(output_lines))
    print(f"Summary written to {out_file}")

if __name__ == "__main__":
    collect()
