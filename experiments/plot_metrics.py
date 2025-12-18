import matplotlib.pyplot as plt
import numpy as np
import ast
import os

# --- Configuration ---
FILE_BASELINE = 'out/output_base.txt'
FILE_PH = 'out/output_ph.txt'
START_NOISE_LEVEL = 5  # The starting noise level from your experiment
# ---------------------

def read_last_line_matrix(filename):
    """Reads the last line of the file and parses the list of confusion matrices."""
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return []

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            print(f"Error: File {filename} is empty.")
            return []
        
        last_line = lines[-1].strip()
        
        # specific parsing for the format: "Confusion Matrices X: [...]"
        if ":" in last_line:
            try:
                # Split by the first colon and take the right part
                data_str = last_line.split(':', 1)[1].strip()
                # Safely evaluate the string as a Python list
                data = ast.literal_eval(data_str)
                return data
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing data from {filename}: {e}")
                return []
        else:
            print(f"Error: Line format in {filename} unexpected. Expected 'Title: [...]'")
            return []

def calculate_metrics(cm):
    """
    Calculates metrics from a confusion matrix [[TP, FN], [FP, TN]].
    Returns: precision, recall, accuracy, f1
    """
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1] # Usually 0 in this specific line detection context
    
    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        
    # Recall (Sensitivity): TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
    # Accuracy: (TP + TN) / Total
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0
        
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    return precision, recall, accuracy, f1

def main():
    # 1. Load Data
    cms_baseline = read_last_line_matrix(FILE_BASELINE)
    cms_ph = read_last_line_matrix(FILE_PH)

    if not cms_baseline or not cms_ph:
        print("Could not load data. Exiting.")
        return

    # Ensure both have the same length for plotting
    min_len = min(len(cms_baseline), len(cms_ph))
    cms_baseline = cms_baseline[:min_len]
    cms_ph = cms_ph[:min_len]

    # Generate Noise Levels (X-axis)
    noise_levels = list(range(START_NOISE_LEVEL, START_NOISE_LEVEL + min_len))

    # 2. Calculate Metrics
    metrics_base = [calculate_metrics(cm) for cm in cms_baseline]
    metrics_ph = [calculate_metrics(cm) for cm in cms_ph]

    # Unzip into separate lists for plotting
    # Structure: [(prec, rec, acc, f1), ...] -> (prec_list, rec_list, acc_list, f1_list)
    p_base, r_base, a_base, f1_base = zip(*metrics_base)
    p_ph, r_ph, a_ph, f1_ph = zip(*metrics_ph)

    # 3. Print Results Table
    print("\n" + "="*80)
    print("PRECISION AND ACCURACY (%) - By Noise Level")
    print("="*80)
    print(f"{'Noise':<8} {'Precision (%)':<30} {'Accuracy (%)':<30}")
    print(f"{'Level':<8} {'Baseline':<15} {'Topological':<15} {'Baseline':<15} {'Topological':<15}")
    print("-"*80)
    
    for i, noise in enumerate(noise_levels):
        print(f"{noise:<8} {p_base[i]*100:<15.2f} {p_ph[i]*100:<15.2f} {a_base[i]*100:<15.2f} {a_ph[i]*100:<15.2f}")
    
    print("-"*80)
    print(f"{'Mean':<8} {np.mean(p_base)*100:<15.2f} {np.mean(p_ph)*100:<15.2f} {np.mean(a_base)*100:<15.2f} {np.mean(a_ph)*100:<15.2f}")
    print("="*80 + "\n")

    # 4. Plotting (only Precision and Accuracy)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performance Metrics: Topological vs Baseline Hough Transform', fontsize=16)

    # Helper function for subplots
    def plot_metric(ax, y_base, y_ph, title, ylabel):
        ax.plot(noise_levels, y_base, 'o--', label='Baseline', color='tab:blue', alpha=0.7)
        ax.plot(noise_levels, y_ph, 'o-', label='Topological (PH)', color='tab:orange', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Noise Level')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plot_metric(axs[0], p_base, p_ph, 'Precision (TP / (TP + FP))', 'Precision')
    plot_metric(axs[1], a_base, a_ph, 'Accuracy (TP / Total)', 'Accuracy')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    save_path = 'out/metrics_comparison1.png'
    plt.savefig(save_path)
    print(f"Plots saved successfully to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
