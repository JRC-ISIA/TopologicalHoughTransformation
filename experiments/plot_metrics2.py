import matplotlib.pyplot as plt
import numpy as np
import ast
import os

# --- Configuration ---
FILE_BASELINE = 'out/output_base2.txt'
FILE_PH = 'out/output_ph2.txt'

# The experiment loop was: range(500, 150, -50) -> [500, 450, ... 200]
# We list them here to map the data correctly.
POINT_COUNTS_DESC = list(range(500, 50, -50)) 

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.color': '#cccccc'
})

COLOR_PH = '#6495ED'      # Cornflower Blue (for the "better" method/squares)
COLOR_BASE = '#C71585'    # Medium Violet Red (for the "baseline"/circles)

MARKER_PH = 's'           # Square
MARKER_BASE = 'o'         # Circle

def read_last_line_matrix(filename):
    """Reads the last line of the file and parses the confusion matrices."""
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return []

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            print(f"Error: File {filename} is empty.")
            return []
        
        last_line = lines[-1].strip()
        if ":" in last_line:
            try:
                data_str = last_line.split(':', 1)[1].strip()
                return ast.literal_eval(data_str)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing data from {filename}: {e}")
                return []
    return []

def calculate_metrics(cm):
    """Returns precision, recall, accuracy, f1"""
    TP, FN = cm[0][0], cm[0][1]
    FP, TN = cm[1][0], cm[1][1]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, accuracy, f1

def main():
    cms_baseline = read_last_line_matrix(FILE_BASELINE)
    cms_ph = read_last_line_matrix(FILE_PH)

    if not cms_baseline or not cms_ph:
        print("Could not load data. Exiting.")
        return

    min_len = min(len(cms_baseline), len(cms_ph), len(POINT_COUNTS_DESC))
    cms_baseline = cms_baseline[:min_len]
    cms_ph = cms_ph[:min_len]
    x_axis_raw = POINT_COUNTS_DESC[:min_len]

    metrics_base = [calculate_metrics(cm) for cm in cms_baseline]
    metrics_ph = [calculate_metrics(cm) for cm in cms_ph]
    combined_base = sorted(zip(x_axis_raw, metrics_base))
    combined_ph = sorted(zip(x_axis_raw, metrics_ph))

    x_sorted = [x for x, _ in combined_base]
    
    prec_base = [m[0] for _, m in combined_base]
    rec_base  = [m[1] for _, m in combined_base]
    acc_base  = [m[2] for _, m in combined_base]
    f1_base   = [m[3] for _, m in combined_base]

    prec_ph = [m[0] for _, m in combined_ph]
    rec_ph  = [m[1] for _, m in combined_ph]
    acc_ph  = [m[2] for _, m in combined_ph]
    f1_ph   = [m[3] for _, m in combined_ph]

    # Print Results Table
    print("\n" + "="*80)
    print("PRECISION AND ACCURACY (%) - By Point Count")
    print("="*80)
    print(f"{'Point':<8} {'Precision (%)':<30} {'Accuracy (%)':<30}")
    print(f"{'Count':<8} {'Baseline':<15} {'Our Method':<15} {'Baseline':<15} {'Our Method':<15}")
    print("-"*80)
    
    for i, pc in enumerate(x_sorted):
        print(f"{pc:<8} {prec_base[i]*100:<15.2f} {prec_ph[i]*100:<15.2f} {acc_base[i]*100:<15.2f} {acc_ph[i]*100:<15.2f}")
    
    print("-"*80)
    print(f"{'Mean':<8} {np.mean(prec_base)*100:<15.2f} {np.mean(prec_ph)*100:<15.2f} {np.mean(acc_base)*100:<15.2f} {np.mean(acc_ph)*100:<15.2f}")
    print("="*80 + "\n")

    # Plotting (only Precision and Accuracy)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    def style_plot(ax, y_base, y_ph, bottom_title):
        ax.plot(x_sorted, y_ph, color=COLOR_PH, marker=MARKER_PH, 
                markersize=8, linestyle='-', label='Our Method', linewidth=1.5)
        
        ax.plot(x_sorted, y_base, color=COLOR_BASE, marker=MARKER_BASE, 
                markersize=8, linestyle='-', label='Baseline', linewidth=1.5)
        
        ax.set_xlabel(r'Pointcount $n_2$', labelpad=10)
        
        ax.text(0.5, -0.3, bottom_title, transform=ax.transAxes, 
                ha='center', va='top', fontsize=14, family='serif')
        
        ax.set_ylim(-0.05, 1.05)
        
        ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')

    style_plot(axs[0], prec_base, prec_ph, r'(a) Precision')
    style_plot(axs[1], acc_base, acc_ph, r'(b) Accuracy')

    # Add a single legend to the figure (optional, or per plot)
    axs[0].legend(loc='best', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.4) 
    
    save_path = 'out/metrics_comparison2.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Styled plots saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
