import matplotlib.pyplot as plt
import numpy as np
import ast
import os

# --- Configuration ---
FILE_BASELINE = 'out/output_base.txt'
FILE_PH = 'out/output_ph.txt'

NOISE_LEVELS = list(range(5, 20))

COLOR_BLUE = '#648FFF'
COLOR_RED = '#DC267F'  

MARKER_PH = 's'           
MARKER_BASE = 'o'

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.linewidth': 0.8,
})

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

    min_len = min(len(cms_baseline), len(cms_ph), len(NOISE_LEVELS))
    cms_baseline = cms_baseline[:min_len]
    cms_ph = cms_ph[:min_len]
    x_axis_raw = NOISE_LEVELS[:min_len]

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

    print("\n" + "="*80)
    print("PRECISION AND ACCURACY (%) - By Noise Level")
    print("="*80)
    print(f"{'Noise':<8} {'Precision (%)':<30} {'Accuracy (%)':<30}")
    print(f"{'Level':<8} {'Baseline':<15} {'Our Method':<15} {'Baseline':<15} {'Our Method':<15}")
    print("-"*80)

    for i, noise in enumerate(x_sorted):
        print(f"{noise:<8} {prec_base[i]*100:<15.2f} {prec_ph[i]*100:<15.2f} {acc_base[i]*100:<15.2f} {acc_ph[i]*100:<15.2f}")

    print("-"*80)
    print(f"{'Mean':<8} {np.mean(prec_base)*100:<15.2f} {np.mean(prec_ph)*100:<15.2f} {np.mean(acc_base)*100:<15.2f} {np.mean(acc_ph)*100:<15.2f}")
    print("="*80 + "\n")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    def style_plot(ax, y_base, y_ph, title, show_ylabel=True):
        """Style plot to match paper's tikz specifications."""
        ax.plot(x_sorted, y_base, color=COLOR_RED, marker=MARKER_BASE,
                markersize=6, linestyle='-', linewidth=2,
                label='baseline', markeredgewidth=0)

        ax.plot(x_sorted, y_ph, color=COLOR_BLUE, marker=MARKER_PH,
                markersize=6, linestyle='-', linewidth=2,
                label='our method', markeredgewidth=0)

        ax.set_xlim(5, 19)
        ax.set_ylim(0.4, 1.1)

        ax.set_xlabel(r'Noise level $\varepsilon$', fontsize=11)
        if show_ylabel:
            ax.set_ylabel('Value', fontsize=11)
        else:
            ax.set_yticklabels([])

        ax.grid(True, which='both', alpha=1.0)
        ax.grid(True, which='major', linewidth=0.6, color='gray', alpha=0.5)
        ax.grid(True, which='minor', linewidth=0.3, color='gray', alpha=0.2)
        ax.minorticks_on()

        ax.set_title(title, fontsize=11, pad=10)

        return ax

    style_plot(axs[0], acc_base, acc_ph, 'Accuracy', show_ylabel=True)
    style_plot(axs[1], prec_base, prec_ph, 'Precision', show_ylabel=False)
    style_plot(axs[2], f1_base, f1_ph, 'F1 Score', show_ylabel=False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2,
               frameon=True, fancybox=False, edgecolor='black',
               bbox_to_anchor=(0.5, -0.05), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.25)

    save_path = 'out/metrics_comparison1.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Publication-quality plots saved to {save_path}")
    # plt.show()  # Commented out to prevent hanging

if __name__ == "__main__":
    main()
