"""
RobustnessTest.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Experiment to test the robustness of the Topological Hough
                Transform against noise in the input data.
Note: This experiment was not part of the publication.
License: MIT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from src.topological_hough_transform import TopologicalHoughTransform
from src.eval import get_conf_matrix
from src.transform import slope_intercept_to_rho_theta
from utils.baseline_hough_transform import baseline_detect_lines
from utils.data_generator import (generate_image, generate_line)
from utils.plotting import draw_lines_on_image, plot_hough_with_loci, \
    plot_persistence_diagram
from utils.parser import create_parser

if __name__ == '__main__':
    # Create parser and get args
    parser = create_parser()
    args = parser.parse_args()
    
    filenames = []
    
    # Use output directory from args
    output_dir = getattr(args, 'output_directory', './out/')
    subfolder_path = os.path.join(output_dir, 'experiment_robustness')
    workingfolder_path = os.path.join(subfolder_path, 'individual_plots')

    intercept1 = 200
    intercept2 = 100
    slope = 0.1
    rho1, theta1 = slope_intercept_to_rho_theta((slope, intercept1))
    rho2, theta2 = slope_intercept_to_rho_theta((slope, intercept2))
    true_lines = [(rho1, theta1), (rho2, theta2)]
    print(f"True line coordinates: {true_lines}")

    # Ausgabefolder und Tempfolder erstellen wenn nötig
    os.makedirs(subfolder_path, exist_ok=True)
    os.makedirs(workingfolder_path, exist_ok=True)

    cm_baseline = [[0, 0], [0, 0]]
    cm_PH = [[0, 0], [0, 0]]

    print(f"Running robustness experiment...")
    print(f"Output will be saved to: {subfolder_path}")
    
    for noise_value in range(8, 10):
        print(f"Testing noise level: {noise_value}")
        for experiment_idx in range(0, 10):
            print(f"  Experiment {experiment_idx + 1}/10 for noise level {noise_value}")
            coordinates = generate_line(
                args=args, slope=slope, intercept=intercept1,
                noise_lvl=noise_value, num_points=150)

            coordinates += generate_line(
                args=args, slope=slope, intercept=intercept2,
                noise_lvl=noise_value, num_points=120)

            image = generate_image(coordinates, args)
            edges = np.array(image)
            original_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            hough_transformer = TopologicalHoughTransform(
                image, value_threshold=150,
                pers_limit=150, three_periods=True)
            img_with_lines_PH = draw_lines_on_image(hough_transformer)
            img_with_lines, lines = baseline_detect_lines(
                original_image=original_image, img_edges=edges, threshold=20)

            fig, axs = plt.subplots(3, 2, figsize=(10, 12),
                                    gridspec_kw={'height_ratios': [5, 2, 5]})
            fig.suptitle(f'Noise Level: {noise_value}')

            axs[0][0].imshow(img_with_lines_PH)
            axs[0][0].title.set_text('Our Method')
            axs[0][1].imshow(img_with_lines)
            axs[0][1].title.set_text('Baseline Method')
            filename = f'plot_{noise_value}{experiment_idx}.png'

            plot_persistence_diagram(hough_transformer,
                                     ax=axs[2][0], show_limit=True)

            my_true_lines, my_other_lines = plot_hough_with_loci(
                hough_transformer, true_lines=true_lines, other_lines=lines,
                show='none', my_ax=axs[2][1]
            )

            cm = get_conf_matrix(noise_value, my_true_lines, my_other_lines)
            cm_baseline = [
                [cm_baseline[i][j] + cm[i][j]
                 for j in range(len(cm_baseline[i]))]
                for i in range(len(cm_baseline))
            ]

            cm = get_conf_matrix(noise_value, my_true_lines,
                                 hough_transformer.get_lines())
            cm_PH = [
                [cm_PH[i][j] + cm[i][j]
                 for j in range(len(cm_PH[i]))]
                for i in range(len(cm_PH))
            ]

            plt.tight_layout()
            full_filepath = os.path.join(workingfolder_path, filename)
            plt.savefig(full_filepath, dpi=150, bbox_inches='tight')
            filenames.append(filename)  # Keep just filename for later use
            plt.close()  # Remove plt.show() to avoid blocking

            if args.log_level == "DEBUG":
                for line in lines:
                    print(f"Lines found with opencv: {line}")

    # Print experiment summary
    print(f"\nExperiment Summary:")
    print(f"Individual plots saved in: {workingfolder_path}")
    print(f"Total plots created: {len(filenames)}")
    print(f"Baseline confusion matrix: {cm_baseline}")
    print(f"PH method confusion matrix: {cm_PH}")

    # Create GIF from individual plots
    print(f"Creating animated GIF from {len(filenames)} individual plots...")
    images = [
        Image.open(os.path.join(workingfolder_path, filename))
        for filename in filenames
    ]
    gif_path = os.path.join(subfolder_path, 'robustness_animation.gif')
    images[0].save(gif_path, save_all=True, append_images=images[1:], 
                   optimize=False, duration=700, loop=0)
    print(f"Animated GIF saved to: {gif_path}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Line search results: ')

    sns.heatmap(cm_PH, annot=True, fmt='d', cmap='magma', ax=axs[0],
                annot_kws={"size": 10}, cbar=False,
                square=True, linewidths=0.5, linecolor='gray',
                xticklabels=['found', 'not found'],
                yticklabels=['existing', 'not existing'])

    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('Actual')
    axs[0].set_title('Our method')

    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='magma',
                ax=axs[1], annot_kws={"size": 10}, cbar=False,
                square=True, linewidths=0.5, linecolor='gray',
                xticklabels=['found', 'not found'],
                yticklabels=['existing', 'not existing'])

    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('Actual')
    axs[1].set_title('Baseline method')

    plt.tight_layout()
    confusion_matrix_path = os.path.join(subfolder_path, 'Confusion_Matrix_Results.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix results saved to: {confusion_matrix_path}")
    plt.close()
