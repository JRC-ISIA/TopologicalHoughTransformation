"""
Precision.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Topological Hough Transform implementation using the
  superlevel-set filtration with persistence homology.
Note: This experiment was not part of the publication.
License: MIT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.topological_hough_transform import TopologicalHoughTransform
from src.eval import find_closest_line
from utils.baseline_hough_transform import baseline_detect_lines
from utils.data_generator import (generate_image, generate_hough_line)
from utils.plotting import draw_lines_on_image
from utils.parser import create_parser

if __name__ == '__main__':
    # Create parser and get args
    parser = create_parser()
    args = parser.parse_args()
    
    filenames = []  # used to store the plots for gif generation
    
    # Use output directory from args, with fallback to default
    output_dir = getattr(args, 'output_directory', './output_folder/')
    subfolder_path = output_dir
    workingfolder_path = os.path.join(output_dir, 'individual_plots')

    # Create output directories
    os.makedirs(subfolder_path, exist_ok=True)
    os.makedirs(workingfolder_path, exist_ok=True)

    rho_diffs_PH = []
    theta_diffs_PH = []
    rho_diffs_BL = []
    theta_diffs_BL = []

    cumulative_rho_diffs = {'BL': [], 'PH': []}
    cumulative_theta_diffs = {'BL': [], 'PH': []}

    # Use num_sim_rounds from args, default to 10 for faster testing
    num_iterations = getattr(args, 'num_sim_rounds', 10)
    print(f"Running precision experiment with {num_iterations} iterations...")

    for iteration in range(num_iterations):
        rho = random.randint(0, 100)
        theta = random.uniform(0, np.pi)
        noise = random.randint(0, 10)
        coordinates = generate_hough_line(
            rho=rho, theta=theta, args=args, noise_lvl=noise,
            origin_shift=True, num_points=220)

        image = generate_image(coordinates, args)
        edges = np.array(image)
        original_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # adjust rho to mach the origin shift
        true_rho = rho + (127 * np.cos(theta) + 128 * np.sin(theta))
        true_line = [
            (rho + (127 * np.cos(theta) + 128 * np.sin(theta))), theta
        ]

        hough_transformer = TopologicalHoughTransform(
            image, value_threshold=150, pers_limit=150, three_periods=True)

        baseline_img, lines_found_by_opencv = baseline_detect_lines(
            original_image=original_image, img_edges=edges, threshold=20)

        lines_found_by_PH = hough_transformer.get_lines_rho_theta()

        transformed_lines = []
        if lines_found_by_PH is not None:
            for rho, theta in lines_found_by_PH:
                if theta < 0:
                    theta += np.pi  # Add π to theta
                    rho = -rho  # Change the sign of rho to keep the same line
                transformed_lines.append((rho, theta))

        img_PH = draw_lines_on_image(hough_transformer)
        
        # Save individual comparison plots for each iteration with better styling
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Baseline method plot
        axs[0].imshow(baseline_img, cmap='gray')
        axs[0].set_title("Baseline Method (OpenCV Hough)", fontsize=14, fontweight='bold')
        # Count lines safely
        opencv_count = len(lines_found_by_opencv) if lines_found_by_opencv is not None else 0
        ph_count = len(lines_found_by_PH) if lines_found_by_PH is not None else 0
        
        axs[0].text(0.02, 0.98, f"Lines detected: {opencv_count}", 
                   transform=axs[0].transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axs[0].axis('off')
        
        # PH method plot  
        axs[1].imshow(img_PH, cmap='gray')
        axs[1].set_title("Topological Hough Transform", fontsize=14, fontweight='bold')
        axs[1].text(0.02, 0.98, f"Lines detected: {ph_count}", 
                   transform=axs[1].transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axs[1].axis('off')
        
        # Add detailed title with true line parameters
        fig.suptitle(f"Iteration {iteration+1}/{num_iterations} | True Line: ρ={true_line[0]:.2f}, θ={true_line[1]:.3f} rad | Noise Level: {noise}", 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save with iteration number for easy sorting
        comparison_filename = f'{workingfolder_path}/comparison_{iteration+1:03d}.png'
        plt.savefig(comparison_filename, dpi=200, bbox_inches='tight')
        filenames.append(comparison_filename)  # Store for potential GIF creation
        plt.close()  # Important: close the figure to free memory
        
        print(f"Iteration {iteration+1}/{num_iterations} - True Lines: {true_line}")
        if args.log_level == "DEBUG":
            print(f" Lines found by Open CV: {lines_found_by_opencv}")
            print(f" Lines found by PH: {transformed_lines}")

        if lines_found_by_opencv is not None and lines_found_by_PH is not None:
            closest_BL_line = find_closest_line(true_line,
                                                lines_found_by_opencv)
            if closest_BL_line:
                found_rho, found_theta = closest_BL_line

                cumulative_rho_diffs['BL'].append(
                    abs(true_line[0] - found_rho))

                cumulative_theta_diffs['BL'].append(
                    abs(true_line[1] - found_theta))

            closest_PH_line = find_closest_line(true_line, transformed_lines)
            if closest_PH_line:
                found_rho, found_theta = closest_PH_line
                cumulative_rho_diffs['PH'].append(
                    abs(true_line[0] - found_rho))
                cumulative_theta_diffs['PH'].append(
                    abs(true_line[1] - found_theta))

    # Print statistics
    print(f"\nResults Summary:")
    print(f"Total iterations: {num_iterations}")
    print(f"Baseline method - Rho differences: {len(cumulative_rho_diffs['BL'])} measurements")
    print(f"PH method - Rho differences: {len(cumulative_rho_diffs['PH'])} measurements")
    print(f"Baseline method - Theta differences: {len(cumulative_theta_diffs['BL'])} measurements")
    print(f"PH method - Theta differences: {len(cumulative_theta_diffs['PH'])} measurements")
    
    # Calculate and print mean differences if we have data
    if cumulative_rho_diffs['BL']:
        print(f"Baseline - Mean Rho difference: {np.mean(cumulative_rho_diffs['BL']):.3f}")
    if cumulative_rho_diffs['PH']:
        print(f"PH Method - Mean Rho difference: {np.mean(cumulative_rho_diffs['PH']):.3f}")
    if cumulative_theta_diffs['BL']:
        print(f"Baseline - Mean Theta difference: {np.mean(cumulative_theta_diffs['BL']):.3f}")
    if cumulative_theta_diffs['PH']:
        print(f"PH Method - Mean Theta difference: {np.mean(cumulative_theta_diffs['PH']):.3f}")
    
    print(f"Individual comparison plots saved in: {workingfolder_path}")
    print(f"Total comparison plots created: {len(filenames)}")

    # Plotting the results
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Create a boxplot for cumulative rho differences
    if cumulative_rho_diffs['BL'] and cumulative_rho_diffs['PH']:
        # Create boxplots - using explicit positions to avoid warnings
        positions = [1, 2]
        bp1 = axs[0].boxplot([cumulative_rho_diffs['BL'], cumulative_rho_diffs['PH']], 
                            positions=positions, tick_labels=['Baseline', 'PH Hough'])
        axs[0].set_title('Cumulative Rho Differences')
        axs[0].set_ylabel('Absolute Rho Difference')
        axs[0].grid(True, alpha=0.3)

        bp2 = axs[1].boxplot([cumulative_theta_diffs['BL'], cumulative_theta_diffs['PH']], 
                            positions=positions, tick_labels=['Baseline', 'PH Hough'])
        axs[1].set_title('Cumulative Theta Differences')
        axs[1].set_ylabel('Absolute Theta Difference')
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{subfolder_path}/precision_comparison_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {subfolder_path}/precision_comparison_results.png")
    else:
        print("Warning: No valid comparisons found. Check if lines were detected by both methods.")
    
    plt.close()
