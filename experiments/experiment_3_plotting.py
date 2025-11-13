"""
experiment_3_plotting.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Plotting script for the Lipschitz continuity experiment
using matplotlib and pandas.
License: MIT
"""
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Plot Lipschitz continuity experiment results')
    parser.add_argument('--csv-file', type=str, default='out/distance_experiment.csv',
                        help='Path to the CSV file with experiment results')
    parser.add_argument('--output-dir', type=str, default='out',
                        help='Directory to save the plots')
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        print("Available CSV files:")
        if os.path.exists('out/distance_experiment.csv'):
            print("  out/distance_experiment.csv")
        if os.path.exists('test_out/distance_experiment.csv'):
            print("  test_out/distance_experiment.csv")
        return
    
    # Read data
    print(f"Reading data from {args.csv_file}")
    data = pd.read_csv(args.csv_file)
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    d_W = data["W1 Image"]
    d_B = data["dB Pers"]
    d_B_img = data["dB Img"]
    
    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: W1 vs dB Persistence
    axs[0,0].scatter(d_W, d_B, c='blue', alpha=0.7, s=20)
    axs[0,0].set_xlabel('$d_W$ (W1 Distance in Image Space)')
    axs[0,0].set_ylabel('$d_B$ (Bottleneck Distance in Persistence Space)')
    axs[0,0].set_title('Stability: W1 vs Bottleneck Distance')
    axs[0,0].grid(True, alpha=0.3)
    
    # Plot 2: W1 vs dB Image (log scale due to very small values)
    axs[0,1].scatter(d_W, d_B_img, c='green', alpha=0.7, s=20)
    axs[0,1].set_xlabel('$d_W$ (W1 Distance in Image Space)')
    axs[0,1].set_ylabel('$d_B$ (Bottleneck Distance in Image Space)')
    axs[0,1].set_title('W1 vs Bottleneck Distance (Image Space)')
    axs[0,1].set_yscale('log')
    axs[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Ratio plot (W1/dB_Pers)
    ratio = d_W / d_B
    axs[1,0].scatter(d_W, ratio, c='red', alpha=0.7, s=20)
    axs[1,0].set_xlabel('$d_W$ (W1 Distance in Image Space)')
    axs[1,0].set_ylabel('$d_W / d_B$ (Ratio)')
    axs[1,0].set_title('Lipschitz Ratio: W1 / Bottleneck Distance')
    axs[1,0].grid(True, alpha=0.3)
    
    # Plot 4: dB Persistence vs dB Image
    axs[1,1].scatter(d_B, d_B_img, c='purple', alpha=0.7, s=20)
    axs[1,1].set_xlabel('$d_B$ (Bottleneck Distance in Persistence Space)')
    axs[1,1].set_ylabel('$d_B$ (Bottleneck Distance in Image Space)')
    axs[1,1].set_title('Persistence vs Image Space Distances')
    axs[1,1].set_yscale('log')
    axs[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'lipschitz_continuity_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"W1 Distance range: {d_W.min():.1f} - {d_W.max():.1f}")
    print(f"dB Persistence range: {d_B.min():.3f} - {d_B.max():.3f}")
    print(f"dB Image range: {d_B_img.min():.2e} - {d_B_img.max():.2e}")
    print(f"Lipschitz ratio (W1/dB_Pers) range: {ratio.min():.1f} - {ratio.max():.1f}")

if __name__ == '__main__':
    main()
