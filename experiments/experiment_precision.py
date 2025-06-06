"""
Precision.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Topological Hough Transform implementation using the
  superlevel-set filtration with persistence homology.
Note: This experiment was not part of the publication.
License: MIT
"""
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.baseline_hough_transform import baseline_detect_lines
from topologicalhoughtransform.TopologicalHoughTransform import \
    TopologicalHoughTransform

from utils.plotting import draw_lines_on_image
from utils.test_data_generator import (generate_image, generate_hough_line)
from topologicalhoughtransform.utils.eval import find_closest_line

if __name__ == '__main__':
    filenames = []  # used to store the plots for gif generation
    subfolder_path = './output_folder/'
    workingfolder_path = './temp/'

    # Ausgabefolder und Tempfolder erstellen wenn nötig
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if not os.path.exists(workingfolder_path):
        os.makedirs(workingfolder_path)

    rho_diffs_PH = []
    theta_diffs_PH = []
    rho_diffs_BL = []
    theta_diffs_BL = []

    cumulative_rho_diffs = {'BL': [], 'PH': []}
    cumulative_theta_diffs = {'BL': [], 'PH': []}

    for _ in range(100):
        rho = random.randint(0, 100)
        theta = random.uniform(0, np.pi)
        noise = random.randint(0, 10)
        coordinates = generate_hough_line(
            rho=rho, theta=theta, noise_lvl=noise,
            origin_shift=True, num_points=220)

        image = generate_image(coordinates)
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
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(baseline_img)
        axs[0].set_title("Baseline")
        axs[1].imshow(img_PH)
        axs[1].set_title("PH Hough")
        plt.show()
        print(f" True Lines: {true_line}")
        print(f" Lines found by Open CV: {lines_found_by_opencv}")
        print(f" Lines found by PH:{transformed_lines}")

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

    # Plotting the results
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Create a boxplot for cumulative rho differences
    sns.boxplot(ax=axs[0],
                data=[cumulative_rho_diffs['BL'],
                      cumulative_rho_diffs['PH']])
    axs[0].set_xticklabels(['Baseline', 'PH Hough'])
    axs[0].set_title('Cumulative Rho Differences')

    # Create a boxplot for cumulative theta differences
    sns.boxplot(ax=axs[1],
                data=[cumulative_theta_diffs['BL'],
                      cumulative_theta_diffs['PH']])
    axs[1].set_xticklabels(['Baseline', 'PH Hough'])
    axs[1].set_title('Cumulative Theta Differences')
    plt.show()
