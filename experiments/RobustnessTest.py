"""
RobustnessTest.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Experiment to test the robustness of the Topological Hough
                Transform against noise in the input data.
Note: This experiment was not part of the publication.
License: MIT
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from utils.baseline_hough_transform import baseline_detect_lines
from topologicalhoughtransform.TopologicalHoughTransform import \
    TopologicalHoughTransform

from utils.plotting import draw_lines_on_image, plot_hough_with_loci, \
    plot_persistence_diagram
from utils.test_data_generator import (generate_image, generate_line)
from topologicalhoughtransform.utils.eval import get_conf_matrix
from topologicalhoughtransform.utils.transform import \
    slope_intercept_to_rho_theta

if __name__ == '__main__':
    filenames = []
    subfolder_path = './output_folder/'
    workingfolder_path = './temp/'

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

    for noise_value in range(8, 10):
        for experiment_idx in range(0, 10):
            coordinates = generate_line(
                slope=slope, intercept=intercept1,
                noise_lvl=noise_value, num_points=150)

            coordinates += generate_line(
                slope=slope, intercept=intercept2,
                noise_lvl=noise_value, num_points=120)

            image = generate_image(coordinates)
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

            cm = get_conf_matrix(my_true_lines, my_other_lines,
                                 hough_transformer.get_lines())
            cm_baseline = [
                [cm_baseline[i][j] + cm[i][j]
                 for j in range(len(cm_baseline[i]))]
                for i in range(len(cm_baseline))
            ]

            cm = get_conf_matrix(0, my_true_lines,
                                 hough_transformer.get_lines())
            cm_PH = [
                [cm_PH[i][j] + cm[i][j]
                 for j in range(len(cm_PH[i]))]
                for i in range(len(cm_PH))
            ]

            plt.tight_layout()
            plt.savefig(workingfolder_path+filename)
            filenames.append(filename)
            plt.show()
            plt.close()

            for line in lines:
                print(f"Lines found with opencv: {line}")

    images = [
        Image.open(os.path.join(workingfolder_path, filename))
        for filename in filenames
    ]
    images[0].save(os.path.join(subfolder_path, 'plots.gif'),
                   save_all=True, append_images=images[1:], optimize=False,
                   duration=700, loop=0)

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
    plt.savefig(os.path.join(workingfolder_path, 'Confusion Matrix'))
    plt.show()
    plt.close()
