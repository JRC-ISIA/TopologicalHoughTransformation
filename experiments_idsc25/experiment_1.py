"""
experiment_1.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Experiment to compare the performance of the Topological Hough
Transform against the baseline OpenCV Hough Transform in terms of
precision and recall for different noise levels.
License: MIT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.topological_hough_transform import TopologicalHoughTransform
from src.eval import get_conf_matrix
from src.transform import slope_intercept_to_rho_theta, line_to_pts
from utils.baseline_hough_transform import baseline_detect_lines
from utils.colors import pth_color
from utils.parser import create_parser
from utils.plotting import draw_dashed_line, draw_lines_on_image, \
    plot_hough_with_loci, plot_persistence_diagram
from utils.data_generator import (generate_image, generate_line)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.info(args)

    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, 'tmp'), exist_ok=True)

    # hard settings
    args.line_2_slope = args.line_1_slope

    cms_baseline, cms_PH = [], []

    for noise_idx, noise_level in enumerate(args.noise_levels):
        cm_baseline = [[0, 0], [0, 0]]
        cm_PH = [[0, 0], [0, 0]]

        for sim_run in range(args.num_sim_rounds):
            logging.debug("Noiselvl: %d; run: %d",
                          noise_level, sim_run)

            offset = random.randint(50, 100)

            rho1, theta1 = slope_intercept_to_rho_theta(
                (args.line_1_slope, args.line_1_intercept + offset))
            rho2, theta2 = slope_intercept_to_rho_theta(
                (args.line_2_slope, args.line_2_intercept - offset))

            true_lines = [(rho1, theta1), (rho2, theta2)]
            logging.debug("True line coordinates at (%f, %f), (%f, %f)",
                          rho1, theta1, rho2, theta2)

            coordinates = generate_line(
                args, slope=args.line_1_slope,
                intercept=args.line_1_intercept+offset,
                noise_lvl=noise_level, num_points=args.n_point_line_1
            )

            coordinates += generate_line(
                args, slope=args.line_2_slope,
                intercept=args.line_2_intercept-offset,
                noise_lvl=noise_level, num_points=args.n_point_line_2
            )

            image = generate_image(coordinates, args)
            edges = np.array(image)
            original_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            hough_transformer = TopologicalHoughTransform(
                image, value_threshold=150, pers_limit=150, three_periods=True)
            img_with_lines_PH = draw_lines_on_image(hough_transformer)

            img_with_lines, lines = baseline_detect_lines(
                original_image=original_image, img_edges=edges,
                threshold=args.opencv_threshold)

            for line in lines:
                pt1, pt2 = line_to_pts(line[0])
                # Draw the dashed line
                draw_dashed_line(img_with_lines_PH, pt1, pt2, color=pth_color)

            # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            # axs[0].imshow(img_with_lines_PH)
            # axs[0].title.set_text('Image Space')

            # plot_persistence_diagram(hough_transformer, ax=axs[1],
            #                          show_limit=True, show_nums=False)

            my_true_lines, my_other_lines = plot_hough_with_loci(
                hough_transformer, true_lines=true_lines, other_lines=lines,
                show='none', my_ax=None)

            # # "Fake"-Handles für die Legende erstellen
            # handles = [
            #     plt.Line2D([0], [0], color='red', label='baseline'),
            #     plt.Line2D([0], [0], color='blue', label='our method')
            # ]

            # plt.tight_layout()
            # plt.savefig(os.path.join(args.output_directory, "tmp",
            #                          f'plot_{noise_level}{sim_run}.png'))
            plt.close('all')

            for line in lines:
                logging.debug("Lines found with opencv: (%f, %f)",
                              line[0][0], line[0][1])

            cm = get_conf_matrix(noise_level, my_true_lines, my_other_lines)
            cm_baseline = [
                [cm_baseline[i][j] + cm[i][j]
                 for j, _ in enumerate(cm_baseline[i])]
                for i, _ in enumerate(cm_baseline)]

            cm = get_conf_matrix(noise_level, my_true_lines,
                                 hough_transformer.get_lines())
            cm_PH = [
                [cm_PH[i][j] + cm[i][j]
                 for j, _ in enumerate(cm_PH[i])]
                for i, _ in enumerate(cm_PH)
            ]

        logging.info("Confusion Matrix Baseline for "
                     "noise level=%s: %s", noise_level, cm_baseline)
        logging.info("Confusion Matrix PH for d=%s: %s", noise_level, cm_PH)
        cms_PH.append(cm_PH)
        cms_baseline.append(cm_baseline)

    logging.debug("Confusion Matrices Baseline: %s", cms_baseline)
    logging.debug("Confusion Matrices PH: %s", cms_PH)

    with open(os.path.join(args.output_directory,
                           "output_base.txt"), "a", encoding='utf-8') as f:
        f.write(f"Confusion Matrices Baseline: {cms_baseline}\n")

    with open(os.path.join(args.output_directory,
                           "output_ph.txt"), "a", encoding='utf-8') as f:
        f.write(f"Confusion Matrices PH: {cms_PH}\n")
