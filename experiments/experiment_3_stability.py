"""
experiment_3_stability.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Stability experiment for the Topological Hough Transform
using the Lipschitz continuity property.
License: MIT
"""
import logging
import os
import random
import tqdm

import cv2
import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topologicalhoughtransform.topological_hough_transform import \
    TopologicalHoughTransform
from utils.parser import create_parser
from utils.data_generator import generate_image, generate_line

NUM_MOVE_POINTS = 36


def generate_rand_y_delta():
    # Generate a normally distributed change with mu=0 and sigma=10,
    # ensuring the change is at least 1 in magnitude
    delta = np.random.normal(0, 10)
    if np.abs(delta) < 1:
        delta += np.sign(delta) * 1
    delta = round(delta)
    return delta


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.info(args)

    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, 'tmp'), exist_ok=True)

    pers_array_0, line_points_0 = [], []
    w1_distances, db_distances_pd, db_distances_img = [], [], []

    for sim_run in tqdm.tqdm(range(args.num_sim_rounds)):
        w1_dist = np.array(
            [generate_rand_y_delta() for _ in range(0, NUM_MOVE_POINTS)]
        )

        line_points = []

        coordinates = generate_line(args, slope=args.line_1_slope,
                                    intercept=args.line_1_intercept,
                                    num_points=args.n_point_line_1,
                                    noise_lvl=0)

        for move_point_nb in tqdm.tqdm(range(0, NUM_MOVE_POINTS)):
            if move_point_nb > 0:
                 # Randomly select one of these points
                selected_point_idx = np.random.randint(
                    low=0, high=args.n_point_line_1)

                delta_y = w1_dist[move_point_nb]

                # Apply the change to the y-coordinate
                coordinates[selected_point_idx][1] += delta_y

            logging.debug("# of moved Points: %d; run: %d",
                          move_point_nb, sim_run)

            image = generate_image(coordinates, args)
            edges = np.array(image)
            original_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Plot the image
            if args.log_level == "DEBUG":
                plt.figure(figsize=(6, 6))
                plt.imshow(original_image, cmap="gray")
                plt.title("Generated Image with Moved Points")
                plt.show()

            hough_transformer = TopologicalHoughTransform(
                image, value_threshold=150, pers_limit=120,
                three_periods=True, normalize=False)

            pers_array = hough_transformer.get_persistence_array()
            line_points = [(x, y + 1000) for x, y in line_points]

            if move_point_nb == 0:
                pers_array_0 = pers_array
                line_points_0 = line_points
            else:
                # Convert persistence diagrams to numpy arrays with
                # float values
                logging.debug("Calculating distance...")
                dgm1 = np.array(pers_array_0, dtype=np.float64)
                dgm2 = np.array(pers_array, dtype=np.float64)

                # Compute Bottleneck Distance of Persistence Arrays
                dB_pd = gd.bottleneck_distance(dgm1, dgm2)
                logging.debug("Bottleneck Distance $d_B$ in Hough Space: %f",
                              dB_pd)
                db_distances_pd.append(dB_pd)

                # Compute Bottleneck Distance of Input images ($d_B$ between
                #   input space)
                dgm1_inp = np.array(line_points_0, dtype=np.float64)
                dgm2_inp = np.array(line_points, dtype=np.float64)

                # Compute Bottleneck Distance
                dB_img = gd.bottleneck_distance(dgm1_inp, dgm2_inp)
                logging.debug("Bottleneck Distance $d_B$ in Image Space: %f",
                              dB_img)
                db_distances_img.append(dB_img)

                w1_distances.append(int(np.abs(w1_dist).sum()))

    logging.info(f"d_B Distance Hough Space: {db_distances_pd}")
    logging.info(f"d_B Distance Image Space: {db_distances_img}")
    logging.info(f"W1 Distance Image Space: {w1_distances}")

    # Create a DataFrame
    df = pd.DataFrame({
        "W1 Image": w1_distances,
        "dB Pers": db_distances_pd,
        "dB Img": db_distances_img,
    })

    # Define file path
    csv_file_path = os.path.join(args.output_directory,
                                 "distance_experiment.csv")

    # Save to CSV
    df.to_csv(csv_file_path, index=False)
    logging.info(f"CSV file saved as {csv_file_path}")
