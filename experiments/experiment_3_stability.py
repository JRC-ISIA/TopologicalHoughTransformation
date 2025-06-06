import logging
import os
import random

import cv2
import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topologicalhoughtransform.TopologicalHoughTransform import \
    TopologicalHoughTransform
from utils.parser import create_parser
from utils.test_data_generator import generate_image, generate_line


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.info(args)

    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, 'tmp'), exist_ok=True)

    pers_array_0, line_points_0 = [], []
    w1_distances, db_distances_pd, db_distances_img = [], [], []

    for sim_run in range(args.num_sim_rounds):
        w1_dist = 0

        line_points = []
        coordinates = generate_line(args, slope=args.line_1_slope,
                                    intercept=args.line_1_intercept,
                                    num_points=args.n_point_line_1,
                                    noise_lvl=0)

        for move_point_idx in range(0, 36):
            if move_point_idx > 0:
                # Filter points where the second value is 120
                valid_points = [p for p in coordinates if p[1] == 120]

                # Randomly select one of these points
                selected_point = random.choice(valid_points)

                # Generate a normally distributed change with mu=0 and sigma=10
                delta_y = 0
                while delta_y == 0:
                    delta_y = int(np.random.normal(0, 10))

                # Apply the change to the y-coordinate
                new_y = selected_point[1] + delta_y

                # Update the original list
                coordinates = [
                    (x, new_y)
                    if (x, y) == selected_point else
                    (x, y) for x, y in coordinates
                ]
                logging.debug(f"Updated points: {coordinates}")
                w1_dist += np.abs(delta_y)

            logging.debug(f"# of moved Points: {move_point_idx}, run:{sim_run}")

            image = generate_image(coordinates, args)
            edges = np.array(image)
            original_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Plot the image
            plt.figure(figsize=(6, 6))
            plt.imshow(original_image, cmap="gray")
            plt.title("Generated Image with Moved Points")
            plt.show()

            hough_transformer = TopologicalHoughTransform(
                image, value_threshold=150, pers_limit=120,
                three_periods=True, normalize=False)

            pers_array = hough_transformer.get_persistence_array()
            line_points = [(x, y + 1000) for x, y in line_points]

            if move_point_idx == 0:
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
                logging.debug(f"Bottleneck Distance $d_B$ in Hough Space: {dB_pd}")
                db_distances_pd.append(dB_pd)

                # Compute Bottleneck Distance of Input images ($d_B$ between
                #   input space)
                dgm1_inp = np.array(line_points_0, dtype=np.float64)
                dgm2_inp = np.array(line_points, dtype=np.float64)

                # Compute Bottleneck Distance
                dB_img = gd.bottleneck_distance(dgm1_inp, dgm2_inp)
                logging.debug(f"Bottleneck Distance $d_B$ in Image Space: {dB_img}")
                db_distances_img.append(dB_img)

                w1_distances.append(w1_dist)

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
    csv_file_path = os.path.join("output_folder",
                                 "distance_experiment.csv")

    # Save to CSV
    df.to_csv(csv_file_path, index=False)
    logging.info(f"CSV file saved as {csv_file_path}")
