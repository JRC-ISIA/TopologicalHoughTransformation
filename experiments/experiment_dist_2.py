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

    pers_array_0 = []
    line_points_0 = []
    distances_i = []
    distances_p = []

    for i in range(0, args.num_sim_rounds):
        dist_picture = 0

        line_points = []
        coordinates = generate_line(slope=args.line_1_slope,
                                    intercept=args.line_1_intercept,
                                    num_points=args.n_point_line_1)

        for j in range(0, 36):
            if j > 0:
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
                    (x, new_y) if (x, y) == selected_point else
                    (x, y) for x, y in coordinates
                ]
                logging.debug(f"Updated points: {coordinates}")
                dist_picture += np.abs(delta_y)

            logging.debug(f"# of moved Points: {j}, run:{i}\n")

            image = generate_image(coordinates)
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

            if j == 0:
                pers_array_0 = pers_array
                line_points_0 = line_points
            else:
                # Convert persistence diagrams to numpy arrays with
                #   float values
                logging.debug("Calculating distance...")
                dgm1 = np.array(pers_array_0, dtype=np.float64)
                dgm2 = np.array(pers_array, dtype=np.float64)
                # Compute Bottleneck Distance
                distance_p = gd.bottleneck_distance(dgm1, dgm2)
                logging.debug(f"DistanceHough: {distance_p}")
                distances_p.append(distance_p)
                distances_i.append(dist_picture)

    logging.info(f"Distance Hough Array: {distances_p}")
    logging.info(f"Distance Image Array: {distances_i}")

    df = pd.DataFrame({
        "Distance_image": distances_i,
        "Distance_pers": distances_p,
    })

    # Define file path
    csv_file_path = "bottleneck_distances_2.csv"
    df.to_csv(csv_file_path, index=False)
    logging.info(f"CSV file saved as {csv_file_path}")

    # Scatter plot for each unique Epsilon
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Distance_image"], df["Distance_pers"], alpha=0.6)
    plt.xlabel("Cout of Pixels points were moved")
    plt.ylabel("Bottleneck Distance Persistence Diagram")
    plt.title("Stability")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
