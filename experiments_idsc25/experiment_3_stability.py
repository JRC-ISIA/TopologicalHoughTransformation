"""
experiment_3_stability.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Experiment to test the robustness of the Topological Hough Transform
License: MIT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import os
import random
import tqdm
import cv2
import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topologicalhoughtransform.topological_hough_transform import TopologicalHoughTransform
from utils.parser import create_parser
from utils.data_generator import generate_image, generate_line

NUM_MOVE_POINTS = 36

def generate_rand_y_delta():
    """Generates a random delta. Ensures magnitude >= 1."""
    delta = np.random.normal(0, 10)
    if np.abs(delta) < 1:
        delta += np.sign(delta) * 1
    return int(round(delta))

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    os.makedirs(args.output_directory, exist_ok=True)

    w1_distances, db_distances_pd, db_distances_img = [], [], []

    for sim_run in tqdm.tqdm(range(args.num_sim_rounds), desc="Simulation"):

        # 1. Pre-calculate deltas
        w1_deltas_array = np.array(
            [generate_rand_y_delta() for _ in range(NUM_MOVE_POINTS)]
        )

        # 2. Generate Initial Line
        # (Cap points to image width to prevent indexing crashes)
        coordinates = generate_line(args, slope=args.line_1_slope,
                                    intercept=args.line_1_intercept,
                                    num_points=min(args.n_point_line_1, args.image_width),
                                    noise_lvl=0)

        # 3. Establish Baseline (State 0)
        base_image = generate_image(coordinates, args)


        hough_transformer = TopologicalHoughTransform(
            base_image,
            value_threshold=150,
            pers_limit=120,
            three_periods=True, normalize=False)
        pers_array_0 = hough_transformer.get_persistence_array()

        line_points_0 = coordinates.copy()
        current_w1_accum = 0

        # 4. Iterative Move Loop
        for move_point_nb in range(NUM_MOVE_POINTS):

            if move_point_nb > 0:

                valid_indices = [i for i, p in enumerate(coordinates) if p[1] == 120]
                if not valid_indices:
                    valid_indices = range(len(coordinates))

                selected_point_idx = random.choice(valid_indices)
                delta_y = w1_deltas_array[move_point_nb]

                # Bound Checking
                old_x, old_y = coordinates[selected_point_idx]
                target_y = old_y + delta_y
                clamped_y = int(max(0, min(target_y, args.image_height - 1)))

                # Update Accumulator with ACTUAL move
                actual_delta = clamped_y - old_y
                coordinates[selected_point_idx] = (old_x, clamped_y)
                current_w1_accum += np.abs(actual_delta)

                # Generate new image
                image = generate_image(coordinates, args)


                hough_transformer = TopologicalHoughTransform(
                    image,
                    value_threshold=150, # Restored to 150
                    pers_limit=120,
                    three_periods=True, normalize=False)
                pers_array = hough_transformer.get_persistence_array()

                dgm1 = np.array(pers_array_0, dtype=np.float64)
                dgm2 = np.array(pers_array, dtype=np.float64)
                if len(dgm1) == 0: dgm1 = np.empty((0, 2))
                if len(dgm2) == 0: dgm2 = np.empty((0, 2))

                dB_pd = gd.bottleneck_distance(dgm1, dgm2)

                dgm1_inp = np.array(line_points_0, dtype=np.float64)
                dgm2_inp = np.array(coordinates, dtype=np.float64)
                dB_img = gd.bottleneck_distance(dgm1_inp, dgm2_inp)

                # Store Results
                w1_distances.append(current_w1_accum)
                db_distances_pd.append(dB_pd)
                db_distances_img.append(dB_img)

    logging.info("Experiment Complete.")

    df = pd.DataFrame({
        "W1 Image": w1_distances,
        "dB Pers": db_distances_pd,
        "dB Img": db_distances_img,
    })

    csv_file_path = os.path.join(args.output_directory, "distance_experiment.csv")
    df.to_csv(csv_file_path, index=False)
    logging.info(f"Saved to {csv_file_path}")
