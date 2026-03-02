"""
optimize_thresholds.py
Description: Find optimal opencv_threshold and pers_limit for F1 score
             with n_point_line_1=150 and n_point_line_2=120
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import os
import random
from itertools import product

import cv2
import numpy as np

from topologicalhoughtransform.topological_hough_transform import TopologicalHoughTransform
from topologicalhoughtransform.eval import get_conf_matrix
from topologicalhoughtransform.transform import slope_intercept_to_rho_theta
from utils.baseline_hough_transform import baseline_detect_lines
from utils.data_generator import generate_image, generate_line
from utils.plotting import plot_hough_with_loci


class Args:
    """Simple args container"""
    def __init__(self):
        self.image_width = 256
        self.image_height = 256
        self.line_1_slope = 1.0
        self.line_2_slope = 1.0
        self.line_1_intercept = 0.0
        self.line_2_intercept = 0.0
        self.n_point_line_1 = 400
        self.n_point_line_2 = 350


def calc_f1(tp, fp, fn):
    """Calculate F1 score from confusion matrix components"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


def run_experiment(args, noise_levels, num_rounds, opencv_thresh, pers_limit):
    """Run experiment with given thresholds and return F1 scores"""

    total_tp_baseline, total_fp_baseline, total_fn_baseline = 0, 0, 0
    total_tp_ph, total_fp_ph, total_fn_ph = 0, 0, 0

    for noise_level in noise_levels:
        for _ in range(num_rounds):
            offset = random.randint(50, 100)

            rho1, theta1 = slope_intercept_to_rho_theta(
                (args.line_1_slope, args.line_1_intercept + offset))
            rho2, theta2 = slope_intercept_to_rho_theta(
                (args.line_2_slope, args.line_2_intercept - offset))

            true_lines = [(rho1, theta1), (rho2, theta2)]

            coordinates = generate_line(
                args, slope=args.line_1_slope,
                intercept=args.line_1_intercept + offset,
                noise_lvl=noise_level, num_points=args.n_point_line_1
            )
            coordinates += generate_line(
                args, slope=args.line_2_slope,
                intercept=args.line_2_intercept - offset,
                noise_lvl=noise_level, num_points=args.n_point_line_2
            )

            image = generate_image(coordinates, args)
            edges = np.array(image)
            original_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # PH method
            hough_transformer = TopologicalHoughTransform(
                image, value_threshold=150, pers_limit=pers_limit, three_periods=True)

            # Baseline
            _, lines = baseline_detect_lines(
                original_image=original_image, img_edges=edges,
                threshold=opencv_thresh)

            # Get corrected lines for evaluation
            my_true_lines, my_other_lines = plot_hough_with_loci(
                hough_transformer, true_lines=true_lines, other_lines=lines,
                show='none', my_ax=None)
            import matplotlib.pyplot as plt
            plt.close('all')

            # Baseline confusion matrix
            cm = get_conf_matrix(noise_level, my_true_lines.copy(), my_other_lines.copy())
            total_tp_baseline += cm[0][0]
            total_fn_baseline += cm[0][1]
            total_fp_baseline += cm[1][0]

            # PH confusion matrix
            cm = get_conf_matrix(noise_level, my_true_lines.copy(),
                               list(hough_transformer.get_lines()))
            total_tp_ph += cm[0][0]
            total_fn_ph += cm[0][1]
            total_fp_ph += cm[1][0]

    f1_baseline, prec_baseline, rec_baseline = calc_f1(
        total_tp_baseline, total_fp_baseline, total_fn_baseline)
    f1_ph, prec_ph, rec_ph = calc_f1(
        total_tp_ph, total_fp_ph, total_fn_ph)

    return {
        'baseline': {'f1': f1_baseline, 'precision': prec_baseline, 'recall': rec_baseline,
                     'tp': total_tp_baseline, 'fp': total_fp_baseline, 'fn': total_fn_baseline},
        'ph': {'f1': f1_ph, 'precision': prec_ph, 'recall': rec_ph,
               'tp': total_tp_ph, 'fp': total_fp_ph, 'fn': total_fn_ph}
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)

    args = Args()

    # Test across all noise levels
    noise_levels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    num_rounds = 10  # Rounds per noise level

    # Parameter ranges to search - adjusted for higher point counts (400/350)
    opencv_thresholds = [30, 35, 40, 45, 50, 55, 60]
    pers_limits = [140, 150, 160, 170, 180, 190, 200]

    print("=" * 80)
    print(f"Optimizing thresholds for n_point_line_1={args.n_point_line_1}, "
          f"n_point_line_2={args.n_point_line_2}")
    print(f"Noise levels: {noise_levels}, Rounds per level: {num_rounds}")
    print("=" * 80)

    best_f1_ph = 0
    best_params_ph = {}
    best_f1_baseline = 0
    best_params_baseline = {}

    results = []

    for opencv_thresh, pers_limit in product(opencv_thresholds, pers_limits):
        print(f"\nTesting opencv_threshold={opencv_thresh}, pers_limit={pers_limit}...",
              end=" ", flush=True)

        result = run_experiment(args, noise_levels, num_rounds, opencv_thresh, pers_limit)

        print(f"Baseline F1={result['baseline']['f1']:.3f}, "
              f"PH F1={result['ph']['f1']:.3f}")

        results.append({
            'opencv_threshold': opencv_thresh,
            'pers_limit': pers_limit,
            **result
        })

        if result['ph']['f1'] > best_f1_ph:
            best_f1_ph = result['ph']['f1']
            best_params_ph = {'opencv_threshold': opencv_thresh, 'pers_limit': pers_limit,
                             **result['ph']}

        if result['baseline']['f1'] > best_f1_baseline:
            best_f1_baseline = result['baseline']['f1']
            best_params_baseline = {'opencv_threshold': opencv_thresh, 'pers_limit': pers_limit,
                                   **result['baseline']}

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    print("\nBest parameters for BASELINE:")
    print(f"  opencv_threshold = {best_params_baseline['opencv_threshold']}")
    print(f"  F1 = {best_params_baseline['f1']:.4f}")
    print(f"  Precision = {best_params_baseline['precision']:.4f}")
    print(f"  Recall = {best_params_baseline['recall']:.4f}")

    print("\nBest parameters for PH METHOD:")
    print(f"  pers_limit = {best_params_ph['pers_limit']}")
    print(f"  F1 = {best_params_ph['f1']:.4f}")
    print(f"  Precision = {best_params_ph['precision']:.4f}")
    print(f"  Recall = {best_params_ph['recall']:.4f}")

    # Also find best combined (for fair comparison)
    print("\n" + "-" * 80)
    print("Top 10 configurations by PH F1 score:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x['ph']['f1'], reverse=True)
    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1}. opencv={r['opencv_threshold']:2d}, pers_limit={r['pers_limit']:3d} | "
              f"PH: F1={r['ph']['f1']:.3f} P={r['ph']['precision']:.3f} R={r['ph']['recall']:.3f} | "
              f"Base: F1={r['baseline']['f1']:.3f}")
