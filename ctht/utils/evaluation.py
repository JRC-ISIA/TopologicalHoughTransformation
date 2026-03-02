import os.path

import numpy as np

from abc import ABC, abstractmethod


def angle_between_lines(theta1, theta2):
    # get direction vectors from angles
    phi1 = theta1 + np.pi/2
    phi2 = theta2 + np.pi/2

    # Calculate the absolute difference between the angles
    alpha = abs(phi1 - phi2)

    # Normalize to [0, π]
    alpha = alpha % np.pi

    # Return acute angle
    return min(alpha, np.pi - alpha)


def line_midpoint(line, img_width):
    rho, theta = line

    xmin, xmax, ymin, ymax = 0, img_width, 0, img_width
    points = []

    # Intersect with x = xmin
    y = (rho - xmin * np.cos(theta)) / np.sin(theta)
    if ymin <= y <= ymax:
        points.append([xmin, y/img_width])

    # Intersect with x = xmax
    y = (rho - xmax * np.cos(theta)) / np.sin(theta)
    if ymin <= y <= ymax:
        points.append([1, y/img_width])

    # Intersect with y = ymin
    x = (rho - ymin * np.sin(theta)) / np.cos(theta)
    if xmin <= x <= xmax:
        points.append([x/img_width, 0])

    # Intersect with y = ymax
    x = (rho - ymax * np.sin(theta)) / np.cos(theta)
    if xmin <= x <= xmax:
        points.append([x/img_width, ymax/img_width])

    if len(points) != 2:
        raise ValueError("Line does not intersect box in two points.")

    midpoint = np.mean(points, axis=0)
    return midpoint


def ead_metric(detected_line, ground_truth_line, img_width):
    """
    Calculate the EAD metric between detected lines and ground truth lines.

    Parameters:
    - detected_lines: List of tuples (rho, theta) for detected lines.
    - ground_truth_lines: List of tuples (rho, theta) for ground truth lines.

    Returns:
    - ead: The EAD metric value.
    """
    alpha = angle_between_lines(detected_line[1], ground_truth_line[1])
    S_theta = 1- alpha / (np.pi / 2)

    midpoint_detected = line_midpoint(detected_line, img_width)
    midpoint_ground_truth = line_midpoint(ground_truth_line, img_width)

    S_d = 1 - np.linalg.norm(midpoint_detected - midpoint_ground_truth)

    return (S_theta * S_d)**2


def hough_space_param_distance(gt, ht):
    gt, ht = np.array(gt), np.array(ht)

    if gt.shape != ht.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape}, ht {ht.shape}")

    # Prepare three versions of ht: θ, θ+π, θ−π
    rho, theta = ht[:, 0], ht[:, 1]
    variants = get_comp_variants(ht, rho, theta)

    # Compute pairwise distances: Δ = variants − gt[:, None, :]
    deltas = np.abs(variants - gt[:, np.newaxis, :])  # shape (N, 3, 2)
    dists = np.linalg.norm(deltas, axis=2)  # shape (N, 3)
    argmin_dists = dists.argmin(axis=1)  # shape (N,)

    idx = np.arange(gt.shape[0])
    min_dist_values = deltas[idx, argmin_dists, :]
    return min_dist_values.reshape(-1, 2)


def hough_space_distance(gt, ht, method='euclidean', normalize_diag=None):
    if method == 'euclidean':

        gt, ht = convert_to_np(gt, ht)

        if gt.shape != ht.shape:
            raise ValueError(f"Shape mismatch: gt {gt.shape}, ht {ht.shape}")

        # Prepare three versions of ht: θ, θ+π, θ−π
        rho, theta = ht[:, 0], ht[:, 1]
        if normalize_diag is not None:
            rho = (rho + normalize_diag) / (2*normalize_diag)
            theta = theta / np.pi
            gt = gt.copy()
            gt[:, 0] = (gt[:, 0] + normalize_diag) / (2 * normalize_diag)
            gt[:, 1] = gt[:, 1] / np.pi

        variants = get_comp_variants(ht, rho, theta)

        # Compute pairwise distances: Δ = variants − gt[:, None, :]
        deltas = variants - gt[:, np.newaxis, :]  # shape (N, 3, 2)
        dists = np.linalg.norm(deltas, axis=2)  # shape (N, 3)
        min_dists = dists.min(axis=1)  # shape (N,)

        return float(min_dists.sum())

    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'euclidean'.")


def convert_to_np(gt, ht):
    if type(gt) != np.ndarray:
        gt = np.array(gt.true_line_params)
    if type(ht) != np.ndarray:
        ht = np.array(ht.detected_line_params)
    return gt, ht


def get_comp_variants(ht, rho, theta):
    variants = np.stack([
        ht,
        np.column_stack((rho, theta + 2 * np.pi)),
        np.column_stack((-rho, theta + np.pi)),
        np.column_stack((-rho, theta - np.pi)),
        np.column_stack((rho, theta - 2 * np.pi))
    ], axis=1)  # shape (N, 3, 2)
    return variants


class Metric(ABC):
    def __init__(self, metric_name, max_num_noise_sigma, num_exp):
        self.metric_name = metric_name
        self.our_method = np.zeros((max_num_noise_sigma, num_exp))
        self.baseline = np.zeros((max_num_noise_sigma, num_exp))
        self.timings_our_method = np.zeros((max_num_noise_sigma, num_exp))
        self.timings_baseline = np.zeros((max_num_noise_sigma, num_exp))

    @abstractmethod
    def compute_ours(self, noise, exp_id, data_generator, ht_method, timing=None):
        pass

    @abstractmethod
    def compute_baseline(self, noise, exp_id, data_generator, ht_method, timing=None):
        pass


class EuclideanDistance(Metric):
    def __init__(self, max_num_noise_sigma, num_exp):
        super().__init__('Euclidean Distance', max_num_noise_sigma, num_exp)
        self.num_quads = np.ones(num_exp) * np.nan
        self.num_lines_detected = np.ones((max_num_noise_sigma, num_exp)) * np.nan
        self.num_lines_detected_baseline = np.ones((max_num_noise_sigma, num_exp)) * np.nan

    def compute_ours(self, noise, exp_id, data_generator, ht_method, timing=None,
                     faulty_detection=False):
        if faulty_detection == True:
            return

        self.our_method[noise, exp_id] = hough_space_distance(
            data_generator, ht_method, normalize_diag=data_generator.diag_len
        )
        self.num_quads[exp_id] = len(ht_method.quads)
        self.num_lines_detected[noise, exp_id] = ht_method.num_parameter
        if timing is not None:
            self.timings_our_method[noise, exp_id] = timing

    def compute_baseline(self, noise, exp_id, data_generator, ht_method, timing=None,
                         faulty_detection=False):
        if faulty_detection:
            return

        self.baseline[noise, exp_id] = hough_space_distance(
            data_generator, ht_method, normalize_diag=data_generator.diag_len
        )
        self.num_lines_detected_baseline[noise, exp_id] = ht_method.num_parameter
        if timing is not None:
            self.timings_baseline[noise, exp_id] = timing



    def normalize(self):
        if self.our_method.sum() != 0:
            self.our_method = self.our_method / self.our_method[0,:].mean()

        if self.baseline.sum() != 0:
            self.baseline = self.baseline / self.baseline[0,:].mean()

    def store_results(self, filepath):
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        np.savez_compressed(
            filepath,
            our_method=self.our_method,
            baseline=self.baseline,
            timings_our_method=self.timings_our_method,
            timings_baseline=self.timings_baseline,
            n_quads=self.num_quads,
            num_lines_detected=self.num_lines_detected,
            num_lines_detected_baseline=self.num_lines_detected_baseline,
        )

    def load_results(self, filepath):
        data = np.load(filepath)
        self.our_method = data['our_method']
        self.baseline = data['baseline']
        self.timings_our_method = data['timings_our_method']
        self.timings_baseline = data['timings_baseline']
        self.num_quads = data['n_quads']
        self.num_lines_detected = data['num_lines_detected']
        self.num_lines_detected_baseline = data['num_lines_detected_baseline']


if __name__ == '__main__':

    X = np.array([[10, np.pi/3], [30, np.pi-0.1], [50, 0.2]])
    X_prime = np.array([[10, np.pi/3], [35, 0.1], [60, np.pi-0.1]])

    print(hough_space_distance(X_prime, X))