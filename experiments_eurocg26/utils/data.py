import numpy as np
from PIL import Image

from ctht.utils.convert import theta_to_k, adjust_detected_line_params
from ctht.utils.evaluation import hough_space_param_distance


class DataGenerator(object):

    def __init__(self, img_width: int):
        self.img_width = img_width
        self.diag_len = np.hypot(img_width, img_width)

        self.coordinates = np.empty((0, 2), dtype=int)
        self.true_line_params = []
        self.mask = None
        self.original_coordinates = None

    def shift_points_orthogonal(self, pts: np.array, theta: float, s: float) -> np.array:
        """
        Shift a point (x0, y0) orthogonally to the line defined by slope k by a distance s (in pixel).
        Parameters:
        - pts: numpy array of shape (N, 2) containing points (x, y).
        - theta: angle of the line in radians.
        - s: distance to shift the point orthogonally.
        Returns:
        - x_new: new x-coordinate of the point after shifting.
        - y_new: new y-coordinate of the point after shifting.
        """
        k = theta_to_k(theta)  # get the slope

        v = np.array([-k, 1])
        v = v / np.linalg.norm(v)

        # Shift point
        shift_ = s * v.reshape(2, -1)
        pts += shift_.T

        return pts#.round().astype(int)

    def add_noised_line(self, num_points:int=10, sigma:int=2, min_limit:int=0.8,
                        min_dist_param_space:int=2, line_param=None) -> None:
        """
        Generate points along a line with added Gaussian noise.
        Parameters:
        - img_width: width of the image space.
        - num_points: number of points to generate.
        - sigma: standard deviation of the Gaussian noise.
        Returns:
        - None.
        Note:
            safety_cnt is used to ensure that we generate enough points, but
            still  being able to exit the loop if it takes too long. So we are
            relaxing the desired num_points condition a bit.
        """
        n_gen_pts = 0
        x_vals, y_vals = 0 , 0
        theta_rad, rho = 0, 0

        min_line_dist = 0

        def gen_theta_rho():
            theta_rad = np.random.uniform(low=0, high=np.pi)  # [low, high)
            rho = np.random.uniform(low=-(self.img_width//2-1), high=self.img_width//2)  # [low, high)
            return rho, theta_rad

        while n_gen_pts <= num_points * min_limit and min_line_dist < min_dist_param_space:
            if line_param is None:
                theta_rad, rho = gen_theta_rho()
            else:
                rho, theta_rad = line_param

            rho, x_vals, y_vals = self.get_x_y_values(num_points, rho, theta_rad)
            n_gen_pts = x_vals.shape[0]

            if len(self.true_line_params) > 0:
                min_line_dist = np.min([
                    np.abs(hough_space_param_distance([[rho, theta_rad]], [tlp]))
                    for tlp in self.true_line_params
                ])
            else:
                min_line_dist = 1.0

        points = np.column_stack((x_vals, y_vals))

        # Shift the point orthogonally by a random distance
        s = np.random.uniform(-sigma, sigma, size=points.shape[0])
        if self.original_coordinates is None:
            self.original_coordinates = points.copy()
        else:
            self.original_coordinates = np.concatenate(
                [self.original_coordinates, points.copy()], axis=0)

        points = self.shift_points_orthogonal(points, theta_rad, s)
        mask = self._return_points_in_image(points)
        if self.mask is None:
            self.mask = mask.copy()
        else:
            self.mask = np.concatenate([self.mask, mask.copy()], axis=0)
        points = points[mask]

        #points = points.round().astype(int)

        if points.shape[0] > 0:
            self.coordinates = np.concatenate([self.coordinates, points], axis=0)
            self.true_line_params.append((rho, theta_rad))

        self.true_line_params = adjust_detected_line_params(self.true_line_params)

    def add_fix_line(self, rho, theta, num_points) -> None:
        """
        Generate points along a fixed line without added Gaussian noise.
        Parameters:
        - rho: distance from the origin to the line.
        - theta: angle of the line in radians.
        Returns:
        - None.
        Note:
            safety_cnt is used to ensure that we generate enough points, but
            still  being able to exit the loop if it takes too long. So we are
            relaxing the desired num_points condition a bit.
        """

        rho, x_vals, y_vals = self.get_x_y_values(num_points, rho, theta)

        points = np.column_stack((x_vals, y_vals))
        mask = self._return_points_in_image(points)
        points = points[mask]
        #points = points.round().astype(int)

        if points.shape[0] > 0:
            self.coordinates = np.concatenate([self.coordinates, points], axis=0)
            self.true_line_params.append((rho, theta))

        self.true_line_params = adjust_detected_line_params(self.true_line_params)

    def get_x_y_values(self, num_points, rho, theta_rad):
        img_size = self.img_width // 2
        if np.isclose(np.sin(theta_rad), 0, atol=1e-9):
            x_vals = np.ones(num_points) * rho
            y_vals = np.random.randint(-img_size, img_size, num_points)
        else:
            b1 = np.clip(
                ((rho - img_size * np.sin(theta_rad)) / np.cos(
                    theta_rad)).round().astype(int),
                -img_size, img_size)
            b2 = np.clip(
                ((rho + img_size * np.sin(theta_rad)) / np.cos(
                    theta_rad)).round().astype(int),
                -img_size, img_size)

            x_min = np.min([b1, b2])
            x_max = np.max([b1, b2])
            if num_points > (x_max - x_min):
                x_vals = np.arange(x_min, x_max, 1)
            else:
                x_vals = np.random.randint(x_min, x_max, num_points)

            y_vals = (rho - x_vals * np.cos(theta_rad)) / np.sin(theta_rad)

        return rho, x_vals, y_vals

    def _return_points_in_image(self, points: np.array) -> np.array:
        mask = (
            (points[:, 0] >= -self.img_width//2 - 1) &
            (points[:, 0] <  self.img_width//2) &
            (points[:, 1] >= -self.img_width//2 - 1) &
            (points[:, 1] <  self.img_width//2)
        )
        return mask

    def get_image(self) -> np.array:
        """Create a black image and insert coordinates as white pixels."""
        image = Image.new('L', (self.img_width, self.img_width), 0)
        imwh = self.img_width // 2

        if self.coordinates is not None:
            for x, y in self.coordinates:
                x, y = int(x), int(y)
                image.putpixel((x+imwh, y+imwh), 255)  # Set pixel to white

        return np.array(image)


def shift_line_rho_theta(rho, theta, d):
    """
    Return (rho', theta') of the same line after shifting the origin
    by (dx, dy) in the +x, +y directions.
    """
    rho_new = rho + d * (np.cos(theta) + np.sin(theta))
    return rho_new, theta


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    num_noise = 10

    fig, axs = plt.subplots(nrows=2, ncols=5)
    axs = axs.flatten()

    for idx, ax in enumerate(axs):
        data_gen = DataGenerator(img_width=64)
        data_gen.add_noised_line(num_points=40, sigma=(idx+1))
        ax.imshow(data_gen.get_image())
        ax.set_aspect('equal')
        ax.set_xlim(0, 63)
        ax.set_ylim(0, 63)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f'Sigma={(idx+1)}')
    plt.savefig("out/data_example.png", dpi=300)
