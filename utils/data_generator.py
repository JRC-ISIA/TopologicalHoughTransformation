"""
data_generator.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Methods to generate synthetic test data for image processing
             tasks.
License: MIT
"""
import random

import numpy as np
from PIL import Image


def _round_np_to_int(arr):
    """Round numpy array to integers."""
    return np.round(arr).astype(np.int32)


def rgb2gray(rgb):
    """Convert RGB image to grayscale using the luminosity method."""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def generate_image(coordinates, args):
    """Create a black image and insert coordinates as white pixels."""
    image = Image.new('L', (args.image_width, args.image_height), 0)

    for c in coordinates:
        image.putpixel(c, 255)  # Set pixel to white

    return image


def generate_noise(args, num_points=100):
    """Generate random white points in the image."""
    coordinates = np.random.randint(
        low=0,
        high=args.image_width-1,
        size=(num_points, 2)
    ).astype(np.uint8)
    return coordinates.tolist()


def point_in_image(x, y, args):
    """Check if a point (x, y) is within the image boundaries."""
    return (0 <= x < args.image_width) and (0 <= y < args.image_height)


def generate_line(args, noise_lvl=5, slope=1, intercept=0,
                  num_points=100, equaldist=False):
    """Generate coordinates for a line with noise."""

    def _get_point_coord_in_img(args, noise_lvl, slope, num_points, equaldist):
        theta = np.arctan(slope)  # Calculate angle of line

        # Calculate orthogonal direction components
        dx = np.sin(theta)  # Component along x that's orthogonal to line
        dy = -np.cos(theta)  # Component along y that's orthogonal to line

        x = np.random.randint(low=0, high=args.image_width-1, size=num_points)
        if noise_lvl == 0:
            assert num_points <= args.image_width, \
                "Number of points exceeds image width for zero noise level."
            x = np.arange(0, args.image_width)
            x = np.random.permutation(x)[:num_points]

        # Generate orthogonal noise components
        if equaldist:
            noise_mag = np.random.uniform(low=0, high=noise_lvl, size=num_points)
        else:
            noise_mag = np.random.normal(loc=0, scale=noise_lvl, size=num_points)

        noise_x, noise_y = noise_mag * dx, noise_mag * dy
        noise_x, noise_y = _round_np_to_int(noise_x), _round_np_to_int(noise_y)

        new_x = x + noise_x
        new_y = _round_np_to_int(slope * x + intercept) + noise_y

        point_coordinates = np.stack((new_x, new_y), axis=1)

        # check for duplicate points
        point_coordinates = np.unique(point_coordinates, axis=0)

        # check if all points are in the image
        in_img = np.apply_along_axis(
            lambda point: point_in_image(point[0], point[1], args),
            1, point_coordinates
        )
        point_coordinates = point_coordinates[in_img]
        return point_coordinates

    point_coordinates = []
    while len(point_coordinates) < num_points:
        pts = _get_point_coord_in_img(args, noise_lvl, slope,
                                      num_points-len(point_coordinates),
                                      equaldist).tolist()

        point_coordinates.extend(pts)

    return point_coordinates


def generate_line_angle(args, noise_lvl=5, angle=1, offset=0, num_points=100):
    """Generate coordinates for a line based on angle and offset."""
    point_coordinates = []

    while len(point_coordinates) < num_points:
        x = random.randint(0, args.image_width - 1)
        noise = random.randint(-noise_lvl, noise_lvl)

        y = int(x * np.tan(np.deg2rad(angle)) + offset + noise)

        if args.image_height > y > 0:
            point_coordinates.append((x, y))

    return point_coordinates


def generate_vertical_line(args, noise_lvl=5, x_position=0):
    """Generate coordinates for a vertical line with noise."""
    coordinates = []
    for y in range(0, args.image_height, args.image_height // 100):
        noisy_x = x_position + random.randint(-noise_lvl, noise_lvl)
        coordinates.append((noisy_x, y))
    return coordinates


def generate_line_bend(args, noise=5, slope=1.2, slope2=0.5,
                       intercept=0, num_points=100):
    """Generate coordinates for a bent line with noise."""
    point_coordinates = []

    # straight line parameters
    for _ in range(num_points // 2):
        x = random.randint(0, args.image_width // 2 - 1)
        noise_error = random.randint(-noise, noise)
        y = int(slope * x + intercept) + noise_error

        # Ensure y is within image boundaries
        y = min(args.image_height - 1, max(0, y))

        point_coordinates.append((x, y))

    # bent line parameters
    for _ in range(num_points // 2):
        x = random.randint(0, args.image_width//2 - 1)
        noise_error = random.randint(-noise, noise)

        y = int(slope * args.image_width//2 +
                (x*slope2 + intercept) + noise_error)

        # Ensure y is within image boundaries
        y = min(args.image_height - 1, max(0, y))

        point_coordinates.append((int(x + args.image_width / 2), y))

    return point_coordinates


def generate_circle(args, noise=5, num_points=100):
    """Generate coordinates for a noisy circle."""

    # define circle parameters for a circle centered in the image
    radius = min(args.image_width, args.image_height) // 2 - 45
    center_x, center_y = args.image_width // 2, args.image_height // 2

    # generate points on the circle with some added noise
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    point_coordinates = np.column_stack(
        [center_x + radius * np.cos(angles) + random.randint(-noise, noise),
         center_y + radius * np.sin(angles) + random.randint(-noise, noise)]
    ).astype(np.int32)

    return point_coordinates


def generate_arc(args, curvature=0, noise=5, position=50, num_points=100):
    """Generate coordinates for a noisy arc."""
    point_coordinates = []

    # Line parameters - adjust the starting point
    start_x, start_y = 10, args.image_height // 2 + position

    # Adjust the ending point
    end_x, end_y = args.image_width - 10, args.image_height // 2 + position

    # Generate points on the line
    x_values = np.linspace(start_x, end_x, num_points)
    y_values = np.linspace(start_y, end_y, num_points)

    # Add curvature
    y_values += curvature * (x_values - start_x) * (x_values - end_x)

    # Draw points on the image
    point_coordinates = [
        (int(x) + random.randint(-noise, noise),
         int(y) + random.randint(-noise, noise))
        for x, y in zip(x_values, y_values)
    ]

    return list(set(point_coordinates))


def generate_hough_line(rho, theta, args, noise_lvl=0,
                        num_points=100, origin_shift=True):
    """Generate coordinates for a line based on Hough parameters."""
    point_coordinates = []

    # Cosine and Sine of theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Check if sin(theta) is nearly zero, indicating a vertical line
    if np.isclose(sin_theta, 0, atol=1e-9):

        # x-coordinate for a vertical line
        x = 127 if origin_shift else int(rho)

        for _ in range(num_points):
            y = random.randint(0, args.image_height - 1)
            noisy_x = x + random.randint(-noise_lvl, noise_lvl)
            if 0 <= y < args.image_height:
                point_coordinates.append((noisy_x, y))
    else:
        # Normal line processing
        if origin_shift:
            x0 = 127
            y0 = 128
        else:
            x0 = rho * cos_theta
            y0 = rho * sin_theta

        t_max = max(args.image_width, args.image_height)
        t_min = -t_max

        for _ in range(num_points):
            t = random.uniform(t_min, t_max)
            x = int(x0 + t * -sin_theta)
            y = int(y0 + t * cos_theta)

            # Add orthogonal noise
            noise_mag = random.randint(-noise_lvl, noise_lvl)
            noisy_x = x + int(noise_mag * (-sin_theta))
            noisy_y = y + int(noise_mag * (-cos_theta))

            # Ensure points are within boundaries
            if point_in_image(noisy_x, noisy_y, args):
                point_coordinates.append((noisy_x, noisy_y))

    return point_coordinates
