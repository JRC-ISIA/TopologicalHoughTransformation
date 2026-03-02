"""
transform.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Methods to convert between slope-intercept and polar coordinates
                (rho, theta) for lines, and to convert lines to points.
License: MIT
"""
import numpy as np


def slope_intercept_to_rho_theta(line):
    """
    Convert a line defined by slope (k) and y-intercept (d) to polar
    coordinates (rho and theta).
    :param line: A tuple (k, d) where k is the slope and d is the y-intercept.
    :return: A tuple (rho, theta) where rho is the distance from the origin to
             the line and theta is the angle in degrees.
    """
    k, d = line

    # Avoid division by zero for vertical lines
    if abs(k) == float('inf'):
        # a vertical line corresponds to theta = 0 or 180 degrees and rho is
        # the negative value of d
        theta = 0
        rho = -d
    else:
        # Calculate theta in radians
        if k == 0:
            # Horizontal line
            theta = np.pi / 2
        else:
            theta = np.arctan(-1 / k)

        # Ensure theta is within [0, π]
        while theta < 0:
            theta += np.pi
        rho = d * np.sin(theta)

    return rho, np.rad2deg(theta)


def rho_theta_to_slope_intercept(line):
    """
    Convert a line defined by polar coordinates (rho and theta) to cartesian
    coordinates (slope (k) and y-intercept (d)).
    :param line: A tuple (rho, theta) where rho is the distance from the origin
                 to the line and theta is the angle in degrees.
    :return: A tuple (k, d) where k is the slope and d is the y-intercept.
    """
    rho, theta = line

    theta_rad = np.deg2rad(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    if np.isclose(sin_theta, 0, atol=1e-9):
        # if sin_theta is close to 0, the line is vertical and we handle it
        # separately
        k = float('inf') if cos_theta > 0 else -float('inf')
        d = -np.abs(rho) if k > 0 else np.abs(rho)
    else:
        k = -cos_theta / sin_theta
        d = rho / sin_theta

    return k, d


def line_to_pts(line, shift=10):
    """
    Convert a line defined by rho and theta to two points (x1, y1) and
        (x2, y2), where the line is crossing through.
    :param line: A tuple (rho, theta) where rho is the distance from the origin
                 to the line and theta is the angle in degrees.
    :param shift: Distance from the center of the image to the points on the
                  line. Default is 10 pixels.
    :return: Two points (x1, y1) and (x2, y2) that define landmarks of the
             line.
    """
    rho, theta = line
    a, b = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))

    x0, y0 = a * rho, b * rho

    def get_point(shift_):
        return round(x0 + shift_ * (-b)), round(y0 + shift_ * a)

    pt1 = get_point(shift)
    pt2 = get_point(-shift)
    return pt1, pt2
