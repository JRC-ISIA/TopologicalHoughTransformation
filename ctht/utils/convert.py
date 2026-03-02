import numpy as np

def rho_theta_to_slope_intercept(line):
    """
    Convert a line defined by polar coordinates (rho and theta) to cartesian
    coordinates (slope (k) and y-intercept (d)).
    :param line: A tuple (rho, theta) where rho is the distance from the origin
                 to the line and theta is the angle in degrees.
    :return: A tuple (k, d) where k is the slope and d is the y-intercept.
    """
    rho, theta_rad = line

    #theta_rad = np.deg2rad(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    try:
        if np.isclose(sin_theta, 0, atol=1e-9):
            # if sin_theta is close to 0, the line is vertical and we handle it
            # separately
            k = float('inf') if cos_theta > 0 else -float('inf')
            d = -np.abs(rho) if k > 0 else np.abs(rho)
        else:
            k = -cos_theta / sin_theta
            d = rho / sin_theta
    except ValueError:
        print(sin_theta)
        raise ValueError

    return k, d


def theta_to_k(theta):

    if np.isclose(np.sin(theta), 0, atol=1e-9):
        # if sin_theta is close to 0, the line is vertical and we handle it
        # separately
        k = float('inf') if np.cos(theta) > 0 else -float('inf')
    else:
        k = -np.cos(theta) / np.sin(theta)

    return k

def slope_intercept_to_rho_theta(line):
    """
    Convert a line defined by slope (k) and y-intercept (d) to polar coordinates
    (rho and theta).
    :param line: A tuple (k, d) where k is the slope and d is the y-intercept.
    :return: A tuple (rho, theta) where rho is the distance from the origin to the line
    and theta is the angle in degrees.
    """
    k, d = line

    # Avoid division by zero for vertical lines
    if abs(k) == float('inf'):
        # a vertical line corresponds to theta = 0 or 180 degrees and rho is the
        # negative value of d
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

    return rho, theta


def _adjust_line_parameter(pt):
    rho, theta = pt
    while theta < 0:
        rho, theta = -rho, theta + np.pi

    while theta >= np.pi:
        rho, theta = -rho, theta - np.pi
    return rho, theta


def adjust_detected_line_params(detected_line_params):
    detected_line_params = [
        _adjust_line_parameter(pt) for pt in detected_line_params
    ]
    return detected_line_params


def rt2xy(point):
    rho, theta = point
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def values_to_closest_index(values, grid):
    values = np.abs(grid - values.reshape(-1, 1))
    return np.argmin(values, axis=1)
