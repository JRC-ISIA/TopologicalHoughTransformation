import numpy as np


def slope_intercept_to_rho_theta(k, d):
    # Avoid division by zero for vertical lines
    if k == float('inf') or k == -float('inf'):
        theta = 0  # Vertical line corresponds to theta = 0 or 180 degrees
        rho = -d  # Assuming d for vertical lines is calculated as -rho in the original method
    else:
        # Calculate theta in radians
        theta = np.arctan(-1 / k)
        # Ensure theta is within [0, π]
        if theta < 0:
            theta += np.pi
        # Calculate rho
        rho = d * np.sin(theta)
    return rho, np.rad2deg(theta)  # Return theta in degrees


def rho_theta_to_slope_intercept(theta, rho):
    # Steigung und Offset von Linie rechnen
    # Konvertierung des Winkels von Grad in Bogenmaß
    theta_rad = np.deg2rad(theta)

    # Berechnung von sin und cos des Winkels
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Berechnung der Liniensteigung m und des y-Achsenabschnitts b
    if np.abs(sin_theta) < 1e-5:  # Wenn sin_theta nahe 0 ist, ist die Linie vertikal
        k = float('inf') if cos_theta > 0 else -float('inf')
        if k>0:
            d = -np.abs(rho)
        else:
            d=np.abs(rho)
    else:
        k = -cos_theta / sin_theta
        d = rho / sin_theta
    return k, d


def line_to_pts(line):
    """
    Convert a line defined by rho and theta to two points (x1, y1) and (x2, y2).
    :param line: A tuple (rho, theta) where rho is the distance from the origin to the line
                 and theta is the angle in degrees.
    :return: Two points (x1, y1) and (x2, y2) that define the line.
    """
    rho, theta = line
    a, b = np.cos(theta), np.sin(theta)

    x0, y0 = a * rho, b * rho

    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
    return pt1, pt2