import numpy as np


def get_conf_matrix(noise, true_lines, lines):

    # eq. 1° in the angle direction
    limit_theta = np.arctan2(2*noise, 256)+2

    # equals 1 px normaldist in origin
    limit_rho = noise+2
    found = 0

    true_lines_loc = true_lines.copy()

    for line in true_lines[:]:
        for baseline in lines[:]:
            if (abs(line[0] - baseline[0]) <= limit_rho and
                    abs(line[1] - baseline[1]) <= limit_theta):
                # found line - already removed?
                if line in true_lines_loc:
                    true_lines_loc.remove(line)
                    lines.remove(baseline)
                    found += 1

    # build confusion matrix
    confusion_matrix = [[found, len(true_lines_loc)],
                        [len(lines), 0]]

    return confusion_matrix


def find_closest_line(true_line, detected_lines):
    """Finds the detected line closest to the true line based on rho and
    theta."""
    min_diff = float('inf')
    closest_line = None

    for line in detected_lines:
        # If the line is a numpy array typically from OpenCV
        if isinstance(line, np.ndarray):
            # Flatten in case it's multi-dimensional
            line = line.flatten()
            rho, theta = line[0], line[1]
        else:
            rho, theta = line

        # Calculate the difference and update the closest line if it's smaller
        diff = abs(true_line[0] - rho) + abs(true_line[1] - theta)
        if diff < min_diff:
            min_diff = diff
            closest_line = (rho, theta)

    return closest_line
