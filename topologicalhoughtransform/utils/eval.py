import logging
import numpy as np
import seaborn as sns


def get_conf_matrix(noise, true_lines, lines, ax=None, cm_title=None):
    limit_ang=np.arctan2(2*noise, 256)+2     #1 entspricht 1°
    limit_rho=noise+2    #1 entspricht 1 px normalabstand ursprung
    found=0
    i=0
    true_lines_loc=true_lines.copy()
    for line in true_lines[:]:
        for baseline in lines[:]:
            if abs(line[0] - baseline[0]) <= limit_rho and abs(line[1] - baseline[1]) <= limit_ang:
                #found line
                #schon entfernt?
                if line in true_lines_loc:
                    true_lines_loc.remove(line)
                    lines.remove(baseline)
                    found+=1
        i+=1
    logging.debug(f"Found: {found}, Not found: {len(true_lines_loc)}, Wrong: {len(lines)}")
    # Matrix bauen
    confusion_matrix = [[found, len(true_lines_loc)],
                        [len(lines), 0]]
    if not ax == None:
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='magma', ax=ax, annot_kws={"size": 10}, cbar=False,
                    square=True, linewidths=0.5, linecolor='gray', xticklabels=['found','not found'], yticklabels=['existing','not existing'])

        #sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='magma', ax=ax)
        ax.set_xlabel(f'Predicted ({cm_title})')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        ax.set_aspect(0.5)
    return confusion_matrix


def find_closest_line(true_line, detected_lines):
    """Finds the detected line closest to the true line based on rho and theta."""
    min_diff = float('inf')
    closest_line = None

    for line in detected_lines:
        if isinstance(line, np.ndarray):  # If the line is a numpy array typically from OpenCV
            line = line.flatten()  # Flatten in case it's multi-dimensional
            rho, theta = line[0], line[1]
        else:
            rho, theta = line

        # Calculate the difference and update the closest line if it's smaller
        diff = abs(true_line[0] - rho) + abs(true_line[1] - theta)
        if diff < min_diff:
            min_diff = diff
            closest_line = (rho, theta)

    return closest_line
