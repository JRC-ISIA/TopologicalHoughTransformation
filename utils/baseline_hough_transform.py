import logging
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common.colors import pth_color
from topologicalhoughtransform.utils.math import line_to_pts
from utils.io import load_wireframe_data
from utils.plotting import display_image_with_wireframe_and_pointlines


def baseline_detect_lines(original_image, edges, threshold = 150, save_output=False):
    """
    Detects lines in an image using the Hough Transform and optionally saves the output.

    Parameters:
    - original_image: The original image (before edge detection).
    - edges: The edge-detected image.
    - theta: Angular resolution in radians of the Hough grid.
    - threshold: Minimum number of votes (intersections in a Hough grid cell).
    - save_output: Boolean indicating whether to save the output image as a JPG file.
    Returns:
    - lines_found: The image with detected lines overlaid.
    - lines: The lines found by the Hough Transform as a numpy array containing coordinates
    """
    # Code adapted from: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
    # Create an empty image to draw lines on
    accumulator_array = np.zeros_like(original_image)

    # Detect lines using the Hough Transform
    lines_p = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    if lines_p is None:
        logging.debug("No lines were detected.")
        lines_p = []

    for line in lines_p:
        pt1, pt2 = line_to_pts(line[0])
        cv2.line(accumulator_array, pt1, pt2, pth_color, 2)

    # Combine the original image with the line image
    lines_found = cv2.addWeighted(original_image, 0.8,
                                  accumulator_array, 1, 0)

    if save_output:
        cv2.imwrite('FoundLines.jpg', lines_found)

    return lines_found, lines_p


if __name__ == '__main__':
    out_dir = 'out/baseline-hough/'
    os.makedirs(out_dir, exist_ok=True)

    #Display Labeled Image.
    file_path = 'TestImages/00030043.pkl'
    data = load_wireframe_data(file_path)
    ground_truth = display_image_with_wireframe_and_pointlines(data)
    cv2.imwrite(os.path.join(out_dir, '01_GroundTruth.jpg'), ground_truth)

    # Use Edge Detection on the Image.
    img_original = cv2.imread('TestImages/00030043.jpg')
    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(img, 10, 150)
    plt.imshow(edges)
    plt.title('Edge Detection')
    plt.savefig(os.path.join(out_dir, '02_EdgeDetection.jpg'))

    #######Baseline:########

    lines_found, lines = baseline_detect_lines(img_original, edges, save_output=True)
    # Display the image with lines
    plt.imshow(cv2.cvtColor(lines_found, cv2.COLOR_BGR2RGB))
    plt.title('Detected Lines (Baseline Model)')
    plt.savefig(os.path.join(out_dir, '03_DetectedLines.jpg'))
