"""
colors.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Some colors used in the project.
License: MIT
"""
import matplotlib.pyplot as plt
import matplotlib.colors

cmap_norm = plt.Normalize(0, 255)
cmap_s = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["white", "blue"])

baseline_color = (220, 38, 127)
baseline_color_str = "#DC267F"
pth_color = (100, 143, 255)
pth_color_str = "#648FFF"
