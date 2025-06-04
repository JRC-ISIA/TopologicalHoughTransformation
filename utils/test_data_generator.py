import random

import numpy as np
from PIL import Image

# Image Size
width = 256
height = 256
num_points = 100


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


# Erstelle ein schwarzes Bild und füge Koordianten als weiße Pixeln ein
def generate_image(coordinates):
    image = Image.new('L', (width, height), 0)  # 'L' ist Graustufenbild

    if coordinates == None:
        for x in range(width):
            for y in range(height):
                image.putpixel((x, y), 255)
    else:
        for x, y in coordinates:
            image.putpixel((x, y), 255)  # Set pixel to white
    return image


# Erstelle Rauschen
def generate_noise():
    white_points = []
    for _ in range(num_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        white_points.append((x, y))
    return white_points


# Erstelle die Koordianten einer Linie
def generate_line(noise_lvl=5, slope=1, intercept=0, num_points=100,
                  equaldist = False):

    theta = np.arctan(slope)  # Calculate angle of line

    white_points = []
    # Calculate orthogonal direction components
    dx = np.sin(theta)  # Component along x that's orthogonal to line
    dy = -np.cos(theta)  # Component along y that's orthogonal to line

    for _ in range(num_points):
        valid_point = False
        while not valid_point:
            x = random.randint(0, width - 1)

            # Generate orthogonal noise components
            if equaldist == True:
                noise_mag = (np.random.uniform(0, noise_lvl))
            else:
                noise_mag = (np.random.normal(0, noise_lvl))
            noise_x = int(noise_mag * dx)
            noise_y = int(noise_mag * dy)

            # Calculate new x and y considering the noise
            new_x = x + noise_x
            new_y = int(slope * x + intercept) + noise_y

            # Check if the point is within image boundaries
            if 0 <= new_x < width and 0 <= new_y < height:
                if (new_x, new_y) not in white_points:  # Check before appending
                    white_points.append((new_x, new_y))
                    valid_point = True

    return white_points


# Erstelle die Koordinaten einer Line via Winkel und offset
def generate_line_angle(noise_lvl=5, angle=1, offset=0, num_points=100):
    white_points = []
    i = 0
    while i < num_points:
        x = random.randint(0, width - 1)
        noise = random.randint(-noise_lvl, noise_lvl)
        y = int(x * np.tan(np.deg2rad(angle)) + offset + noise)
        if y < height and y > 0:
            i += 1
            white_points.append((x, y))
    return white_points


# Erstelle eine vertikale Linie
def generate_vertical_line(noise_lvl=5, x_position=0):
    coordinates = []
    for y in range(0, height, height // 100):  # Ensure approximately 100 points
        noisy_x = x_position + random.randint(-noise_lvl, noise_lvl)  # Add noise to x
        coordinates.append((noisy_x, y))
    return coordinates


# Erstelle Koordinaten einer geknickten, verrauschten Linie
def generate_line_bend(noise=5, slope=1.2, slope2=0.5, intercept=0):
    white_points = []
    # Gerade
    for _ in range(int(num_points / 2)):
        x = random.randint(0, width / 2 - 1)
        noise_error = random.randint(-1 * noise, noise)
        y = int(slope * x + intercept) + noise_error
        y = min(height - 1, max(0, y))  # Ensure y is within image boundaries
        white_points.append((x, y))
    # Verbogen weiter
    for _ in range(int(num_points / 2)):
        x = random.randint(0, width / 2 - 1)
        noise_error = random.randint(-5, 5)
        y = int(slope * (width / 2) + (x * slope2 + intercept) + noise)
        y = min(height - 1, max(0, y))  # Ensure y is within image boundaries
        white_points.append((int(x + width / 2), y))
    return white_points


# Erstelle Koordinaten eines verrauschten Kreises
def generate_circle(noise=5):
    white_points = []
    # Kreisparameter
    radius = min(width, height) // 2 - 45  # Kreis halbweg groß
    center_x, center_y = width // 2, height // 2

    # Generiere Punkte auf dem Kreis
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    circle_points = np.column_stack([center_x + radius * np.cos(angles),
                                     center_y + radius * np.sin(angles)]).astype(np.int32)

    # Zeichne die Punkte auf das Bild
    for point in circle_points:
        noisex = random.randint(-1 * noise, noise)
        noisey = random.randint(-1 * noise, noise)
        white_points.append((point[0] + noisex, point[1] + noisey))

    return white_points


# Erstelle Koordiaten eines Bogens
def generate_arc(curvature=0, noise=5, position=50):
    white_points = []
    # Line parameters
    start_x, start_y = 10, height // 2 + position  # Adjust the starting point
    end_x, end_y = width - 10, height // 2 + position  # Adjust the ending point

    # Generate points on the line
    x_values = np.linspace(start_x, end_x, num_points)
    y_values = np.linspace(start_y, end_y, num_points)

    # Add curvature
    y_values += curvature * (x_values - start_x) * (x_values - end_x)

    # Draw points on the image
    for x, y in zip(x_values, y_values):
        noisex = random.randint(-1 * noise, noise)
        noisey = random.randint(-1 * noise, noise)
        x_coord = int(x) + noisex
        y_coord = int(y) + noisey

        # Check if the coordinates are within the image boundaries
        if 0 <= x_coord < width and 0 <= y_coord < height:
            white_points.append((x_coord, y_coord))

    return white_points


def generate_hough_line(rho, theta, noise_lvl=0, num_points=100, origin_shift=True):
    white_points = []

    # Cosine and Sine of theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if np.isclose(sin_theta, 0, atol=1e-9):  # Check if sin(theta) is nearly zero, indicating a vertical line
        x = 127 if origin_shift else int(rho)  # x-coordinate for a vertical line
        for _ in range(num_points):
            y = random.randint(0, height - 1)
            noise_mag = random.randint(-noise_lvl, noise_lvl)
            noisy_x = x + noise_mag  # Adding noise to y-coordinate
            if 0 <= y < height:
                white_points.append((noisy_x, y))
    else:
        # Normal line processing
        if origin_shift:
            x0 = 127
            y0 = 128
        else:
            x0 = rho * cos_theta
            y0 = rho * sin_theta
        t_min = -max(width, height)  # Extend range to ensure full coverage
        t_max = max(width, height)

        for _ in range(num_points):
            t = random.uniform(t_min, t_max)
            x = int(x0 + t * -sin_theta)
            y = int(y0 + t * cos_theta)

            # Add orthogonal noise
            noise_mag = random.randint(-noise_lvl, noise_lvl)
            noise_x = int(noise_mag * (-sin_theta))
            noise_y = int(noise_mag * (-cos_theta))

            noisy_x = x + noise_x
            noisy_y = y + noise_y

            # Ensure points are within boundaries
            if 0 <= noisy_x < width and 0 <= noisy_y < height:
                white_points.append((noisy_x, noisy_y))
    return white_points
