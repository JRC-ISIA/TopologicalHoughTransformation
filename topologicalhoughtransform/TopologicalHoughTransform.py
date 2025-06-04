# Hogh Transformation for Lines. Line in original Picture -> Point after Transformation
import logging

import numpy as np

from topologicalhoughtransform.ph.PersistenceHomologie import persistence
from topologicalhoughtransform.utils.math import rho_theta_to_slope_intercept


class TopologicalHoughTransform(object):

    def __init__(self, image, angle_step=1, lines_are_white=True,
                 three_periods=False, value_threshold=5,
                 pers_limit=150, img_color=None, normalize=True):

        self.img = np.array(image)
        if img_color is None:
            self.img_plot = self.img
        else:
            self.img_plot = np.array(img_color)

        self.angle_step = angle_step
        self.lines_are_white = lines_are_white
        self.three_periods = three_periods
        self.value_threshold = value_threshold
        self.pers_limit = pers_limit

        self.lines = []
        self.line_coordinates = []

        self._do_transform(normalize)
        self.g0 = persistence(np.array(self.hough_image),
                              persistence_neighborship_construction=moebius_neighborship_construction)
        self._search_lines()
        self._calc_line_list()

    def get_lines(self):
        """Get the lines found in the image."""
        return self.lines

    def get_image(self):
        """Get the original image."""
        return self.img

    def get_hough_image(self):
        """Get the Hough transformed image."""
        return self.hough_image

    def get_lines_rho_theta(self):
        """Get the extracted lines in rho-theta format."""
        return [
            (rho_index - self.hough_image.shape[0] / 2, np.deg2rad(theta_index - 90))
            for rho_index, theta_index in self.lines
        ]

    def _do_transform(self, normalize = True):
        """Perform the Hough transformation on the image."""
        width, height = self.img.shape
        diag_len = int(np.hypot(width, height))

        # Rho and Theta ranges
        self.thetas = np.deg2rad(np.linspace(-90.0, 90.0, int(180 / self.angle_step)))
        self.rhos = np.linspace(-diag_len, diag_len, len(self.thetas))

        # Cache some resuable values
        cos_t = np.cos(self.thetas)
        sin_t = np.sin(self.thetas)
        num_thetas = len(self.thetas)

        # Hough accumulator array of theta vs rho
        if self.three_periods:
            self.hough_image = np.zeros((2 * diag_len, 3 * num_thetas), dtype=np.uint8)
        else:
            self.hough_image = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
        # (row, col) indexes to edges
        are_edges = self.img > self.value_threshold if self.lines_are_white else self.img < self.value_threshold
        y_idxs, x_idxs = np.nonzero(are_edges)

        # Vote in the hough accumulator
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for t_idx in range(num_thetas):
                # Calculate rho. diag_len is added for a positive index
                rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
                if self.three_periods == True:
                    self.hough_image[-rho, t_idx] += 1
                    self.hough_image[rho, num_thetas + t_idx] += 1
                    self.hough_image[-rho, 2 * num_thetas + t_idx] += 1
                else:
                    self.hough_image[rho, t_idx] += 1

        # Normalize the accumulator array, if required
        if normalize:
            self.hough_image = self.hough_image * (255 / np.max(self.hough_image))


    def get_persistence_array(self):
        persistence_values = []
        for (p_birth, bl, pers, _) in self.g0:
            if self.three_periods:
                if p_birth[1] < self.hough_image.shape[1] / 3:
                    continue
                if p_birth[1] > self.hough_image.shape[1] / 3 * 2:
                    continue

            if pers > 0:
                persistence_values.append((bl - pers, bl))

        return persistence_values

    def _search_lines(self):
        found_lines = []
        for homclass in self.g0:
            p_birth, bl, pers, p_death = homclass
            if pers <= self.pers_limit:
                continue
            y, x = p_birth
            found_lines.append((y, x))

        if self.three_periods:
            corrected_found_lines = []
            for (y, x) in found_lines:
                logging.debug(f'Birth Locus: x={x}, y={y}')
                x = x - 180
                if 0 <= x <= 180:
                    # In Originalperiode gefunden
                    logging.debug(f'Corrected Birth Locus: x={x}, y={y}')
                    corrected_found_lines.append((y, x))
                elif -45 <= x < 0:
                    y = -y + self.hough_image.shape[0]
                    x += 180
                    logging.debug(f'Corrected Birth Locus: x={x}, y={y}')
                elif 180 < x <= 225:
                    y = - y + self.hough_image.shape[0]
                    x -= 180
                    logging.debug(f'Corrected Birth Locus: x={x}, y={y}')
            found_lines = corrected_found_lines
        self.lines = found_lines



    def _calc_line_list(self):
        for i, p_birth in enumerate(self.lines):
            y, x = p_birth
            logging.debug(f'Transformierte Koordinaten: x: {x}, y: {y}')

            # Theta und Rho auf originalskala umrechnen
            x = (90 - x)
            y = (y - self.hough_image.shape[0] / 2)
            logging.debug(f'Korrigierte Koordinaten: x: {x}, y: {y}')

            # Inverstransformation
            k, d = rho_theta_to_slope_intercept(x, y)
            logging.debug(f'Linie: k={k}, d={d}')

            # An gespiegeltes Bild anpassen
            d = d * -1
            k = k * -1

            # Linie Rechnen
            x1 = np.arange(0, self.img.shape[1])
            senkrecht = False
            if k == float('inf') or k == float('-inf'):
                senkrecht = True
            else:
                y1 = (k * x1 + d)

            # Was ausserhalb des Bildes is wegschmeissen
            x1_filtered = []
            y1_filtered = []
            if not senkrecht:
                for xi, yi in zip(x1, y1):
                    if self.img.shape[0] >= yi >= 0 and xi > 0:
                        x1_filtered.append(xi)
                        y1_filtered.append(yi)
            else:
                # Ungültige Linien ausblenden
                for j in range(self.img.shape[0]):
                    x1_filtered.append(d)
                    y1_filtered.append(j)
            line = [x1_filtered, y1_filtered]
            self.line_coordinates.append(line)


def moebius_neighborship_construction(p, w, h, **kwargs):
    logging.debug("Moebius strip construction is used, which may not be "
                  "suitable for all applications.")
    y, x = p
    neigh = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]

    # construct moebius strip - this is the addendum to the 4-way
    neigh = [(-j + h - 1, w - 1) if i < 0 else (j, i) for j, i in neigh]
    neigh = [(-j + h - 1, 0) if i >= w else (j, i) for j, i in neigh]
    return neigh