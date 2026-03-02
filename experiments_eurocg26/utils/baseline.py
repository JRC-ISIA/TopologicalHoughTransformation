import cv2
import numpy as np

from ctht.utils.convert import adjust_detected_line_params


class HTBaseline(object):
    def __init__(self, data_generator,
                 rho_resolution=None,
                 theta_resolution=None,
                 opencv_threshold=50):

        if rho_resolution is None:
            rho_resolution = 1

        if theta_resolution is None:
            theta_resolution = np.pi / 180

        image = data_generator.get_image()

        lines_p = cv2.HoughLinesWithAccumulator(
            image,
            rho=rho_resolution,
            theta=theta_resolution,
            threshold=opencv_threshold
        )

        self.detected_line_params= []

        if lines_p is not None:
            lines_p = lines_p.reshape(-1, 3)

            rho, theta = lines_p[:, 0], lines_p[:, 1]

            rho = rho - data_generator.img_width // 2 *(np.cos(theta) + np.sin(theta))

            rho[rho == 0.] = 1e-8  # Avoid division by zero

            self.detected_line_params = [(rho_, theta_) for rho_, theta_ in
                                         zip(rho, theta)]

            self.detected_line_params = adjust_detected_line_params(self.detected_line_params)
            self.full_data = lines_p