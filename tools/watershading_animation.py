"""
Watershading_Animation.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Tool to generate an animation of the watershading process.
License: MIT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image

from topologicalhoughtransform.topological_hough_transform import TopologicalHoughTransform
from utils.data_generator import generate_image, generate_line
from utils.parser import create_parser


def store_frame(xx, yy, im, filepath_):
    """Store a single frame of the watershading animation."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xx, yy, im, rstride=1, cstride=1,
                    cmap=plt.cm.jet, linewidth=0)
    ax.set_zlim(0, 250)
    plt.savefig(filepath_)
    plt.title("Watershading")
    plt.show()
    plt.close()


def main():
    """Main function to generate the watershading animation."""
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.info(args)

    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, 'tmp'), exist_ok=True)

    filepaths = []  # used to store the plots for gif generation

    coordinates = generate_line(args, slope=0.15, intercept=100,
                                noise_lvl=15, num_points=150)
    image = generate_image(coordinates, args)

    hough_transformer = TopologicalHoughTransform(
        image, value_threshold=150, pers_limit=150)

    im_orig = hough_transformer.hough_image
    xx, yy = np.meshgrid(
        np.linspace(0, im_orig.shape[1], im_orig.shape[1]),
        np.linspace(0, im_orig.shape[0], im_orig.shape[0])
    )

    max_im = int(im_orig.max())

    with tempfile.TemporaryDirectory() as tmp_folder:
        for filtration_value in tqdm.tqdm(
                range(max_im, -1, -10),
                desc="Generating Watershading Animation"):
            im = im_orig.copy()
            im[im < filtration_value] = filtration_value

            # Create a 3D plot
            filepath = os.path.join(tmp_folder,
                                    f"Watershading_{filtration_value}.png")
            filepaths.append(filepath)
            store_frame(xx, yy, im, filepath)
            del im

        images = [Image.open(fp) for fp in filepaths]
        images[0].save(
            os.path.join(args.output_directory, 'Watershading_animation.gif'),
            save_all=True, append_images=images[1:], optimize=False,
            duration=700, loop=0
        )


if __name__ == '__main__':
    main()
