import os
import tqdm
import tempfile
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from HoughforPoints import Hough
from TestDataGenerator import (generate_image, generate_line)


def store_frame(xx, yy, im, filepath_):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xx, yy, im, rstride=1, cstride=1,
                    cmap=plt.cm.jet, linewidth=0)
    plt.savefig(filepath_)
    plt.title("Watershading")
    plt.show()
    plt.close()


def main():
    filepaths = [] #used to store the plots for gif generation
    subfolder_path = '../output_folder/'

    # Ausgabefolder und Tempfolder erstellen wenn nötig
    os.makedirs(subfolder_path, exist_ok=True)

    coordinates = generate_line(slope=0.15, intercept=100, noise_lvl=15, num_points= 150)
    image = generate_image(coordinates)

    hough_transformer = Hough(image, value_threshold=150, pers_limit=150)
    im_orig = hough_transformer.hough_image
    x_range = np.linspace(0, im_orig.shape[1], im_orig.shape[1])
    y_range = np.linspace(0, im_orig.shape[0], im_orig.shape[0])
    xx, yy = np.meshgrid(x_range, y_range)

    max = im_orig.max()
    with tempfile.TemporaryDirectory() as workingfolder_path:
        for i in tqdm.tqdm(range(int(max),-1,-10),
                           desc="Generating Watershading Animation"):
            im=im_orig.copy()
            for x in range(im.shape[0]):
                for y in range(im.shape[1]):
                    if im[x,y]<i:
                        im[x,y]=i
            im[0][0]=0

            # Create a 3D plot
            filepath = os.path.join(workingfolder_path, f"Watershading{i}.png")
            filepaths.append(filepath)
            store_frame(xx, yy, im, filepath)

            del im

        images = [Image.open(fp) for fp in filepaths]
        images[0].save(os.path.join(subfolder_path, 'Watershading_animation.gif'),
                       save_all=True, append_images=images[1:], optimize=False,
                       duration=700, loop=0)


if __name__ == '__main__':
    main()