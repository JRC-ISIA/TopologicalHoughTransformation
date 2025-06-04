import cv2
import logging
import numpy as np
from matplotlib import pyplot as plt

from common.colors import baseline_color_str
from common.colors import cmap_s, pth_color_str

from matplotlib.lines import Line2D


def display_image_with_wireframe_and_pointlines(data):
    # Function to display the image with wireframe and pointlines overlay
    
    # Extract necessary data
    image = data['img']
    points = np.array(data['points'])
    lines = np.array(data['lines'])
    pointlines_index = data['pointlines_index']

    # Convert the image to a format suitable for display if necessary
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    # Draw each line on the image
    for line in lines:
        pt1 = tuple(points[line[0]].astype(int))
        pt2 = tuple(points[line[1]].astype(int))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    # Draw pointlines for each point
    for pl_index in pointlines_index:
        for line_index in pl_index:
            line = lines[line_index]
            pt1 = tuple(points[line[0]].astype(int))
            pt2 = tuple(points[line[1]].astype(int))
            # Different color for pointlines
            cv2.line(image, pt1, pt2, (255, 0, 0), 1)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Hide axes ticks
    plt.title('Labeled Image Ground Truth')
    plt.show()

    return image


def draw_dashed_line(image, pt1, pt2, color=(0, 0, 255), thickness=2, dash_length=20):
    """
    Draws a dashed line between pt1 and pt2 on the given image.

    Parameters:
    - image: Image to draw on.
    - pt1, pt2: Start and end points of the line.
    - color: Color of the dashed line (default: red).
    - thickness: Thickness of the dashes.
    - dash_length: Length of each dash.
    """
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    vec = (np.array(pt2) - np.array(pt1)) / dist  # Direction vector
    num_dashes = int(dist // (2 * dash_length))  # Number of dashes

    for i in range(num_dashes):
        start = (pt1 + vec * (2 * i * dash_length)).astype(int)
        end = (pt1 + vec * ((2 * i + 1) * dash_length)).astype(int)
        cv2.line(image, tuple(start), tuple(end), color, thickness)


def draw_lines_on_image(tht):
    # Make a copy of the image to avoid modifying the original
    img_with_lines = 255 - tht.img.copy()

    # Check if the image is grayscale and convert it to BGR to draw colored lines
    if len(img_with_lines.shape) == 2 or img_with_lines.shape[2] == 1:
        img_with_lines = cv2.cvtColor(img_with_lines, cv2.COLOR_GRAY2BGR)

    for ln in tht.line_coordinates:
        x1, y1, x2, y2 = int(ln[0][0]), int(ln[1][0]), int(ln[0][-1]), int(
            ln[1][-1])
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (100, 143, 255), 2)
    return img_with_lines


# Hough-Diagramm mit Loci plotten
def plot_hough_with_loci(tht, show=all, true_lines=None, other_lines=None,
                         my_ax=None, legend = True):
    corr_true_lines=[]
    corr_other_lines=[]
    # Plote Tranformiertes bild
    # Flip the image horizontally
    # im_flipped = np.flip(im, axis=0)
    found_lines = []
    if my_ax == None:
        if tht.three_periods:
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)
    else:
        ax=my_ax
    if my_ax == None or not tht.three_periods:
        ax.set_title("Hough Space $S$")

        ax.imshow(tht.hough_image, aspect='auto', cmap=cmap_s)

        ax.set_xlabel(r"$\theta$ in °")
        ax.set_ylabel(r"$\rho$ in pixel")

        #plt.contourf(xx, yy, self.hough_image, cmap='Purples')
        if tht.three_periods:
            plt.axvline(tht.hough_image.shape[1] / 3, color='g', linestyle='-')  # Interessante Periode markieren
            plt.axvline(tht.hough_image.shape[1] / 3 * 2, color='g', linestyle='-')  # Interessante Periode markieren
        for i, homclass in enumerate(tht.g0):
            p_birth, bl, pers, p_death = homclass
            if pers <= tht.pers_limit:
                continue
            y, x = p_birth
            logging.debug(f"Line found with PH: {p_birth} ")
            #ax.plot([x], [y], '.', c=pth_color, label='Our Method')
            #ax.plot([x], [y], '.', c='r',label='PH + hough')
            #ax.text(x, y + 0.25, str(i + 1), color='r')
        if other_lines is not None:
            for i in range(0, len(other_lines)):
                y = (other_lines[i][0][0] * (-1) + tht.hough_image.shape[0] / 2)  # wieso -1?
                x = np.rad2deg(other_lines[i][0][1]) - 90
                corr_other_lines.append([y,x])
                if tht.three_periods:
                    x+=180
                ax.plot(x, y, 'v', c='yellow', alpha=0.3)
                #ax.plot(x, y, '.', c='yellow', label='Baseline hough')
                logging.debug(f"opencv base lines transformed {x, y}")
        if true_lines is not None:
            for line in true_lines:
                y = (line[0] * (-1) + tht.hough_image.shape[0] / 2)  # wieso -1?
                x = (line[1] - 90)
                corr_true_lines.append([y,x])
                if tht.three_periods:
                    x+=180
                ax.plot(x, y, 'x', c='fuchsia')
                #ax.plot(x, y, '.', c='lime', label='Reference Lines')
                logging.debug(f"True lines transformed {x, y}")
        plt.gca().invert_yaxis()
        #if legend:
        #    legend_handles = [
        #        Line2D([0], [0], marker='x', color='w', markerfacecolor='fuchsia', markersize=10, linestyle='None', label='Reference Lines'),
        #        Line2D([0], [0], marker='.', color='w', markerfacecolor='r', markersize=10, linestyle='None', label='Our Method'),
        #        Line2D([0], [0], marker='v', color='w', markerfacecolor='yellow', markersize=10, linestyle='None', label='Baseline Method')
        #    ]
        #    ax.legend(handles=legend_handles)
        change_axes(ax, tht.hough_image)
        if show == 'one' or show == 'all':
            plt.show()

    # wenn mehrere Perioden: gefundene Punkte in den Ausschnitt bringen und nochmals Plotten
    if tht.three_periods:
        corr_other_lines=[]
        corr_true_lines=[]
        im = tht.hough_image

        if show != 'three':
            im = im[:, im.shape[1] // 3:im.shape[1] * 2 // 3]

        if my_ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = my_ax

        ax.set_title("Hough Space $S$")


        ax.set_xlabel(r"$\theta$ in °")
        ax.set_ylabel(r"$\rho$ in pixel")

        ax.imshow(im, aspect='auto', cmap=cmap_s)

        if true_lines is not None:
            for line in true_lines:
                y = (line[0] * (-1) + tht.hough_image.shape[0] / 2)  # wieso -1?
                x = (line[1] - 90)
                corr_true_lines.append([y,x])
                #if self.three_periods:
                #    x += 180
                ax.plot(x, y, 'x', c='black')
                # ax.plot(x, y, '.', c='lime', label='Reference Lines')
                logging.debug(f"True lines transformed {x, y}")

        if other_lines is not None:
            for i in range(0, len(other_lines)):
                y = (other_lines[i][0][0] * (-1) + tht.hough_image.shape[0] / 2)  # wieso -1?
                x = np.rad2deg(other_lines[i][0][1]) - 90
                corr_other_lines.append([y,x])
                #if self.three_periods:
                #    x += 180
                ax.plot(x, y, 'v', c=baseline_color_str, alpha=0.3)
                logging.debug(f"opencv base lines transformed {x, y}")

        corrected_found_lines = []
        for i, p_birth in enumerate(tht.lines):
            y, x = p_birth
            logging.debug(f'Birth Locus: x={x}, y={y}')
            #x = x - 180
            if 0 <= x <= 180:
                # In Originalperiode gefunden
                #Draw 1st point
                #TODO: FIX: Punkt wird auch im Hough Bild mit 1 periode gezeichnet
                y = -y+ im.shape[0] if show == 'three' else y #quickfix
                ax.plot([x], [y], '.', c=pth_color_str)
                #ax.text(x, y + 0.25, str(i + 1), color='r')
                #logging.debug(f'Corrected Birth Locus: x={x}, y={y}')
                corrected_found_lines.append((y, x))
                #Quickfix JF für Präsentation
                if show == 'three':
                    #Draw 2nd point
                    x += 180
                    y = -y + im.shape[0]
                    ax.plot([x], [y], '.', c='r')
                    ax.text(x, y + 0.25, str(i + 1), color='r')
                    # Draw the 3rd point:
                    x += 180
                    y = -y + im.shape[0]
                    ax.plot([x], [y], '.', c='r')
                    ax.text(x, y + 0.25, str(i + 1), color='r')

            elif -90 <= x < 0:
                if show != 'three':
                    y = -y + im.shape[0]
                    #y = + im.shape[0]
                    x += 180
                    #if i == 0:
                    ax.plot([x], [y], '.', c='b')
                    #ax.text(x, y + 0.25, str(i), color='y')
                    logging.debug(f'Corrected Birth Locus: x={x}, y={y}')
                corrected_found_lines.append((y, x))
            elif 180 < x <= 270:
                if show != 'three':
                    y = - y + im.shape[0]
                    x -= 180
                    #if i == 0:
                    ax.plot([x], [y], '.', c='b')
                    #ax.text(x, y + 0.25, str(i), color='b')
                    logging.debug(f'Corrected Birth Locus: x={x}, y={y}')
                corrected_found_lines.append((y, x))

        found_lines = corrected_found_lines
        plt.gca().invert_yaxis()
        if legend:
            legend_handles = [
                Line2D([0], [0], marker='x', color='black', markerfacecolor='black', markersize=10, linestyle='None', label='Ground Truth'),
                Line2D([0], [0], marker='.', color='w', markerfacecolor=pth_color_str, markersize=10, linestyle='None', label='Our Method'),
                Line2D([0], [0], marker='v', color='w', markerfacecolor=baseline_color_str, markersize=10, linestyle='None', label='Baseline Method')
            ]
            ax.legend(handles=legend_handles)
        change_axes(ax, im)
    tht.lines = found_lines
    if show == 'sec' or show == 'all':
        plt.show()
    return corr_true_lines, corr_other_lines


def change_axes(ax, im):
    """Change the axes of the plot to fit the Hough image."""

    # Adjusting the y-axis to invert and show labels from -max/2 to max/2
    ax.set_ylim(im.shape[0], 0)
    ax.set_yticks(np.linspace(0, im.shape[0], 9))
    ax.set_yticklabels([int(y - im.shape[0] / 2) for y in ax.get_yticks()])

    # Adjusting the x-axis to show labels from -max/2 to max/2
    ax.set_xlim(0, im.shape[1])
    ax.set_xticks(np.linspace(0, im.shape[1], 5))
    ax.set_xticklabels([int(x - im.shape[1] / 2) for x in ax.get_xticks()])