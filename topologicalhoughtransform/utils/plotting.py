import numpy as np
from matplotlib import pyplot as plt


def plot_persistence_diagram(tht, ax=None, show_limit=False,
                             show_nums=True, three_periods=False):
    """
    Plot the persistence diagram from the Hough transform results.
    Args:
        ax:
        show_limit:
        show_nums:

    Returns:

    """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    point_index_counter = 1

    for i, homclass in enumerate(tht.g0):
        p_birth, bl, pers, p_death = homclass
        if three_periods:
            if p_birth[1] < tht.hough_image.shape[1]/3:
                continue
            if p_birth[1] > tht.hough_image.shape[1]/3*2:
                continue

        if pers <= 5.0:
            continue
        x, y = bl, bl - pers
        ax.plot([x], [y], '.', c='k')

        if show_nums:
            ax.text(y, x + 2, str(point_index_counter), color='b')

        point_index_counter += 1

    # Limit einzeichnen
    # Add line with slope 1 starting at (0, 10)
    if show_limit:
        ax.plot([250, tht.pers_limit],
                [255-tht.pers_limit, 0],
                '--', c='#648FFF')

    # plot diagonal
    ax.plot([0, 255], [0, 255], '--', c='black')

    ax.set_ylabel("Death")
    ax.set_xlabel("Birth")
    ax.set_xlim(260, -5)
    ax.set_ylim(260, -5)

    yticks = np.linspace(0, 250, 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([r'$-\
        infty$' if tick == 0 else str(int(tick)) for tick in yticks])
    ax.set_title('Persistence Diagram')

    return ax
