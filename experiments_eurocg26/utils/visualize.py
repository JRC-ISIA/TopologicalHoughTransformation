import gudhi as gd
import numpy as np
from matplotlib import pyplot as plt

from ctht.utils.convert import values_to_closest_index

opencv_color = "#ff7f0e"
tht_color = 'papayawhip'
our_color = "#2ca02c"
gt_color = "#1f77b4"


def viz_pd(y_, ax):
    cubical_complex = gd.CubicalComplex(top_dimensional_cells=y_)
    cubical_complex.compute_persistence()
    gd.plot_persistence_diagram(cubical_complex.persistence(), axes=ax)


def rho_theta_to_xy(rho, theta, tht):
    x = np.argmin(np.abs(tht.theta_values_rad - theta))
    y = np.argmin(np.abs(tht.rho_values - rho))
    return x, y


def vizualize_experiments(tht, cht, data_generator, pd=False):
    """
    Visualizes the results of the Hough Transform experiment.
    Parameters:
    - tht: TopoGaussHT object containing the Hough Transform results.
    - cht: ClassHoughTransform object containing the Hough Transform results.
    """
    n_cols_ = 4 if pd else 3
    width_ = 12 if pd else 9
    img_dim = data_generator.img_width // 2
    img_indcs = np.arange(-img_dim+1, img_dim)

    fig, ax = plt.subplots(ncols=n_cols_, figsize=(width_, 3))

    # ##############  Data points
    ax[0].plot(data_generator.coordinates[:, 0], data_generator.coordinates[:, 1], 'k.')
    ax[0].plot(img_indcs, np.zeros_like(img_indcs), "k-.", alpha=0.1)
    ax[0].plot(np.zeros_like(img_indcs), img_indcs, "k-.", alpha=0.1)

    # ##############  Ours: TopoGaussHT
    if tht is not None:
        common_kwargs = dict(linestyle='-.', color=our_color)
        for it, (d_rho, d_theta) in enumerate(tht.detected_line_params):
            d_y = (d_rho - img_indcs * np.cos(d_theta)) / np.sin(d_theta)
            common_kwargs['label'] = 'ours' if it == 0 else None
            ax[0].plot(img_indcs, d_y, **common_kwargs)

    # ##############  True: Baseline
    common_kwargs = dict(linestyle='--', color=gt_color)
    for it, (true_rho, true_theta) in enumerate(data_generator.true_line_params):
        true_y = (true_rho - img_indcs * np.cos(true_theta)) / np.sin(true_theta)
        common_kwargs['label'] = 'true' if it == 0 else None
        ax[0].plot(img_indcs, true_y, **common_kwargs)

    # ##############  OpenCV: ClassicalHoughTransform
    if cht is not None:
        common_kwargs = dict(linestyle=':', color=opencv_color)
        for it, (d_rho, d_theta) in enumerate(cht.detected_line_params):
            d_y = (d_rho - img_indcs * np.cos(d_theta)) / np.sin(d_theta)
            common_kwargs['label'] = 'OpenCV' if it == 0 else None
            ax[0].plot(img_indcs, d_y, **common_kwargs)

    ax[0].set_title('Image Space $I$')
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')
    ax[0].set_xlim(-img_dim, img_dim)
    ax[0].set_ylim(-img_dim, img_dim)
    ax[0].legend(loc='upper left')

    if tht is not None:
        ax[1].imshow(tht.hough_image, origin='lower', cmap='viridis')

    def _set_axis_labeling(ax_):
        ax_.set_title('Hough Space $S$')
        ax_.set_xlabel('$\\theta$')
        ax_.set_ylabel('$\\rho$')
        # set the tick positions (in image‐pixel coords) ...

        if tht is not None:
            idxs = np.linspace(0, tht.hough_image.shape[1] - 1, 5).astype(int)
            ax_.set_xticks(idxs)
            ax_.set_yticks(idxs)
            # ... and the labels (in degrees)
            ax_.set_xticklabels([f"{np.rad2deg(tht.theta_values_rad[i]):.1f}°" for i in idxs])
            ax_.set_yticklabels([f"{tht.rho_values[i]:.1f}" for i in idxs])

    def _plot_markers(ax_):
        common_kwargs = dict(marker='x', markersize=5)

        if tht is not None:
            params = np.array(tht.detected_line_params)

            rho_mapped = values_to_closest_index(params[:, 0], tht.rho_values)
            theta_mapped = values_to_closest_index(params[:, 1], tht.theta_values_rad)

            ax_.plot(theta_mapped, rho_mapped, **common_kwargs,
                     label='ours', color=our_color)

        if cht is not None:
            params = np.array(cht.detected_line_params)
            rho_mapped = values_to_closest_index(params[:, 0], tht.rho_values)
            theta_mapped = values_to_closest_index(params[:, 1],
                                                   tht.theta_values_rad)
            ax_.plot(theta_mapped, rho_mapped, **common_kwargs,
                     label='OpenCV', color=opencv_color)

        common_kwargs['marker'] = '.'
        params = np.array(data_generator.true_line_params)
        rho_mapped = values_to_closest_index(params[:, 0], tht.rho_values)
        theta_mapped = values_to_closest_index(params[:, 1],
                                               tht.theta_values_rad)
        ax_.plot(theta_mapped, rho_mapped, **common_kwargs,
                 label='true', color=gt_color)

    _plot_markers(ax[1])
    _set_axis_labeling(ax[1])
    ax[1].legend(loc='upper left')

    if tht is not None:
        ax[2].contour(tht.hough_image, levels=10, cmap='viridis')
        _plot_markers(ax[2])
        _set_axis_labeling(ax[2])
        ax[2].legend(loc='upper left')

    if pd:
        ax[3].imshow(data_generator.get_image())
        r, t = cht.detected_line_params[0]
        pt0 = (0, (r) / np.sin(t))
        pt1 = (data_generator.img_width-1, (r - (data_generator.img_width - 1) * np.cos(t)) / np.sin(t))
        pts = np.array([pt0, pt1])
        ax[3].plot(pts[:,0], pts[:,1], color=opencv_color)
        ax[3].invert_yaxis()
        img_indcs = np.arange(0, data_generator.img_width)
        ax[3].plot(img_indcs, np.ones_like(img_indcs) * img_dim, "k-.", alpha=0.5)
        ax[3].plot(np.ones_like(img_indcs) * img_dim, img_indcs, "k-.", alpha=0.5)

    plt.tight_layout()
    plt.show()
    plt.close('all')
