import numpy as np
from matplotlib import pyplot as plt


def subdivide_quads(quads, r_values, theta_vals):
    """
    Subdivide a list of quads in (θ, r) space based on intersections
    with Hough transform curves.

    Parameters
    ----------
    quads : list of lists of floats
        Each quad is represented as [θ_min, θ_max, r_min, r_max].
    r_values : ndarray of shape (n_points, n_theta)
        Array containing r(θ) values for each point's Hough curve.
    theta_vals : ndarray of shape (n_theta,)
        Discretized theta values corresponding to each column of r_values.

    Returns
    -------
    next_quads : list of list of float
        Subdivided quads to be processed in the next iteration.
    final_quads : list of list of float
        Quads that are considered final (either empty or single-intersection).

    Notes
    -----
    - A quad is subdivided if it contains intersections with more than one curve.
    - Empty quads (no intersections) are kept as final.
    - Quads containing exactly one intersection are also finalized.
    """
    next_quads, final_quads = [], []

    for quad in quads:
        # check if ANY curve passes through the quad
        intersect = check_intersect_discrete(quad, r_values, theta_vals)

        if intersect:
            # intersect: split
            split_quads = split_quad_centric(quad)
            next_quads.extend(split_quads)
        else:
            # if not intersection, quad is final
            final_quads.append(quad)  # Empty quad is final

    return next_quads, final_quads


def check_intersect_discrete(quad, r_values, theta_vals) -> bool:
    t0, t1, r0, r1 = quad
    for r_curve in r_values:
        # condition if to split or not
        if np.any((theta_vals >= t0) & (theta_vals <= t1) &
                  (r_curve >= r0) & (r_curve <= r1)):
            return True

    return False


def split_quad_centric(quad):
    t0, t1, r0, r1 = quad

    t_mid = (t0 + t1) / 2
    r_mid = (r0 + r1) / 2
    return [
        [t0, t_mid, r0, r_mid],
        [t_mid, t1, r0, r_mid],
        [t0, t_mid, r_mid, r1],
        [t_mid, t1, r_mid, r1]
    ]


def iterative_quads(initial_quads, r_values, theta_vals, max_depth=12):
    """
    Iteratively apply `subdivide_quads` for a fixed number of iterations.

    Parameters
    ----------
    initial_quads : list of list of float
        List of starting quads in (θ, r) space.
    r_values : ndarray of shape (n_points, n_theta)
        Hough transform r-values for all input points.
    theta_vals : ndarray of shape (n_theta,)
        Discretized θ values.
    max_depth : int
        Maximum number of subdivision iterations to perform.

    Returns
    -------
    all_quads_by_iteration : list of list of list of float
        A list of quad lists per iteration, including both active and final quads.

    Notes
    -----
    - The process terminates early if there are no active quads left.
    - At each iteration, both newly subdivided and finalized quads are tracked.
    """
    active_quads = initial_quads
    final_quads_total = []

    for _ in range(max_depth):
        active_quads, final_quads = subdivide_quads(active_quads, r_values, theta_vals)
        final_quads_total.extend(final_quads)
        if not active_quads:
            break

    return active_quads + final_quads_total


def plot_points_and_hough(iterations, points, theta_vals, r_values, filename=None):
    """
    Plot the input points and their Hough transform curves side by side.

    Parameters
    ----------
    iterations : list of list of list of float
        Output from `iterative_quads`, containing quads for each iteration.
    iteration_number : int
        The index of the iteration to visualize.
    filename : str, optional
        If provided, the plot is saved to this filename (SVG format recommended).

    Returns
    -------
    None
        Displays a Matplotlib figure with two subplots.

    Notes
    -----
    - The left subplot shows the original random points in (x, y) space.
    - The right subplot shows their Hough transform curves with current
      quadtree subdivisions as black rectangles.
    """

    # Assign a distinct color to each point/curve
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: original points
    ax_points = axes[0]
    for i, (x, y) in enumerate(points):
        ax_points.scatter(x, y, color=colors[i], s=80)
    ax_points.set_title('Points')
    ax_points.set_xlabel('x')
    ax_points.set_ylabel('y')
    ax_points.set_aspect('equal')
    ax_points.set_xlim(0, 1)
    ax_points.set_ylim(0, 1)
    ax_points.grid(True)

    # Right subplot: Hough transform with quadtree quads
    ax_hough = axes[1]
    for i, r_curve in enumerate(r_values):
        ax_hough.plot(theta_vals, r_curve, color=colors[i])
    for quad in iterations:
        t0, t1, r0, r1 = quad
        rect = plt.Rectangle((t0, r0), t1 - t0, r1 - r0,
                             edgecolor='black', facecolor='none', lw=1)
        ax_hough.add_patch(rect)
    ax_hough.set_title(f'Hough Transform with Quadtree')
    ax_hough.set_xlabel('Theta')
    ax_hough.set_ylabel('r')
    ax_hough.set_aspect('auto')
    ax_hough.grid(False)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    plt.show()


def quad2points(quads):
    """
    Convert quads into a flat list of unique corner points (rho, theta).
    Each quad is (t0, t1, r0, r1). Corners become tuples so they are hashable.
    Preserves the first-seen order.
    """
    points = []
    for (t0, t1, r0, r1) in quads:
        # create hashable corner tuples (rho, theta)
        corners = ((r0, t0), (r0, t1), (r1, t1), (r1, t0))
        points.extend(corners)

    points = list(set(points))

    return points


def softmax_temperature(weights, temperature=1.0):
    """
    Apply softmax with temperature to smooth/blend weights.

    Args:
        weights: array-like of weight values
        temperature: float > 0
            - Lower (< 1): sharpens distribution (winner-takes-more)
            - Higher (> 1): smooths distribution (more uniform)
            - 1.0: standard softmax

    Returns:
        Smoothed weights that sum to 1
    """
    weights = np.array(weights, dtype=np.float64)
    scaled = weights / temperature
    centered = scaled - scaled.max()  # subtract max for numerical stability

    if centered.sum() == 0:  # the case where all weights are equal
        return np.ones_like(weights) / len(weights)

    exp_weights = np.exp(centered)
    return exp_weights / exp_weights.sum()
