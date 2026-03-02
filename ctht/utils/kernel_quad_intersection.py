import logging
import numpy as np

def kernel_intersects_quad(x, y, w, quad):
    """
    Analytic test whether the kernel support of a single image-space point
    intersects a quad in (theta, rho) Hough space.

    The Hough transform of a point (x, y) is the sinusoid:
        rho(theta) = x cos(theta) + y sin(theta)

    This can be rewritten in amplitude–phase form as:
        rho(theta) = A cos(theta - phi)

    where:
        A   = sqrt(x^2 + y^2)          (distance of the point from the origin)
        phi = atan2(y, x)              (angle of the point in image space)

    Using this form makes it easy to reason about extrema of rho(theta)
    and to test intersection with axis-aligned quads analytically.
    """

    # ------------------------------------------------------------
    # Convert (x, y) to amplitude–phase representation
    # ------------------------------------------------------------
    # A is the maximum absolute value attained by rho(theta)
    # over all theta; geometrically, it is the distance of the
    # point from the origin.
    A = np.hypot(x, y)

    # phi is the phase shift of the sinusoid. It tells us at which
    # theta the Hough curve reaches its maximum:
    #     rho(phi) = A
    phi = np.arctan2(y, x)

    # ------------------------------------------------------------
    # 1. Check intersection with vertical quad boundaries
    #    (theta = t0 and theta = t1)
    # ------------------------------------------------------------
    # At fixed theta, the kernel support is the interval:
    #     [rho(theta) - w, rho(theta) + w]
    for t in (quad.t0, quad.t1):
        r_center = A * np.cos(t - phi)
        if (r_center + w >= quad.r0) and (r_center - w <= quad.r1):
            return True

    # ------------------------------------------------------------
    # 2. Check intersection with horizontal quad boundaries
    # ------------------------------------------------------------
    # Over a theta interval [t0, t1], rho(theta) can only attain
    # its extrema at:
    #   - the interval endpoints t0, t1
    #   - the global maximum at theta = phi
    #   - the global minimum at theta = phi + pi
    candidates = [quad.t0, quad.t1]

    # Maximum of cos(theta - phi)
    if quad.t0 <= phi <= quad.t1:
        candidates.append(phi)

    # Minimum of cos(theta - phi)
    phi2 = phi + np.pi
    if quad.t0 <= phi2 <= quad.t1:
        candidates.append(phi2)

    # Evaluate rho(theta) at all candidate locations
    r_vals = A * np.cos(np.array(candidates) - phi)

    # Expand extrema by kernel support width
    r_min = np.min(r_vals) - w
    r_max = np.max(r_vals) + w

    # Intersection occurs if the thickened curve overlaps
    # the rho-interval of the quad
    return (r_max >= quad.r0) and (r_min <= quad.r1)


def predicate_function(quad, ht, max_kernels=1, gauss_k=2.0):
    """
    Split quad if it intersects more than `max_kernels` kernel supports.

    Args:
        quad (list): [theta_min, theta_max, rho_min, rho_max]
        ht (ContHT): ContHT instance
        max_kernels (int): intersection threshold
        gauss_k (float): Gaussian truncation in sigma units
    """

    # ------------------------------------------
    # Kernel support radius
    # ------------------------------------------
    if ht.kernel.lower() == 'hat':
        w = ht.lambda_value
        lambda_k = 1 / w
    elif ht.kernel.lower() == 'gauss':
        w = gauss_k * ht.lambda_value
        lambda_k = 1 / (ht.lambda_value * np.sqrt(2))
        raise ValueError("Still needs to be checked!")
    else:
        raise ValueError(f"Unknown kernel: {ht.kernel}")

    #+ -> PB
    pb = np.array([p for p in ht.points if kernel_intersects_quad(p[0], p[1], w, quad)])
    if pb.ndim == 1:
        return False
    pb_abs = np.hypot(pb[:,0], pb[:,1])
    D = pb_abs.max()

    lambda_score = lambda_k * np.sqrt(1 + D**2)
    cal_eps = lambda_score * quad.diag/2
    return cal_eps > ht.eps
