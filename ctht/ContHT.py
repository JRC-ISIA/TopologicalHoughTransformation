import logging
from typing import DefaultDict

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np

from ctht.BaseHT import BaseHT
from ctht.Quad import Quad
from ctht.utils.kernel_quad_intersection import predicate_function
from ctht.utils.quadtree import quad2points
from ctht.utils.convert import rho_theta_to_slope_intercept
from experiments_eurocg26.utils.data import DataGenerator


class ContHT(BaseHT):
    def __init__(self, img_width, lambda_value=5, eps=5, min_persistence=10.,
                 kernel='Hat', predicate_fct=None, debug=False):
        super().__init__(img_width)

        self.lambda_value = lambda_value
        self.kernel = kernel
        self.eps = eps
        self.predicate_fct = predicate_fct
        self.debug = debug

        self.max_depth = 10
        self.depth = 2
        self.min_persistence = min_persistence

        self.diag = np.hypot(img_width, img_width)

        self.r0 = self.diag
        if self.kernel.lower() == 'hat':
            # width 'distance' lambda^2 zero anyway
            self.r0 += self.lambda_value
        elif self.kernel.lower() == 'gauss':
            # for Gaussian, consider significant up to eps, so when is K(x) <= eps
            add_factor = 0
            if self.eps <= 1:
                add_factor = self.lambda_value * np.sqrt(-np.log(self.eps)) * np.sqrt(2)
            self.r0 += add_factor
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        quad = Quad(depth=0, bounding_box=[0, np.pi, -self.r0, self.r0])
        quads = quad.subdivide()
        self.quads = [sub for q in quads for sub in q.subdivide()]

        self.intermediate_quads = []
        self._cached_f = DefaultDict()

    def fit(self, pts):
        self.points = pts

        self._iterative_quads()
        self._build_dual_graph()
        lines = self._graph_filtration()

        def bb_to_param(bbs):
            lines = []
            for bb in bbs:
                quad_idx = bb['birth_simplex']
                quad = self.quads[quad_idx]
                lines.append(quad.get_center())
            return lines

        # TODO: lines to detected_line_params
        self.detected_line_params = list(set(bb_to_param(lines)))
        self.detected_line_params =[(r, t) for t, r in self.detected_line_params]
        self.double_check_lines()

    def double_check_lines(self):
        for idx, lines in enumerate(self.detected_line_params):
            rho, theta = lines
            if theta < 0:
                self.detected_line_params[idx] = ( -rho, theta + np.pi)
            if theta > np.pi:
                self.detected_line_params[idx] = (-rho, theta - np.pi)

    def _iterative_quads(self):
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

        Returns
        -------
        all_quads_by_iteration : list of list of list of float
            A list of quad lists per iteration, including both active and final quads.

        Notes
        -----
        - The process terminates early if there are no active quads left.
        - At each iteration, both newly subdivided and finalized quads are tracked.
        """
        active_quads = self.quads
        final_quads = []

        for depth in range(self.depth, self.max_depth):
            logging.info('Iteration %d', depth)
            active_quads, quads_finalized = self.subdivide_quads(active_quads)
            final_quads.extend(quads_finalized)
            self.intermediate_quads.append(active_quads + final_quads)
            self.quads = active_quads + final_quads
            self.depth = depth

            if not active_quads:
                logging.info("No active quads left, stopping subdivision.")
                break

            # We don't care about pixels at the moment
            #if self.diag <= 2**(depth+2):  # starting of with two large quads
            #    logging.info(f"Minimum quad size reached at {depth}, "
            #                 f"stopping subdivision.")
            #    break


    def subdivide_quads(self, quads):
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
            # check if splitting condition is met
            if self.predicate_fct:
                logging.debug("Using custom predicate function for splitting quads.")
                split_condition = self.predicate_fct(quad, self)
            else:
                logging.debug("Using default predicate function for splitting quads.")
                split_condition = self.predicate(quad)

            if split_condition:
                # intersect: split
                split_quads = quad.subdivide()
                #split_quads = split_quad_centric(quad)
                next_quads.extend(split_quads)
            else:
                # if not intersection, quad is final
                final_quads.append(quad)  # Empty quad is final

        return next_quads, final_quads

    def _get_function_value_at(self, theta, rho):
        assert theta.shape[0] == rho.shape[0]
        fct_values = np.zeros(theta.shape[0])
        for p in self.points:
            # point, theta, rho
            fct_values += self._transform_point(p, theta, rho)
        return fct_values

    def _transform_point(self, pt, desired_theta_rad, desired_rho):
        x, y = pt
        rho = x * np.cos(desired_theta_rad) + y * np.sin(desired_theta_rad)
        delta = np.abs(desired_rho - rho)
        val = 0.0

        if self.kernel.lower() == 'hat':
            # quadratic 'hat' consistent with `fit`
            z_ = 1 - (delta / self.lambda_value)
            val = np.maximum(z_, 0.0)

        elif self.kernel.lower() == 'gauss':
            z_ = - (delta**2) / (2*(self.lambda_value**2))
            val = np.exp(z_)

        return val / len(self.points)

    def _build_dual_graph(self):
        logging.info("Building dual graph of quadtree...")

        def filtration(x):
            return -x

        self.simplex_tree = gd.SimplexTree()
        num_graph_nodes = len(self.quads)

        param = np.array([self.quads[idx].get_center() for idx in range(num_graph_nodes)]) # [:,0] theta, [:,1] rho
        theta, rho = param[:,0], param[:,1]

        fct_values = self._get_function_value_at(theta, rho)

        for idx in range(num_graph_nodes):
            self.quads[idx].index = idx
            self.quads[idx].center_function_value = fct_values[idx]

        logging.info("Building up edges for %d graph nodes...", num_graph_nodes)
        num_inserted_edges = 0

        # Build edges
        for q in self.quads:
            for neighbor in q.neighbors:

                # to avoid adding two times the same edge
                try:
                    if q.index > neighbor.index:
                        continue
                except ValueError:
                    pass

                num_inserted_edges += 1

                # add an edge between a and b
                edge_weight = min(q.center_function_value,
                                  neighbor.center_function_value)

                self.simplex_tree.insert([q.index, neighbor.index],
                                         filtration=filtration(edge_weight))

        # iterate over the whole graph and set the filtration value for each node
        for q in self.quads:
            self.simplex_tree.assign_filtration(
                [q.index], filtration=filtration(q.center_function_value))

        logging.info("Done building dual graph with %d nodes and %d edges.",
                     num_graph_nodes, num_inserted_edges)


    def _graph_filtration(self):
        """Get lower-star 0D persistence sorted by persistence value.

            Returns:
                list of dict: Each dict has 'pers' and 'birth_simplex' keys.
            """
        logging.info("Starting persistence computation on dual graph...")
        _ = self.simplex_tree.make_filtration_non_decreasing()

        self.simplex_tree.compute_persistence()

        self._diag = self.simplex_tree.persistence(persistence_dim_max=False)
        pairs = self.simplex_tree.persistence_pairs()

        d = [
            {
                'pers': np.abs(self.simplex_tree.filtration(destroyer) - self.simplex_tree.filtration(creator) if self.simplex_tree.filtration(destroyer) != float('inf') else float('inf')),
                'birth_simplex': creator[0],
                'death_simplex': destroyer[0] if destroyer else None,
                'birth_value': interval[0],
                'death_value': interval[1] if interval[1] != float(
                    'inf') else None
            }
            for (dim, interval), (creator, destroyer) in zip(self._diag, pairs)
            if dim == 0
        ]

        d = [
            m for m in d
            if m['pers'] > self.min_persistence
        ]
        asc = sorted(d, key=lambda x: x['pers'], reverse=True)
        self.pers = [d_i['pers'] for d_i in asc]
        #diff_pers_pos = np.diff(self.pers[1:]).argmin()  # min since negative
        #asc = asc[:diff_pers_pos+2]  # keep only most persistent features
        logging.info("Done persistence computation, found %d features.", len(asc))
        return asc


def discretize_cont_ht_at_points(ht, n_theta=800, n_rho=800):
    T, R = np.meshgrid(np.linspace(0, np.pi, n_rho), np.linspace(-ht.r0, ht.r0, n_theta))

    Z = np.zeros((n_theta, n_rho))
    for pt in ht.points:
        Z += ht._transform_point(pt, T, R)
    return R, T, Z


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()]
    )

    img_width = 32
    data_generator = DataGenerator(img_width=img_width)
    data_generator.add_noised_line(num_points=10, sigma=0)
    data_generator.add_noised_line(num_points=8, sigma=0)

    #data_generator.add_noised_line(num_points=4, sigma=0)

    cont_ht = ContHT(img_width=img_width, lambda_value=3, kernel='Hat',
                     predicate_fct=predicate_function, eps=5,
                     min_persistence=0.07, debug=False)
    #cont_ht.max_depth = 4
    cont_ht.fit(pts=data_generator.coordinates)

    logging.info("True     line parameters (rho, theta): %s", data_generator.true_line_params)
    logging.info("Detected line parameters (rho, theta): %s", cont_ht.detected_line_params)

    def check_if_all_detected_param_in_limit(detected, true):
        limit_rho = cont_ht.diag / (2**cont_ht.depth)
        limit_theta = np.pi/(2**cont_ht.depth)

        results = []
        for t_rho, t_theta in true:
            found = False
            for d_rho, d_theta in detected:
                if abs(t_rho - d_rho) <= limit_rho and \
                        abs(t_theta - d_theta) <= limit_theta:
                    found = True
                    break
            results += [found]
        return all(results)

    if check_if_all_detected_param_in_limit(cont_ht.detected_line_params,
                                            data_generator.true_line_params):
        logging.info("All in Limit")
    else:
        logging.warning("Not all in Limit")

    R, T, Z = discretize_cont_ht_at_points(cont_ht, n_theta=400, n_rho=400)

    num_quad_it = len(cont_ht.intermediate_quads)

    fig, axs = plt.subplots(nrows=2, ncols=int((num_quad_it+2)//2),
                            figsize=((num_quad_it+2)//2*4, 7))
    axs = axs.flatten()

    for it, ax in enumerate(axs[:-2]):
        c = ax.pcolormesh(T, R, Z, cmap='viridis', shading='auto')
        ax.set_aspect(np.pi/(2*cont_ht.diag), adjustable='box')
        ax.set_ylabel(r'$r$')
        ax.set_xlabel(r'$\theta$')

        for quad in cont_ht.intermediate_quads[it]:
            rect = plt.Rectangle((quad.t0, quad.r0), quad.t1 - quad.t0, quad.r1 - quad.r0,
                                 edgecolor='red', facecolor='none', lw=1)
            ax.add_patch(rect)
            ax.set_title(f'Intermediate quadtree subdivision\n(iteration {it}, '
                         f'#quads={len(cont_ht.intermediate_quads[it])})')

    c = axs[-2].pcolor(T, R, Z, cmap='viridis', shading='auto')
    axs[-2].set_aspect(np.pi/(2*cont_ht.diag), adjustable='box')
    axs[-2].set_ylabel(r'$r$')
    axs[-2].set_xlabel(r'$\theta$')

    detected_params = np.array(cont_ht.detected_line_params)
    true_params = np.array(data_generator.true_line_params)

    c = axs[-2].pcolor(T, R, Z, cmap='viridis', shading='auto')
    #fig.colorbar(c, ax=axs[-2])
    axs[-2].scatter(detected_params[:,1], detected_params[:,0], marker="x", color='magenta', label='Detected lines')
    axs[-2].scatter(true_params[:,1], true_params[:,0], marker="X", color='red', label='True lines')
    axs[-2].set_title('Final detected lines in Hough space')
    axs[-2].legend(loc='upper right')


    # draw the dual graph
    for idx, quad in enumerate(cont_ht.quads):
        rect = plt.Rectangle((quad.t0, quad.r0), quad.t1 - quad.t0,
                             quad.r1 - quad.r0,
                             edgecolor='red', facecolor='none', lw=1)
        #axs[-2].add_patch(rect)

        center_g = quad.get_center()
        #axs[-2].text(quad.t0, center_g[1], s=f"{quad._center_function_value:.3f}",
        #             color='white', fontsize=2)

    axs[-1].scatter(data_generator.coordinates[:, 0], data_generator.coordinates[:, 1], marker='o', color='black')

    # plot true lines
    for rho, theta in data_generator.true_line_params:
        k, d = rho_theta_to_slope_intercept((rho, theta))
        ppts = np.array([(x, k*x + d) for x in [-img_width//2, img_width//2]])
        axs[-1].plot(ppts[:,0], ppts[:,1], 'r-')

    # plot detected lines
    for rho, theta in cont_ht.detected_line_params:
        k, d = rho_theta_to_slope_intercept((rho, theta))
        ppts = np.array([(x, k*x + d) for x in [-img_width//2, img_width//2]])
        axs[-1].plot(ppts[:,0], ppts[:,1], 'g-')

    axs[-1].set_xlim(-img_width//2, img_width//2)
    axs[-1].set_ylim(-img_width//2, img_width//2)
    axs[-1].set_aspect('equal')
    axs[-1].set_title('Image Space')

    plt.tight_layout()
    plt.savefig('out/cont_ht_quadtree_iterations.png', dpi=300)
    plt.show()
    plt.close('all')
    logging.info(f'Saved figure out/cont_ht_quadtree_iterations.png')
