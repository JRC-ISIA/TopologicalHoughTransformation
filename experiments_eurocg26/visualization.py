import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from ctht.ContHT import ContHT, discretize_cont_ht_at_points
from ctht.utils.kernel_quad_intersection import predicate_function
from experiments_eurocg26.common_exp_settings import CommonExportSettings
from experiments_eurocg26.utils.baseline import HTBaseline
from experiments_eurocg26.utils.data import DataGenerator
from experiments_eurocg26.utils.visualize import opencv_color, gt_color, our_color

# setup logging to info
logging.basicConfig(level=logging.INFO)
DEBUG = True

os.makedirs("out", exist_ok=True)

img_width = 64
rhos = [10, 20, -5]
thetas = [np.pi * 2.3 / 3, np.pi / 5, np.pi / 2]

num_points = np.random.randint(8, 20, (len(rhos))).tolist()
data_generator = DataGenerator(img_width=img_width)
for idx, (rho, theta, n_points) in enumerate(zip(rhos, thetas, num_points)):
    # data_generator.add_fix_line(rho=rho, theta=theta, num_points=n_points)
    data_generator.add_noised_line(num_points=n_points,
                                   sigma=2, line_param=(rho, theta))

cont_ht = ContHT(img_width=img_width,
                 lambda_value=5,
                 predicate_fct=predicate_function,
                 eps=5, min_persistence=0.1)

# coordinates for direct reproducibility for the results of the EuroCG Paper
if DEBUG:
    stored = [[13.41302783, 26.9998197], [8.39039942, 23.4781484],
              [-6.50454502, 6.72436823], [-8.81183837, 6.79084793],
              [-22.08429338, -9.41281172], [-32.84992276, -19.82956109],
              [2.3778954, 16.82573199], [4.83899405, 22.65381504],
              [-23.81903211, -11.87287908], [-11.25374503, 2.95650089],
              [25.55204674, -4.18827914], [2.29000897, 31.48397234],
              [26.78238956, -5.39730721], [2.94737583, 29.85865287],
              [11.53492874, 15.06863079], [19.16050839, 7.99139202],
              [8.74427194, 21.45279774], [4.93042415, 27.09357292],
              [16.57030425, 14.52119635], [13.57778539, 14.44992857],
              [6.02738096, 23.68470979], [26.92630304, -3.18982349],
              [21.96164642, 5.82068903], [31.60960757, -6.09597693],
              [15.84149188, 11.88875871], [28.30012168, -2.19168583],
              [10.36328227, 18.4232287], [14.19105234, 14.8954931],
              [10.01208011, 22.37391429], [-8., -6.47625981],
              [28., -4.43293955],
              [-3., -5.57987055], [-28., -4.49724166], [13., -4.02893521],
              [11., -3.40874792], [1., -5.3027744], [25., -4.18062903],
              [31., -5.82886766], [19., -6.88505379], [21., -4.73316808],
              [7., -5.00947827], [30., -5.33986435], [-30., -5.2083524]]

    data_generator.coordinates = np.array(stored)

cont_ht.fit(data_generator.coordinates)

cht = HTBaseline(data_generator, opencv_threshold=5)


# %%

# python
def reorder_legend(ax, desired_label_order, **legend_kwargs):
    """
    Reorder legend entries on axis `ax` to follow `desired_label_order`.
    `desired_label_order` is a list of label strings in the order you want.
    Additional kwargs are passed to ax.legend.
    """
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {lbl: h for h, lbl in zip(handles, labels)}
    ordered_handles = []
    ordered_labels = []
    for lbl in desired_label_order:
        if lbl in label_to_handle:
            ordered_handles.append(label_to_handle[lbl])
            ordered_labels.append(lbl)
    # include any remaining items (optional)
    for h, l in zip(handles, labels):
        if l not in ordered_labels:
            ordered_handles.append(h)
            ordered_labels.append(l)
    ax.legend(ordered_handles, ordered_labels, **legend_kwargs)


## Persistence diagram
points = cont_ht._diag
birth_values = [pt[1][0] for pt in points]
death_values = [pt[1][1] for pt in points if pt[1][1] != float('inf')]
persistences = [pt[1][1] - pt[1][0] for pt in points]
max_pers = persistences[2]

min_birth = min(birth_values)
min_death = min(death_values)

points_detected = np.array(
    [pt[1] for pt in points if (pt[1][1] - pt[1][0]) >= max_pers])
points_detected[0, 1] = 0.0  # adjust for visualization
points_not_detected = np.array(
    [pt[1] for pt in points if (pt[1][1] - pt[1][0]) < max_pers])
points_detected[:, 0] += -min_birth
points_not_detected[:, 0] += -min_birth
points_detected[:, 1] += -min_birth
points_not_detected[:, 1] += -min_birth

points_detected[0, 1] = np.ceil(
    points_detected[0, 1] * 10) / 10  # adjust for visualization

# if points_not_detected[:,0].min() > points_detected[:,0].max():
#    plt.close('all')
#    continue

# if (points_not_detected[:, 0] <= points_detected[:,0].max()).sum() < 2:
#    plt.close('all')
#    continue


min_pers = (points_not_detected[:, 1] - points_not_detected[:, 0]).max()
opencv_threshold = points_detected[:, 0].max() + 0.005

# if (max_pers - min_pers) < 0.1:
#    plt.close('all')
#    continue


n_points = 1200
R, T, Z = discretize_cont_ht_at_points(
    cont_ht, n_theta=n_points, n_rho=n_points
)

# % viz
fs = 12
n_figs = 4
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
axs = axs.flatten()


def rho_theta_to_line(rho, theta, x_vals):
    y_vals = (rho - x_vals * np.cos(theta)) / np.sin(theta)
    return y_vals


x_vals = np.array([-40, 40])
for idx, (rho, theta) in enumerate(cht.detected_line_params):
    kwargs = {'color': opencv_color, 'alpha': 0.4, 'linewidth': 1.5,
              'markersize': 12}
    if idx == 0:
        kwargs[
            'label'] = f'{len(cht.detected_line_params)} detected Lines (OpenCV)'
    ax = axs[3] if n_figs == 4 else axs[0]
    ax.plot(x_vals, rho_theta_to_line(rho, theta, x_vals), **kwargs)

axs[0].scatter(data_generator.coordinates[:, 0],
               data_generator.coordinates[:, 1],
               color='k', s=12, label='Sampled Points')

if n_figs == 4:
    axs[3].scatter(data_generator.coordinates[:, 0],
                   data_generator.coordinates[:, 1],
                   color='k', s=12, label='Sampled Points')

for idx, (rho, theta) in enumerate(zip(rhos, thetas)):
    kwargs = {'color': gt_color, 'alpha': 0.85, 'linewidth': 2.0,
              'markersize': 12}
    if idx == 0:
        kwargs['label'] = 'True Lines'
    axs[0].plot(x_vals, rho_theta_to_line(rho, theta, x_vals), **kwargs)
    axs[1].plot(theta, rho, '.', **kwargs)
    if n_figs == 4:
        axs[3].plot(x_vals, rho_theta_to_line(rho, theta, x_vals), **kwargs)

for idx, (rho, theta) in enumerate(cont_ht.detected_line_params):
    kwargs = {'color': our_color, 'alpha': 0.9, 'linewidth': 2.0,
              'markersize': 12}
    if idx == 0:
        kwargs[
            'label'] = f'{points_detected.shape[0]} detected Lines (our method)'
    ax = axs[3] if n_figs == 4 else axs[0]
    ax.plot(x_vals, rho_theta_to_line(rho, theta, x_vals), **kwargs)
    kwargs['marker'] = 'x'
    axs[1].plot(theta, rho, '.', **kwargs)

for ax in [axs[0], axs[3]]:
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x', fontsize=fs)
    ax.set_ylabel('y', fontsize=fs)
    ax.legend(loc='lower right')
axs[0].set_title("Primal Space")
axs[3].set_title("Primal Space with detected lines")

c = axs[1].pcolor(T, R, Z, cmap='Greys', alpha=0.5, vmin=0, shading='auto')
axs[1].set_ylabel(r'$r$', fontsize=fs)
axs[1].set_xlabel(r'$\Theta$', fontsize=fs)
axs[1].set_title("Continous Hough Space")
axs[1].set_xticks([0, np.pi / 2, np.pi, ])
axs[1].set_xticklabels(['$0$', '$\\frac{\\pi}{2}$', '$\\pi$'])

axs[1].legend(loc='lower right')
axs[1].set_ylim(-cont_ht.diag, cont_ht.diag)
axs[1].set_xlim(0, np.pi)
axs[1].set_aspect(np.pi / (2 * cont_ht.diag), adjustable='box')

# gd.plot_persistence_diagram(points_not_detected, axes=axs[2])
axs[2].scatter(points_detected[:, 0], points_detected[:, 1], axes=axs[2],
               color=gt_color, alpha=0.9, label='True Lines')
axs[2].scatter(points_not_detected[:, 0], points_not_detected[:, 1],
               axes=axs[2], color='red', alpha=0.4, label='False Lines')

axs[2].set_title("Persistence Diagram")
axs[2].set_ylabel("Death", fontsize=fs)
axs[2].set_xlabel("Birth", fontsize=fs)

axs[2].plot([-0.6, 0.6], [-0.6, 0.6], 'k', alpha=0.5)  # diagonal line
axs[2].fill_between(x=[-0.6, 0.6], y1=[-0.6, 0.6], y2=[-0.5 - 3, 0.5 - 3],
                    color='black', alpha=0.1)

axs[2].plot([-50, 10], [-50 + max_pers, 10 + max_pers], 'g--', alpha=0.5,
            label='Persistence Threshold Area')
axs[2].plot([-50, 10], [-50 + min_pers, 10 + min_pers], 'g--', alpha=0.5)
axs[2].fill_between(x=[-50, 10], y1=[-50 + min_pers, 10 + min_pers],
                    y2=[-50 + max_pers, 10 + max_pers], color='green',
                    alpha=0.05)
axs[2].axvline(opencv_threshold, color=opencv_color, linestyle='--', alpha=0.6,
               label='Hough Transform Threshold')
axs[2].legend(loc='lower right', fontsize=fs - 3)

max_birth = max(points_detected[:, 0].max(), points_not_detected[:, 0].max())
max_dead = max(points_detected[:, 1].max(), points_not_detected[:, 1].max())
lim = max(max_birth, max_dead)
axs[2].set_xlim(-0.005, lim)
axs[2].set_ylim(-0.0, lim * 1.05)

xlims = axs[2].get_xlim()
ylims = axs[2].get_ylim()
axs[2].set_aspect(abs(xlims[1] - xlims[0]) / abs(ylims[1] - ylims[0]))

# points_detected[points_detected[:,1] == points_detected[:,1].max(), 1] = lim

ticks = np.arange(0, lim, 0.1)

# ensure 'lim' is included as the final tick
if ticks.size == 0 or not np.isclose(ticks[-1], lim):
    ticks = np.append(ticks, lim)
axs[2].set_xticks(ticks)

axs[2].set_xticklabels([f"{tick:.1f}" for tick in ticks])
axs[2].set_yticks(ticks)
axs[2].set_yticklabels([f"{tick:.1f}" if np.abs(tick) < points_detected[:,
                                                        1].max() else r'$+\infty$'
                        for tick in ticks])
axs[2].axhline(points_detected[:, 1].max(), color='black', alpha=0.25)

# Example usage (insert after your plotting code and before plt.tight_layout()):
reorder_legend(
    axs[3],
    ['Sampled Points', 'True Lines',
     f'{points_detected.shape[0]} detected Lines (our method)',
     f'{len(cht.detected_line_params)} detected Lines (OpenCV)'],
    loc='lower right'
)

plt.tight_layout()
plt.savefig("out/comparison_methods.png", dpi=CommonExportSettings.dpi)
plt.show()
plt.close('all')
