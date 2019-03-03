"""
======================================================
Plot a confidence ellipse of a two-dimensional dataset
======================================================

This example shows how to plot a confidence ellipse of a
two-dimensional dataset, using its pearson correlation coefficient.

The approach that is used to obtain the correct geometry is
explained and proved here:

https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
"""

#############################################################################
#
# The plotting function itself
# """"""""""""""""""""""""""""
#
# This function plots the confidence ellipse of the covariance of the given
# array-like variables x and y. The ellipse is plotted into the given
# axes-object ax.
#
# The radiuses of the ellipse can be controlled by n_std which is the number
# of standard deviations. The default value is 3 which makes the ellipse
# enclose 99.7% of the points (given the data is normally distributed
# like in these examples).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=3.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data

    ax : matplotlib.axes object to the ellipse into

    n_std : number of standard deviations to determine the ellipse's radiuses

    Returns
    -------
    None

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


#############################################################################
#
# A helper function to create a correlated dataset
# """"""""""""""""""""""""""""""""""""""""""""""""
#
# Creates a random two-dimesional dataset with the specified
# two-dimensional mean (mu) and dimensions (scale).
# The correlation can be controlled by the param 'dependency',
# a 2x2 matrix.

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


#############################################################################
#
# Positive correlation
# """"""""""""""""""""
#

fig, ax_pos = plt.subplots(figsize=(6, 6))
np.random.seed(1234)

dependency_pos = np.array([
    [0.85, 0.35],
    [0.15, -0.65]
])
mu = np.array([2, 4]).T
scale = np.array([3, 5]).T

# Indicate the x- and y-axis
ax_pos.axvline(c='grey', lw=1)
ax_pos.axhline(c='grey', lw=1)

x, y = get_correlated_dataset(500, dependency_pos, mu, scale)
confidence_ellipse(x, y, ax_pos, facecolor='none', edgecolor='red')

# Also plot the dataset itself, for reference
ax_pos.scatter(x, y, s=0.5)
# Mark the mean ("mu")
ax_pos.scatter([mu[0]], [mu[1]], c='red', s=3)
ax_pos.set_title(f'Positive correlation')
plt.show()


#############################################################################
#
# Negative correlation
# """"""""""""""""""""
#

fig, ax_neg = plt.subplots(figsize=(6, 6))
dependency_neg = np.array([
    [0.9, -0.4],
    [0.1, -0.6]
])
mu = np.array([2, 4]).T
scale = np.array([3, 5]).T

ax_neg.axvline(c='grey', lw=1)
ax_neg.axhline(c='grey', lw=1)

x, y = get_correlated_dataset(500, dependency_neg, mu, scale)
confidence_ellipse(x, y, ax_neg, facecolor='none', edgecolor='red')

ax_neg.scatter(x, y, s=0.5)

ax_neg.scatter([mu[0]], [mu[1]], c='red', s=3)
ax_neg.set_title(f'Negative correlation')
plt.show()


#############################################################################
#
# Weak correlation
# """"""""""""""""
#
# This is still an ellipse, not a circle because x and y
# are differently scaled. However, the fact that x and y are uncorrelated
# is shown by the axes of the ellipse being aligned with the x- and y-axis
# of the coordinate system.

fig, ax_uncorrel = plt.subplots(figsize=(6, 6))

in_dependency = np.array([
    [1, 0],
    [0, 1]
])
mu = np.array([2, 4]).T
scale = np.array([5, 3]).T

ax_uncorrel.axvline(c='grey', lw=1)
ax_uncorrel.axhline(c='grey', lw=1)

x, y = get_correlated_dataset(500, in_dependency, mu, scale)
confidence_ellipse(x, y, ax_uncorrel, facecolor='none', edgecolor='red')
ax_uncorrel.scatter(x, y, s=0.5)
ax_uncorrel.scatter([mu[0]], [mu[1]], c='red', s=3)
ax_uncorrel.set_title(f'Weak correlation')


#############################################################################
#
# Different number of standard deviations
# """""""""""""""""""""""""""""""""""""""
#
# A plot with n_std = 3 (gray), 2 (blue) and 1 (red)

fig, ax_nstd = plt.subplots(figsize=(6, 6))

dependency_nstd = np.array([
    [0.8, 0.75],
    [-0.2, 0.35]
])
mu = np.array([0, 0]).T
scale = np.array([8, 5]).T

ax_nstd.axvline(c='grey', lw=1)
ax_nstd.axhline(c='grey', lw=1)

x, y = get_correlated_dataset(500, dependency_nstd, mu, scale)
ax_nstd.scatter(x, y, s=0.5)

confidence_ellipse(x, y, ax_nstd, n_std=1,
    facecolor='none', edgecolor='red')
confidence_ellipse(x, y, ax_nstd, n_std=2,
    facecolor='none', edgecolor='blue')
confidence_ellipse(x, y, ax_nstd, n_std=3,
    facecolor='none', edgecolor='gray', linestyle='--')

ax_nstd.scatter([mu[0]], [mu[1]], c='red', s=3)
ax_nstd.set_title(f'Different standard deviations')
plt.show()


#############################################################################
#
# Using the keyword arguments
# """""""""""""""""""""""""""
#
# Use the kwargs specified for matplotlib.patches.Patch in order
# to have the ellipse rendered in different ways.

fig, ax_kwargs = plt.subplots(figsize=(6, 6))
dependency_kwargs = np.array([
    [-0.8, 0.5],
    [-0.2, 0.5]
])
mu = np.array([2, -3]).T
scale = np.array([6, 5]).T

ax_kwargs.axvline(c='grey', lw=1)
ax_kwargs.axhline(c='grey', lw=1)

x, y = get_correlated_dataset(500, dependency_kwargs, mu, scale)
# Now plot the dataset first ("under" the ellipse) in order to
# demonstrate the transparency of the ellipse (alpha).
ax_kwargs.scatter(x, y, s=0.5)
confidence_ellipse(x, y, ax_kwargs,
    alpha=0.5, facecolor='pink', edgecolor='purple')

ax_kwargs.scatter([mu[0]], [mu[1]], c='red', s=3)
ax_kwargs.set_title(f'Using kwargs')

fig.subplots_adjust(hspace=0.25)
plt.show()
