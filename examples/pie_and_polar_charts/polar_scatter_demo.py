"""
Demo of scatter plot on a polar axis.

Size increases radially in this example and color increases with angle (just to
verify the symbols are being scattered correctly).
"""
import numpy as np
import matplotlib.pyplot as plt


def polar_scatter_demo(ax, theta, r, area, colors):
    """
    produces a randomly colored polar plot given three arrays,
    theta, radii,width .

    Parameters
    ----------
    ax :  PolarAxesSubplot
          Axes on which to plot polar_scatter

    theta : array
            Angles at which to plot points

    r : array
            radii at which to plot points

    area : array
           sizes of points

    colors : array
           color values of points.

    Returns
    -------
    c : artist object returned

    """
    c = ax.scatter(theta, r, c=colors, s=area, cmap=plt.cm.hsv)
    c.set_alpha(0.75)
    return c


# example data
N = 150
ex_r = 2 * np.random.rand(N)
ex_theta = 2 * np.pi * np.random.rand(N)
ex_area = 200 * r ** 2 * np.random.rand(N)
ex_colors = theta

ax = plt.subplot(111, polar=True)

polar_scatter_demo(ax, ex_theta, ex_r, ex_area, ex_colors)

plt.show()
