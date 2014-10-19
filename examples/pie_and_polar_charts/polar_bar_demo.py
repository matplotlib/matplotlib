"""
Demo of bar plot on a polar axis.
"""
import numpy as np
import matplotlib.pyplot as plt


def polar_bar_demo(ax, theta, radii, width):
    """
    produces a randomly colored polar plot given three arrays,
    theta, radii,width .

    Parameters
    ----------
    ax :  PolarAxesSubplot
          Axes on which to plot polar_bar

    theta : array
            Angles at which to plot polar bars

    radii : array
            lengths of polar bar

    width : array
            widths of polars bars

    Returns
    -------
    bars : artist object returned

         Returns axes for further modification.
    """

    bars = ax.bar(theta, radii, width=width, bottom=0.0)

# Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.5)

    return bars

# Generate Example Data.
N = 20
ex_theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
ex_radii = 10 * np.random.rand(N)
ex_width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(111, polar=True)
polar_bar_demo(ax, ex_theta, ex_radii, ex_width)

plt.show()
