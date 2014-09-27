"""
Demo of bar plot on a polar axis.
"""
import numpy as np
import matplotlib.pyplot as plt


N = 20
ex_theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
ex_radii = 10 * np.random.rand(N)
ex_width = np.pi / 4 * np.random.rand(N)


def polar_bar_demo(theta,radii,width):

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, radii, width=width, bottom=0.0)

# Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.5)

    plt.show()
    return ax

polar_bar_demo(ex_theta,ex_radii,ex_width)
