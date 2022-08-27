"""
==============
Markevery Demo
==============

The ``markevery`` property of `.Line2D` allows drawing markers at a subset of
data points.

The list of possible parameters is specified at `.Line2D.set_markevery`.
In short:

- A single integer N draws every N-th marker.
- A tuple of integers (start, N) draws every N-th marker, starting at data
  index *start*.
- A list of integers draws the markers at the specified indices.
- A slice draws the markers at the sliced indices.
- A float specifies the distance between markers as a fraction of the Axes
  diagonal in screen space. This will lead to a visually uniform distribution
  of the points along the line, irrespective of scales and zooming.
"""

import numpy as np
import matplotlib.pyplot as plt

# define a list of markevery cases to plot
cases = [
    None,
    8,
    (30, 8),
    [16, 24, 32],
    [0, -1],
    slice(100, 200, 3),
    0.1,
    0.4,
    (0.2, 0.4)
]


def plot_with_markevery(
            x, y, shape="o", ls="-", ms=4, scale=None,
            x_lim=None, y_lim=None, **kwargs
        ):
    fig, axs = plt.subplots(3, 3, figsize=(10, 6), constrained_layout=True, subplot_kw=kwargs)
    
    for ax, markevery in zip(axs.flat, cases):
        ax.set_title(f"{markevery=}")
        ax.plot(x, y, shape, ls=ls, ms=ms, markevery=markevery)
        
        if scale:
            ax.set_xscale(scale)
            ax.set_yscale(scale)
        
        if x_lim:
            ax.set_xlim(x_lim)
        
        if y_lim:
            ax.set_ylim(y_lim)


# data points
delta = 0.11
x = np.linspace(0, 10 - 2 * delta, 200) + delta
y = np.sin(x) + 1.0 + delta

###############################################################################
# markevery with linear scales
# ----------------------------
plot_with_markevery(x, y)


###############################################################################
# markevery with log scales
# -------------------------
#
# Note that the log scale causes a visual asymmetry in the marker distance for
# when subsampling the data using an integer. In contrast, subsampling on
# fraction of figure size creates even distributions, because it's based on
# fractions of the Axes diagonal, not on data coordinates or data indices.

plot_with_markevery(x, y, scale="log")

###############################################################################
# markevery on zoomed plots
# -------------------------
#
# Integer-based *markevery* specifications select points from the underlying
# data and are independent on the view. In contrast, float-based specifications
# are related to the Axes diagonal. While zooming does not change the Axes
# diagonal, it changes the displayed data range, and more points will be
# displayed when zooming.

plot_with_markevery(x, y, x_lim=(6, 6.7), y_lim=(1.1, 1.7))

###############################################################################
# markevery on polar plots
# ------------------------

r = np.linspace(0, 3.0, 200)
theta = 2 * np.pi * r

plot_with_markevery(theta, r, projection="polar")

plt.show()
