"""
==================
Animated histogram
==================

Use histogram's `.BarContainer` to draw a bunch of rectangles for an animated
histogram.
"""

import functools

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

# Setting up a random number generator with a fixed state for reproducibility.
rng = np.random.default_rng(seed=19680801)
# Fixing bin edges.
HIST_BINS = np.linspace(-4, 4, 100)

# Histogram our data with numpy.
data = rng.standard_normal(1000)
n, _ = np.histogram(data, HIST_BINS)

# %%
# To animate the histogram, we need an ``animate`` function, which generates
# a random set of numbers and updates the heights of rectangles. The ``animate``
# function updates the `.Rectangle` patches on an instance of `.BarContainer`.


def animate(frame_number, bar_container):
    # Simulate new data coming in.
    data = rng.standard_normal(1000)
    n, _ = np.histogram(data, HIST_BINS)
    for count, rect in zip(n, bar_container.patches):
        rect.set_height(count)

    return bar_container.patches

# %%
# Using :func:`~matplotlib.pyplot.hist` allows us to get an instance of
# `.BarContainer`, which is a collection of `.Rectangle` instances.  Since
# `.FuncAnimation` will only pass the frame number parameter to the animation
# function, we use `functools.partial` to fix the ``bar_container`` parameter.

# Output generated via `matplotlib.animation.Animation.to_jshtml`.

fig, ax = plt.subplots()
_, _, bar_container = ax.hist(data, HIST_BINS, lw=1,
                              ec="yellow", fc="green", alpha=0.5)
ax.set_ylim(top=55)  # set safe limit to ensure that all data is visible.

anim = functools.partial(animate, bar_container=bar_container)
ani = animation.FuncAnimation(fig, anim, 50, repeat=False, blit=True)
plt.show()

# %%
# .. tags:: plot-type: histogram, animation
