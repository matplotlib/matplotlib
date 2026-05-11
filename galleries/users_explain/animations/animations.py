"""
.. redirect-from:: /tutorials/introductory/animation_tutorial

.. _animations:

===========================
Animations using Matplotlib
===========================

Based on its plotting functionality, Matplotlib also provides an interface to
generate animations using the `~matplotlib.animation` module. An
animation is a sequence of frames where each frame corresponds to a plot on a
`~matplotlib.figure.Figure`. This tutorial covers a general guideline on
how to create such animations and the different options available. More
information is available in the API description:
`~matplotlib.animation`
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# %%
# Animation classes
# =================
#
# The animation process in Matplotlib can be thought of in 2 different ways:
#
# - `~matplotlib.animation.FuncAnimation`: Generate data for first
#   frame and then modify this data for each frame to create an animated plot.
#
# - `~matplotlib.animation.ArtistAnimation`: Generate a list (iterable)
#   of artists that will draw in each frame in the animation.
#
# `~matplotlib.animation.FuncAnimation` is more efficient in terms of
# speed and memory as it draws an artist once and then modifies it. On the
# other hand `~matplotlib.animation.ArtistAnimation` is flexible as it
# allows any iterable of artists to be animated in a sequence.
#
# ``FuncAnimation``
# -----------------
#
# The `~matplotlib.animation.FuncAnimation` class allows us to create an
# animation by passing a function that iteratively modifies the data of a plot.
# This is achieved by using the *setter* methods on various
# `~matplotlib.artist.Artist` (examples: `~matplotlib.lines.Line2D`,
# `~matplotlib.collections.PathCollection`, etc.).
#
# Animating using `.FuncAnimation` typically requires these steps:
#
# 1) Plot the initial figure as you would in a static plot.
# 2) Create an animation function that updates the artists.
# 3) Create a `.FuncAnimation`.
# 4) Save or show the animation.
#

fig, ax = plt.subplots()

t = np.linspace(0, 3, 40)

g = -9.81
v0 = 12
z = g * t**2 / 2 + v0 * t

v02 = 5
z2 = g * t**2 / 2 + v02 * t

scat = ax.scatter(
    t[0],
    z[0],
    c="b",
    s=5,
    label=f"v0 = {v0} m/s",
)

line2 = ax.plot(
    t[0],
    z2[0],
    label=f"v0 = {v02} m/s",
)[0]

ax.set(
    xlim=(0, 3),
    ylim=(-4, 10),
    xlabel="Time [s]",
    ylabel="Z [m]",
)

ax.legend()


def update(frame):
    x = t[:frame]
    y = z[:frame]

    data = np.stack([x, y]).T
    scat.set_offsets(data)

    line2.set_xdata(t[:frame])
    line2.set_ydata(z2[:frame])

    return (scat, line2)


ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=40,
    interval=30,
)

plt.show()

# %%
# Animations in Jupyter Notebooks
# ===============================
#
# Animations can also be displayed inside Jupyter notebooks by converting
# them to HTML and displaying them using `IPython.display.HTML`, e.g.
# ``HTML(ani.to_jshtml(fps=20))``.
#
# You should explicitly close the figure (``plt.close(fig)``). Otherwise
# it will be picked up by IPython's auto-show figure functionality and
# the static plot will be shown in addition to the animation.
#
# Example:

from IPython.display import HTML

fig, ax = plt.subplots()

x = np.linspace(0, 2 * np.pi)

line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_data(x, np.sin(x - 2 * np.pi * i / 50))
    return (line,)


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=50,
)

plt.close(fig)

HTML(ani.to_jshtml(fps=20))

# This adds interactive playback controls directly in the notebook.

# %%
# ``ArtistAnimation``
# -------------------
#
# `~matplotlib.animation.ArtistAnimation` can be used
# to generate animations if there is data stored on various different artists.
#

fig, ax = plt.subplots()

rng = np.random.default_rng(19680801)

data = np.array([20, 20, 20, 20])
x = np.array([1, 2, 3, 4])

artists = []

colors = [
    "tab:blue",
    "tab:red",
    "tab:green",
    "tab:purple",
]

for i in range(20):
    data += rng.integers(
        low=0,
        high=10,
        size=data.shape,
    )

    container = ax.barh(
        x,
        data,
        color=colors,
    )

    artists.append(container)

ani = animation.ArtistAnimation(
    fig=fig,
    artists=artists,
    interval=400,
)

plt.show()
