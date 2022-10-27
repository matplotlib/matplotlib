"""
===========================
Animations using matplotlib
===========================

Animations in matplotlib can be created using the `~matplotlib.Animation`
module. This module provides 2 main animation classes -
`~matplotlib.animation.FuncAnimation` and
`~matplotlib.animation.ArtistAnimation`.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

###############################################################################
# :class:`~matplotlib.animation.FuncAnimation`
# --------------------------------------------

# :class:`~matplotlib.animation.FuncAnimation` uses a user-provided function by
# repeatedly calling the function with at a regular *interval*. This allows
# dynamic generation of data by using generators.


def init():
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 10)
    xdata.clear()
    ydata.clear()
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []


def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(2 * np.pi * frame / 30))
    xmin, xmax = ax.get_xlim()

    if frame >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

ani = animation.FuncAnimation(
    fig=fig, func=update, frames=240, init_func=init, interval=30
)
plt.show()

###############################################################################
# Animating :class:`~matplotlib.lines.PathCollection`
# ---------------------------------------------------

fig, ax = plt.subplots()
rng = np.random.default_rng()

scat = ax.scatter(
    rng.uniform(low=0, high=1, size=100),
    rng.uniform(low=0, high=1, size=100),
    c='b'
)
ax.grid()


def update(frame):
    x = rng.uniform(low=0, high=1, size=100)
    y = rng.uniform(low=0, high=1, size=100)
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    return scat,

ani = animation.FuncAnimation(
    fig=fig, func=update, frames=240, interval=300
)
plt.show()


###############################################################################
# Animating :class:`~matplotlib.image.AxesImage`
# ---------------------------------------------------

fig, ax = plt.subplots()
rng = np.random.default_rng()

aximg = ax.imshow(rng.uniform(low=0, high=1, size=(10, 10)), cmap="Blues")


def update(frame):
    data = rng.uniform(low=0, high=1, size=(10, 10))
    aximg.set_data(data)
    return aximg,

ani = animation.FuncAnimation(
    fig=fig, func=update, frames=240, interval=200
)
plt.show()

###############################################################################
# :class:`~matplotlib.animation.ArtistAnimation`
# ----------------------------------------------
#
# :class:`~matplotlib.animation.ArtistAnimation` uses a list of artists to
# iterate over and animate.


fig, ax = plt.subplots()
ax.grid()
rng = np.random.default_rng()

x_frames = rng.uniform(low=0, high=1, size=(100, 120))
y_frames = rng.uniform(low=0, high=1, size=(100, 120))
artists = [
    [
        ax.scatter(x_frames[:, i], y_frames[:, i], c="b")
    ]
    for i in range(x_frames.shape[-1])
]

ani = animation.ArtistAnimation(fig=fig, artists=artists, repeat_delay=1000)
plt.show()

###############################################################################
# Animation Writers
# -------------------------------------------

fig, ax = plt.subplots()
ax.grid()
rng = np.random.default_rng()

scat = ax.scatter(
    rng.uniform(low=0, high=1, size=100),
    rng.uniform(low=0, high=1, size=100),
    c='b'
)


def update(frame):
    x = rng.uniform(low=0, high=1, size=100)
    y = rng.uniform(low=0, high=1, size=100)
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    return scat,

ani = animation.FuncAnimation(
    fig=fig, func=update, frames=240, interval=200
)
# ani.save(filename="/tmp/pillow_example.gif", writer="pillow")
# ani.save(filename="/tmp/pillow_example.apng", writer="pillow")

# Why does HTMLWriter documentation have a variable named
# supported_formats = ['png', 'jpeg', 'tiff', 'svg']
# ani.save(filename="/tmp/html_example.html", writer="html")
# ani.save(filename="/tmp/html_example.htm", writer="html")
# ani.save(filename="/tmp/html_example.png", writer="html")

# Since the frames are piped out to ffmpeg, this supports all formats
# supported by ffmpeg
# ani.save(filename="/tmp/ffmpeg_example.mkv", writer="ffmpeg")
# ani.save(filename="/tmp/ffmpeg_example.mp4", writer="ffmpeg")
# ani.save(filename="/tmp/ffmpeg_example.mjpeg", writer="ffmpeg")

# Imagemagick
# ani.save(filename="/tmp/imagemagick_example.gif", writer="imagemagick")
