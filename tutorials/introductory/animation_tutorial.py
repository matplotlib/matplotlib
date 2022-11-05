"""
===========================
Animations using matplotlib
===========================

Based on its plotting functionality, matplotlib also provides an interface to
generate animations using the :class:`~matplotlib.animation` module. An
animation is a sequence of frames where each frame corresponds to a plot on a
:class:`~matplotlib.figure.Figure`. This tutorial covers a general guideline on
how to create such animations and the different options available.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

###############################################################################
# Animation Classes
# =================
#
# The process of animation in matplotlib can be thought about in 2 different
# ways:
#
# - :class:`~matplotlib.animation.FuncAnimation`: Generate data for first
#   frame and then modify this data for each frame to create an animated plot.
#
# - :class:`~matplotlib.animation.FuncAnimation`: Generate a list (iterable)
#   of artists that will draw in each frame in the animation.
#
# :class:`~matplotlib.animation.FuncAnimation` is more efficient in terms of
# speed and memory as it draws an artist once and then modifies it. On the
# other hand :class:`~matplotlib.animation.ArtistAnimation` is flexible as it
# allows any iterable of artists to be animated in a sequence.
#
# :class:`~matplotlib.animation.FuncAnimation`
# --------------------------------------------
#
# :class:`~matplotlib.animation.FuncAnimation` class allows us to create an
# animation by passing a function that iteratively modifies the data of a plot.
# This is achieved by using the *setter* methods on various
# :class:`~matplotlib.artist.Artist`
# (examples: :class:`~matplotlib.lines.Line2D`,
# :class:`~matplotlib.collections.PathCollection`, etc.). A usual
# :class:`~matplotlib.animation.FuncAnimation` object takes a
# :class:`~matplotlib.figure.Figure` that we want to animate and a function
# *func* that modifies the data plotted on the figure. It uses the *frames*
# parameter to determine the length of the animation. The *interval* parameter
# is used to determine time in milliseconds between drawing of two frames.
# We will now look at some examples of using
# :class:`~matplotlib.animation.FuncAnimation` with different artists.

###############################################################################
# Animating Lines
# ^^^^^^^^^^^^^^^
#
# `.pyplot.plot` returns a :class:`~matplotlib.lines.Line2D` collection. The
# data on this collection can be modified by using the
# `.lines.Line2D.set_data` function. Therefore we can use this to modify the
# plot using the function for every frame.

fig, ax = plt.subplots()
rng = np.random.default_rng()

xdata = np.arange(0, 2 * np.pi, 0.01)
(line,) = ax.plot(xdata, np.sin(xdata), c="b")
ax.set_ylim(-1.1, 1.1)


def update(frame):
    # .set_ydata resets the y-data for the line, so we add the new point to
    # the existing line x-data and calculate y again.
    line.set_ydata(np.sin(xdata + frame / 50))
    return (line,)


ani = animation.FuncAnimation(fig=fig, func=update,
                              interval=30)
plt.show()

###############################################################################
# Animating Markers
# ^^^^^^^^^^^^^^^^^
#
# `.pyplot.scatter` returns a :class:`~matplotlib.collections.PathCollection`
# that can similarly be modified by using the
# `.collections.PathCollection.set_offsets` function.

fig, ax = plt.subplots()
t = np.linspace(-4, 4, 400)
a, b = 3, 2
delta = np.pi / 2

scat = ax.scatter(np.sin(a * t[0] + delta), np.sin(b * t[0]), c="b", s=2)
ax.grid()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)


def update(frame):
    # .set_offsets also resets the offset data for the entire collection with
    # the new values. Therefore, to also carry forward the previously
    # calculated information, we use the data from the first to the current
    # frame to set the new offsets.
    x = np.sin(a * t[:frame] + delta)
    y = np.sin(b * t[:frame])
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    return (scat,)


ani = animation.FuncAnimation(fig=fig, func=update, frames=400, interval=30)
plt.show()


###############################################################################
# Animating Images
# ^^^^^^^^^^^^^^^^
#
# When we plot an image using `.pyplot.imshow`, it returns an
# :class:`~matplotlib.image.AxesImage` object. The data in this object can also
# similarly be modified by using the `.image.AxesImage.set_data` method.


def f(x, y, mean, cov):
    dev_x = x - mean
    dev_y = y - mean
    maha = -0.5 * (((x-mean)/cov)**2 + ((y-mean)/cov)**2)
    return (1/(np.pi * cov)) * np.exp(maha)

fig, ax = plt.subplots()

x, y = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
mean = 0
cov = 0.1
data = f(x, y, mean, cov)
aximg = ax.imshow(data)


def update(frame):
    x, y = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    mean = 0
    cov = 0.01 * frame + 1e-6
    data = f(x, y, mean, cov)

    aximg.set_data(data)
    return (aximg,)


ani = animation.FuncAnimation(fig=fig, func=update, frames=None, interval=100)
plt.show()

###############################################################################
# :class:`~matplotlib.animation.ArtistAnimation`
# ----------------------------------------------
#
# On the other hand, :class:`~matplotlib.animation.ArtistAnimation` can be used
# to generate animations if we have data on various different artists. This
# list of artists is then converted frame by frame into an animation.


def f(x, y, mean, cov):
    dev_x = x - mean
    dev_y = y - mean
    maha = -0.5 * (((x-mean)/cov)**2 + ((y-mean)/cov)**2)
    return (1/(np.pi * cov)) * np.exp(maha)

fig, ax = plt.subplots()

x, y = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
mean = 0
cov = 0.1
data = f(x, y, mean, cov)
aximg = ax.imshow(data)

artists = [
    [ax.imshow(f(x, y, mean, 0.01 * frame + 1e-6))]
    for frame in range(120)
]

ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=100)
plt.show()

###############################################################################
# Animation Writers
# =================
#
# Animation objects can be saved to disk using various multimedia writers
# (ex: Pillow, *ffpmeg*, *imagemagick*). Not all video formats are supported
# by all writers. There are 4 major types of writers:
#
# - :class:`~matplotlib.animation.PillowWriter` - Uses the Pillow library to
#   create the animation.
#
# - :class:`~matplotlib.animation.HTMLWriter` - Used to create JS-based
#   animations.
#
# - Pipe-based writers - :class:`~matplotlib.animation.FFMpegWriter` and
#   :class:`~matplotlib.animation.ImageMagickWriter` are pipe based writers.
#   These writers pipe each frame to the utility (*ffmpeg* / *imagemagick*)
#   which then stitches all of them together to create the animation.
#
# - File-based writers - :class:`~matplotlib.animation.FFMpegFileWriter` and
#   :class:`~matplotlib.animation.ImageMagickFileWriter` are examples of
#   file-based writers. These writers are slower than their pipe-based
#   alternatives but are more useful for debugging as they save each frame in
#   a file before stitching them together into an animation.
#
# Saving Animations
# -----------------
#
# ================================================  ===========================
# Writer                                            Supported Formats
# ================================================  ===========================
# :class:`~matplotlib.animation.PillowWriter`       .gif, .apng
# :class:`~matplotlib.animation.HTMLWriter`         .htm, .html, .png
# :class:`~matplotlib.animation.FFMpegWriter`       All formats supported by
#                                                   *ffmpeg*
# :class:`~matplotlib.animation.ImageMagickWriter`  .gif
# ================================================  ===========================
#
# To save animations using any of the writers, we can use the
# `.animation.Animation.save` method. It takes the *filename* that we want to
# save the animation as and the *writer*, which is either a string or a writer
# object. It also takes an *fps* argument. This argument is different than the
# *interval* argument that `~.animation.FuncAnimation` or
# `~.animation.ArtistAnimation` uses. *fps* determines the frame rate that the
# **saved** animation uses, whereas *interval* determines the frame rate that
# the **displayed** animation uses.

fig, ax = plt.subplots()
ax.grid()
rng = np.random.default_rng()

scat = ax.scatter(
    rng.uniform(low=0, high=1, size=100),
    rng.uniform(low=0, high=1, size=100),
    c="b"
)


def update(frame):
    x = rng.uniform(low=0, high=1, size=100)
    y = rng.uniform(low=0, high=1, size=100)
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    return (scat,)


ani = animation.FuncAnimation(fig=fig, func=update, frames=240, interval=200)

###############################################################################
# Pillow writers::
#
#   ani.save(filename="/tmp/pillow_example.gif", writer="pillow")
#   ani.save(filename="/tmp/pillow_example.apng", writer="pillow")
#
# HTML writers::
#
#   ani.save(filename="/tmp/html_example.html", writer="html")
#   ani.save(filename="/tmp/html_example.htm", writer="html")
#   ani.save(filename="/tmp/html_example.png", writer="html")
#
# FFMpegWriter - Since the frames are piped out to ffmpeg, this option supports
# all formats supported by ffmpeg::
#
#   ani.save(filename="/tmp/ffmpeg_example.mkv", writer="ffmpeg")
#   ani.save(filename="/tmp/ffmpeg_example.mp4", writer="ffmpeg")
#   ani.save(filename="/tmp/ffmpeg_example.mjpeg", writer="ffmpeg")
#
# Imagemagick writers::
#
#   ani.save(filename="/tmp/imagemagick_example.gif", writer="imagemagick")
