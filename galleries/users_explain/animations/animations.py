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
how to create such animations and the different options available.

.. inheritance-diagram:: matplotlib.animation.FuncAnimation matplotlib.animation.ArtistAnimation
   :parts: 1

"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

# %%
# Animation classes
# =================
#
# ``Animation`` A base class for Animations.
# The animation process in Matplotlib can be thought of in 2 different ways:
#
# - `~matplotlib.animation.FuncAnimation`: Generate data for first
#   frame and then modify this data for each frame to create an animated plot.
#   :doc:`TimedAnimation <https://matplotlib.org/devdocs/api/_as_gen/matplotlib.animation.TimedAnimation.html#matplotlib.animation.TimedAnimation>` subclass that makes an animation by repeatedly calling a function func.
#
#
# - `~matplotlib.animation.ArtistAnimation`: Generate a list (iterable)
#   of artists that will draw in each frame in the animation.
#   :doc:`TimedAnimation <https://matplotlib.org/devdocs/api/_as_gen/matplotlib.animation.TimedAnimation.html#matplotlib.animation.TimedAnimation>` subclass that creates an animation by using a fixed set of Artist objects.
#
# In both cases it is critical to keep a reference to the instance
# object.  The animation is advanced by a timer (typically from the host
# GUI framework) which the `Animation` object holds the only reference
# to.  If you do not hold a reference to the `Animation` object, it (and
# hence the timers) will be garbage collected which will stop the
# animation.
# To save an animation use `Animation.save`, `Animation.to_html5_video`,
# or `Animation.to_jshtml`.
#
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
# `~matplotlib.collections.PathCollection`, etc.). A usual
# `~matplotlib.animation.FuncAnimation` object takes a
# `~matplotlib.figure.Figure` that we want to animate and a function
# *func* that modifies the data plotted on the figure. It uses the *frames*
# parameter to determine the length of the animation. The *interval* parameter
# is used to determine time in milliseconds between drawing of two frames.
# Animating using `.FuncAnimation` would usually follow the following
# structure:
#
# - Plot the initial figure, including all the required artists. Save all the
#   artists in variables so that they can be updated later on during the
#   animation.
# - Create an animation function that updates the data in each artist to
#   generate the new frame at each function call.
# - Create a `.FuncAnimation` object with the `.Figure` and the animation
#   function, along with the keyword arguments that determine the animation
#   properties.
# - Use `.animation.Animation.save` or `.pyplot.show` to save or show the
#   animation.
#
# The inner workings of `FuncAnimation` is more-or-less::
# 
for d in frames:
    artists = func(d, *fargs)
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(interval)
# 
# with details to handle 'blitting' (to dramatically improve the live
# performance), to be non-blocking, not repeatedly start/stop the GUI
# event loop, handle repeats, multiple animated axes, and easily save
# the animation to a movie file.
# 
# 'Blitting' is a `standard technique
# <https://en.wikipedia.org/wiki/Bit_blit>`__ in computer graphics.  The
# general gist is to take an existing bit map (in our case a mostly
# rasterized figure) and then 'blit' one more artist on top.  Thus, by
# managing a saved 'clean' bitmap, we can only re-draw the few artists
# that are changing at each frame and possibly save significant amounts of
# time.  When we use blitting (by passing ``blit=True``), the core loop of
# `FuncAnimation` gets a bit more complicated::
ax = fig.gca()

def update_blit(artists):
    fig.canvas.restore_region(bg_cache)
    for a in artists:
        a.axes.draw_artist(a)

    ax.figure.canvas.blit(ax.bbox)

artists = init_func()

for a in artists:
    a.set_animated(True)

fig.canvas.draw()
bg_cache = fig.canvas.copy_from_bbox(ax.bbox)

for f in frames:
    artists = func(f, *fargs)
    update_blit(artists)
    fig.canvas.start_event_loop(interval)
# 
# This is of course leaving out many details (such as updating the
# background when the figure is resized or fully re-drawn).  However,
# this hopefully minimalist example gives a sense of how ``init_func``
# and ``func`` are used inside of `FuncAnimation` and the theory of how
# 'blitting' works.
# 
# .. note::
# 
#     The zorder of artists is not taken into account when 'blitting'
#     because the 'blitted' artists are always drawn on top.
# 
# The expected signature on ``func`` and ``init_func`` is very simple to
# keep `FuncAnimation` out of your book keeping and plotting logic, but
# this means that the callable objects you pass in must know what
# artists they should be working on.  There are several approaches to
# handling this, of varying complexity and encapsulation.  The simplest
# approach, which works quite well in the case of a script, is to define the
# artist at a global scope and let Python sort things out.  For example:: 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()
# 
# The second method is to use `functools.partial` to pass arguments to the
# function::
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

fig, ax = plt.subplots()
line1, = ax.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return line1,

def update(frame, ln, x, y):
    x.append(frame)
    y.append(np.sin(frame))
    ln.set_data(x, y)
    return ln,

ani = FuncAnimation(
    fig, partial(update, ln=line1, x=[], y=[]),
    frames=np.linspace(0, 2*np.pi, 128),
    init_func=init, blit=True)

plt.show()
# 
# A third method is to use closures to build up the required
# artists and functions.  A fourth method is to create a class.
#
# Examples
# ^^^^^^^^
# 
# * :doc:`../gallery/animation/animate_decay`
# * :doc:`../gallery/animation/bayes_update`
# * :doc:`../gallery/animation/double_pendulum`
# * :doc:`../gallery/animation/animated_histogram`
# * :doc:`../gallery/animation/rain`
# * :doc:`../gallery/animation/random_walk`
# * :doc:`../gallery/animation/simple_anim`
# * :doc:`../gallery/animation/strip_chart`
# * :doc:`../gallery/animation/unchained`
#
# The update function uses the ``set_*`` function for different artists to
# modify the data. The following table shows a few plotting methods, the artist
# types they return and some methods that can be used to update them.
#
# ========================================  =============================  ===========================
# Plotting method                           Artist                         Set method
# ========================================  =============================  ===========================
# `.Axes.plot`                              `.lines.Line2D`                `~.lines.Line2D.set_data`
# `.Axes.scatter`                           `.collections.PathCollection`  `~.collections.\
#                                                                          PathCollection.set_offsets`
# `.Axes.imshow`                            `.image.AxesImage`             ``AxesImage.set_data``
# `.Axes.annotate`                          `.text.Annotation`             `~.text.Annotation.\
#                                                                          update_positions`
# `.Axes.barh`                              `.patches.Rectangle`           `~.Rectangle.set_angle`,
#                                                                          `~.Rectangle.set_bounds`,
#                                                                          `~.Rectangle.set_height`,
#                                                                          `~.Rectangle.set_width`,
#                                                                          `~.Rectangle.set_x`,
#                                                                          `~.Rectangle.set_y`,
#                                                                          `~.Rectangle.set_xy`
# `.Axes.fill`                              `.patches.Polygon`             `~.Polygon.set_xy`
# `.Axes.add_patch`\(`.patches.Ellipse`\)   `.patches.Ellipse`             `~.Ellipse.set_angle`,
#                                                                          `~.Ellipse.set_center`,
#                                                                          `~.Ellipse.set_height`,
#                                                                          `~.Ellipse.set_width`
# ========================================  =============================  ===========================
#
# Covering the set methods for all types of artists is beyond the scope of this
# tutorial but can be found in their respective documentations. An example of
# such update methods in use for `.Axes.scatter` and `.Axes.plot` is as follows.

fig, ax = plt.subplots()
t = np.linspace(0, 3, 40)
g = -9.81
v0 = 12
z = g * t**2 / 2 + v0 * t

v02 = 5
z2 = g * t**2 / 2 + v02 * t

scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
ax.legend()


def update(frame):
    # for each frame, update the data stored on each artist.
    x = t[:frame]
    y = z[:frame]
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # update the line plot:
    line2.set_xdata(t[:frame])
    line2.set_ydata(z2[:frame])
    return (scat, line2)


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
plt.show()


# %%
# ``ArtistAnimation``
# -------------------
#
# `~matplotlib.animation.ArtistAnimation` can be used
# to generate animations if there is data stored on various different artists.
# This list of artists is then converted frame by frame into an animation. For
# example, when we use `.Axes.barh` to plot a bar-chart, it creates a number of
# artists for each of the bar and error bars. To update the plot, one would
# need to update each of the bars from the container individually and redraw
# them. Instead, `.animation.ArtistAnimation` can be used to plot each frame
# individually and then stitched together to form an animation. A barchart race
# is a simple example for this.


fig, ax = plt.subplots()
rng = np.random.default_rng(19680801)
data = np.array([20, 20, 20, 20])
x = np.array([1, 2, 3, 4])

artists = []
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
for i in range(20):
    data += rng.integers(low=0, high=10, size=data.shape)
    container = ax.barh(x, data, color=colors)
    artists.append(container)


ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
plt.show()

# %%
# Animation writers
# =================
#
# Animation objects can be saved to disk using various multimedia writers
# (ex: Pillow, *ffpmeg*, *imagemagick*). Not all video formats are supported
# by all writers. There are 4 major types of writers:
#
# - `~matplotlib.animation.PillowWriter` - Uses the Pillow library to
#   create the animation.
#
# - `~matplotlib.animation.HTMLWriter` - Used to create JavaScript-based
#   animations.
#
# - Pipe-based writers - `~matplotlib.animation.FFMpegWriter` and
#   `~matplotlib.animation.ImageMagickWriter` are pipe based writers.
#   These writers pipe each frame to the utility (*ffmpeg* / *imagemagick*)
#   which then stitches all of them together to create the animation.
#
# - File-based writers - `~matplotlib.animation.FFMpegFileWriter` and
#   `~matplotlib.animation.ImageMagickFileWriter` are examples of
#   file-based writers. These writers are slower than their pipe-based
#   alternatives but are more useful for debugging as they save each frame in
#   a file before stitching them together into an animation.
#
# Saving Animations
# -----------------
#
# .. list-table::
#    :header-rows: 1
#
#    * - Writer
#      - Supported Formats
#    * - `~matplotlib.animation.PillowWriter`
#      - .gif, .apng, .webp
#    * - `~matplotlib.animation.HTMLWriter`
#      - .htm, .html, .png
#    * - | `~matplotlib.animation.FFMpegWriter`
#        | `~matplotlib.animation.FFMpegFileWriter`
#      - All formats supported by |ffmpeg|_: ``ffmpeg -formats``
#    * - | `~matplotlib.animation.ImageMagickWriter`
#        | `~matplotlib.animation.ImageMagickFileWriter`
#      - All formats supported by |imagemagick|_: ``magick -list format``
#
# .. _ffmpeg: https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features
# .. |ffmpeg| replace:: *ffmpeg*
#
# .. _imagemagick: https://imagemagick.org/script/formats.php#supported
# .. |imagemagick| replace:: *imagemagick*
#
# To save animations using any of the writers, we can use the
# `.animation.Animation.save` method. It takes the *filename* that we want to
# save the animation as and the *writer*, which is either a string or a writer
# object. It also takes an *fps* argument. This argument is different than the
# *interval* argument that `~.animation.FuncAnimation` or
# `~.animation.ArtistAnimation` uses. *fps* determines the frame rate that the
# **saved** animation uses, whereas *interval* determines the frame rate that
# the **displayed** animation uses.
#
# Below are a few examples that show how to save an animation with different
# writers.
#
#
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
# FFMpegWriter::
#
#   ani.save(filename="/tmp/ffmpeg_example.mkv", writer="ffmpeg")
#   ani.save(filename="/tmp/ffmpeg_example.mp4", writer="ffmpeg")
#   ani.save(filename="/tmp/ffmpeg_example.mjpeg", writer="ffmpeg")
#
# Imagemagick writers::
#
#   ani.save(filename="/tmp/imagemagick_example.gif", writer="imagemagick")
#   ani.save(filename="/tmp/imagemagick_example.webp", writer="imagemagick")
#   ani.save(filename="apng:/tmp/imagemagick_example.apng",
#            writer="imagemagick", extra_args=["-quality", "100"])
#
# (the ``extra_args`` for *apng* are needed to reduce filesize by ~10x)
