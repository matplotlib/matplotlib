"""
Matplotlib used to recompute autoscaled limits after every plotting (plot(),
bar(), etc.) call. It now only does so when actually rendering the canvas, or
when the user queries the Axes limits and that is possible through autoscale
feature. This is an improvement over the previous case when user has to
manually autoscale the Axes (or axis) according to the data. This particular
method is a part of matplotlib.axes.Axes class. It is used to scale the
Axes (or axis) according to the data limits. Before autoscale one could
have to manually struggle with the Axes to scale itself according to
data. Whenever we use autoscale, the Axes (or axis) recalculate its
limits with respect to the data.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.collections import EllipseCollection


def autoscale():
    """Experiment autoscale feature by switching it on/off for a curve"""
    curve = np.sin(np.linspace(0, 6))
    _, axis = plt.subplots(ncols=2)

    # At First we explicitly enable autoscaling (However it is by-default
    # enabled for pyplot artist.)
    axis[0].autoscale(True)
    axis[0].plot(curve)

    # Now let's disable it.
    axis[1].autoscale(False)
    axis[1].plot(curve)


def autoscale_and_margins():
    """Utility function to describe autoscaling feature with Margins."""
    arr = np.arange(-np.pi / 2, (3 * np.pi) / 2, 0.1)
    f_t = np.cos(arr)
    _, axis = plt.subplots(ncols=2)
    for a_x in [axis[0], axis[1]]:
        a_x.plot(arr, f_t, color="red")
        a_x.axhline(y=0, color="magenta", alpha=0.7)
        a_x.margins(0.2, 0.2)

    axis[1].autoscale(tight=True)
    # "tight" argument in ax.margins is by default set to True but unable
    # to automatically autoscale the plot when zoomed-in or zoomed-out.
    plt.tight_layout()


def autoscale_and_collections():
    """Function to showcase the relation between autoscale feature and
    matplotlib's collections objects.
    """
    fig, axis = plt.subplots(ncols=2)
    x_arr = np.arange(10)
    y_arr = np.arange(15)
    x_arr, y_arr = np.meshgrid(x_arr, y_arr)

    x_y = np.column_stack((x_arr.ravel(), y_arr.ravel()))
    ec_1 = EllipseCollection(
        10,
        10,
        5,
        units="y",
        offsets=x_y * 0.5,
        transOffset=axis[0].transData,
        cmap="jet",
    )
    ec_1.set_array((x_arr * y_arr).ravel())
    ec_2 = EllipseCollection(
        10,
        10,
        5,
        units="y",
        offsets=x_y * 0.5,
        transOffset=axis[1].transData,
        cmap="jet",
    )
    ec_2.set_array((x_arr * y_arr).ravel())

    # when autoscaling is enabled then axis is autoscaled to the value
    # specified in the offsets argument in the above two commands.
    axis[0].add_collection(ec_1)
    axis[1].add_collection(ec_2)
    axis[1].autoscale_view()
    fig.canvas.draw()


def autoscale_and_patches():
    """Function to showcase relation between autoscale feature and matplotlib's
    patches objects.
    """
    fig, axis = plt.subplots(ncols=2)
    vertices = [(0, 0), (0, 3), (1, 0), (0, 0)]
    codes = [Path.MOVETO] + [Path.LINETO] * 2 + [Path.MOVETO]
    vertices = np.array(vertices, float)
    path_1 = Path(vertices, codes)
    patches_1 = PathPatch(path_1, facecolor="magenta", alpha=0.7)
    # matplotlib.patches.Patch is Base class of PathPatch and it
    # does not support autoscaling by-default.

    path_2 = Path(vertices, codes)
    patches_2 = PathPatch(path_2, facecolor="magenta", alpha=0.7)

    # As re-using of the artists is not supported, hence creating two
    # different Path and PathPatch objects.

    axis[0].add_patch(patches_1)
    axis[1].add_patch(patches_2)
    axis[1].autoscale()
    fig.canvas.draw()


def autoscale_disable():
    """Function to explicitly disable autoscale feature"""
    arr = np.arange(10)

    # Disable Autoscaling
    plt.autoscale(False)
    plt.xlim(0, 2)
    plt.plot(arr)


# There are some cases when we have to explicitly enable or disable autoscaling
# feature and we would see some of those cases later on in this tutorial. Let's
# just discuss how we can explicitly enable or disable autoscaling:

autoscale()

# Whenever we set margins our axes remains invariant of the change caused by
# it. Hence we use autoscaling if we want data to be bound with the axes
# irrespective of the margin set. Let's look at an example to understand this:

autoscale_and_margins()

# Collection and Patch subclasses of Artist class does not support autoscaling
# by-default. Autoscaling should be explicitly enabled for Collection and Patch
# objects to support it. See Axes limits in the below plots. For further
# reference on relation of Autoscale feature with Collection Class see-
# https://matplotlib.org/3.2.1/api/prev_api_changes/api_changes_3.2.0/behavior.html#autoscaling

# Let's look at the relation between autoscale feature and Collection object:
autoscale_and_collections()


# Let's look at the relation between autoscale feature and Patch object:
autoscale_and_patches()

# Some of the subclasses under Collection class are :
# https://matplotlib.org/3.1.1/_images/inheritance-1d05647d989bf64e3e438a24b19fee19432184da.png
# reference site : https://matplotlib.org/3.1.1/api/collections_api.html. Till
# now we have seen how we can enable autoscaling. Now lets just discuss under
# which case we possibly need to disable it:

# Suppose we want to plot a line at 45 degrees to x and y axes and we want to
# plot the data from a specified range and hence we have to set limits for x
# and y axes and in that case we have to first disable autoscaling and then
# we can set the limits:

autoscale_disable()
