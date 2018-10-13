"""
==============
Angle Arc Demo
==============

Draw an angle arc between two vectors
"""
from math import atan2, sqrt, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots()

vec1 = ax.arrow(0, 0, 2, 1, head_width=0.05)
vec2 = ax.arrow(0, 0, -1, 2, head_width=0.05)


def get_vector_angle(trans, vec):
    """
    Finds the angle of a FancyArrow. Probably
    a better way to do this.

    Parameters
    ----------
    trans : matplotlib.transforms.CompositeGenericTransform
        Transformation from data coords to axes coords
    vec : matplotlib.patches.FancyArrow

    Returns
    -------
    float
        Angle in radians between +x axis
        and vec.
    """
    #Shift axes coords to -0.5, 0.5 for proper angle calculation
    xy = trans.transform(vec.get_xy()) - 0.5
    dx = max(xy[:, 0], key=abs) - min(xy[:, 0], key=abs)
    dy = max(xy[:, 1], key=abs) - min(xy[:, 1], key=abs)
    return atan2(dy, dx)*180/pi


def draw_arc_between_vectors(ax, vec1, vec2):
    """
    Draws a scale-invariant arc between two vectors.

    Arc will be drawn counterclockwise if the angle
    of vec1 is smaller than the angle of vec2 and
    will be drawn clockwise otherwise. Arc will be
    drawn as a mpatches.Arc on the provided axes.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes on which vec1 and vec2 are drawn
    vec1 : matplotlib.patches.FancyArrow
        Vector 1
    vec2 : matplotlib.patches.FancyArrow
        Vector 2
    """
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))
    dx = x1 - x0
    dy = y1 - y0
    d = sqrt(dx**2 + dy**2)
    width = 0.1*d/dx
    height = 0.1*d/dy
    trans = ax.transData + ax.transAxes.inverted()
    theta1 = get_vector_angle(trans, vec1)
    theta2 = get_vector_angle(trans, vec2)
    arc = mpatches.Arc(
        trans.transform((0, 0)),
        width, height,
        theta1=theta1, theta2=theta2,
        gid="angle_arc",
        transform=ax.transAxes)
    [p.remove() for p in ax.patches if p.get_gid() == "angle_arc"]
    ax.add_patch(arc)


def fig_resize(event):
    draw_arc_between_vectors(ax, vec1, vec2)


def axes_resize(ax):
    draw_arc_between_vectors(ax, vec1, vec2)

fig.canvas.mpl_connect("resize_event", fig_resize)
ax.callbacks.connect("xlim_changed", axes_resize)
ax.callbacks.connect("ylim_changed", axes_resize)

plt.xlim(-3, 3)
plt.ylim(-3, 3)

plt.show()
