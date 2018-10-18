import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.text import Text
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox


class AngleMarker(Arc, Text):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, vec1, vec2, size=100, units="pixels", ax=None,
                 text='', **kwargs):
        """
        Params
        ------
        xy, vec1, vec2 : tuple or array of two floats
            center position and two points. Angle marker is drawn between the
            two vectors connecting vec1 and vec2 with xy, respectively. Units
            are data coordinates.

        size : float
            diameter of the angle marker in units specified by ``units``.

        units : string
            One of the following strings to specify the units of ``size``:
                * "pixels" : pixels
                * "points" : points, use points instead of pixels to not have a
                                        dependence of the dpi
                * "axes width", "axes height" : relative units of axes
                  width, height
                * "axes min", "axes max" : minimum or maximum of relative axes
                  width, height

        ax : `matplotlib.axes.Axes`
            The axes to add the angle marker to

        kwargs :
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc of the arc.

        """
        self._xydata = xy  # in data coordinates
        self.ax = ax or plt.gca()
        self.vec1 = vec1  # tuple or array of absolute coordinates
        self.vec2 = vec2  # tuple or array of absolute coordinates
        self.size = size
        self.units = units

        Arc.__init__(self, self._xydata, size, size, angle=0.0,
                     theta1=self.theta1, theta2=self.theta2, **kwargs)
        Text.__init__(self, x=self._x, y=self._y, text=text, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_artist(self)

    def get_size(self):
        factor = 1.
        if self.units == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.units[:4] == "axes":
            b = TransformedBbox(Bbox.from_bounds(0, 0, 1, 1),
                                self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.units[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """ return center in pixels """
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """ set center in data coordinates """
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    def get_x_text(self):
        return self._xydata[0] + 3*self.size

    def get_y_text(self):
        return self._xydata[1] + 3*self.size

    def set_xy_text(self, xy):
        pass

    def set_color(self, color):
        Arc.set_color(self, color)
        Text.set_color(self, color)

    def draw(self, renderer):
        Arc.draw(self, renderer)
        Text.draw(self, renderer)

    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)
    _x = property(get_x_text, set_xy_text)
    _y = property(get_y_text, set_xy_text)


fig, ax = plt.subplots()

ax.plot([2, .5, 1], [0, .2, 1])
am = AngleMarker((.5, .2), (2, 0), (1, 1), size=0.25, units="axes max", ax=ax,
                 text=r"$\theta$")
plt.show()

'''
def testing(size=0.25, units="axes fraction", dpi=100, fs=(6.4, 5),
            show=False):

    fig, axes = plt.subplots(2, 2, sharex="col", sharey="row", dpi=dpi,
                             figsize=fs,
                             gridspec_kw=dict(width_ratios=[1, 3],
                                              height_ratios=[3, 1]))

    def plot_angle(ax, pos, vec1, vec2, acol="C0", **kwargs):
        ax.plot([vec1[0], pos[0], vec2[0]], [vec1[1], pos[1], vec2[1]],
                color=acol)
        am = AngleMarker(pos, vec1, vec2, ax=ax, text=r"$\theta$", **kwargs)

    tx = "figsize={}, dpi={}, arcsize={} {}".format(fs, dpi, size, units)
    axes[0, 1].set_title(tx, loc="right", size=9)
    kw = dict(size=size, units=units)
    p = (.5, .2), (2, 0), (1, 1)
    plot_angle(axes[0, 0], *p, **kw)
    plot_angle(axes[0, 1], *p, **kw)
    plot_angle(axes[1, 1], *p, **kw)
    kw.update(acol="limegreen")
    plot_angle(axes[0, 0], (1.2, 0), (1, -1), (1.3, -.8), **kw)
    plot_angle(axes[1, 1], (0.2, 1), (0, 0), (.3, .2), **kw)
    plot_angle(axes[0, 1], (0.2, 0), (0, -1), (.3, -.8), **kw)
    kw.update(acol="crimson")
    plot_angle(axes[1, 0], (1, .5), (1, 1), (2, .5), **kw)

    fig.tight_layout()
    fig.savefig(tx.replace("=", "_") + ".png")
    fig.savefig(tx.replace("=", "_") + ".pdf")
    if show:
        plt.show()


s = [(0.25, "axes min"), (0.25, "axes max"),
     (0.25, "axes width"), (0.25, "axes height"),
     (100, "pixels"), (72, "points")]
d = [72, 144]
f = [(6.4, 5), (12.8, 10)]

import itertools

for (size, unit), dpi, fs in itertools.product(s, d, f):
    testing(size=size, units=unit, dpi=dpi, fs=fs)
'''
