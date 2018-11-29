import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.text import Annotation
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox


class AngleMarker(Arc):
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

        if self.theta1 > self.theta2:
            self.vec1, self.vec2 = self.vec2, self.vec1

        Arc.__init__(self, self._xydata, size, size, angle=0.0,
                     theta1=self.theta1, theta2=self.theta2, **kwargs)
        self.set_transform(IdentityTransform())

        if units == "pixels" or units == "points":
            textcoords = "offset " + units
        else:
            textcoords = "offset pixels"

        annotation = Annotation(
            text,
            self._xydata,
            xytext=self._text_pos,
            xycoords="data",
            textcoords=textcoords,
            **kwargs)

        self.ax.add_patch(self)
        self.ax.add_artist(annotation)

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

    def get_text_pos(self):
        theta = np.deg2rad((self.theta2 + self.theta1)/2)
        x = self.width*np.cos(theta)
        y = self.height*np.sin(theta)
        return (x, y)

    def set_text_pos(self, xy):
        pass

    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)
    _text_pos = property(get_text_pos, set_text_pos)


if __name__ == "__main__":
    fig, ax = plt.subplots()

    ax.plot([2, .5, -1], [1, .2, 1])
    am = AngleMarker((.5, .2), (2, 1), (-1, 1), size=50, units="pixels", ax=ax,
                     text=r"$\theta$")
    plt.show()
