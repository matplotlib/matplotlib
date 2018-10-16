import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform


class AngleArc(Arc):
    def __init__(self, xy, vec1, vec2, size=100, units="pixels",
                 ax=None, fig=None, **kwargs):
        self._xydata = xy  # in data coordinates
        self.ax = ax or plt.gca()
        self.fig = fig or plt.gcf()
        self.vec1 = vec1  # tuple or array of coordinates, relative to xy
        self.vec2 = vec2  # tuple or array of coordinates, relative to xy
        self.size = size

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())

        if units == "relative":
            fig.canvas.mpl_connect("resize_event", self._resize)

        self.ax.add_patch(self)

    def _resize(self, event):
        x0, y0 = self.ax.transAxes.transform((0, 0))
        x1, y1 = self.ax.transAxes.transform((1, 1))
        dx = x1 - x0
        dy = y1 - y0
        smallest = min(dx, dy)
        self.width = 0.25*dx
        self.height = 0.25*dx

    def get_center_pixels(self):
        """ return center in pixel coordinates """
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """ set center in data coordinates """
        self._xydata = xy

    _center = property(get_center_pixels, set_center)

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)


fig, ax = plt.subplots()

ax.plot([2, .5, 1], [0, .2, 1])
am = AngleArc((0.5, 0.2), (2, 0), (1, 1), size=100, units="relative",
              ax=ax, fig=fig)
plt.show()
