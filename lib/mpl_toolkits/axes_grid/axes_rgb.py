from mpl_toolkits.axes_grid1.axes_rgb import (
    make_rgb_axes, imshow_rgb, RGBAxesBase)
from mpl_toolkits.axisartist.axislines import Axes


class RGBAxes(RGBAxesBase):
    _defaultAxesClass = Axes
