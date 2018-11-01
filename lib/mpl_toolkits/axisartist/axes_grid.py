import mpl_toolkits.axes_grid1.axes_grid as axes_grid_orig
from .axislines import Axes


class CbarAxes(axes_grid_orig.CbarAxesBase, Axes):
    def __init__(self, *args, orientation, **kwargs):
        self.orientation = orientation
        self._default_label_on = False
        self.locator = None
        super().__init__(*args, **kwargs)

    def cla(self):
        super().cla()
        self._config_axes()


class Grid(axes_grid_orig.Grid):
    _defaultAxesClass = Axes


class ImageGrid(axes_grid_orig.ImageGrid):
    _defaultAxesClass = Axes
    _defaultCbarAxesClass = CbarAxes


AxesGrid = ImageGrid
