import mpl_toolkits.axes_grid1.axes_grid as axes_grid_orig
from .axes_divider import LocatableAxes


class CbarAxes(axes_grid_orig.CbarAxesBase, LocatableAxes):
    def __init__(self, *args, orientation, **kwargs):
        self.orientation = orientation
        self._default_label_on = False
        self.locator = None
        super().__init__(*args, **kwargs)

    def cla(self):
        super().cla()
        self._config_axes()


class Grid(axes_grid_orig.Grid):
    _defaultLocatableAxesClass = LocatableAxes


class ImageGrid(axes_grid_orig.ImageGrid):
    _defaultLocatableAxesClass = LocatableAxes
    _defaultCbarAxesClass = CbarAxes


AxesGrid = ImageGrid
