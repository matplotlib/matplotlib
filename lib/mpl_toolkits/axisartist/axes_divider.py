from matplotlib import cbook

from mpl_toolkits.axes_grid1.axes_divider import (
    Divider, AxesLocator, SubplotDivider, AxesDivider, locatable_axes_factory,
    make_axes_locatable)

from mpl_toolkits.axisartist.axislines import Axes as _Axes


@cbook.deprecated('3.0',
                  alternative='mpl_toolkits.axisartist.axislines.Axes')
class Axes(_Axes):
    pass


@cbook.deprecated('3.0',
                  alternative='mpl_toolkits.axisartist.axislines.Axes')
class LocatableAxes(_Axes):
    pass
