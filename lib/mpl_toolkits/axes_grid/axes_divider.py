from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from mpl_toolkits.axes_grid1.axes_divider import Divider, AxesLocator, SubplotDivider, \
     AxesDivider, locatable_axes_factory, make_axes_locatable

from mpl_toolkits.axes_grid.axislines import Axes
LocatableAxes = locatable_axes_factory(Axes)
