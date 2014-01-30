"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import warnings
from matplotlib.cbook import mplDeprecation

# import Gcf/pyplot stuff
from matplotlib._pylab_helpers import Gcf


# import Gcf stuff from QT4 backend
from .backend_qt4 import show
from .backend_qt4 import draw_if_interactive

# import the backend with call backs included from
from .backend_qt4 import FigureManagerQT


from .base_backend_qt4 import (TimerQT,
                            SubplotToolQt,
                            backend_version)


# import qtAgg stuff
from .base_backend_qt4agg import (new_figure_manager,
                              new_figure_manager_given_figure,
                              NavigationToolbar2QT,
                              FigureCanvasQTAgg)


class NavigationToolbar2QTAgg(NavigationToolbar2QT):
    def __init__(*args, **kwargs):
        warnings.warn('This class has been deprecated in 1.4 ' +
                      'as it has no additional functionality over ' +
                      '`NavigationToolbar2QT`.  Please change your code to '
                      'use `NavigationToolbar2QT` instead',
                    mplDeprecation)
        NavigationToolbar2QT.__init__(*args, **kwargs)


FigureCanvas = FigureCanvasQTAgg
FigureManager = FigureManagerQT
