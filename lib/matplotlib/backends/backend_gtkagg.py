"""
Render to gtk from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from matplotlib._pylab_helpers import Gcf
# import the Gcf free parts
from ._backend_gtkagg import (gtk,
                              FigureCanvasGTKAgg,
                              FigureManagerGTKAgg,
                              NavigationToolbar2GTKAgg,
                              new_figure_manager,
                              new_figure_manager_given_figure)

# import the gcg contaminated parts
from .backend_gtk import (show,
                          draw_if_interactive,
                          _gtk_cleanup,
                          key_press_handler)


FigureCanvas = FigureCanvasGTKAgg
FigureManager = FigureManagerGTKAgg

# set the call backs
FigureManager._key_press_handler = staticmethod(key_press_handler)
FigureManager._destroy_callback = staticmethod(Gcf.destroy)
FigureManager._gtk_cleanup = staticmethod(_gtk_cleanup)
