"""
GTK+ Matplotlib interface using cairo (not GDK) drawing operations.
Author: Steve Chaplin
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from matplotlib._pylab_helpers import Gcf

# import the gcg contaminated parts
from .backend_gtk import (show,
                          draw_if_interactive,
                          _gtk_cleanup,
                          key_press_handler)

from ._backend_gtkcairo import (gtk,
                                new_figure_manager,
                                new_figure_manager_given_figure,
                                RendererGTKCairo,
                                FigureCanvasGTKCairo,
                                FigureManagerGTKCairo,
                                NavigationToolbar2GTKCairo,
                                backend_version)

FigureCanvas = FigureCanvasGTKCairo
FigureManager = FigureManagerGTKCairo

# set the call backs
FigureManager._key_press_handler = staticmethod(key_press_handler)
FigureManager._destroy_callback = staticmethod(Gcf.destroy)
FigureManager._gtk_cleanup = staticmethod(_gtk_cleanup)
