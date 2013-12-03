from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import sys

import matplotlib
from matplotlib._pylab_helpers import Gcf
# pull in Gcf contaminate parts
from matplotlib.backend_bases import (ShowBase,
                                      key_press_handler)
# pull in Gcf free parts
from ._backend_gtk3 import (TimerGTK3,
                           FigureCanvasGTK3,
                           FigureManagerGTK3,
                           NavigationToolbar2GTK3,
                           FileChooserDialog,
                           DialogLineprops,
                           error_msg_gtk,
                           PIXELS_PER_INCH,
                           backend_version, Gtk)

_debug = False
#_debug = True


def draw_if_interactive():
    """
    Is called after every pylab drawing command
    """
    if matplotlib.is_interactive():
        figManager = Gcf.get_active()
        if figManager is not None:
            figManager.canvas.draw_idle()


class Show(ShowBase):
    def mainloop(self):
        if Gtk.main_level() == 0:
            Gtk.main()

show = Show()


def _gtk_cleanup():
    if Gcf.get_num_fig_managers() == 0 and \
       not matplotlib.is_interactive() and \
       Gtk.main_level() >= 1:
        Gtk.main_quit()


FigureCanvas = FigureCanvasGTK3
FigureManager = FigureManagerGTK3

FigureManager._key_press_handler = staticmethod(key_press_handler)
FigureManager._destroy_callback = staticmethod(Gcf.destroy)
FigureManager._gtk_cleanup = staticmethod(_gtk_cleanup)
