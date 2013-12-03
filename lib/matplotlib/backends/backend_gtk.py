from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import matplotlib
# pull in everything from the backing file, should probably be more selective
from ._backend_gtk import (new_figure_manager,
                           new_figure_manager_given_figure,
                           TimerGTK,
                           FigureCanvasGTK,
                           FigureManagerGTK,
                           NavigationToolbar2GTK,
                           FileChooserDialog,
                           DialogLineprops,
                           error_msg_gtk,
                           PIXELS_PER_INCH,
                           backend_version)

# import gtk from the backing file
from ._backend_gtk import gtk
# pull in Gcf
from matplotlib._pylab_helpers import Gcf

# pull in Gcf contaminate parts
from matplotlib.backend_bases import (ShowBase,
                                      key_press_handler)


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
        if gtk.main_level() == 0:
            gtk.main()

show = Show()


def _gtk_cleanup():
    """
    tests if we have closed the last window in a non-interactive session,
    if so close gtk
    """
    if (Gcf.get_num_fig_managers() == 0 and
            not matplotlib.is_interactive() and
            gtk.main_level() >= 1):
        gtk.main_quit()


FigureCanvas = FigureCanvasGTK
FigureManager = FigureManagerGTK

FigureManager._key_press_handler = staticmethod(key_press_handler)
FigureManager._destroy_callback = staticmethod(Gcf.destroy)
FigureManager._gtk_cleanup = staticmethod(_gtk_cleanup)
