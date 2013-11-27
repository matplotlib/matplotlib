from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

try:
    import gi
except ImportError:
    raise ImportError("Gtk3 backend requires pygobject to be installed.")

try:
    gi.require_version("Gtk", "3.0")
except AttributeError:
    raise ImportError(
        "pygobject version too old -- it must have require_version")
except ValueError:
    raise ImportError(
        "Gtk3 backend requires the GObject introspection bindings for Gtk 3 "
        "to be installed.")

try:
    from gi.repository import Gtk
except ImportError:
    raise ImportError("Gtk3 backend requires pygobject to be installed.")

import matplotlib
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import ShowBase

from . backend_gtk3_gui import FigureManagerGTK3Gui, FigureCanvasGTK3Gui


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


class FigureManagerGTK3(FigureManagerGTK3Gui):
    """
    Public attributes

    canvas      : The FigureCanvas instance
    num         : The Figure number
    toolbar     : The Gtk.Toolbar  (gtk only)
    vbox        : The Gtk.VBox containing the canvas and toolbar (gtk only)
    window      : The Gtk.Window   (gtk only)
    """
    def __init__(self, canvas, num):
        FigureManagerGTK3Gui.__init__(self, canvas)

        self.num = num
        self.set_window_title("Figure %d" % num)

        def destroy(*args):
            Gcf.destroy(num)
        self.window.connect("destroy", destroy)
        self.window.connect("delete_event", destroy)
        if matplotlib.is_interactive():
            self.window.show()

    def destroy(self, *args):
        FigureManagerGTK3Gui.destroy(self, *args)

        if Gcf.get_num_fig_managers() == 0 and \
               not matplotlib.is_interactive() and \
               Gtk.main_level() >= 1:
            Gtk.main_quit()


FigureCanvasGTK3 = FigureCanvasGTK3Gui
