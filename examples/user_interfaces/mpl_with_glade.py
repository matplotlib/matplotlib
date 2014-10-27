#!/usr/bin/env python

from __future__ import print_function
import matplotlib
matplotlib.use('GTK')

from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar
from matplotlib.widgets import SpanSelector

from numpy import arange, sin, pi
import gtk
import gtk.glade


def simple_msg(msg, parent=None, title=None):
    dialog = gtk.MessageDialog(
        parent=None,
        type=gtk.MESSAGE_INFO,
        buttons=gtk.BUTTONS_OK,
        message_format=msg)
    if parent is not None:
        dialog.set_transient_for(parent)
    if title is not None:
        dialog.set_title(title)
    dialog.show()
    dialog.run()
    dialog.destroy()
    return None


class GladeHandlers(object):
    def on_buttonClickMe_clicked(event):
        simple_msg('Nothing to say, really',
                   parent=widgets['windowMain'],
                   title='Thanks!')


class WidgetsWrapper(object):
    def __init__(self):
        self.widgets = gtk.glade.XML('mpl_with_glade.glade')
        self.widgets.signal_autoconnect(GladeHandlers.__dict__)

        self['windowMain'].connect('destroy', lambda x: gtk.main_quit())
        self['windowMain'].move(10, 10)
        self.figure = Figure(figsize=(8, 6), dpi=72)
        self.axis = self.figure.add_subplot(111)

        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axis.plot(t, s)
        self.axis.set_xlabel('time (s)')
        self.axis.set_ylabel('voltage')

        self.canvas = FigureCanvas(self.figure)  # a gtk.DrawingArea
        self.canvas.show()
        self.canvas.set_size_request(600, 400)
        self.canvas.set_events(
            gtk.gdk.BUTTON_PRESS_MASK |
            gtk.gdk.KEY_PRESS_MASK |
            gtk.gdk.KEY_RELEASE_MASK
            )
        self.canvas.set_flags(gtk.HAS_FOCUS | gtk.CAN_FOCUS)
        self.canvas.grab_focus()

        def keypress(widget, event):
            print('key press')

        def buttonpress(widget, event):
            print('button press')

        self.canvas.connect('key_press_event', keypress)
        self.canvas.connect('button_press_event', buttonpress)

        def onselect(xmin, xmax):
            print(xmin, xmax)

        span = SpanSelector(self.axis, onselect, 'horizontal', useblit=False,
                            rectprops=dict(alpha=0.5, facecolor='red'))

        self['vboxMain'].pack_start(self.canvas, True, True)
        self['vboxMain'].show()

        # below is optional if you want the navigation toolbar
        self.navToolbar = NavigationToolbar(self.canvas, self['windowMain'])
        self.navToolbar.lastDir = '/var/tmp/'
        self['vboxMain'].pack_start(self.navToolbar)
        self.navToolbar.show()

        sep = gtk.HSeparator()
        sep.show()
        self['vboxMain'].pack_start(sep, True, True)

        self['vboxMain'].reorder_child(self['buttonClickMe'], -1)

    def __getitem__(self, key):
        return self.widgets.get_widget(key)

widgets = WidgetsWrapper()
gtk.main()
