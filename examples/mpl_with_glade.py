import matplotlib
matplotlib.use('GTK')

from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from matplotlib.backends.backend_gtk import FigureCanvasGTK, NavigationToolbar

from Numeric import arange, sin, pi
import gtk
import gtk.glade


def simple_msg(msg, parent=None, title=None):
    dialog = gtk.MessageDialog(
        parent         = None,
        type           = gtk.MESSAGE_INFO,
        buttons        = gtk.BUTTONS_OK,
        message_format = msg)
    if parent is not None:
        dialog.set_transient_for(parent)
    if title is not None:
        dialog.set_title(title)
    dialog.show()
    dialog.run()
    dialog.destroy()
    return None



class GladeHandlers:
    def on_buttonClickMe_clicked(event):
        simple_msg('Nothing to say, really',
                   parent=widgets['windowMain'],
                   title='Thanks!')

class WidgetsWrapper:
    def __init__(self):
        self.widgets = gtk.glade.XML('mpl_with_glade.glade')
        self.widgets.signal_autoconnect(GladeHandlers.__dict__)

        self.figure = Figure(figsize=(8,6), dpi=72)
        self.axis = Subplot(self.figure, 111)
        self.figure.add_axis(self.axis)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)
        self.axis.plot(t,s)
        self.axis.set_xlabel('time (s)')
        self.axis.set_ylabel('voltage')

        self.canvas = FigureCanvasGTK(self.figure) # a gtk.DrawingArea
        self.canvas.show()
        self['vboxMain'].pack_start(self.canvas, gtk.TRUE, gtk.TRUE)
        self['vboxMain'].show()
        
        # below is optional if you want the navigation toolbar
        self.navToolbar = NavigationToolbar(self.canvas, self['windowMain'])
        self.navToolbar.lastDir = '/var/tmp/'
        self['vboxMain'].pack_start(self.navToolbar)
        self.navToolbar.show()

        sep = gtk.HSeparator()
        sep.show()
        self['vboxMain'].pack_start(sep, gtk.TRUE, gtk.TRUE)


        self['vboxMain'].reorder_child(self['buttonClickMe'],-1)

    def __getitem__(self, key):
        return self.widgets.get_widget(key)

widgets = WidgetsWrapper()
gtk.mainloop ()
