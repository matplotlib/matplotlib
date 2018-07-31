"""
===============
GTK Spreadsheet
===============

Example of embedding matplotlib in an application and interacting with
a treeview to store data.  Double click on an entry to update plot
data

"""
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gtk, Gdk

from matplotlib.backends.backend_gtk3agg import FigureCanvas
# from matplotlib.backends.backend_gtk3cairo import FigureCanvas

from numpy.random import random
from matplotlib.figure import Figure


class DataManager(Gtk.Window):
    numRows, numCols = 20, 10

    data = random((numRows, numCols))

    def __init__(self):
        Gtk.Window.__init__(self)
        self.set_default_size(600, 600)
        self.connect('destroy', lambda win: Gtk.main_quit())

        self.set_title('GtkListStore demo')
        self.set_border_width(8)

        vbox = Gtk.VBox(False, 8)
        self.add(vbox)

        label = Gtk.Label('Double click a row to plot the data')

        vbox.pack_start(label, False, False, 0)

        sw = Gtk.ScrolledWindow()
        sw.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
        sw.set_policy(Gtk.PolicyType.NEVER,
                      Gtk.PolicyType.AUTOMATIC)
        vbox.pack_start(sw, True, True, 0)

        model = self.create_model()

        self.treeview = Gtk.TreeView(model)
        self.treeview.set_rules_hint(True)

        # matplotlib stuff
        fig = Figure(figsize=(6, 4))

        self.canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
        vbox.pack_start(self.canvas, True, True, 0)
        ax = fig.add_subplot(111)
        self.line, = ax.plot(self.data[0, :], 'go')  # plot the first row

        self.treeview.connect('row-activated', self.plot_row)
        sw.add(self.treeview)

        self.add_columns()

        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.KEY_PRESS_MASK |
                        Gdk.EventMask.KEY_RELEASE_MASK)

    def plot_row(self, treeview, path, view_column):
        ind, = path  # get the index into data
        points = self.data[ind, :]
        self.line.set_ydata(points)
        self.canvas.draw()

    def add_columns(self):
        for i in range(self.numCols):
            column = Gtk.TreeViewColumn(str(i), Gtk.CellRendererText(), text=i)
            self.treeview.append_column(column)

    def create_model(self):
        types = [float]*self.numCols
        store = Gtk.ListStore(*types)

        for row in self.data:
            store.append(tuple(row))
        return store


manager = DataManager()
manager.show_all()
Gtk.main()
