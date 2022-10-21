"""
================
GTK4 Spreadsheet
================

Example of embedding Matplotlib in an application and interacting with a
treeview to store data.  Double click on an entry to update plot data.
"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk

from matplotlib.backends.backend_gtk4agg import FigureCanvas  # or gtk4cairo.

from numpy.random import random
from matplotlib.figure import Figure


class DataManager(Gtk.ApplicationWindow):
    num_rows, num_cols = 20, 10

    data = random((num_rows, num_cols))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_default_size(600, 600)

        self.set_title('GtkListStore demo')

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, homogeneous=False,
                       spacing=8)
        self.set_child(vbox)

        label = Gtk.Label(label='Double click a row to plot the data')
        vbox.append(label)

        sw = Gtk.ScrolledWindow()
        sw.set_has_frame(True)
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sw.set_hexpand(True)
        sw.set_vexpand(True)
        vbox.append(sw)

        model = self.create_model()
        self.treeview = Gtk.TreeView(model=model)
        self.treeview.connect('row-activated', self.plot_row)
        sw.set_child(self.treeview)

        # Matplotlib stuff
        fig = Figure(figsize=(6, 4), constrained_layout=True)

        self.canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)
        vbox.append(self.canvas)
        ax = fig.add_subplot()
        self.line, = ax.plot(self.data[0, :], 'go')  # plot the first row

        self.add_columns()

    def plot_row(self, treeview, path, view_column):
        ind, = path  # get the index into data
        points = self.data[ind, :]
        self.line.set_ydata(points)
        self.canvas.draw()

    def add_columns(self):
        for i in range(self.num_cols):
            column = Gtk.TreeViewColumn(str(i), Gtk.CellRendererText(), text=i)
            self.treeview.append_column(column)

    def create_model(self):
        types = [float] * self.num_cols
        store = Gtk.ListStore(*types)
        for row in self.data:
            # Gtk.ListStore.append is broken in PyGObject, so insert manually.
            it = store.insert(-1)
            store.set(it, {i: val for i, val in enumerate(row)})
        return store


def on_activate(app):
    manager = DataManager(application=app)
    manager.show()


app = Gtk.Application(application_id='org.matplotlib.examples.GTK4Spreadsheet')
app.connect('activate', on_activate)
app.run()
