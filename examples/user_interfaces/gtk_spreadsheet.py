#!/usr/bin/env python
"""
Example of embedding matplotlib in an application and interacting with
a treeview to store data.  Double click on an entry to update plot
data

"""
import pygtk
pygtk.require('2.0')
import gtk
from gtk import gdk

import matplotlib
matplotlib.use('GTKAgg')  # or 'GTK'
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas

from numpy.random import random
from matplotlib.figure import Figure


class DataManager(gtk.Window):
    numRows, numCols = 20,10

    data = random((numRows, numCols))

    def __init__(self):
        gtk.Window.__init__(self)
        self.set_default_size(600, 600)
        self.connect('destroy', lambda win: gtk.main_quit())

        self.set_title('GtkListStore demo')
        self.set_border_width(8)

        vbox = gtk.VBox(False, 8)
        self.add(vbox)

        label = gtk.Label('Double click a row to plot the data')

        vbox.pack_start(label, False, False)

        sw = gtk.ScrolledWindow()
        sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        sw.set_policy(gtk.POLICY_NEVER,
                      gtk.POLICY_AUTOMATIC)
        vbox.pack_start(sw, True, True)

        model = self.create_model()

        self.treeview = gtk.TreeView(model)
        self.treeview.set_rules_hint(True)


        # matplotlib stuff
        fig = Figure(figsize=(6,4))

        self.canvas = FigureCanvas(fig)  # a gtk.DrawingArea
        vbox.pack_start(self.canvas, True, True)
        ax = fig.add_subplot(111)
        self.line, = ax.plot(self.data[0,:], 'go')  # plot the first row

        self.treeview.connect('row-activated', self.plot_row)
        sw.add(self.treeview)

        self.add_columns()

        self.add_events(gdk.BUTTON_PRESS_MASK |
                        gdk.KEY_PRESS_MASK|
                        gdk.KEY_RELEASE_MASK)


    def plot_row(self, treeview, path, view_column):
        ind, = path  # get the index into data
        points = self.data[ind,:]
        self.line.set_ydata(points)
        self.canvas.draw()


    def add_columns(self):
        for i in range(self.numCols):
            column = gtk.TreeViewColumn('%d'%i, gtk.CellRendererText(), text=i)
            self.treeview.append_column(column)


    def create_model(self):
        types = [float]*self.numCols
        store = gtk.ListStore(*types)

        for row in self.data:
            store.append(row)
        return store


manager = DataManager()
manager.show_all()
gtk.main()
