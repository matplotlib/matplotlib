"""
=================
Embedding in GTK4
=================

Demonstrate adding a FigureCanvasGTK4Agg widget to a Gtk.ScrolledWindow using
GTK4 accessed via pygobject.
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from matplotlib.backends.backend_gtk4agg import (
    FigureCanvasGTK4Agg as FigureCanvas)
from matplotlib.figure import Figure
import numpy as np


def on_activate(app):
    win = Gtk.ApplicationWindow(application=app)
    win.set_default_size(400, 300)
    win.set_title("Embedding in GTK4")

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    t = np.arange(0.0, 3.0, 0.01)
    s = np.sin(2*np.pi*t)
    ax.plot(t, s)

    # A scrolled margin goes outside the scrollbars and viewport.
    sw = Gtk.ScrolledWindow(margin_top=10, margin_bottom=10,
                            margin_start=10, margin_end=10)
    win.set_child(sw)

    canvas = FigureCanvas(fig)  # a Gtk.DrawingArea
    canvas.set_size_request(800, 600)
    sw.set_child(canvas)

    win.show()


app = Gtk.Application(application_id='org.matplotlib.examples.EmbeddingInGTK4')
app.connect('activate', on_activate)
app.run(None)
