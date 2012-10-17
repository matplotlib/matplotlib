#!/usr/bin/env python
"""
demonstrate NavigationToolbar with GTK3 accessed via pygobject
"""

from gi.repository import Gtk

from matplotlib.figure import Figure
from numpy import arange, sin, pi
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar

win = Gtk.Window()
win.connect("delete-event", Gtk.main_quit )
win.set_default_size(400,300)
win.set_title("Embedding in GTK")

f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(1,1,1)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)
a.plot(t,s)

vbox = Gtk.VBox()
win.add(vbox)

# Add canvas to vbox
canvas = FigureCanvas(f)  # a Gtk.DrawingArea
vbox.pack_start(canvas, True, True, 0)

# Create toolbar
toolbar = NavigationToolbar(canvas, win)
vbox.pack_start(toolbar, False, False, 0)

win.show_all()
Gtk.main()

