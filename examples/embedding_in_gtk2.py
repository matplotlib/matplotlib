#!/usr/bin/env python
# show how to add a matplotlib FigureCanvasGTK widget and a toolbar to a gtk.Window

from matplotlib.numerix import arange, sin, pi

import matplotlib
matplotlib.use('GTKAgg')  # or 'GTK'

# switch comments for gtk over gtkagg
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas

# or NavigationToolbar for classic
from matplotlib.backends.backend_gtk import NavigationToolbar2GTK as NavigationToolbar

from matplotlib.axes import Subplot
from matplotlib.figure import Figure

import gtk

win = gtk.Window()
win.set_title("Embedding in GTK")
win.connect("destroy", lambda x: gtk.main_quit())

vbox = gtk.VBox()
win.add(vbox)

fig = Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

ax.plot(t,s)

canvas = FigureCanvas(fig)  # a gtk.DrawingArea
vbox.pack_start(canvas)

toolbar = NavigationToolbar(canvas, win)
vbox.pack_start(toolbar, gtk.FALSE, gtk.FALSE)

win.show_all()
gtk.main()
