#!/usr/bin/env python
# show how to add a matplotlib FigureCanvasGTK widget to a gtk.Window

from matplotlib.numerix import arange, sin, pi

import matplotlib
matplotlib.use('GTKAgg')  # or 'GTK'

from matplotlib.axes import Subplot

# switch comments for gtk over gtkagg
#from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.figure import Figure

import gtk

win = gtk.Window()
win.set_default_size(400,300)
win.set_title("Embedding in GTK")
win.connect("destroy", lambda x: gtk.main_quit())

f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)
a.plot(t,s)

canvas = FigureCanvas(f)  # a gtk.DrawingArea
win.add(canvas)

win.show_all()
gtk.main()
