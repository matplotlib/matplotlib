#!/usr/bin/env python
from matplotlib.numerix import arange, sin, pi

import matplotlib
matplotlib.use('GTKAgg')  # or 'GTK'

from matplotlib.axes import Subplot

# swith comments for gtk over gtkagg
#from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.figure import Figure

import gtk

win = gtk.Window()
win.set_title("Embedding in GTK")
win.connect("destroy", gtk.mainquit)

vbox = gtk.VBox(spacing=3)
win.add(vbox)

f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

a.plot(t,s)


canvas = FigureCanvas(f)  # a gtk.DrawingArea
vbox.pack_start(canvas)

#button = gtk.Button('Quit')
#button.connect('clicked', lambda b: gtk.mainquit())
#button.show()
#vbox.pack_start(button)

win.show_all()
#gtk.mainloop()
gtk.main()
