#!/usr/bin/env python
from matplotlib.numerix import arange, sin, pi

import matplotlib
matplotlib.use('GTK')

from matplotlib.axes import Subplot
from matplotlib.backends.backend_gtk import FigureCanvasGTK
from matplotlib.figure import Figure

import gtk

win = gtk.Window()
win.set_name("Embedding in GTK")
win.connect("destroy", gtk.mainquit)
win.set_border_width(5)

vbox = gtk.VBox(spacing=3)
win.add(vbox)
vbox.show()

f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

a.plot(t,s)


canvas = FigureCanvasGTK(f)  # a gtk.DrawingArea
canvas.show()
vbox.pack_start(canvas)

button = gtk.Button('Quit')
button.connect('clicked', lambda b: gtk.mainquit())
button.show()
vbox.pack_start(button)

win.show()
gtk.mainloop()
