#!/usr/bin/env python
from matplotlib.numerix import arange, sin, pi

import matplotlib
matplotlib.use('GTKAgg')  # or 'GTK'

# swith comments for gtk over gtkagg
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas

# or NavigationToolbar for classic
from matplotlib.backends.backend_gtk import NavigationToolbar2GTK as NavigationToolbar

from matplotlib.axes import Subplot
from matplotlib.figure import Figure
import gtk

win = gtk.Window()
win.set_title("Embedding in GTK")
win.connect("destroy", gtk.mainquit)
win.set_border_width(5)

vbox = gtk.VBox(spacing=3)
vbox.show()
win.add(vbox)

fig = Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

ax.plot(t,s)


canvas = FigureCanvas(fig)  # a gtk.DrawingArea
canvas.show()
vbox.pack_start(canvas)

toolbar = NavigationToolbar(canvas)
toolbar.show()
vbox.pack_start(toolbar, gtk.FALSE, gtk.FALSE)

#buttonQuit = gtk.Button('Quit')
#buttonQuit.connect('clicked', gtk.mainquit)
#buttonQuit.show()
#vbox.pack_start(buttonQuit)


if __name__=='__main__':
    win.show()
#    gtk.mainloop()
    gtk.main()
