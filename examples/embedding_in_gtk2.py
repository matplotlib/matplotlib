#!/usr/bin/env python
"""
show how to add a matplotlib FigureCanvasGTK or FigureCanvasGTKAgg widget and
a toolbar to a gtk.Window
"""

from matplotlib.axes import Subplot
from matplotlib.figure import Figure
from matplotlib.numerix import arange, sin, pi
from matplotlib.widgets import HorizontalSpanSelector

# switch comments for gtk over gtkagg
#from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
# or NavigationToolbar for classic
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar

import gtk

win = gtk.Window()
win.connect("destroy", lambda x: gtk.main_quit())
win.set_default_size(400,300)
win.set_title("Embedding in GTK")

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
vbox.pack_start(toolbar, False, False)

def onselect(*args): pass
span = HorizontalSpanSelector(ax, onselect, useblit=False,
                              rectprops=dict(alpha=0.5, facecolor='red') )

win.show_all()
gtk.main()
