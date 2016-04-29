"""
demonstrate adding a FigureCanvasGTK3Agg widget to a Gtk.ScrolledWindow
using GTK3 accessed via pygobject
"""

from gi.repository import Gtk

from matplotlib.figure import Figure
from numpy import arange, sin, pi
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

win = Gtk.Window()
win.connect("delete-event", Gtk.main_quit)
win.set_default_size(400, 300)
win.set_title("Embedding in GTK")

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0, 3.0, 0.01)
s = sin(2*pi*t)
a.plot(t, s)

sw = Gtk.ScrolledWindow()
win.add(sw)
# A scrolled window border goes outside the scrollbars and viewport
sw.set_border_width(10)

canvas = FigureCanvas(f)  # a Gtk.DrawingArea
canvas.set_size_request(800, 600)
sw.add_with_viewport(canvas)

win.show_all()
Gtk.main()
