from Numeric import arange, sin, pi

import matplotlib
matplotlib.use('GTK')
from matplotlib.backends.backend_gtk import FigureCanvasGTK, NavigationToolbar

from matplotlib.axes import Subplot
from matplotlib.figure import Figure
import gtk

win = gtk.Window()
win.set_name("Embedding in GTK")
win.connect("destroy", gtk.mainquit)
win.set_border_width(5)

vbox = gtk.VBox(spacing=3)
win.add(vbox)
vbox.show()

fig = Figure(figsize=(5,4), dpi=100)
ax = Subplot(fig, 111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

ax.plot(t,s)
fig.add_axis(ax)

canvas = FigureCanvasGTK(fig)  # a gtk.DrawingArea
canvas.show()
vbox.pack_start(canvas)


toolbar = NavigationToolbar(canvas, win)
toolbar.show()
vbox.pack_start(toolbar, gtk.FALSE, gtk.FALSE)

buttonQuit = gtk.Button('Quit')
buttonQuit.connect('clicked', gtk.mainquit)
buttonQuit.show()
vbox.pack_start(buttonQuit)



if __name__=='__main__':
    win.show()
    gtk.mainloop()
