import matplotlib
matplotlib.use('GTK')

from matplotlib.backends.backend_gtk import Figure, NavigationToolbar

from matplotlib.axes import Subplot
import Numeric as numpy
import gtk

win = gtk.Window()
win.set_name("Embedding in GTK")
win.connect("destroy", gtk.mainquit)
win.set_border_width(5)

vbox = gtk.VBox(spacing=3)
win.add(vbox)
vbox.show()

f = Figure(figsize=(5,4), dpi=100)
a = Subplot(f, 111)
t = numpy.arange(0.0,3.0,0.01)
s = numpy.sin(2*numpy.pi*t)

a.plot(t,s)
f.add_axis(a)
f.show()
vbox.pack_start(f)

toolbar = NavigationToolbar(f, win)
toolbar.show()
vbox.pack_start(toolbar, gtk.FALSE, gtk.FALSE)

button = gtk.Button('Quit')
button.connect('clicked', lambda b: gtk.mainquit())
button.show()
vbox.pack_start(button)

win.show()
gtk.mainloop()
