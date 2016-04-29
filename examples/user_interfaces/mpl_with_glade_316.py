from gi.repository import Gtk

from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from numpy import arange, sin, pi
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas


class Window1Signals(object):
    def on_window1_destroy(self, widget):
        Gtk.main_quit()


def main():
    builder = Gtk.Builder()
    builder.add_objects_from_file("mpl_with_glade_316.glade", ("window1", ""))
    builder.connect_signals(Window1Signals())
    window = builder.get_object("window1")
    sw = builder.get_object("scrolledwindow1")

    # Start of Matplotlib specific code
    figure = Figure(figsize=(8, 6), dpi=71)
    axis = figure.add_subplot(111)
    t = arange(0.0, 3.0, 0.01)
    s = sin(2*pi*t)
    axis.plot(t, s)

    axis.set_xlabel('time [s]')
    axis.set_ylabel('voltage [V]')

    canvas = FigureCanvas(figure)  # a Gtk.DrawingArea
    canvas.set_size_request(800, 600)
    sw.add_with_viewport(canvas)
    # End of Matplotlib specific code

    window.show_all()
    Gtk.main()

if __name__ == "__main__":
    main()
