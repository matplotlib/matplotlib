"""
===========
Embed in Qt
===========

Simple Qt application embedding Matplotlib canvases.  This program will work
equally well using any Qt binding (PyQt6, PySide6, PyQt5, PySide2).  The
binding can be selected by setting the :envvar:`QT_API` environment variable to
the binding name, or by first importing it.
"""

import sys
import time

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        layout.addWidget(NavigationToolbar(static_canvas, self))
        layout.addWidget(static_canvas)

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        layout.addWidget(NavigationToolbar(dynamic_canvas, self))

        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")

        self._dynamic_ax = dynamic_canvas.figure.subplots()
        # Set up a Line2D.
        self.xdata = np.linspace(0, 10, 101)
        self._update_ydata()
        self._line, = self._dynamic_ax.plot(self.xdata, self.ydata)
        # The below two timers must be attributes of self, so that the garbage
        # collector won't clean them after we finish with __init__...

        # The data retrieval may be fast as possible (Using QRunnable could be
        # even faster).
        self.data_timer = dynamic_canvas.new_timer(1)
        self.data_timer.add_callback(self._update_ydata)
        self.data_timer.start()
        # Drawing at 50Hz should be fast enough for the GUI to feel smooth, and
        # not too fast for the GUI to be overloaded with events that need to be
        # processed while the GUI element is changed.
        self.drawing_timer = dynamic_canvas.new_timer(20)
        self.drawing_timer.add_callback(self._update_canvas)
        self.drawing_timer.start()

    def _update_ydata(self):
        # Shift the sinusoid as a function of time.
        self.ydata = np.sin(self.xdata + time.time())

    def _update_canvas(self):
        self._line.set_data(self.xdata, self.ydata)
        # It should be safe to use the synchronous draw() method for most drawing
        # frequencies, but it is safer to use draw_idle().
        self._line.figure.canvas.draw_idle()


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
