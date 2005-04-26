#! /usr/bin/env python
# Copyright (C) 2005 Florent Rougon
#
# This file is an example program for matplotlib. It may be used,
# distributed and modified without limitation.
import sys, os, random
from qt import *

from matplotlib.numerix import arange, sin, pi

# The QApplication has to be created before backend_qt is imported, otherwise
# it will create one itself.
QApplication.setColorSpec(QApplication.NormalColor)
app = QApplication(sys.argv)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

TRUE  = 1
FALSE = 0

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class FloMplCanvas(FigureCanvas):
    """Ultimately, this is a qWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width,height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.compute_initial_figure()
        
        FigureCanvas.__init__(self, self.fig)
        self.reparent(parent, QPoint(0, 0))

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def sizeHint(self):
        w, h = self.fig.get_width_height()
        return QSize(w, h)

    def minimumSizeHint(self):
        return QSize(10, 10)


class FloCanvas1(FloMplCanvas):
    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)

class FloCanvas2(FloMplCanvas):
    def __init__(self, *args, **kwargs):
        FloMplCanvas.__init__(self, *args, **kwargs)
        timer = QTimer(self, "Canvas update timer")
        QObject.connect(timer, SIGNAL("timeout()"), self.update_figure)
        timer.start(1000, FALSE)

    def compute_initial_figure(self):
         self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        l = []
        for i in range(4):
            l.append(random.randint(0, 10))

        self.axes.lines = []
        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()


class ApplicationWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self, None,
                             "application main window",
                             Qt.WType_TopLevel | Qt.WDestructiveClose)

        self.file_menu = QPopupMenu(self)
        self.file_menu.insertItem('&Quit', self.fileQuit, Qt.CTRL + Qt.Key_Q)
        self.menuBar().insertItem('&File', self.file_menu)

#        self.file_menu.insertSeparator()

        self.help_menu = QPopupMenu(self)
        self.menuBar().insertSeparator()
        self.menuBar().insertItem('&Help', self.help_menu)

        self.help_menu.insertItem('&About', self.about)

        self.main_widget = QWidget(self, "Main widget")

        l = QVBoxLayout(self.main_widget)
        c1 = FloCanvas1(self.main_widget, width=5, height=4, dpi=100)
        c2 = FloCanvas2(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(c1)
        l.addWidget(c2)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().message("Ready", 2000)

    def fileQuit(self):
        qApp.exit(0)

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QMessageBox.about(self, "About %s" % progname,
"""%(prog)s version %(version)s
Copyright (c) 2005 Florent Rougon

This program is a monitor for the Conrad Charge Manager 2010
""" % {"prog": progname, "version": progversion})


def main():
    mw = ApplicationWindow()
    mw.setCaption("%s" % progname)
    qApp.setMainWidget(mw)
    mw.show()
    sys.exit(app.exec_loop())


if __name__ == "__main__": main()
