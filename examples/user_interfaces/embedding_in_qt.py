#! /usr/bin/env python

# embedding_in_qt.py --- Simple Qt application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

import sys, os, random
from qt import *

from numpy import arange, sin, pi
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# This seems to be what PyQt expects, according to the examples shipped in
# its distribution.
TRUE  = 1
FALSE = 0

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

# Note: color-intensive applications may require a different color allocation
# strategy.
#QApplication.setColorSpec(QApplication.NormalColor)
app = QApplication(sys.argv)

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, self.fig)
        self.reparent(parent, QPoint(0, 0))

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def sizeHint(self):
        w, h = self.get_width_height()
        return QSize(w, h)

    def minimumSizeHint(self):
        return QSize(10, 10)


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QTimer(self, "canvas update timer")
        QObject.connect(timer, SIGNAL("timeout()"), self.update_figure)
        timer.start(1000, FALSE)

    def compute_initial_figure(self):
         self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = [ random.randint(0, 10) for i in xrange(4) ]

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

        self.help_menu = QPopupMenu(self)
        self.menuBar().insertSeparator()
        self.menuBar().insertItem('&Help', self.help_menu)

        self.help_menu.insertItem('&About', self.about)

        self.main_widget = QWidget(self, "Main widget")

        l = QVBoxLayout(self.main_widget)
        sc = MyStaticMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(sc)
        l.addWidget(dc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().message("All hail matplotlib!", 2000)

    def fileQuit(self):
        qApp.exit(0)

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QMessageBox.about(self, "About %s" % progname,
u"""%(prog)s version %(version)s
Copyright \N{COPYRIGHT SIGN} 2005 Florent Rougon

This program is a simple example of a Qt application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation."""
                          % {"prog": progname, "version": progversion})


aw = ApplicationWindow()
aw.setCaption("%s" % progname)
qApp.setMainWidget(aw)
aw.show()
sys.exit(qApp.exec_loop())
