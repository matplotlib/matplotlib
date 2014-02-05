#!/usr/bin/env python

# embedding_in_qt4.py --- Simple Qt4 application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

from __future__ import unicode_literals, division
import sys
import os

from matplotlib.backends.qt4_compat import QtCore, QtGui
import numpy as np
from matplotlib.backends._backend_qt4agg import (FigureCanvas,
                                                 FigureManager,
                                                 new_figure_manager)

from matplotlib.figure import Figure

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


def demo_plot(ax):
    """
    Plots sin waves with random period
    """
    k = np.random.random_integers(1, 10)
    th = np.linspace(0, 2*np.pi, 1024)
    ax.plot(th, np.sin(th * k))


class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        # QT boiler plate to set up main window
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtGui.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtGui.QWidget(self)

        l = QtGui.QVBoxLayout(self.main_widget)
        button = QtGui.QPushButton("make window A")
        button.clicked.connect(self.new_window_hard_way)
        l.addWidget(button)

        buttonB = QtGui.QPushButton("make window B")
        buttonB.clicked.connect(self.new_window_easy_way)
        l.addWidget(buttonB)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self._figs = []
        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def new_window_hard_way(self):
        # make a figure
        fig = Figure()
        # make a canvas
        canvas = FigureCanvas(fig)
        # make a manager from the canvas
        manager = FigureManager(canvas, 1)
        # grab an axes in the figure
        ax = fig.gca()
        # plot some demo code
        demo_plot(ax)
        # show the window
        manager.show()

    def new_window_easy_way(self):
        # make a new manager
        manager = new_figure_manager(2)
        # grab a reference to the figure
        fig = manager.canvas.figure
        # get an axes object in the figure
        ax = fig.gca()
        # plot some demo-data
        demo_plot(ax)
        # show the window
        manager.show()

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtGui.QMessageBox.about(self, "About",
"""embedding_in_qt4.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2013 Thomas Caswell

This program is a simple example of a Qt4 application making use of
the FigureManager class.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation."""
)


qApp = QtGui.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % progname)
aw.show()
sys.exit(qApp.exec_())
#qApp.exec_()
