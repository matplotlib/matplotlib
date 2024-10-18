"""====================================
Combining with VTK in a Qt interface
====================================

`VTK <https://www.vtk.org/>`_ is a library for 3D visualization.

This example shows how to embed an interactive Matplotlib figure along
side a VTK renderer in a Qt UI.

TODO:

 - use more meaningful example data
 - change cylinder to isosurface view of data

"""


import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from PyQt5 import QtWidgets, QtCore

import vtk

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def make_vtk_plane(ren):

    # create source
    source = vtk.vtkPlaneSource()
    source.SetPoint1(-10, 10, 0)
    source.SetPoint2(10, -10, 0)
    source.SetOrigin(-10, -10, 0)
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())

    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # assign actor to the renderer
    ren.AddActor(actor)

    return source, mapper, actor


def make_vtk_cylinder(ren):

    # Create source
    source = vtk.vtkCylinderSource()
    source.SetCenter(0, 0, 0)
    source.SetRadius(10.0)
    source.SetHeight(10)

    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren.AddActor(actor)

    return source, mapper, actor


class AppForm(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.data = np.ones((21, 10, 10))
        for j in range(len(self.data)):
            self.data[j] *= j - 10

        self.create_main_frame()

    def create_main_frame(self):
        self.main_frame = QtWidgets.QWidget()

        self.fig = Figure((5.0, 4.0), dpi=100)
        ax = self.fig.subplots()
        self.im = ax.imshow(self.data[10], vmin=-10, vmax=10)
        self.fig.colorbar(self.im, ax=ax)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.setFocus()

        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.canvas)  # the matplotlib canvas
        vbox.addWidget(self.mpl_toolbar)

        # make the vtk widget
        self.frame = QtWidgets.QFrame()
        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        vbox.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.cyl = make_vtk_cylinder(self.ren)
        self.pln = make_vtk_plane(self.ren)

        self.ren.ResetCamera()

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(-10)
        self.slider.setMaximum(10)
        vbox.addWidget(self.slider)

        def move_plane(x):
            self.pln[2].SetPosition(0, 0, x)
            self.im.set_array(self.data[int(x) + 10])
            self.vtkWidget.update()
            self.canvas.draw_idle()

        self.slider.valueChanged.connect(move_plane)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

        self.show()
        self.iren.Initialize()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    form = AppForm()
    sys.exit(app.exec_())
