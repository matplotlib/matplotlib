"""
===================================
Embedding in a Paraview application
===================================


`ParaView <https://www.paraview.org/>`__ is a powerful 3D
visualization tool.  It has both a Qt client application and python
integration.

This example shows how to embed an interactive Matplotlib figure into
the Qt UI.

ParaView (via VTK) embeds a python interpreter into the primary
application.  This code is intended to be imported into the
interpreter running within ParaView, find the main window and
inject a dock widget with a Matplotlib Figure in it.  To use this,
in the ParaView python shell ::

 >> import sys
 >> sys.path.append(path_to_examples)
 >> import embed_in_paraview_sgksip as eip
 >> fig = eip.make_dock_figure()
 >> ax = fig.add_subplot(111)  # or ax = fig.subplots()
 >> ln, = ax.plot(range(5))

"""


from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT


def get_main_window(target_name="pqClientMainWindow"):
    """Find the main window

    This assumes that the object name on the main window (where we
    will be able to add a dock widget) has the object name

    Parameters
    ----------
    target_name : str
        The same of the MainWindow object of interest

    Returns
    -------
    QtWidgets.QMainWindow

    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError(
            "No running QApplication found. " "May not be running inside of ParaView"
        )

    return next(
        iter(
            w
            for w in app.topLevelWidgets()
            if (
                isinstance(w, QtWidgets.QMainWindow)
                and w.objectName() == "pqClientMainWindow"
            )
        )
    )


def make_dock_figure():
    """Create and inject a Figure into the UI

    The user is responsible for the Figure and widget life-cycle.

    Returns
    -------
    f : Figure
    mpl_dock : QtWidgets.QDockWidget

    """
    # create the Figure
    f = Figure()
    f.stale_callback = lambda obj, val: obj.canvas.draw_idle()

    # set up the Qt widgets we need to inject into the UI
    mpl_dock = QtWidgets.QDockWidget("Matplotlib")
    layout = QtWidgets.QVBoxLayout()
    qw = QtWidgets.QWidget()
    qw.setLayout(layout)
    mpl_dock.setWidget(qw)

    # create the Canvas and toolbar
    cv = FigureCanvas(f)
    mpl_toolbar = NavigationToolbar2QT(cv, qw)

    # add to the layout
    layout.addWidget(cv)
    layout.addWidget(mpl_toolbar)

    # find the main window and add the dock
    mw = get_main_window()
    mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, mpl_dock)

    return f, mpl_dock
