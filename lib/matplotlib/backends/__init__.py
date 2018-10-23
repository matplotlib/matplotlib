import importlib
import logging
import os
import sys
import traceback

import matplotlib
from matplotlib import cbook
from matplotlib.backend_bases import _Backend

_log = logging.getLogger(__name__)


# NOTE: plt.switch_backend() (called at import time) will add a "backend"
# attribute here for backcompat.


def _get_running_interactive_framework():
    """
    Return the interactive framework whose event loop is currently running, if
    any, or "headless" if no event loop can be started, or None.

    Returns
    -------
    Optional[str]
        One of the following values: "qt5", "qt4", "gtk3", "wx", "tk",
        "macosx", "headless", ``None``.
    """
    QtWidgets = (sys.modules.get("PyQt5.QtWidgets")
                 or sys.modules.get("PySide2.QtWidgets"))
    if QtWidgets and QtWidgets.QApplication.instance():
        return "qt5"
    QtGui = (sys.modules.get("PyQt4.QtGui")
             or sys.modules.get("PySide.QtGui"))
    if QtGui and QtGui.QApplication.instance():
        return "qt4"
    Gtk = (sys.modules.get("gi.repository.Gtk")
           or sys.modules.get("pgi.repository.Gtk"))
    if Gtk and Gtk.main_level():
        return "gtk3"
    wx = sys.modules.get("wx")
    if wx and wx.GetApp():
        return "wx"
    tkinter = sys.modules.get("tkinter")
    if tkinter:
        for frame in sys._current_frames().values():
            while frame:
                if frame.f_code == tkinter.mainloop.__code__:
                    return "tk"
                frame = frame.f_back
    if 'matplotlib.backends._macosx' in sys.modules:
        if sys.modules["matplotlib.backends._macosx"].event_loop_is_running():
            return "macosx"
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return "headless"
    return None


@cbook.deprecated("3.0")
def pylab_setup(name=None):
    """
    Return new_figure_manager, draw_if_interactive and show for pyplot.

    This provides the backend-specific functions that are used by pyplot to
    abstract away the difference between backends.

    Parameters
    ----------
    name : str, optional
        The name of the backend to use.  If `None`, falls back to
        ``matplotlib.get_backend()`` (which return :rc:`backend`).

    Returns
    -------
    backend_mod : module
        The module which contains the backend of choice

    new_figure_manager : function
        Create a new figure manager (roughly maps to GUI window)

    draw_if_interactive : function
        Redraw the current figure if pyplot is interactive

    show : function
        Show (and possibly block) any unshown figures.
    """
    # Import the requested backend into a generic module object.
    if name is None:
        name = matplotlib.get_backend()
    backend_name = (name[9:] if name.startswith("module://")
                    else "matplotlib.backends.backend_{}".format(name.lower()))
    backend_mod = importlib.import_module(backend_name)
    # Create a local Backend class whose body corresponds to the contents of
    # the backend module.  This allows the Backend class to fill in the missing
    # methods through inheritance.
    Backend = type("Backend", (_Backend,), vars(backend_mod))

    # Need to keep a global reference to the backend for compatibility reasons.
    # See https://github.com/matplotlib/matplotlib/issues/6092
    global backend
    backend = name

    _log.debug('backend %s version %s', name, Backend.backend_version)
    return (backend_mod,
            Backend.new_figure_manager,
            Backend.draw_if_interactive,
            Backend.show)
