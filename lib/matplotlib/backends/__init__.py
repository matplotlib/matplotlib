from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import importlib
import logging
import os
import sys
import traceback

import matplotlib
from matplotlib import rcParams
from matplotlib.backend_bases import _Backend


_log = logging.getLogger(__name__)


def _get_current_event_loop():
    """Return the currently running event loop if any, or "headless", or None.

    "headless" indicates that no event loop can be started.

    Returns
    -------
    Optional[str]
        A value in {"qt5", "qt4", "gtk3", "gtk2", "tk", "headless", None}
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
    gtk = sys.modules.get("gtk")
    # gtk3 will also insert gtk into sys.modules :/
    if not Gtk and gtk and gtk.main_level():
        return "gtk2"
    tkinter = sys.modules.get("tkinter") or sys.modules.get("Tkinter")
    if tkinter and any(frame.f_code.co_filename == tkinter.__file__
                       and frame.f_code.co_name == "mainloop"
                       for frame in sys._current_frames().values()):
        return "tk"
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return "headless"


def pylab_setup(name=None):
    '''return new_figure_manager, draw_if_interactive and show for pyplot

    This provides the backend-specific functions that are used by
    pyplot to abstract away the difference between interactive backends.

    Parameters
    ----------
    name : str or List[str], optional
        The name of the backend to use.  If a list of backends, they will be
        tried in order until one successfully loads.  If ``None``, use
        ``rcParams['backend']``.

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

    Raises
    ------
    ImportError
        If a backend cannot be loaded because a different event loop has
        already started, or if a third-party backend fails to import.

    '''
    if name is None:
        name = matplotlib.rcParams["backend"]

    if not isinstance(name, six.string_types):
        for n in name:
            try:
                _log.info("Trying to load backend %s.", n)
                return pylab_setup(n)
            except ImportError as exc:
                _log.info("Loading backend %s failed: %s", n, exc)
        else:
            raise ValueError("No suitable backend among {}".format(name))

    backend_name = (name[9:] if name.startswith("module://")
                    else "matplotlib.backends.backend_{}".format(name.lower()))

    backend_mod = importlib.import_module(backend_name)
    Backend = type(str("Backend"), (_Backend,), vars(backend_mod))
    _log.info("Loaded backend %s version %s.", name, Backend.backend_version)

    required_event_loop = Backend.required_event_loop
    current_event_loop = _get_current_event_loop()
    if (current_event_loop and required_event_loop
            and current_event_loop != required_event_loop):
        raise ImportError(
            "Cannot load backend {!r} which requires the {!r} event loop, as "
            "the {!r} event loop is currently running".format(
                name, required_event_loop, current_event_loop))

    rcParams["backend"] = name

    # need to keep a global reference to the backend for compatibility
    # reasons. See https://github.com/matplotlib/matplotlib/issues/6092
    global backend
    backend = name

    # We want to get functions out of a class namespace and call them *without
    # the first argument being an instance of the class*.  This works directly
    # on Py3.  On Py2, we need to remove the check that the first argument be
    # an instance of the class.  The only relevant case is if `.im_self` is
    # None, in which case we need to use `.im_func` (if we have a bound method
    # (e.g. a classmethod), everything is fine).
    def _dont_check_first_arg(func):
        return (func.im_func if getattr(func, "im_self", 0) is None
                else func)

    return (backend_mod,
            _dont_check_first_arg(Backend.new_figure_manager),
            _dont_check_first_arg(Backend.draw_if_interactive),
            _dont_check_first_arg(Backend.show))
