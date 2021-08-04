"""
Common code for GTK3 and GTK4 backends.
"""

import logging

import matplotlib as mpl
from matplotlib import cbook
from matplotlib.backend_bases import (
    _Backend,
)

# The GTK3/GTK4 backends will have already called `gi.require_version` to set
# the desired GTK.
from gi.repository import Gio, Gtk


_log = logging.getLogger(__name__)

backend_version = "%s.%s.%s" % (
    Gtk.get_major_version(), Gtk.get_minor_version(), Gtk.get_micro_version())

# Placeholder
_application = None


def _shutdown_application(app):
    # The application might prematurely shut down if Ctrl-C'd out of IPython,
    # so close all windows.
    for win in app.get_windows():
        win.destroy()
    # The PyGObject wrapper incorrectly thinks that None is not allowed, or we
    # would call this:
    # Gio.Application.set_default(None)
    # Instead, we set this property and ignore default applications with it:
    app._created_by_matplotlib = True
    global _application
    _application = None


def _create_application():
    global _application

    if _application is None:
        app = Gio.Application.get_default()
        if app is None or getattr(app, '_created_by_matplotlib'):
            # display_is_valid returns False only if on Linux and neither X11
            # nor Wayland display can be opened.
            if not mpl._c_internal_utils.display_is_valid():
                raise RuntimeError('Invalid DISPLAY variable')
            _application = Gtk.Application.new('org.matplotlib.Matplotlib3',
                                               Gio.ApplicationFlags.NON_UNIQUE)
            # The activate signal must be connected, but we don't care for
            # handling it, since we don't do any remote processing.
            _application.connect('activate', lambda *args, **kwargs: None)
            _application.connect('shutdown', _shutdown_application)
            _application.register()
            cbook._setup_new_guiapp()
        else:
            _application = app

    return _application


class _BackendGTK(_Backend):
    @staticmethod
    def mainloop():
        global _application
        if _application is None:
            return

        try:
            _application.run()  # Quits when all added windows close.
        finally:
            # Running after quit is undefined, so create a new one next time.
            _application = None
