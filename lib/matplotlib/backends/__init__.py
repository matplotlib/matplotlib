import importlib
import logging
import traceback

import matplotlib
from matplotlib.backend_bases import _Backend

_log = logging.getLogger(__name__)

backend = matplotlib.get_backend()
_backend_loading_tb = "".join(
    line for line in traceback.format_stack()
    # Filter out line noise from importlib line.
    if not line.startswith('  File "<frozen importlib._bootstrap'))


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
