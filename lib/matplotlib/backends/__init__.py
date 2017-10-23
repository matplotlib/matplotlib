from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

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
    '''return new_figure_manager, draw_if_interactive and show for pyplot

    This provides the backend-specific functions that are used by
    pyplot to abstract away the difference between interactive backends.

    Parameters
    ----------
    name : str, optional
        The name of the backend to use.  If `None`, falls back to
        ``matplotlib.get_backend()`` (which return ``rcParams['backend']``)

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

    '''
    # Import the requested backend into a generic module object
    if name is None:
        name = matplotlib.get_backend()
    backend_name = (name[9:] if name.startswith("module://")
                    else "matplotlib.backends.backend_{}".format(name.lower()))

    backend_mod = importlib.import_module(backend_name)
    Backend = type(str("Backend"), (_Backend,), vars(backend_mod))
    _log.info('backend %s version %s', name, Backend.backend_version)

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
