"""Module for managing if we are "interactive" or not."""
from matplotlib import is_interactive as _is_interact, interactive as _interactive


def is_interactive():
    """
    Return whether plots are updated after every plotting command.

    The interactive mode is mainly useful if you build plots from the command
    line and want to see the effect of each command while you are building the
    figure.

    In interactive mode:

    - newly created figures will be shown immediately;
    - figures will automatically redraw on change;
    - `mpl_gui.show` will not block by default.
    - `mpl_gui.FigureContext` will not block on ``__exit__`` by default.

    In non-interactive mode:

    - newly created figures and changes to figures will not be reflected until
      explicitly asked to be;
    - `mpl_gui.show` will block by default.
    - `mpl_gui.FigureContext` will block on ``__exit__`` by default.

    See Also
    --------
    ion : Enable interactive mode.
    ioff : Disable interactive mode.
    show : Show all figures (and maybe block).
    """
    return _is_interact()


class _IoffContext:
    """
    Context manager for `.ioff`.

    The state is changed in ``__init__()`` instead of ``__enter__()``. The
    latter is a no-op. This allows using `.ioff` both as a function and
    as a context.
    """

    def __init__(self):
        self.wasinteractive = is_interactive()
        _interactive(False)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if self.wasinteractive:
            _interactive(True)
        else:
            _interactive(False)


class _IonContext:
    """
    Context manager for `.ion`.

    The state is changed in ``__init__()`` instead of ``__enter__()``. The
    latter is a no-op. This allows using `.ion` both as a function and
    as a context.
    """

    def __init__(self):
        self.wasinteractive = is_interactive()
        _interactive(True)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.wasinteractive:
            _interactive(False)
        else:
            _interactive(True)


def ioff():
    """
    Disable interactive mode.

    See `.is_interactive` for more details.

    See Also
    --------
    ion : Enable interactive mode.
    is_interactive : Whether interactive mode is enabled.
    show : Show all figures (and maybe block).

    Notes
    -----
    For a temporary change, this can be used as a context manager::

        # if interactive mode is on
        # then figures will be shown on creation
        mg.ion()
        # This figure will be shown immediately
        fig = mg.figure()

        with mg.ioff():
            # interactive mode will be off
            # figures will not automatically be shown
            fig2 = mg.figure()
            # ...

    To enable usage as a context manager, this function returns an
    ``_IoffContext`` object. The return value is not intended to be stored
    or accessed by the user.
    """
    return _IoffContext()


def ion():
    """
    Enable interactive mode.

    See `.is_interactive` for more details.

    See Also
    --------
    ioff : Disable interactive mode.
    is_interactive : Whether interactive mode is enabled.
    show : Show all figures (and maybe block).

    Notes
    -----
    For a temporary change, this can be used as a context manager::

        # if interactive mode is off
        # then figures will not be shown on creation
        mg.ioff()
        # This figure will not be shown immediately
        fig = mg.figure()

        with mg.ion():
            # interactive mode will be on
            # figures will automatically be shown
            fig2 = mg.figure()
            # ...

    To enable usage as a context manager, this function returns an
    ``_IonContext`` object. The return value is not intended to be stored
    or accessed by the user.
    """
    return _IonContext()
